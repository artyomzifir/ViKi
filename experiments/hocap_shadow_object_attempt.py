"""Fail-closed G10_2 object-relative Shadow Hand IK diagnostic.

Uses final fitted hand joints and the object track; rejects infeasible floor
and self-collision constraints instead of producing a display-only plan.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from viki.benchmark.hocap_provenance import require_hand_fit_clean
from experiments.shadow_hand_fit import fit_shadow, HUMAN_FROM_SHADOW
from viki.retarget.frames import calibration_to_robot, calibration_rotation_to_robot, robot_to_calibration
from viki.retarget.robots import normalize_robot
from viki.retarget.solver import PinocchioKinematics, BatchWeights, BatchOptions, solve_trajectory, trajectory_constraint_margins

parser=argparse.ArgumentParser()
parser.add_argument('--solver',choices=['framewise','batch'],default='framewise')
parser.add_argument('--grasp',choices=['hold','anatomical'],default='hold')
parser.add_argument('--closure-m',type=float,default=.005)
args=parser.parse_args()
if not 0 <= args.closure_m <= .012: parser.error('closure must be between 0 and 12 mm')
episode=Path('data/datasets/hocap_demo_object_v4/g10_object2_190')
provenance= require_hand_fit_clean(episode)
source=episode/'cln.npz'
track_path=episode/'intermediates/segmentation/hocap-oracle-v1/object_models.npz'
seed_path=episode/'plan.h5'
model_path=Path('data/_hocap_demo_source/ur10_shadow.urdf')
with np.load(source,allow_pickle=False) as clean, np.load(track_path,allow_pickle=False) as track, h5py.File(seed_path) as seed:
    assert str(clean['coordinate_frame'])=='hocap_tag_1'
    timestamps=np.asarray(clean['timestamps'],dtype=np.int64)
    human_p=np.asarray(clean['hand_fit_positions'],dtype=float)
    human_r=np.asarray(clean['hand_fit_rotations'],dtype=float)
    human_q=np.asarray(clean['hand_fit_joint_angles'][:,7:],dtype=float)
    ids=list(map(int,track['object_ids']))
    col=ids.index(2)
    obj_p=np.asarray(track['translation_world'][:,col],dtype=float)
    obj_r=np.asarray(track['rotation_world_object'][:,col],dtype=float)
    contact=np.asarray(track['contact'][:,col],dtype=bool)
    seed_q=np.asarray(seed['q'],dtype=float)
    base_p=np.asarray(seed['base_position_calibration'],dtype=float)
    base_rpy=np.asarray(seed['base_rpy_deg_calibration'],dtype=float)
    assert np.array_equal(timestamps,seed['timestamps'][:])

if len(timestamps)!=190 or not np.isfinite(human_q).all() or not np.isfinite(human_p).all():
    raise ValueError('expected finite 190-frame final fitted hand')
contact_ids=np.flatnonzero(contact)
if not len(contact_ids) or np.any(np.diff(contact_ids)>1):
    raise ValueError('object 2 contact phase is absent or discontinuous')
first,last=int(contact_ids[0]),int(contact_ids[-1])
initial_z=float(np.median(obj_p[:first,2]))
lift=np.flatnonzero((obj_p[:,2]-initial_z)>0.01)
if not len(lift): raise ValueError('no observed object lift')
anchor=int(lift[0])
carry_start=max(first+1,anchor)
carry_end=int(np.flatnonzero((obj_p[:,2]-initial_z)>0.01)[-1])
if not (first<carry_start<carry_end<last):
    raise ValueError('invalid object-relative phase bounds')
local_p=obj_r[anchor].T@(human_p[anchor]-obj_p[anchor])
local_r=obj_r[anchor].T@human_r[anchor]
object_p=obj_p+np.einsum('tij,j->ti',obj_r,local_p)
object_r=np.einsum('tij,jk->tik',obj_r,local_r)
weights=np.zeros(len(timestamps),dtype=float)
weights[first:carry_start+1]=np.linspace(0,1,carry_start-first+1)
weights[carry_start:carry_end+1]=1
weights[carry_end:last+1]=np.linspace(1,0,last-carry_end+1)
target_p=(1-weights[:,None])*human_p+weights[:,None]*object_p
target_r=[]
for i,w in enumerate(weights):
    pair=Rotation.from_matrix(np.stack([human_r[i],object_r[i]]))
    target_r.append(Slerp([0,1],pair)([w]).as_matrix()[0])
target_r=np.stack(target_r)
# Human palm axes: +x fingers, +y little finger, +z palmar.
# Shadow palm axes: +z fingers, -x little finger, -y palmar.
human_from_shadow=HUMAN_FROM_SHADOW
target_r=np.einsum('tij,jk->tik',target_r,human_from_shadow)

model=pin.buildModelFromUrdf(str(model_path))
collision=pin.buildGeomFromUrdf(model,str(model_path),pin.GeometryType.COLLISION)
visual=pin.buildGeomFromUrdf(model,str(model_path),pin.GeometryType.VISUAL)
robot=SimpleNamespace(model=model,collision_model=collision,visual_model=visual)
shadow_names=[str(n) for n in model.names if str(n).startswith('rh_')]
with np.load(source, allow_pickle=False) as clean:
    joint_map=fit_shadow(clean,model,collision)
if args.grasp == 'hold':
    # Keep one object-relative grip during carry; continuing human articulation
    # on a different-sized robot otherwise opens the measured contact patch.
    data=model.createData()
    palm_id=model.getFrameId('rh_palm',pin.FrameType.BODY)
    tips=[model.getFrameId('rh_'+c+'tip',pin.FrameType.BODY) for c in ['th','ff','mf','rf','lf']]
    indices=[model.idx_qs[model.getJointId(n)] for n in shadow_names]
    neutral=pin.neutral(model)
    x0=np.array([joint_map[n][anchor] for n in shadow_names])
    def tip_positions(x):
        q=neutral.copy();q[indices]=x
        pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
        palm=data.oMf[palm_id]
        return np.stack([palm.rotation.T@(data.oMf[f].translation-palm.translation) for f in tips])
    initial=tip_positions(x0)
    center=target_r[anchor].T@(obj_p[anchor]-target_p[anchor])
    # The object's canonical z is its long axis; close toward that axis,
    # retaining each finger's position along the object.
    axis=target_r[anchor].T@obj_r[anchor][:,2]
    radial=initial-center
    radial-=np.outer(radial@axis,axis)
    goal=initial-args.closure_m*radial/np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-8)
    grasp=least_squares(lambda x:np.r_[(tip_positions(x)-goal).ravel()*100,(x-x0)*.1],x0,
        bounds=(model.lowerPositionLimit[indices]+1e-7,model.upperPositionLimit[indices]-1e-7),max_nfev=80).x
    for k,n in enumerate(shadow_names):
        joint_map[n]=(1-weights)*joint_map[n]+weights*grasp[k]
config=normalize_robot('ur10')
kin=PinocchioKinematics(robot,'rh_palm',collision_pairs=8,collision_min_distance_m=0.001,
    actuated_joint_names=config.joint_names,actuated_position_limits=config.position_limits,
    reference_q=np.zeros(6),passive_joint_positions=joint_map,gripper_prefix='rh_',
    floor_normal=(0,0,1),floor_offset_m=0.0)
solver=BatchOptions(max_iterations=40,collision_enabled=True)
kin.velocity_limit*=1.0
target_robot_p=calibration_to_robot(target_p,base_p,base_rpy)
target_robot_r=calibration_rotation_to_robot(target_r,base_rpy)
print('Shadow IK',len(timestamps),'contact',first,last,'carry',carry_start,carry_end,'anchor',anchor,flush=True)
# Restore palm pose/floor feasibility before the hard whole-trajectory solve.
for t in range(len(seed_q)):
    kin.set_frame(t)
    prior=seed_q[t] if t==0 else seed_q[t-1]
    def residual(q):
        p,r=kin.pose(q)
        floor=kin.floor_margins(q)
        return np.r_[(p-target_robot_p[t])*100, Rotation.from_matrix(target_robot_r[t]@r.T).as_rotvec()*10,
                     np.minimum(floor-.003,0)*1000,(q-prior)*.02]
    seed_q[t]=least_squares(residual,prior,bounds=(kin.q_min,kin.q_max),max_nfev=80).x
print('arm pose initialization complete',flush=True)
seed_margins=trajectory_constraint_margins(kin,seed_q,float(np.median(np.diff(timestamps)))/1e6)
print('seed margins',seed_margins,flush=True)
attempt_path=episode/'shadow_object_attempt.json'
attempt={'schema':'viki_shadow_object_attempt_v1','episode':episode.name,'hand_source':'final_optimized_hand_fit',
    'object_id':2,'contact_window':[first,last+1],'object_relative_window':[carry_start,carry_end+1],
    'anchor_frame':anchor,'seed_margins':seed_margins,'finger_fit':'segment_direction_fit_with_sampled_clearance_objective',
    'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,track_path,seed_path,model_path)}}
attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
try:
    if args.solver == 'batch':
        result=solve_trajectory(kin,target_robot_p,target_robot_r,np.ones(len(timestamps)),
            float(np.median(np.diff(timestamps)))/1e6,BatchWeights(),solver,initial_q=seed_q,
            report=lambda **progress: print(progress,flush=True))
    else:
        # Per-frame IK is accepted only after checking the entire saved path,
        # including velocities between every pair of consecutive frames.
        if min(seed_margins.values()) < -1e-7:
            raise RuntimeError(f'framewise trajectory failed validation: {seed_margins}')
        achieved=[]
        for t,q in enumerate(seed_q):
            kin.set_frame(t);achieved.append(kin.pose(q)[0])
        result=SimpleNamespace(q=seed_q,achieved_position=np.array(achieved),
                               status='framewise_ik_whole_trajectory_validated')
except RuntimeError as exc:
    attempt['status']='infeasible'
    attempt['error']=str(exc)
    attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
    raise
error=np.linalg.norm(result.achieved_position-target_robot_p,axis=1)*1000
margin=seed_margins if args.solver=='framewise' else trajectory_constraint_margins(kin,result.q,float(np.median(np.diff(timestamps)))/1e6)
report={'schema':'viki_shadow_object_retarget_v1','episode':episode.name,
    'hand_source':'final_optimized_hand_fit','object_id':2,
    'object_mode':'tracked_object_relative_wrist_with_contact_and_lift_phases',
    'finger_grasp_mode':args.grasp,'fingertip_closure_m':args.closure_m if args.grasp=='hold' else 0.,
    'contact_window':[first,last+1],'object_relative_window':[carry_start,carry_end+1],
    'anchor_frame':anchor,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,track_path,seed_path,model_path)},
    'seed_role':'UR10 initial joint guess only; target generated from final hand fit and tracked object',
    'joint_map':'22 Shadow joints fitted to FK of final optimized human hand; robot phalanx lengths and URDF bounds; sampled finger clearance objective',
    'finger_fit':'segment_direction_fit_with_sampled_clearance_objective','solver_status':result.status,'position_median_mm':float(np.median(error)),
    'position_mean_mm':float(np.mean(error)),
    'self_collision_clearance_m':.001,'scene_object_collision_constraints':False,
    'position_rmse_mm':float(np.sqrt(np.mean(error**2))),
    'position_p95_mm':float(np.percentile(error,95)),'constraint_margins':margin}
# Retiming preserves all solved poses while respecting arm AND finger speeds.
source_timestamps=timestamps.copy()
commands=np.c_[result.q,np.stack([joint_map[n] for n in shadow_names],axis=1)]
all_names=list(config.joint_names)+shadow_names
speed_limits=np.array([model.velocityLimit[model.idx_vs[model.getJointId(n)]] for n in all_names])
from experiments.shadow_timing import retime_commands
timestamps,all_velocity_margin=retime_commands(timestamps,commands,speed_limits)
report['original_duration_sec']=float((source_timestamps[-1]-source_timestamps[0])/1e6)
report['retimed_duration_sec']=float((timestamps[-1]-timestamps[0])/1e6)
report['all_joint_velocity_margin_rad_s']=all_velocity_margin
report['timing']='per-segment retiming to URDF arm and finger velocity limits; source frame indices preserved'
out=episode/'shadow_object_plan.npz'
np.savez_compressed(out,timestamps=timestamps,source_timestamps=source_timestamps,arm_q=result.q.astype(np.float32),
    shadow_joint_names=np.asarray(shadow_names),shadow_q=np.stack([joint_map[n] for n in shadow_names],axis=1).astype(np.float32),
    target_position=target_p.astype(np.float32),target_rotation=target_r.astype(np.float32),
    achieved_position=robot_to_calibration(result.achieved_position,base_p,base_rpy).astype(np.float32),
    base_position=base_p,base_rpy_deg=base_rpy,report_json=json.dumps(report))
attempt['status']='solved'
attempt['plan']=str(out)
attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

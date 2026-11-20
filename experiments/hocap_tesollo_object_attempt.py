"""Fail-closed G10_2 object-relative Tesollo DG5F IK diagnostic.

Uses final fitted hand joints and the object track; rejects infeasible floor
and self-collision constraints instead of producing a display-only plan.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import time
import os
import resource
import hashlib
import json
import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from viki.benchmark.hocap_provenance import require_hand_fit_clean
from experiments.tesollo_hand_fit import fit_tesollo, HUMAN_FROM_TESOLLO
from experiments.prepare_tesollo_urdf import PALM_ROLL_RAD
from viki.retarget.frames import calibration_to_robot, calibration_rotation_to_robot, robot_to_calibration
from viki.retarget.robots import normalize_robot
from viki.retarget.solver import PinocchioKinematics, BatchWeights, BatchOptions, solve_trajectory, trajectory_constraint_margins

def total_cpu_time():
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+children.ru_utime+children.ru_stime

_profile=[]
_last_wall=time.perf_counter(); _last_cpu=total_cpu_time()
def stage(label):
    global _last_wall,_last_cpu
    wall=time.perf_counter(); cpu=total_cpu_time()
    _profile.append(dict(stage=label,wall_sec=wall-_last_wall,cpu_sec=cpu-_last_cpu))
    print("PROFILE",label,round(wall-_last_wall,3),round(cpu-_last_cpu,3),flush=True)
    _last_wall=wall;_last_cpu=cpu

parser=argparse.ArgumentParser()
parser.add_argument('--solver',choices=['framewise','batch'],default='framewise')
parser.add_argument('--constraint-workers',type=int,default=min(8, os.cpu_count() or 1),help='forked CPU processes for exact whole-trajectory constraint checks')
parser.add_argument('--grasp',choices=['hold','anatomical'],default='hold')
parser.add_argument('--closure-m',type=float,default=.005)
parser.add_argument('--diagnostic-plan',action='store_true',help='save a visibly labeled infeasible plan for GUI diagnosis')
args=parser.parse_args()
if not 0 <= args.closure_m <= .012: parser.error('closure must be between 0 and 12 mm')
if args.constraint_workers < 1: parser.error('constraint workers must be positive')
if args.constraint_workers > (os.cpu_count() or 1): parser.error('constraint workers exceed logical CPU count')
episode=Path('data/datasets/hocap_demo_object_v4/g10_object2_190')
provenance= require_hand_fit_clean(episode)
source=episode/'cln.npz'
track_path=episode/'intermediates/segmentation/hocap-oracle-v1/object_models.npz'
seed_path=episode/'plan.h5'
model_path=Path('data/_hocap_demo_source/ur10_tesollo.urdf')
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

stage('input_load')
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
# Collision-distance scan against the real DG5F and object meshes: the
# unshifted side approach hits with finger 5 before the thumb reaches it.
# Move the approach above and across the object, then carry the offset in
# the object's frame. This remains a derived path from final hand-fit data.
pregrasp_offset=np.array([.030,.030,.020])
grip_offset=np.array([.020,.040,.020])
local_grip_offset=obj_r[anchor].T@grip_offset
pose_offset=np.zeros_like(target_p)
for i in range(len(timestamps)):
    if i < 10: continue
    if i < 23:
        u=(i-10)/(23-10);u=u*u*(3-2*u)
        pose_offset[i]=u*pregrasp_offset
    elif i < anchor:
        u=(i-23)/(anchor-23)
        pose_offset[i]=(1-u)*pregrasp_offset+u*grip_offset
    elif i <= last:
        pose_offset[i]=obj_r[i]@local_grip_offset
    elif i <= last+12:
        pose_offset[i]=(1-(i-last)/12)*(obj_r[last]@local_grip_offset)
target_p+=pose_offset
target_r=[]
for i,w in enumerate(weights):
    pair=Rotation.from_matrix(np.stack([human_r[i],object_r[i]]))
    target_r.append(Slerp([0,1],pair)([w]).as_matrix()[0])
target_r=np.stack(target_r)
# Human palm axes: +x fingers, +y little finger, +z palmar.
# Tesollo palm axes: +z fingers, -y little finger, +x palmar.
human_from_tesollo=HUMAN_FROM_TESOLLO
# Match the selected DG5F twist around the visible UR10 flange axis.
# Gallery orientation A was selected after contact-geometry comparison.
target_r=np.einsum('tij,jk->tik',target_r,human_from_tesollo)
target_r=np.einsum('tij,jk->tik',target_r,Rotation.from_euler('z',PALM_ROLL_RAD).as_matrix())

model=pin.buildModelFromUrdf(str(model_path))
collision=pin.buildGeomFromUrdf(model,str(model_path),pin.GeometryType.COLLISION)
visual=pin.buildGeomFromUrdf(model,str(model_path),pin.GeometryType.VISUAL)
robot=SimpleNamespace(model=model,collision_model=collision,visual_model=visual)
tesollo_names=[str(n) for n in model.names if str(n).startswith('rj_dg_')]
stage('model_geometry_load')
with np.load(source, allow_pickle=False) as clean:
    joint_map=fit_tesollo(clean,model,collision)
stage('final_hand_anatomical_fit')
if args.grasp == 'hold':
    # Keep one object-relative grip during carry; continuing human articulation
    # on a different-sized robot otherwise opens the measured contact patch.
    data=model.createData()
    palm_id=model.getFrameId('rl_dg_palm',pin.FrameType.BODY)
    tips=[model.getFrameId(f'rl_dg_{c}_tip',pin.FrameType.BODY) for c in range(1,6)]
    indices=[model.idx_qs[model.getJointId(n)] for n in tesollo_names]
    neutral=pin.neutral(model)
    x0=np.array([joint_map[n][anchor] for n in tesollo_names])
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
    for k,n in enumerate(tesollo_names):
        joint_map[n]=(1-weights)*joint_map[n]+weights*grasp[k]
stage('object_grasp_closure')
config=normalize_robot('ur10')
kin=PinocchioKinematics(robot,'rl_dg_palm',collision_pairs=8,collision_min_distance_m=0.0,
    actuated_joint_names=config.joint_names,actuated_position_limits=config.position_limits,
    reference_q=np.zeros(6),passive_joint_positions=joint_map,gripper_prefix='rj_dg_',
    floor_normal=(0,0,1),floor_offset_m=0.0)
solver=BatchOptions(max_iterations=40,collision_enabled=True)
kin.velocity_limit*=1.0
target_robot_p=calibration_to_robot(target_p,base_p,base_rpy)
target_robot_r=calibration_rotation_to_robot(target_r,base_rpy)
print('Tesollo IK',len(timestamps),'contact',first,last,'carry',carry_start,carry_end,'anchor',anchor,flush=True)
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
stage('arm_pose_initialization')
np.savez_compressed(episode/'tesollo_seed_diagnostic.npz',arm_q=seed_q,**{f'hand_{n}':joint_map[n] for n in tesollo_names})
print('arm pose initialization complete',flush=True)
seed_margins=trajectory_constraint_margins(kin,seed_q,float(np.median(np.diff(timestamps)))/1e6,
    workers=args.constraint_workers)
stage('whole_trajectory_constraints')
print('seed margins',seed_margins,flush=True)
attempt_path=episode/'tesollo_object_attempt.json'
attempt={'schema':'viki_tesollo_object_attempt_v1','episode':episode.name,'hand_source':'final_optimized_hand_fit',
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
        hard_ok = min(seed_margins[k] for k in ('joint','floor','collision')) >= -1e-7
        if not hard_ok and not args.diagnostic_plan:
            raise RuntimeError(f'framewise trajectory failed validation: {seed_margins}')
        achieved=[]
        for t,q in enumerate(seed_q):
            kin.set_frame(t);achieved.append(kin.pose(q)[0])
        result=SimpleNamespace(q=seed_q,achieved_position=np.array(achieved),
                               status='framewise_ik_whole_trajectory_validated' if hard_ok else 'diagnostic_unvalidated_collision')
except RuntimeError as exc:
    attempt['status']='infeasible'
    attempt['error']=str(exc)
    attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
    raise
stage('solver')
error=np.linalg.norm(result.achieved_position-target_robot_p,axis=1)*1000
margin=seed_margins if args.solver=='framewise' else trajectory_constraint_margins(kin,result.q,float(np.median(np.diff(timestamps)))/1e6,
    workers=args.constraint_workers)
report={'schema':'viki_tesollo_object_retarget_v1','episode':episode.name,
    'hand_source':'final_optimized_hand_fit','object_id':2,
    'object_mode':'tracked_object_relative_wrist_with_contact_and_lift_phases',
    'finger_grasp_mode':args.grasp,'fingertip_closure_m':args.closure_m if args.grasp=='hold' else 0.,
    'approach_offset_world_m':pregrasp_offset.tolist(),'anchor_offset_world_m':grip_offset.tolist(),
    'offset_derivation':'collision-distance scan of DG5F geoms against tracked object-2 mesh',
    'contact_window':[first,last+1],'object_relative_window':[carry_start,carry_end+1],
    'anchor_frame':anchor,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,track_path,seed_path,model_path)},
    'seed_role':'UR10 initial joint guess only; target generated from final hand fit and tracked object',
    'constraint_workers':args.constraint_workers,'constraint_device':'cpu',
    'joint_map':'20 Tesollo joints fitted to FK of final optimized human hand; robot phalanx lengths and URDF bounds; sampled finger clearance objective',
    'finger_fit':'segment_direction_fit_with_sampled_clearance_objective','solver_status':result.status,'position_median_mm':float(np.median(error)),
    'position_mean_mm':float(np.mean(error)),
    'self_collision_clearance_m':0.0,'scene_object_collision_constraints':False,
    'position_rmse_mm':float(np.sqrt(np.mean(error**2))),
    'position_p95_mm':float(np.percentile(error,95)),'constraint_margins':margin}
# Retiming preserves all solved poses while respecting arm AND finger speeds.
source_timestamps=timestamps.copy()
commands=np.c_[result.q,np.stack([joint_map[n] for n in tesollo_names],axis=1)]
all_names=list(config.joint_names)+tesollo_names
speed_limits=np.array([model.velocityLimit[model.idx_vs[model.getJointId(n)]] for n in all_names])
from experiments.shadow_timing import retime_commands
timestamps,all_velocity_margin=retime_commands(timestamps,commands,speed_limits)
report['original_duration_sec']=float((source_timestamps[-1]-source_timestamps[0])/1e6)
report['retimed_duration_sec']=float((timestamps[-1]-timestamps[0])/1e6)
report['all_joint_velocity_margin_rad_s']=all_velocity_margin
if all_velocity_margin < -1e-7: raise RuntimeError(f'retimed velocity limit violated: {all_velocity_margin}')
report['timing']='per-segment retiming to URDF arm and finger velocity limits; source frame indices preserved'
out=episode/'tesollo_object_plan.npz'
np.savez_compressed(out,timestamps=timestamps,source_timestamps=source_timestamps,arm_q=result.q.astype(np.float32),
    tesollo_joint_names=np.asarray(tesollo_names),tesollo_q=np.stack([joint_map[n] for n in tesollo_names],axis=1).astype(np.float32),
    target_position=target_p.astype(np.float32),target_rotation=target_r.astype(np.float32),
    achieved_position=robot_to_calibration(result.achieved_position,base_p,base_rpy).astype(np.float32),
    base_position=base_p,base_rpy_deg=base_rpy,report_json=json.dumps(report))
stage('retiming_and_save')
profile_path=episode/'tesollo_object_profile.json'
profile_path.write_text(json.dumps({'schema':'viki_stage_profile_v1','stages':_profile,'total_wall_sec':sum(x['wall_sec'] for x in _profile),'total_cpu_sec':sum(x['cpu_sec'] for x in _profile),'process_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'constraint_workers':args.constraint_workers,'constraint_device':'cpu'},indent=2)+'\n')
attempt['status']='solved' if result.status == 'framewise_ik_whole_trajectory_validated' else 'diagnostic_unvalidated_collision'
attempt['plan']=str(out)
attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

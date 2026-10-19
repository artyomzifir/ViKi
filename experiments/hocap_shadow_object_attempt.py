"""Fail-closed G10_2 object-relative Shadow Hand IK diagnostic.

Uses final fitted hand joints and the object track; rejects infeasible floor
and self-collision constraints instead of producing a display-only plan.
"""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp
from viki.retarget.frames import calibration_to_robot, calibration_rotation_to_robot, robot_to_calibration
from viki.retarget.robots import normalize_robot
from viki.retarget.solver import PinocchioKinematics, BatchWeights, BatchOptions, solve_trajectory, trajectory_constraint_margins

episode=Path('data/datasets/hocap_demo_object_v4/g10_object2_190')
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
human_from_shadow=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=float)
target_r=np.einsum('tij,jk->tik',target_r,human_from_shadow)

joint_map={}
for finger,code,offset in [('index','FF',4),('middle','MF',8),('ring','RF',12),('pinky','LF',16)]:
    abd,mcp,pip,dip=human_q[:,offset:offset+4].T
    joint_map[f'rh_{code}J4']=abd
    joint_map[f'rh_{code}J3']=mcp
    joint_map[f'rh_{code}J2']=pip
    joint_map[f'rh_{code}J1']=dip
abd,cmc,mcp,ip=human_q[:,0:4].T
joint_map.update({'rh_THJ5':abd,'rh_THJ4':cmc,'rh_THJ3':np.zeros(len(timestamps)),
                  'rh_THJ2':mcp,'rh_THJ1':ip,'rh_LFJ5':np.zeros(len(timestamps))})
model=pin.buildModelFromUrdf(str(model_path))
collision=pin.buildGeomFromUrdf(model,str(model_path),pin.GeometryType.COLLISION)
visual=pin.buildGeomFromUrdf(model,str(model_path),pin.GeometryType.VISUAL)
robot=SimpleNamespace(model=model,collision_model=collision,visual_model=visual)
shadow_names=[str(n) for n in model.names if str(n).startswith('rh_')]
if set(joint_map)!=set(shadow_names):
    raise ValueError(f'Shadow mapping incomplete: {set(shadow_names)-set(joint_map)}')
clipped={}
for name,command in joint_map.items():
    jid=model.getJointId(name)
    lo=float(model.lowerPositionLimit[model.idx_qs[jid]])
    hi=float(model.upperPositionLimit[model.idx_qs[jid]])
    clipped[name]=int(np.sum((command<lo)|(command>hi)))
    joint_map[name]=np.clip(command,lo,hi)
config=normalize_robot('ur10')
kin=PinocchioKinematics(robot,'rh_palm',collision_pairs=8,collision_min_distance_m=0.005,
    actuated_joint_names=config.joint_names,actuated_position_limits=config.position_limits,
    reference_q=np.zeros(6),passive_joint_positions=joint_map,gripper_prefix='rh_',
    floor_normal=(0,0,1),floor_offset_m=0.0)
solver=BatchOptions(max_iterations=40,collision_enabled=True)
kin.velocity_limit*=1.0
target_robot_p=calibration_to_robot(target_p,base_p,base_rpy)
target_robot_r=calibration_rotation_to_robot(target_r,base_rpy)
print('Shadow IK',len(timestamps),'contact',first,last,'carry',carry_start,carry_end,'anchor',anchor,'clipped',clipped,flush=True)
seed_margins=trajectory_constraint_margins(kin,seed_q,float(np.median(np.diff(timestamps)))/1e6)
print('seed margins',seed_margins,flush=True)
attempt_path=episode/'shadow_object_attempt.json'
attempt={'schema':'viki_shadow_object_attempt_v1','episode':episode.name,'hand_source':'final_optimized_hand_fit',
    'object_id':2,'contact_window':[first,last+1],'object_relative_window':[carry_start,carry_end+1],
    'anchor_frame':anchor,'seed_margins':seed_margins,'clipped_samples':clipped,
    'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,track_path,seed_path,model_path)}}
attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
try:
    result=solve_trajectory(kin,target_robot_p,target_robot_r,np.ones(len(timestamps)),
        float(np.median(np.diff(timestamps)))/1e6,BatchWeights(),solver,initial_q=seed_q,
        report=None)
except RuntimeError as exc:
    attempt['status']='infeasible'
    attempt['error']=str(exc)
    attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
    raise
error=np.linalg.norm(result.achieved_position-target_robot_p,axis=1)*1000
margin=trajectory_constraint_margins(kin,result.q,float(np.median(np.diff(timestamps)))/1e6)
report={'schema':'viki_shadow_object_retarget_v1','episode':episode.name,
    'hand_source':'final_optimized_hand_fit','object_id':2,
    'object_mode':'tracked_object_relative_wrist_with_contact_and_lift_phases',
    'contact_window':[first,last+1],'object_relative_window':[carry_start,carry_end+1],
    'anchor_frame':anchor,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,track_path,seed_path,model_path)},
    'seed_role':'UR10 initial joint guess only; target generated from final hand fit and tracked object',
    'joint_map':'human fitted 20-DOF angles mapped and clipped to Shadow right-hand joints; thumb THJ3 and little-finger LFJ5 held neutral',
    'clipped_samples':clipped,'solver_status':result.status,'position_median_mm':float(np.median(error)),
    'position_rmse_mm':float(np.sqrt(np.mean(error**2))),
    'position_p95_mm':float(np.percentile(error,95)),'constraint_margins':margin}
out=episode/'shadow_object_plan.npz'
np.savez_compressed(out,timestamps=timestamps,arm_q=result.q.astype(np.float32),
    shadow_joint_names=np.asarray(shadow_names),shadow_q=np.stack([joint_map[n] for n in shadow_names],axis=1).astype(np.float32),
    target_position=target_p.astype(np.float32),target_rotation=target_r.astype(np.float32),
    achieved_position=robot_to_calibration(result.achieved_position,base_p,base_rpy).astype(np.float32),
    base_position=base_p,base_rpy_deg=base_rpy,report_json=json.dumps(report))
attempt['status']='solved'
attempt['plan']=str(out)
attempt_path.write_text(json.dumps(attempt,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

"""One-scene geometry check before the bounded object-pose placement pool."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.hocap_shadow_polisher import source,make_scene,rollout,EP
from experiments.hocap_shadow_palm_polisher import build_kinematics
from experiments.shadow_object_pose_placement_controls import object_to_palm,compose,inverse,support_center_height
from viki.retarget.frames import base_rotation

INPUT=Path('/app/rl-output/shadow_multifinger_followup_20261025/final_commands.npz')

def main():
 plan,track,names,_,*_=source()
 with np.load(INPUT,allow_pickle=False) as saved:commands=saved['commands'].copy()
 scene,table_z,_=make_scene(plan,track,names)
 report,trajectory=rollout(scene,plan,track,names,commands,table_z,full=True)
 kin=build_kinematics({**plan,'shadow_q':commands[:,6:]})
 br=base_rotation(plan['base_rpy_deg']);bp=np.asarray(plan['base_position'])
 def fk(frame,arm):
  kin.set_frame(frame);p,r=kin.pose(arm)
  return bp+br@p,br@r
 obs=90;end=125
 objp=trajectory['object_positions'][obs];objr=trajectory['object_rotations'][obs]
 palm_actual=fk(obs,trajectory['q_actual'][obs,:6]);rel=object_to_palm(objp,objr,*palm_actual)
 palm_src_end=fk(end,commands[end,:6]);rel_inv=inverse(*rel)
 nominal_obj_end=compose(*palm_src_end,*rel_inv)
 actual_obj_end=(trajectory['object_positions'][end],trajectory['object_rotations'][end])
 col=list(map(int,track['object_ids'])).index(2)
 target_rot=track['rotation_world_object'][-1,col]
 mesh=EP/'intermediates/segmentation/hocap-oracle-v1/meshes/object_2.obj'
 vertices=np.array([[float(y) for y in x.split()[1:4]] for x in mesh.read_text().splitlines() if x.startswith('v ')])
 target=np.array([*track['translation_world'][-1,col,:2],support_center_height(vertices,target_rot,table_z)])
 def pose(p,r):return {'position_m':np.asarray(p).tolist(),'euler_deg':Rotation.from_matrix(r).as_euler('xyz',degrees=True).tolist()}
 result={'observation_frame':obs,'endpoint_frame':end,'table_top_z_m':table_z,
         'mesh_local_min_m':vertices.min(axis=0).tolist(),'mesh_local_max_m':vertices.max(axis=0).tolist(),
         'observed_object':pose(objp,objr),'observed_palm':pose(*palm_actual),
         'object_to_palm':pose(*rel),'nominal_endpoint_object':pose(*nominal_obj_end),
         'actual_endpoint_object':pose(*actual_obj_end),'target_support_object':pose(target,target_rot),
         'nominal_endpoint_position_error_mm':float(np.linalg.norm(nominal_obj_end[0]-target)*1000),
         'actual_endpoint_position_error_mm':float(np.linalg.norm(actual_obj_end[0]-target)*1000),
         'nominal_endpoint_rotation_error_deg':float(np.rad2deg(Rotation.from_matrix(target_rot@nominal_obj_end[1].T).magnitude())),
         'actual_endpoint_rotation_error_deg':float(np.rad2deg(Rotation.from_matrix(target_rot@actual_obj_end[1].T).magnitude()))}
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()

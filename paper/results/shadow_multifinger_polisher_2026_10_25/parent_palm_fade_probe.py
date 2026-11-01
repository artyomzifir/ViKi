import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.hocap_shadow_polisher import source,make_scene,rollout,sha,PLAN,TRACK,URDF,EP
from experiments.hocap_shadow_palm_polisher import build_kinematics,baseline_fk_and_margins,export_commands
from experiments.shadow_palm_controls import object_frame_palm_target,solve_arm_pose,validate_ik_errors
from experiments.shadow_polisher_controls import command_margins
from viki.retarget.frames import base_rotation
out=Path('/app/rl-output/shadow_parent_palm_fade_20261025');out.mkdir(exist_ok=False)
p,t,n,base,lo,hi,vel=source();model,z,xml=make_scene(p,t,n);kin=build_kinematics(p);pp,pr,bf,bc=baseline_fk_and_margins(kin,base);oi=list(t['object_ids']).index(2);ro=np.einsum('ij,tjk->tik',base_rotation(p['base_rpy_deg']).T,t['rotation_world_object'][:,oi]);res=np.array([-.006,0,0,0,0,0]);h={'plan':sha(PLAN),'track':sha(TRACK),'urdf':sha(URDF),'meshes':{str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)}}
protocol={'reason':'isolate whether removing the palm correction at110..125 contributes to late loss/bump','fixed_fades':[[125,151],[140,160]],'rise':[10,23],'palm_residual':res.tolist(),'fingers':'source, unchanged','constraints':'same FK IK, floor selfcollision, segment velocity, physics and target gate','source_hashes':h};(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
x=np.arange(190,dtype=float)
def smooth(a,b):
 u=np.clip((x-a)/(b-a),0,1);return u*u*(3-2*u)
for i,(start,end) in enumerate(protocol['fixed_fades']):
 gate=smooth(10,23)*(1-smooth(start,end));cmd=base.copy();pe=[];re=[];floors=bf.copy();collisions=bc.copy();failure=None
 for frame in np.flatnonzero(gate>0):
  tp,tr=object_frame_palm_target(pp[frame],pr[frame],ro[frame],res,float(gate[frame]));q,_,_=solve_arm_pose(kin,int(frame),base[frame,:6],tp,tr);cmd[frame,:6]=q.astype(np.float32);kin.set_frame(int(frame));ap,ar=kin.pose(cmd[frame,:6]);pe.append(float(np.linalg.norm(tp-ap)));re.append(float(Rotation.from_matrix(tr@ar.T).magnitude()));floors[frame]=kin.floor_margins(cmd[frame,:6]).min();collisions[frame]=kin.collision_margins(cmd[frame,:6]).min()
  if not validate_ik_errors(pe[-1:],re[-1:]) or min(floors[frame],collisions[frame])< -1e-7:failure={'frame':int(frame),'floor':float(floors[frame]),'self':float(collisions[frame]),'poserr':pe[-1],'roterr':re[-1]};break
 pm,vm=command_margins(cmd,p['timestamps'],lo,hi,vel);row=dict(fade=[start,end],position_margin=pm,velocity_margin=vm,failure=failure)
 if failure is None and pm>=-1e-6 and vm>=-1e-6:
  export_commands(out/f'fade_{start}_{end}_commands.npz',cmd,p['timestamps'],n,h,res,i);metrics,traj=rollout(model,p,t,n,cmd,z,full=True);row['metrics']=metrics;np.savez_compressed(out/f'fade_{start}_{end}_trajectory.npz',**traj)
 rows.append(row);print(json.dumps(row),flush=True);(out/'results.json').write_text(json.dumps(rows,indent=2))

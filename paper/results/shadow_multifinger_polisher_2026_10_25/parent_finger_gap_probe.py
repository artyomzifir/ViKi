import json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from experiments.hocap_shadow_polisher import source,make_scene
p,t,n,c,*_=source();m,_,_=make_scene(p,t,n);obj=m.body('object_2').id;og=m.geom('object_geom_2').id;adr=np.array([m.jnt_qposadr[m.joint(x).id] for x in n]);oa=m.jnt_qposadr[m.body_jntadr[obj]];traj=np.load('/app/rl-output/shadow_palm_polisher_20261025/trajectory.npz',allow_pickle=False);out=[]
for frame in (42,60,90,110):
 d=mujoco.MjData(m);d.qpos[adr]=traj['q_actual'][frame];d.qpos[oa:oa+3]=traj['object_positions'][frame];d.qpos[oa+3:oa+7]=Rotation.from_matrix(traj['object_rotations'][frame]).as_quat(scalar_first=True);mujoco.mj_forward(m,d)
 for digit in ('ff','mf','rf','lf'):
  geoms=[i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith('rh_'+digit) and (m.geom_contype[i] or m.geom_conaffinity[i])]
  def dist():
   vals=[]
   for i in geoms:
    ft=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,i,og,.2,ft);vals.append((gap,i,ft.copy()))
   return min(vals,key=lambda x:x[0])
  gap,geom,ft=dist();q0=d.qpos.copy();der=[]
  for j in (4,3,2,1):
   name=f'rh_{digit.upper()}J{j}';idx=m.jnt_qposadr[m.joint(name).id];d.qpos[:]=q0;d.qpos[idx]+=.005;mujoco.mj_forward(m,d);der.append((dist()[0]-gap)/.005)
  d.qpos[:]=q0;mujoco.mj_forward(m,d)
  out.append(dict(frame=frame,digit=digit,closest_body=m.body(m.geom_bodyid[geom]).name,gap_mm=float(gap*1000),finger_to_object_world_mm=((ft[3:]-ft[:3])*1000).tolist(),distance_derivative_m_per_rad_j4_j3_j2_j1=der))
print(json.dumps(out,indent=2))

import json
import mujoco
import numpy as np
from experiments.hocap_shadow_polisher import source, make_scene
plan,track,names,cmd,*_=source(); m,_,_=make_scene(plan,track,names)
cmd=np.load('/app/rl-output/shadow_multifinger_followup_20261025/final_commands.npz',allow_pickle=False)['commands']
a=np.array([m.jnt_qposadr[m.joint(n).id] for n in names]); obj=m.body('object_2').id; wrist=m.body('wrist_3_link').id
out=[]
for stop in (60,90,98):
 d=mujoco.MjData(m); d.qpos[a]=cmd[0];d.ctrl[:]=cmd[0];mujoco.mj_forward(m,d)
 for _ in range(round(.3/m.opt.timestep)):mujoco.mj_step(m,d)
 for t,q in enumerate(cmd[:stop+1]):
  steps=round((plan['timestamps'][t]-plan['timestamps'][t-1])/1e6/m.opt.timestep) if t else 0
  for s in range(steps):
   d.ctrl[:]=cmd[max(0,t-1)]+(q-cmd[max(0,t-1)])*(s+1)/steps;mujoco.mj_step(m,d)
 start=d.xpos[obj].copy();rel0=d.xmat[wrist].reshape(3,3).T@(d.xpos[obj]-d.xpos[wrist]); samples=[]
 d.ctrl[:]=cmd[stop]
 for s in range(round(2/m.opt.timestep)):
  mujoco.mj_step(m,d)
  if (s+1)%1000==0:
   rel=d.xmat[wrist].reshape(3,3).T@(d.xpos[obj]-d.xpos[wrist]);forces=[0.,0.]
   for i,c in enumerate(d.contact):
    bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
    if 'object_2' not in bodies:continue
    other=bodies[1] if bodies[0]=='object_2' else bodies[0];f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
    if other.startswith('rh_th'):forces[0]+=max(0.,float(f[0]))
    elif other.startswith(('rh_ff','rh_mf','rh_rf','rh_lf')):forces[1]+=max(0.,float(f[0]))
   samples.append(dict(seconds=(s+1)*m.opt.timestep,z_drop_mm=float((start[2]-d.xpos[obj,2])*1000),wrist_relative_slip_mm=float(np.linalg.norm(rel-rel0)*1000),forces_n=forces))
 out.append(dict(stop_frame=stop,initial_height_m=float(start[2]),samples=samples))
print(json.dumps(out,indent=2))

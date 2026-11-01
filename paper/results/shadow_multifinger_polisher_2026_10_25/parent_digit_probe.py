import json
from pathlib import Path
import mujoco,numpy as np
from experiments.hocap_shadow_polisher import source,make_scene
p,t,n,baseline,*_=source();m,_,_=make_scene(p,t,n);obj=m.body('object_2').id;adr=np.array([m.jnt_qposadr[m.joint(x).id] for x in n]);digits=['th','ff','mf','rf','lf'];out=[]
selected=np.load('/app/rl-output/shadow_palm_polisher_20261025/final_commands.npz',allow_pickle=False)['commands']
for label,cmd in [('baseline',baseline),('palm_best',selected)]:
 d=mujoco.MjData(m);d.qpos[adr]=cmd[0];d.ctrl[:]=cmd[0];mujoco.mj_forward(m,d)
 for _ in range(round(.3/m.opt.timestep)):mujoco.mj_step(m,d)
 frames=[];carry=[];patches={}
 for f,q in enumerate(cmd):
  steps=round((p['timestamps'][f]-p['timestamps'][f-1])/1e6/m.opt.timestep) if f else 0;forces=[];numcontacts=[]
  for s in range(steps):
   d.ctrl[:]=cmd[max(0,f-1)]+(q-cmd[max(0,f-1)])*(s+1)/steps;mujoco.mj_step(m,d);load=np.zeros(5);count=np.zeros(5,dtype=int)
   for k,c in enumerate(d.contact):
    b1,b2=[m.geom_bodyid[g] for g in (c.geom1,c.geom2)]
    if obj not in (b1,b2):continue
    other=m.body(b2 if b1==obj else b1).name;digit=next((i for i,name in enumerate(digits) if other.startswith('rh_'+name)),None)
    if digit is None:continue
    force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force);load[digit]+=max(0.,force[0]);count[digit]+=force[0]>.05
    if f in (42,60,90,98,105,110,115,125,140) and s==steps-1 and force[0]>.05:
     patches.setdefault(str(f),[]).append(dict(digit=digits[digit],body=other,normal_force=float(force[0]),object_local_contact=(d.xmat[obj].reshape(3,3).T@(c.pos-d.xpos[obj])).tolist()))
   forces.append(load);numcontacts.append(count)
   if 42<=f<126:carry.append(load)
  arr=np.asarray(forces).reshape(-1,5);active=arr>.05
  if len(arr):frames.append(dict(frame=f,mean_force=arr.mean(axis=0).tolist(),max_force=arr.max(axis=0).tolist(),active_fraction=active.mean(axis=0).tolist(),thumb_plus_two_fraction=float((active[:,0]&(active[:,1:].sum(axis=1)>=2)).mean())))
 c=np.asarray(carry);active=c>.05
 out.append(dict(label=label,digits=digits,force_threshold_n=.05,carry_digit_active_fraction=active.mean(axis=0).tolist(),carry_thumb_plus_two_fraction=float((active[:,0]&(active[:,1:].sum(axis=1)>=2)).mean()),carry_three_non_thumb_fraction=float((active[:,1:].sum(axis=1)>=3).mean()),carry_mean_force_n=c.mean(axis=0).tolist(),sample_patches=patches,frames=frames))
print(json.dumps(out,indent=2))

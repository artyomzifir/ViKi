"""Name the offending official geometry pair in a rejected Tesollo seed."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from viki.retarget.robots import normalize_robot
from viki.retarget.solver import PinocchioKinematics

EP=Path('data/datasets/hocap_demo_object_v4/g10_object2_190')
PATH=Path('data/_hocap_demo_source/ur10_tesollo.urdf')
model=pin.buildModelFromUrdf(str(PATH))
collision=pin.buildGeomFromUrdf(model,str(PATH),pin.GeometryType.COLLISION)
with np.load(EP/'tesollo_seed_diagnostic.npz') as z:
    arm=z['arm_q']
    hand={n[5:]:z[n] for n in z.files if n.startswith('hand_')}
kin=PinocchioKinematics(SimpleNamespace(model=model,collision_model=collision),'rl_dg_palm',
    collision_pairs=8,collision_min_distance_m=.001,
    actuated_joint_names=normalize_robot('ur10').joint_names,
    passive_joint_positions=hand,gripper_prefix='rj_dg_',
    floor_normal=(0,0,1),floor_offset_m=0)
data=model.createData();cd=kin.collision_model.createData()
records=[]
for t in range(0,len(arm),5):
    q=arm[t]
    kin.set_frame(t)
    full=kin._configuration(q)
    pin.forwardKinematics(model,data,full)
    pin.updateGeometryPlacements(model,data,kin.collision_model,cd,full)
    pin.computeDistances(model,data,kin.collision_model,cd,full)
    items=sorted((float(d.min_distance), int(p.first), int(p.second)) for p,d in zip(kin.collision_model.collisionPairs,cd.distanceResults))
    for dist,a,b in items[:12]:
        if True:
            records.append((dist,t,kin.collision_model.geometryObjects[a].name,kin.collision_model.geometryObjects[b].name))
    if t%20==0: print('frame',t,'min',items[0][0],flush=True)
print('worst 30:')
for item in sorted(records)[:30]: print(item)

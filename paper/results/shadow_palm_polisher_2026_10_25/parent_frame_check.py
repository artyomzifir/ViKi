import json
import pinocchio as pin
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.hocap_shadow_polisher import source,make_scene,URDF
from viki.retarget.frames import base_rotation
p,t,n,c,*_=source();m,_,_=make_scene(p,t,n);d=mujoco.MjData(m);pm=pin.buildModelFromUrdf(str(URDF));pd=pm.createData();R=base_rotation(p['base_rpy_deg']);a=np.array([m.jnt_qposadr[m.joint(x).id] for x in n]);qa=np.array([pm.idx_qs[pm.getJointId(x)] for x in n]); rows=[]
for frame in (0,23,42,60,98,125,189):
 q=pin.neutral(pm);q[qa]=c[frame];pin.forwardKinematics(pm,pd,q);pin.updateFramePlacements(pm,pd);d.qpos[a]=c[frame];mujoco.mj_forward(m,d)
 errors=[]
 for link in ['shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link','rh_ffproximal','rh_thproximal']:
  f=pm.getFrameId(link,pin.FrameType.BODY);b=m.body(link).id;pose=pd.oMf[f];ep=R@pose.translation+p['base_position'];er=R@pose.rotation
  errors.append(dict(link=link,position_m=float(np.linalg.norm(ep-d.xpos[b])),rotation_rad=float(Rotation.from_matrix(er@d.xmat[b].reshape(3,3).T).magnitude())))
 rows.append(dict(frame=frame,errors=errors))
print(json.dumps(dict(base_position=p['base_position'].tolist(),base_rpy_deg=p['base_rpy_deg'].tolist(),rows=rows),indent=2))

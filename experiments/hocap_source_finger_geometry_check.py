import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint, Bounds
p=Path('/app/data/datasets/hocap_demo_object_v4/g10_object2_190')
m=p/'intermediates/segmentation/hocap-oracle-v1/meshes'
v=np.array([[float(x) for x in line.split()[1:4]] for line in (m/'object_2.obj').read_text().splitlines() if line.startswith('v ')])*1000
h=ConvexHull(v);A=h.equations[:,:3];b=-h.equations[:,3]
c=np.load(p/'cln.npz');g=np.load(p/'gt/joints3d_world.npz');o=np.load(p/'gt/object_poses_world.npz');a=next(x for x in json.loads((m/'alignment.json').read_text()) if x['id']==2)
T=o['poses_world'][:,1]@np.linalg.inv(a['track_from_mesh'])
r=c['hand_fit_capsule_radii'][1:]*1000
ref=np.load('/app/rl-output/source_finger_audit_20261026/finger_gaps.npz')['annotated_joints_gt_object']*1000
rows=[]
for frame in (42,60,90,120,145):
 for finger,j in enumerate((3,7,11,15,19)):
  bone=finger*3+2;ends=(g['joints_world'][frame,[j,j+1]]-T[frame,:3,3])@T[frame,:3,:3]*1000
  start=ends[0];d=ends[1]-start
  def fun(x):
   diff=x[:3]-start-x[3]*d
   return diff@diff, np.r_[2*diff,-2*diff@d]
  sol=minimize(fun,np.r_[v.mean(axis=0),.5],jac=True,method='SLSQP',constraints=[LinearConstraint(np.c_[A,np.zeros(len(A))],-np.inf,b)],bounds=Bounds([-np.inf]*3+[0],[np.inf]*3+[1]),options={'ftol':1e-9,'maxiter':200})
  dist=np.sqrt(max(sol.fun,0));gap=dist-r[bone]
  # For centerlines outside the hull this directly checks capsule surface gap.
  rows.append(dict(frame=frame,finger=finger,solver_success=bool(sol.success),constraint_violation_mm=float(np.maximum(A@sol.x[:3]-b,0).max()),centerline_distance_mm=float(dist),qp_gap_mm=float(gap),mujoco_gap_mm=float(ref[frame,bone]),difference_mm=float(gap-ref[frame,bone])))
report={'method':'independent convex quadratic minimization between source segment and official OBJ convex hull in object coordinates, no MuJoCo geometry transforms','rows':rows,'max_abs_difference_mm':max(abs(x['difference_mm']) for x in rows if x['centerline_distance_mm']>1e-4)}
Path('/app/rl-output/source_finger_audit_20261026/independent_geometry_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

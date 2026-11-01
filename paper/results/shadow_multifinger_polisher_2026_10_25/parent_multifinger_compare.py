import json
from pathlib import Path
import mujoco,numpy as np
root=Path('/app/rl-output')
pairs=[('multifinger','shadow_multifinger_followup_20261025','trajectory.npz','shadow_multifinger_parent_original_runner_20261025','shadow_multifinger_parent_review_20261025'),('placement','shadow_placement_source_release_20261025','candidate_2_trajectory.npz','shadow_placement_parent_original_runner_20261025','shadow_placement_parent_review_20261025')]
for label,source,filename,old,out in pairs:
 a=np.load(root/source/filename,allow_pickle=False);b=np.load(root/old/'trajectory.npz',allow_pickle=False);arrays={k:bool(np.array_equal(a[k],b[k])) for k in b.files};assert all(arrays.values()),arrays
 m1=mujoco.MjModel.from_xml_path(str(root/source/'scene.xml'));m2=mujoco.MjModel.from_xml_path(str(root/old/'scene.xml'));fields=['body_mass','body_inertia','body_gravcomp','geom_friction','geom_contype','geom_conaffinity','geom_pos','geom_size','jnt_range','actuator_gainprm','actuator_biasprm','actuator_forcerange','exclude_signature'];models={f:bool(np.array_equal(getattr(m1,f),getattr(m2,f))) for f in fields};models.update(no_weld=m1.neq==m2.neq==0,timestep=m1.opt.timestep==m2.opt.timestep,integrator=m1.opt.integrator==m2.opt.integrator);assert all(models.values()),models
 expected=json.loads((root/source/'report.json').read_text())['selected_metrics'];actual=json.loads((root/old/'report.json').read_text());metric={k:v==actual[k] for k,v in expected.items() if k in actual};assert all(metric.values()),metric
 r={'trajectory_arrays_bitwise':arrays,'compiled_physics_identical':models,'shared_metrics_exact':metric};(root/out/'parent_reproduction.json').write_text(json.dumps(r,indent=2)+'\n');print(label,json.dumps(r))

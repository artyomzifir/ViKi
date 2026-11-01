import json
from pathlib import Path
import mujoco,numpy as np
src=Path('/app/rl-output/shadow_palm_polisher_20261025');old=Path('/app/rl-output/shadow_palm_parent_original_runner_20261025');out=Path('/app/rl-output/shadow_palm_parent_review_20261025')
a=np.load(src/'trajectory.npz',allow_pickle=False);b=np.load(old/'trajectory.npz',allow_pickle=False);print('keys',a.files,b.files)
shared=sorted(set(a.files)&set(b.files));arrays={k:bool(np.array_equal(a[k],b[k])) for k in shared};assert all(arrays.values()),arrays
m1=mujoco.MjModel.from_xml_path(str(src/'scene.xml'));m2=mujoco.MjModel.from_xml_path(str(old/'scene.xml'))
fields=['body_mass','body_inertia','body_gravcomp','geom_friction','geom_contype','geom_conaffinity','geom_pos','geom_size','jnt_range','actuator_gainprm','actuator_biasprm','actuator_forcerange','exclude_signature']
models={f:bool(np.array_equal(getattr(m1,f),getattr(m2,f))) for f in fields};models.update(neq=m1.neq==m2.neq==0,timestep=m1.opt.timestep==m2.opt.timestep,integrator=m1.opt.integrator==m2.opt.integrator);assert all(models.values()),models
m=json.loads((src/'manifest.json').read_text());rep=json.loads((old/'report.json').read_text());metrics={k:rep[k]==v for k,v in m['selected_metrics'].items() if k in rep};assert all(metrics.values()),metrics
r=dict(trajectory_arrays_bitwise=arrays,compiled_physics_identical=models,shared_metrics_exact=metrics,accepted=m['accepted'],notes='The independent original replay has a separate simulation loop and XML construction; source physics and commands were preserved.')
(out/'parent_reproduction.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

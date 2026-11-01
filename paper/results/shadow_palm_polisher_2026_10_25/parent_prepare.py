import json,hashlib
from pathlib import Path
import numpy as np
from experiments.hocap_shadow_polisher import source,PLAN,TRACK,URDF,EP,sha
from experiments.shadow_polisher_controls import command_margins,validate_command_export
from experiments.shadow_palm_controls import palm_gate
p,t,n,c,lo,hi,v=source();src=Path('/app/rl-output/shadow_palm_polisher_20261025');out=Path('/app/rl-output/shadow_palm_parent_review_20261025');out.mkdir(parents=True,exist_ok=False)
hashes={'plan':sha(PLAN),'track':sha(TRACK),'urdf':sha(URDF),'meshes':{str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)}}
with np.load(src/'final_commands.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
cmd=a['commands'];ts=a['timestamps'];manifest=json.loads((src/'manifest.json').read_text());zero=np.load(src/'candidates/00.npz',allow_pickle=False)['commands']
checks=dict(zero_bitwise=zero.tobytes()==c.tobytes(),timestamps_exact=np.array_equal(ts,p['timestamps']),fingers_bitwise=cmd[:,6:].tobytes()==c[:,6:].tobytes(),outside_gate_bitwise=cmd[palm_gate()==0].tobytes()==c[palm_gate()==0].tobytes(),shape=cmd.shape==c.shape,dtype=cmd.dtype==c.dtype,source_hashes_unchanged=hashes==manifest['source_hashes'],export_hash_matches=sha(src/'final_commands.npz')==manifest['selected_command_sha256'])
pm,vm=command_margins(cmd,ts,lo,hi,v);checks.update(joint_bounds=pm>=-1e-6,velocity_bounds=vm>=-1e-6)
assert all(checks.values()),checks
report=json.loads(str(p['report_json']));report['solver_status']='parent_review_only_palm_diagnostic';p['arm_q']=cmd[:,:6].copy();p['shadow_q']=cmd[:,6:].copy();p['timestamps']=ts.copy();p['report_json']=np.asarray(json.dumps(report));np.savez_compressed(out/'legacy_candidate_plan.npz',**p)
r=dict(checks=checks,position_margin_rad=pm,velocity_margin_rad_s=vm,selected_index=manifest['selected_index'],selected_residual=manifest['selected_residual'],source_hashes=hashes)
(out/'parent_invariants.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

from pathlib import Path
import json
import numpy as np
from experiments.hocap_shadow_polisher import source,sha
from experiments.shadow_polisher_controls import command_margins
from experiments.shadow_multifinger_controls import multifinger_commands
p,t,n,base,lo,hi,v=source();src=Path('/app/rl-output/shadow_multifinger_followup_20261025');out=Path('/app/rl-output/shadow_multifinger_parent_review_20261025');out.mkdir(exist_ok=False)
f=np.load(src/'final_commands.npz',allow_pickle=False);cmd=f['commands'];palm=np.load('/app/rl-output/shadow_palm_polisher_20261025/final_commands.npz',allow_pickle=False)['commands'];res=f['multifinger_residual'];expected=multifinger_commands(palm,n,res);pm,vm=command_margins(cmd,f['timestamps'],lo,hi,v)
checks={'declared_residual_reproduces_commands_bitwise':expected.tobytes()==cmd.tobytes(),'arm_unchanged_from_palm':cmd[:,:6].tobytes()==palm[:,:6].tobytes(),'timestamps_unchanged':np.array_equal(f['timestamps'],p['timestamps']),'outside_gate_unchanged':np.array_equal(cmd[:11],palm[:11]) and np.array_equal(cmd[151:],palm[151:]),'names_match':list(f['joint_names'])==n,'position_limits':pm>=-1e-6,'velocity_limits':vm>=-1e-6};assert all(checks.values()),checks
p['arm_q']=cmd[:,:6].copy();p['shadow_q']=cmd[:,6:].copy();r=json.loads(str(p['report_json']));r['solver_status']='parent_review_only_multifinger_diagnostic';p['report_json']=np.asarray(json.dumps(r));np.savez_compressed(out/'legacy_candidate_plan.npz',**p)
report={'checks':checks,'joint_position_margin_rad':pm,'command_velocity_margin_rad_s':vm,'selected_residual':res.tolist(),'export_sha256':sha(src/'final_commands.npz')};(out/'parent_invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

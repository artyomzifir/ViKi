import json
from pathlib import Path
import numpy as np
from experiments.hocap_shadow_polisher import source,sha
from experiments.shadow_polisher_controls import command_margins
p,t,n,base,lo,hi,v=source();src=Path('/app/rl-output/shadow_placement_source_release_20261025');out=Path('/app/rl-output/shadow_placement_parent_review_20261025');out.mkdir(exist_ok=False)
f=np.load(src/'selected_commands.npz',allow_pickle=False);cmd=f['commands'];inp=np.load('/app/rl-output/shadow_multifinger_polisher_20261025_v2/final_commands.npz',allow_pickle=False)['commands'];sched=json.loads(str(f['placement_schedule_json']));pm,vm=command_margins(cmd,f['timestamps'],lo,hi,v)
checks={'prefix_preserved':np.array_equal(cmd[:sched['anchor']],inp[:sched['anchor']]),'finger_commands_preserved':np.array_equal(cmd[:,6:],inp[:,6:]),'arm_frozen_until_retreat':np.array_equal(cmd[sched['anchor']:sched['retreat_start']+1,:6],np.broadcast_to(inp[sched['anchor'],:6],(sched['retreat_start']+1-sched['anchor'],6))),'timestamps_unchanged':np.array_equal(f['timestamps'],p['timestamps']),'position_limits':pm>=-1e-6,'velocity_limits':vm>=-1e-6};assert all(checks.values()),checks
p['arm_q']=cmd[:,:6].copy();p['shadow_q']=cmd[:,6:].copy();r=json.loads(str(p['report_json']));r['solver_status']='parent_review_only_placement_diagnostic';p['report_json']=np.asarray(json.dumps(r));np.savez_compressed(out/'legacy_candidate_plan.npz',**p)
report={'checks':checks,'joint_position_margin_rad':pm,'command_velocity_margin_rad_s':vm,'schedule':sched,'export_sha256':sha(src/'selected_commands.npz')};(out/'parent_invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

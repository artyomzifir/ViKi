"""Read-only carry-frame collision audit of the two frozen optimized endpoints."""
import json
from pathlib import Path
import numpy as np
from experiments.hocap_shadow_polisher import source
from experiments.hocap_shadow_multifinger_polisher import hashes,load_palm,kinematic_margins
from experiments.hocap_shadow_palm_polisher import build_kinematics
from experiments.shadow_multifinger_controls import multifinger_commands
from experiments.hocap_shadow_pressure_hint_ablation import full_residual,SNAPSHOTS,BACKTRACK

folder=Path('/app/rl-output/shadow_pressure_hint_ablation_20261026')
rows=[json.loads(x) for x in (folder/'optimization.jsonl').read_text().splitlines()]
plan,track,names,baseline,*_=source()
h=hashes();palm=load_palm(names,plan,baseline,h)
kin=build_kinematics({**plan,'shadow_q':palm[:,6:]})
result=[]
for optimized in rows:
 x0=np.asarray(optimized['x0']);x=np.asarray(optimized['x_opt'])
 for scale in BACKTRACK:
  q=multifinger_commands(palm,names,full_residual(x0+scale*(x-x0)))
  frames=[]
  for frame in SNAPSHOTS:
   floor,self_margin,bad=kinematic_margins(kin,names,q,[frame])
   frames.append({'frame':frame,'floor_margin_m':floor,'screened_self_margin_m':self_margin,'violated':bad is not None})
  result.append({'mode':optimized['mode'],'scale':scale,'carry_frames':frames})
(folder/'carry_frame_screen.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

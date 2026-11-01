from pathlib import Path
import json,hashlib,collections
import numpy as np
import h5py
from viki.benchmark.evaluate import evaluate_episode
from viki.benchmark.hocap_objects import evaluate_object_models
root=Path('/app/data/datasets')
result={'date':'2026-10-25','scope':'Recompute from saved arrays, no fresh extraction or IK','datasets':{},'plans':[],'objects':[]}
for folder in ['hocap_eightview','hocap_stable_fused_v4_one_scene','hocap_stable_fused_v4_smoke_120','hocap_demo_object_v4']:
 rows=[]
 for ep in sorted((root/folder).iterdir()):
  if not (ep/'cln.npz').is_file():continue
  r=evaluate_episode(ep)
  with np.load(ep/'cln.npz',allow_pickle=False) as a:r['has_hand_fit']='hand_fit_joint_angles' in a.files
  r['cln_sha256']=hashlib.sha256((ep/'cln.npz').read_bytes()).hexdigest();rows.append(r)
  if folder=='hocap_eightview':
   with h5py.File(ep/'plan.h5') as p:
    def scalar(k):
     x=p[k][()];return x.decode() if isinstance(x,bytes) else x.item() if isinstance(x,np.generic) else x
    c=json.loads(scalar('config_json'));m=json.loads(scalar('metrics_json'))
    result['plans'].append({'episode':ep.name,'frames':len(p['q']),'source_pose':scalar('source_pose'),'solver_status':scalar('solver_status'),'robot':c['robot'],'object_grasp':m.get('object_grasp'),'position_rmse_mm':m.get('position_rmse_mm'),'orientation_rmse_deg':m.get('orientation_rmse_deg')})
   ob=ep/'intermediates/segmentation/hocap-oracle-v1';meta=json.loads((ob/'meta.json').read_text())
   obj=evaluate_object_models(ep/'gt/object_poses_world.npz',ob/'object_models.npz',{int(k):v for k,v in meta['object_name_by_id'].items()})
   result['objects'].append({'episode':ep.name,**obj})
 summary={}
 for name in ['smoothed_all','fused_observed','hand_fit_joints']:
  scores=[r['prepare'][name] for r in rows if name in r['prepare']]
  if scores:
   n=sum(x['matched_joints'] for x in scores);g=sum(x['gt_joints'] for x in scores)
   summary[name]={'mpjpe_mm':sum(x['matched_joints']*x['mpjpe_mm'] for x in scores)/n,'matched':n,'gt':g}
 result['datasets'][folder]={'episodes':len(rows),'frames':sum(r['frame_count'] for r in rows),'fitted_episodes':sum(r['has_hand_fit'] for r in rows),'summary':summary,'rows':rows}
obs=[o for r in result['objects'] for o in r['objects'].values()]
result['object_summary']={'tracks':len(obs),'macro_translation_mean_mm':float(np.mean([o['translation_mean_mm'] for o in obs])),'macro_rotation_mean_deg':float(np.mean([o['rotation_mean_deg'] for o in obs])),'tracked_frames':sum(o['tracked_frames'] for o in obs),'gt_frames':sum(o['gt_frames'] for o in obs)}
print(json.dumps(result,indent=2))

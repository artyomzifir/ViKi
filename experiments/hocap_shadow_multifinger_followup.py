"""One bounded, data-driven 32-point local follow-up to multifinger pilot."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.hocap_shadow_polisher import source, make_scene, rollout, sha
from experiments.hocap_shadow_multifinger_polisher import (
    PALM_INPUT, hashes, load_palm, expected_export_hashes, kinematic_margins,
    static_hold, full_digit_trace, export_commands, jsonwrite)
from experiments.hocap_shadow_palm_polisher import build_kinematics, sampled_joint_velocity_margin
from experiments.shadow_multifinger_controls import multifinger_commands
from experiments.shadow_polisher_controls import command_margins, validate_command_export

PHASE1=Path('/app/rl-output/shadow_multifinger_polisher_20261025_v2/final_commands.npz')
PROTOCOL={
    'schema':'shadow_multifinger_followup_protocol_v1',
    'chronology':'separate exploratory follow-up after frozen 64 candidates; total hypothesis budget 96',
    'reason':'phase1 MF +0.12rad produced 97.6% carry thumb+two contact but 16.2mm/19.2mm static slip/drop',
    'candidate_count':32,
    'grid':'MF flex [0.08,0.10,0.12,0.14] x RF flex [0,0.04,0.08] x LF flex [0,0.08] radians',
    'thumb_ablation':'MF flex [0.10,0.12] x THJ2 [-0.04,-0.02,0.02,0.04] radians',
    'all_other_parameters':'zero; same -6mm palm, finger gate, physics, constraints and acceptance as first multifinger cycle',
    'shortlist':'top 8 airborne_unpinned by same static score, fallback diagnostic if none',
    'selection':'target gate first, then old score +100*(1-carry thumb+two fraction), comparing phase1 selected control',
    'held_out_grid':{'mass_kg':[.06,.10],'robot_object_friction':[.9,1.2]},
}


def local_candidates():
    result=[]
    for mf in (.08,.10,.12,.14):
        for rf in (0.,.04,.08):
            for lf in (0.,.08):
                v=np.zeros(8);v[:3]=[mf,rf,lf];result.append(v)
    for mf in (.10,.12):
        for th in (-.04,-.02,.02,.04):
            v=np.zeros(8);v[0]=mf;v[7]=th;result.append(v)
    assert len(result)==32
    return result


def phase_errors(trajectory,track):
    col=list(map(int,track['object_ids'])).index(2)
    err=np.linalg.norm(trajectory['object_positions']-track['translation_world'][:,col],axis=1)*1000
    return {name:{'mean_mm':float(np.mean(err[a:b])),'p95_mm':float(np.percentile(err[a:b],95)),
                  'max_mm':float(np.max(err[a:b]))}
            for name,a,b in [('pickup_42_60',42,60),('early_carry_60_99',60,99),
                             ('late_carry_99_126',99,126),('release_126_151',126,151),
                             ('retreat_151_190',151,190)]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/app/rl-output/shadow_multifinger_followup_20261025')
    args=parser.parse_args();out=Path(args.output)
    if any((out/x).exists() for x in ('manifest.json','feasibility.jsonl','static_ledger.jsonl','full_ledger.jsonl')):
        raise RuntimeError('follow-up output already exists')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    plan,track,names,baseline,lower,upper,velocity=source();h=hashes()
    palm=load_palm(names,plan,baseline,h)
    with np.load(PHASE1,allow_pickle=False) as saved:
        phase1,ts,factor=validate_command_export(saved,
            {**expected_export_hashes(h),'palm_input_sha256':h['palm_input']},
            names,plan['timestamps'],baseline.shape,baseline.dtype)
        phase1_residual=saved['multifinger_residual'].copy()
    if factor!=1. or not np.array_equal(phase1,multifinger_commands(palm,names,phase1_residual)):
        raise RuntimeError('phase1 selected command provenance mismatch')
    model,table_z,xml=make_scene(plan,track,names)
    (out/'scene.xml').write_text(xml)
    candidates=local_candidates()
    jsonwrite(out/'protocol.json',{'protocol':PROTOCOL,'candidates':[x.tolist() for x in candidates]})
    manifest={'schema':'shadow_multifinger_followup_v1','protocol':PROTOCOL,'source_hashes':h,
              'phase1_selected_sha256':sha(PHASE1),'accepted':False,
              'physics':'same hocap_shadow_polisher.make_scene and rollout',
              'collision_screen':'per-candidate finger passive trajectories, full changed-frame floor and nearest eight retained URDF self pairs'}
    jsonwrite(out/'manifest.json',manifest)
    kin=build_kinematics(plan)
    candidate_dir=out/'candidates';candidate_dir.mkdir()
    feasibility=[]
    for i,residual in enumerate(candidates):
        command=multifinger_commands(palm,names,residual)
        pm,vm=command_margins(command,plan['timestamps'],lower,upper,velocity)
        row={'index':i,'residual_rad':residual.tolist(),'position_margin_rad':pm,'velocity_margin_rad_s':vm}
        if min(pm,vm)<-1e-6:
            row['status']='rejected_limits'
        else:
            floor,selfmargin,frame=kinematic_margins(kin,names,command,range(11,151))
            row.update(floor_margin_m=floor,screened_self_collision_margin_m=selfmargin)
            if frame is not None:row.update(status='rejected_floor_or_self',rejected_frame=frame)
            else:
                row['status']='feasible';file=candidate_dir/f'{i:02d}.npz'
                export_commands(file,command,plan,names,h,residual,i)
                row.update(command_file=str(file.relative_to(out)),command_sha256=sha(file))
        feasibility.append(row)
        with (out/'feasibility.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        print(json.dumps({'phase':'feasibility','index':i,'status':row['status'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    static=[]
    for row in feasibility:
        if row['status']!='feasible':continue
        with np.load(out/row['command_file'],allow_pickle=False) as saved:command=saved['commands'].copy()
        metrics=static_hold(model,plan,names,command)
        item={'index':row['index'],'metrics':metrics}
        static.append(item)
        with (out/'static_ledger.jsonl').open('a') as file:file.write(json.dumps(item)+'\n')
        print(json.dumps({'phase':'static','index':row['index'],'score':round(metrics['static_score'],2),'pass':metrics['static_pass'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    static.sort(key=lambda x:(x['metrics']['static_score'],x['index']))
    airborne=[x for x in static if x['metrics']['airborne_unpinned']]
    shortlist=[x['index'] for x in (airborne if airborne else static)[:8]]
    full=[]
    for index in shortlist:
        with np.load(out/feasibility[index]['command_file'],allow_pickle=False) as saved:command=saved['commands'].copy()
        metrics=rollout(model,plan,track,names,command,table_z)
        digit,_=full_digit_trace(model,plan,names,command)
        hold=next(x['metrics'] for x in static if x['index']==index)
        rank=float(metrics['score']+100*(1-digit['carry_thumb_plus_two_fraction']))
        item={'index':index,'metrics':metrics,'digits':digit,'static_hold':hold,'selection_score':rank}
        full.append(item)
        with (out/'full_ledger.jsonl').open('a') as file:file.write(json.dumps(item)+'\n')
        print(json.dumps({'phase':'full','index':index,'score':round(rank,2),'target_success':metrics['target_task_success'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    phase1_metrics=rollout(model,plan,track,names,phase1,table_z)
    phase1_digits,_=full_digit_trace(model,plan,names,phase1)
    phase1_hold=static_hold(model,plan,names,phase1)
    phase1_row={'index':'phase1_selected','metrics':phase1_metrics,'digits':phase1_digits,
                'static_hold':phase1_hold,
                'selection_score':float(phase1_metrics['score']+100*(1-phase1_digits['carry_thumb_plus_two_fraction']))}
    pool=full+[phase1_row]
    selected=min(pool,key=lambda x:(not x['metrics']['target_task_success'],x['selection_score'],str(x['index'])))
    if selected['index']=='phase1_selected':
        chosen=phase1;residual=phase1_residual;index=-1;selected_source='phase1_selected'
    else:
        index=selected['index'];residual=candidates[index];selected_source=f'followup_{index}'
        with np.load(out/feasibility[index]['command_file'],allow_pickle=False) as saved:chosen=saved['commands'].copy()
    export_commands(out/'final_commands.npz',chosen,plan,names,h,residual,index)
    selected_metrics,trajectory=rollout(model,plan,track,names,chosen,table_z,full=True)
    selected_digits,digit_trace=full_digit_trace(model,plan,names,chosen)
    selected_hold,hold_trace=static_hold(model,plan,names,chosen,full=True)
    np.savez_compressed(out/'trajectory.npz',**trajectory,**digit_trace)
    np.savez_compressed(out/'static_trace.npz',**hold_trace)
    sensitivity=[]
    for mass in (.06,.10):
        for friction in (.9,1.2):
            varied,z,_=make_scene(plan,track,names,mass=mass,friction=friction)
            for label,cmd in (('baseline',baseline),('palm_control',palm),('selected_diagnostic',chosen)):
                metrics=rollout(varied,plan,track,names,cmd,z)
                digits,_=full_digit_trace(varied,plan,names,cmd)
                sensitivity.append({'mass_kg':mass,'robot_object_friction':friction,
                                    'trajectory':label,'metrics':metrics,'digits':digits})
    jsonwrite(out/'sensitivity.json',sensitivity)
    with np.load(out/'final_commands.npz',allow_pickle=False) as saved:
        checked,_,factor=validate_command_export(saved,
            {**expected_export_hashes(h),'palm_input_sha256':h['palm_input']},
            names,plan['timestamps'],baseline.shape,baseline.dtype)
    if factor!=1. or not np.array_equal(checked,chosen):raise RuntimeError('exported command mismatch')
    replay=rollout(model,plan,track,names,checked,table_z)
    if replay!=selected_metrics:raise RuntimeError('exported trajectory failed independent metric reproduction')
    baseline_metrics,baseline_traj=rollout(model,plan,track,names,baseline,table_z,full=True)
    palm_metrics,palm_traj=rollout(model,plan,track,names,palm,table_z,full=True)
    result={'schema':'shadow_multifinger_followup_report_v1','selection_source':selected_source,'selected_index':index,
            'selected_residual_rad':residual.tolist(),'selected_metrics':selected_metrics,'selected_digits':selected_digits,
            'selected_static_hold':selected_hold,'phase_errors':phase_errors(trajectory,track),
            'baseline_phase_errors':phase_errors(baseline_traj,track),'palm_phase_errors':phase_errors(palm_traj,track),
            'phase1_selected':phase1_row,'baseline_metrics':baseline_metrics,'palm_metrics':palm_metrics,
            'sampled_actual_velocity_margin_rad_s':sampled_joint_velocity_margin(trajectory['q_actual'],plan['timestamps'],velocity),
            'accepted':bool(selected_metrics['target_task_success'] and selected_metrics['full_scene_success'] and selected_hold['static_pass']),
            'replayed_export_exact':True}
    jsonwrite(out/'report.json',result)
    manifest.update(feasible_count=sum(x['status']=='feasible' for x in feasibility),
        static_pass_count=sum(x['metrics']['static_pass'] for x in static),
        airborne_unpinned_count=len(airborne),shortlist_indices=shortlist,
        selected_source=selected_source,selected_index=index,selected_metrics=selected_metrics,
        selected_static_hold=selected_hold,selected_command_sha256=sha(out/'final_commands.npz'),
        accepted=result['accepted'],total_wall_s=time.monotonic()-started)
    jsonwrite(out/'manifest.json',manifest)
    print(json.dumps({'selected_source':selected_source,'target_success':selected_metrics['target_task_success'],
                      'static_pass':selected_hold['static_pass'],'total_wall_s':manifest['total_wall_s']},indent=2),flush=True)

if __name__=='__main__':main()

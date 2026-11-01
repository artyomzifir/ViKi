"""Opt-in bounded multifinger grasp/placement pilot on frozen G10_2 Shadow scene.

Uses the saved -6 mm palm plan, unchanged MuJoCo physics, and old strict task gates.
Every exported trajectory is open-loop and retains all 190 timestamped frames.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from experiments.hocap_shadow_polisher import EP, PLAN, TRACK, URDF, source, make_scene, rollout, sha
from experiments.hocap_shadow_palm_polisher import build_kinematics, sampled_joint_velocity_margin
from experiments.shadow_polisher_controls import command_margins, validate_command_export
from experiments.shadow_multifinger_controls import BOUNDS, DIGITS, FORCE_THRESHOLD_N, finger_gate, multifinger_commands, active_digit_forces, static_hold_metrics

PALM_INPUT=Path('/app/rl-output/shadow_palm_polisher_20261025/final_commands.npz')
SEED=20261025
PROTOCOL={
    'schema':'shadow_multifinger_protocol_v1', 'seed':SEED,
    'basis':'saved palm-best -6 mm object-local x arm; only finger residuals change',
    'parameters':['MF flex J1/2/3','RF flex J1/2/3','LF flex J1/2/3',
                  'MF J4 abduction','RF J4 abduction','LF J4 abduction','LF J5 cup','TH J2 opposition'],
    'finger_flex_weights':{'J1':1.,'J2':.7,'J3':.35},
    'gate':'C1 rise frames 10..30, hold 30..125, fall 125..151; source after151',
    'candidate_count':64,'probes':'each coordinate lower/full, lower/half, upper/half, upper/full, omitting zero duplicates',
    'geometric_seeds':'MF/RF/LF flexion [0.02,0.08,0.18] radians at scales 0.5,1,1.5, based on fixed frame60 distal-gap finite differences',
    'remainder':'seeded 8-D Latin-hypercube',
    'force_threshold_n':FORCE_THRESHOLD_N,
    'static':'replay through frame60 then freeze all 28 controls for 2.0s; require lift>20mm from settled start, no target world/neighbor support, no robot-table contact, peak slip<10mm, peak z-drop<10mm, last-second peak drift<5mm, peak relative rotation<10deg, thumb + >=2 other digits active >=80% last second',
    'static_ranking':'min(300,peak_slip2_mm)+min(300,peak_zdrop2_mm)+min(100,peak_last1drift_mm)+min(100,peak_rotation2_deg)+100*(1-triad_last1_fraction)+1000*(not airborne_unpinned)',
    'shortlist':'8 lowest static scores among airborne_unpinned feasible; if none, 8 lowest overall for explicit diagnostic, plus original and palm-best controls',
    'final_ranking':'target_task_success first, then minimum old frozen score + 100*(1-carry thumb+two digit fraction); static_pass separately required to describe stable grasp',
    'held_out_grid':{'mass_kg':[.06,.10],'robot_object_friction':[.9,1.2]},
    'acceptance':'unchanged target/full-scene gates in hocap_shadow_polisher.rollout',
}


def candidate_parameters():
    values=[np.zeros(8)]
    for axis,(lo,hi) in enumerate(BOUNDS):
        for value in (lo,lo/2,hi/2,hi):
            if value==0:continue
            row=np.zeros(8);row[axis]=value;values.append(row)
    for scale in (.5,1.,1.5):
        row=np.zeros(8);row[:3]=scale*np.array([.02,.08,.18]);values.append(row)
    sampler=qmc.LatinHypercube(d=8,seed=SEED)
    values.extend(qmc.scale(sampler.random(64-len(values)),BOUNDS[:,0],BOUNDS[:,1]))
    assert len(values)==64
    return values


def hashes():
    return {'plan':sha(PLAN),'track':sha(TRACK),'urdf':sha(URDF),
            'meshes':{str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)},
            'palm_input':sha(PALM_INPUT)}


def expected_export_hashes(h):
    return {'schema':'shadow_polisher_commands_v1','plan_sha256':h['plan'],
            'track_sha256':h['track'],'urdf_sha256':h['urdf'],
            'mesh_hashes_json':json.dumps(h['meshes'],sort_keys=True)}


def load_palm(names,plan,baseline,h):
    with np.load(PALM_INPUT,allow_pickle=False) as saved:
        commands,timestamps,factor=validate_command_export(saved,expected_export_hashes(h),names,
            plan['timestamps'],baseline.shape,baseline.dtype)
        residual=np.asarray(saved['palm_residual'],dtype=float)
    if factor!=1. or residual.shape!=(6,) or not np.array_equal(residual,np.array([-.006,0,0,0,0,0])):
        raise ValueError('expected frozen -6 mm palm input, original timestamps')
    if not np.array_equal(commands[:,6:],baseline[:,6:]):
        raise ValueError('palm input unexpectedly changed fingers')
    return commands


def set_passive_fingers(kin,names,commands):
    """Install each candidate's actual finger trajectory in the pilot IK screen."""
    index={int(kin.model.idx_qs[kin.model.getJointId(n)]):i for i,n in enumerate(names)}
    previous=kin._passive_positions
    if len(previous)!=22 or any(qidx not in index for qidx,_ in previous):
        raise RuntimeError('unexpected passive-finger model')
    kin._passive_positions=[(qidx,np.asarray(commands[:,index[qidx]],dtype=float).copy())
                            for qidx,_ in previous]
    kin._collision_configuration_cache=None


def kinematic_margins(kin,names,commands,frames):
    set_passive_fingers(kin,names,commands)
    floor=float('inf');self_margin=float('inf')
    for frame in frames:
        kin.set_frame(int(frame))
        q=commands[frame,:6].astype(float)
        floor=min(floor,float(np.min(kin.floor_margins(q))))
        self_margin=min(self_margin,float(np.min(kin.collision_margins(q))))
        if floor < -1e-7 or self_margin < -1e-7:
            return floor,self_margin,int(frame)
    return floor,self_margin,None


def digit_forces(model,data,objid):
    values=np.zeros(5,dtype=float)
    for ci,c in enumerate(data.contact):
        b1,b2=(model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2])
        if objid not in (b1,b2):continue
        other=model.body(b2 if b1==objid else b1).name
        for j,digit in enumerate(DIGITS):
            if other.startswith('rh_'+digit):
                force=np.zeros(6)
                mujoco.mj_contactForce(model,data,ci,force)
                values[j]+=max(0.,float(force[0]))
                break
    return values


def prepared_data(model,names,commands):
    data=mujoco.MjData(model)
    addresses=np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    data.qpos[addresses]=commands[0]
    data.ctrl[:]=commands[0]
    mujoco.mj_forward(model,data)
    for _ in range(round(.3/model.opt.timestep)):
        mujoco.mj_step(model,data)
    return data,addresses


def advance_frame(model,data,plan,commands,t,observer=None):
    steps=round((plan['timestamps'][t]-plan['timestamps'][t-1])/1e6/model.opt.timestep) if t else 0
    prev=commands[max(t-1,0)]
    current=commands[t]
    for step in range(steps):
        data.ctrl[:]=prev+(current-prev)*(step+1)/steps
        mujoco.mj_step(model,data)
        if observer is not None:observer(t,data)


def relative_pose(data,obj,wrist):
    r=data.xmat[wrist].reshape(3,3).T
    return r@(data.xpos[obj]-data.xpos[wrist]),r@data.xmat[obj].reshape(3,3)


def static_hold(model,plan,names,commands,full=False):
    data,_=prepared_data(model,names,commands)
    obj=model.body('object_2').id;wrist=model.body('wrist_3_link').id
    settled_z=float(data.xpos[obj,2])
    for t in range(61):advance_frame(model,data,plan,commands,t)
    lift=(float(data.xpos[obj,2])-settled_z)*1000
    p,r=relative_pose(data,obj,wrist)
    rel=[p];rot=[r];objpos=[data.xpos[obj].copy()];forces=[];support=[];robot_table=[]
    data.ctrl[:]=commands[60]
    for _ in range(round(2./model.opt.timestep)):
        mujoco.mj_step(model,data)
        p,r=relative_pose(data,obj,wrist)
        rel.append(p);rot.append(r);objpos.append(data.xpos[obj].copy())
        forces.append(digit_forces(model,data,obj))
        target_supported=False;table_hit=False
        for ci,c in enumerate(data.contact):
            b1,b2=(model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2])
            n1,n2=model.body(b1).name,model.body(b2).name
            if obj not in (b1,b2) and 'world' not in (n1,n2):continue
            f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f)
            if f[0]<=1e-6:continue
            if obj in (b1,b2):
                other=n2 if b1==obj else n1
                if other=='world' or other.startswith('object_'):target_supported=True
            if 'world' in (n1,n2):
                other=n2 if n1=='world' else n1
                if other.startswith('rh_') or other.endswith('link'):table_hit=True
        support.append(target_supported);robot_table.append(table_hit)
    rel=np.asarray(rel);rot=np.asarray(rot);objpos=np.asarray(objpos);forces=np.asarray(forces)
    support=np.asarray(support);robot_table=np.asarray(robot_table)
    metrics=static_hold_metrics(rel,rot,objpos,forces,lift,support,robot_table,model.opt.timestep)
    if full:return metrics,{'relative_positions':rel,'relative_rotations':rot,'object_positions':objpos,'digit_forces_n':forces,
                            'target_support_steps':support,'robot_table_steps':robot_table}
    return metrics


def full_digit_trace(model,plan,names,commands):
    data,_=prepared_data(model,names,commands)
    obj=model.body('object_2').id
    forces=[];frames=[];mean_by_frame=[];peak_by_frame=[];frac_by_frame=[];triad_by_frame=[]
    current=[]
    def observer(t,d):
        f=digit_forces(model,d,obj)
        forces.append(f);frames.append(t);current.append(f)
    for t in range(190):
        current=[]
        advance_frame(model,data,plan,commands,t,observer)
        arr=np.asarray(current).reshape(-1,5)
        if len(arr):
            active,triad=active_digit_forces(arr)
            mean_by_frame.append(arr.mean(axis=0));peak_by_frame.append(arr.max(axis=0))
            frac_by_frame.append(active.mean(axis=0));triad_by_frame.append(float(triad.mean()))
        else:
            mean_by_frame.append(np.zeros(5));peak_by_frame.append(np.zeros(5))
            frac_by_frame.append(np.zeros(5));triad_by_frame.append(0.)
    forces=np.asarray(forces);frames=np.asarray(frames)
    active,triad=active_digit_forces(forces)
    carry=(frames>=42)&(frames<126)
    release=(frames>=126)&(frames<151)
    result={'carry_digit_active_fraction':active[carry].mean(axis=0).tolist(),
            'carry_digit_mean_force_n':forces[carry].mean(axis=0).tolist(),
            'carry_thumb_plus_two_fraction':float(triad[carry].mean()),
            'release_digit_active_fraction':active[release].mean(axis=0).tolist(),
            'release_thumb_plus_two_fraction':float(triad[release].mean()),
            'terminal_digit_active_fraction':active[frames>=175].mean(axis=0).tolist()}
    trace={'digit_force_step_n':forces,'digit_step_frame':frames,
           'digit_force_mean_by_frame_n':np.asarray(mean_by_frame),
           'digit_force_peak_by_frame_n':np.asarray(peak_by_frame),
           'digit_active_step_fraction_by_frame':np.asarray(frac_by_frame),
           'thumb_plus_two_step_fraction_by_frame':np.asarray(triad_by_frame)}
    return result,trace


def export_commands(path,commands,plan,names,h,residual,index):
    np.savez_compressed(path,schema='shadow_polisher_commands_v1',commands=commands,
        timestamps=plan['timestamps'],joint_names=np.asarray(names),
        plan_sha256=h['plan'],track_sha256=h['track'],urdf_sha256=h['urdf'],
        mesh_hashes_json=json.dumps(h['meshes'],sort_keys=True),carry_timing_factor=1.,
        palm_input_sha256=h['palm_input'],multifinger_residual=np.asarray(residual,dtype=float),
        multifinger_candidate_index=int(index))


def jsonwrite(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/app/rl-output/shadow_multifinger_polisher_20261025')
    parser.add_argument('--replay',help='independently replay compatible final_commands.npz')
    args=parser.parse_args()
    out=Path(args.output)
    if any((out/name).exists() for name in ('manifest.json','feasibility.jsonl','static_ledger.jsonl','full_ledger.jsonl')):
        raise RuntimeError('output already contains a run; select a fresh directory')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    plan,track,names,baseline,lower,upper,velocity=source()
    h=hashes();palm=load_palm(names,plan,baseline,h)
    model,table_z,xml=make_scene(plan,track,names)
    if args.replay:
        with np.load(args.replay,allow_pickle=False) as saved:
            commands,timestamps,factor=validate_command_export(saved,
                {**expected_export_hashes(h),'palm_input_sha256':h['palm_input']},
                names,plan['timestamps'],baseline.shape,baseline.dtype)
            residual=np.asarray(saved['multifinger_residual'],dtype=float)
        if factor!=1. or not np.array_equal(commands,multifinger_commands(palm,names,residual)):
            raise ValueError('exported commands do not match declared bounded multifinger residual')
        pm,vm=command_margins(commands,timestamps,lower,upper,velocity)
        if pm< -1e-6 or vm< -1e-6:raise ValueError('replayed command violates URDF limits')
        base=rollout(model,plan,track,names,commands,table_z)
        digits,trace=full_digit_trace(model,plan,names,commands)
        hold,holdtrace=static_hold(model,plan,names,commands,full=True)
        report={**base,**digits,'static_hold':hold,'command_sha256':sha(Path(args.replay))}
        jsonwrite(out/'replay_report.json',report)
        np.savez_compressed(out/'replay_digit_trace.npz',**trace)
        np.savez_compressed(out/'replay_static_trace.npz',**holdtrace)
        print(json.dumps(report,indent=2),flush=True)
        return
    jsonwrite(out/'protocol.json',{'protocol':PROTOCOL,'bounds_rad':BOUNDS.tolist(),
        'candidates':[x.tolist() for x in candidate_parameters()]})
    (out/'scene.xml').write_text(xml)
    kin=build_kinematics(plan)
    palm_floor,palm_self,_=kinematic_margins(kin,names,palm,range(190))
    if min(palm_floor,palm_self)<-1e-7:raise RuntimeError('palm control violates floor/self screen')
    manifest={'schema':'shadow_multifinger_pilot_v1','protocol':PROTOCOL,'source_hashes':h,
              'palm_command_sha256':hashlib.sha256(palm.tobytes()).hexdigest(),
              'baseline_command_sha256':hashlib.sha256(baseline.tobytes()).hexdigest(),
              'physics':'unchanged make_scene and rollout from hocap_shadow_polisher.py',
              'collision_screen':'per-candidate passive fingers in Pinocchio; floor plus nearest eight retained URDF self pairs each frame; adjacent/nominal pairs excluded; scene object collisions only measured in MuJoCo',
              'palm_floor_margin_m':palm_floor,'palm_self_margin_m':palm_self,
              'accepted':False}
    jsonwrite(out/'manifest.json',manifest)
    cand_dir=out/'candidates';cand_dir.mkdir()
    candidates=candidate_parameters();feasibility=[]
    for i,residual in enumerate(candidates):
        command=multifinger_commands(palm,names,residual)
        pm,vm=command_margins(command,plan['timestamps'],lower,upper,velocity)
        row={'index':i,'residual_rad':residual.tolist(),'position_margin_rad':pm,'velocity_margin_rad_s':vm}
        if pm< -1e-6 or vm< -1e-6:
            row['status']='rejected_limits'
        else:
            floor,selfmargin,frame=kinematic_margins(kin,names,command,range(11,151))
            row.update(floor_margin_m=floor,screened_self_collision_margin_m=selfmargin)
            if frame is not None:
                row.update(status='rejected_floor_or_self',rejected_frame=frame)
            else:
                row['status']='feasible'
                path=cand_dir/f'{i:02d}.npz'
                export_commands(path,command,plan,names,h,residual,i)
                row.update(command_file=str(path.relative_to(out)),command_sha256=sha(path))
        feasibility.append(row)
        with (out/'feasibility.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        print(json.dumps({'phase':'feasibility','index':i,'status':row['status'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    if feasibility[0]['status']!='feasible':raise RuntimeError('palm zero residual rejected')
    manifest.update(feasible_count=sum(x['status']=='feasible' for x in feasibility),feasibility_wall_s=time.monotonic()-started)
    jsonwrite(out/'manifest.json',manifest)
    static=[]
    for row in feasibility:
        if row['status']!='feasible':continue
        with np.load(out/row['command_file'],allow_pickle=False) as saved:cmd=saved['commands'].copy()
        metrics=static_hold(model,plan,names,cmd)
        item={'index':row['index'],'metrics':metrics}
        static.append(item)
        with (out/'static_ledger.jsonl').open('a') as file:file.write(json.dumps(item)+'\n')
        print(json.dumps({'phase':'static','index':row['index'],'score':round(metrics['static_score'],2),'pass':metrics['static_pass'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    static.sort(key=lambda x:(x['metrics']['static_score'],x['index']))
    airborne=[x for x in static if x['metrics']['airborne_unpinned']]
    shortlist=[x['index'] for x in (airborne if airborne else static)[:8]]
    manifest['shortlist_fallback_no_airborne_candidate']=not bool(airborne)
    # Always include zero palm control in full replay, independent of its shortlist rank.
    full=[]
    for i in list(dict.fromkeys([0]+shortlist)):
        row=feasibility[i]
        with np.load(out/row['command_file'],allow_pickle=False) as saved:cmd=saved['commands'].copy()
        metrics=rollout(model,plan,track,names,cmd,table_z)
        digit,trace=full_digit_trace(model,plan,names,cmd)
        hold=next(x['metrics'] for x in static if x['index']==i)
        rank=float(metrics['score']+100*(1-digit['carry_thumb_plus_two_fraction']))
        item={'index':i,'metrics':metrics,'digits':digit,'static_hold':hold,'selection_score':rank}
        full.append(item)
        with (out/'full_ledger.jsonl').open('a') as file:file.write(json.dumps(item)+'\n')
        print(json.dumps({'phase':'full','index':i,'score':round(rank,2),'target_success':metrics['target_task_success'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    baseline_metrics=rollout(model,plan,track,names,baseline,table_z)
    baseline_digits,baseline_trace=full_digit_trace(model,plan,names,baseline)
    baseline_hold=static_hold(model,plan,names,baseline)
    jsonwrite(out/'baseline_control.json',{'metrics':baseline_metrics,'digits':baseline_digits,'static_hold':baseline_hold})
    palm_control=next(x for x in full if x['index']==0)
    selected=min(full,key=lambda x:(not x['metrics']['target_task_success'],x['selection_score'],x['index']))
    index=selected['index'];residual=candidates[index]
    with np.load(out/feasibility[index]['command_file'],allow_pickle=False) as saved:selected_cmd=saved['commands'].copy()
    export_commands(out/'final_commands.npz',selected_cmd,plan,names,h,residual,index)
    selected_report,trajectory=rollout(model,plan,track,names,selected_cmd,table_z,full=True)
    selected_digits,digit_trace=full_digit_trace(model,plan,names,selected_cmd)
    selected_hold,hold_trace=static_hold(model,plan,names,selected_cmd,full=True)
    np.savez_compressed(out/'trajectory.npz',**trajectory,**digit_trace)
    np.savez_compressed(out/'static_trace.npz',**hold_trace)
    sensitivity=[]
    for mass in (.06,.10):
        for friction in (.9,1.2):
            varied,varied_z,_=make_scene(plan,track,names,mass=mass,friction=friction)
            for label,cmd in (('baseline',baseline),('palm_control',palm),('selected_diagnostic',selected_cmd)):
                metrics=rollout(varied,plan,track,names,cmd,varied_z)
                digits,_=full_digit_trace(varied,plan,names,cmd)
                sensitivity.append({'mass_kg':mass,'robot_object_friction':friction,
                                    'trajectory':label,'metrics':metrics,'digits':digits})
    jsonwrite(out/'sensitivity.json',sensitivity)
    # Replay independently from the exported NPZ, not the in-memory candidate.
    with np.load(out/'final_commands.npz',allow_pickle=False) as saved:
        checked,timestamps,factor=validate_command_export(saved,
            {**expected_export_hashes(h),'palm_input_sha256':h['palm_input']},
            names,plan['timestamps'],baseline.shape,baseline.dtype)
    if factor!=1. or not np.array_equal(checked,selected_cmd):raise RuntimeError('export reproduction mismatch')
    reproduced=rollout(model,plan,track,names,checked,table_z)
    if reproduced!=selected_report:raise RuntimeError('saved command replay changed task metrics')
    report={'schema':'shadow_multifinger_report_v1','selected_index':index,'selected_residual_rad':residual.tolist(),
            'selected_metrics':selected_report,'selected_digits':selected_digits,'selected_static_hold':selected_hold,
            'palm_control':palm_control,'baseline_control':{'metrics':baseline_metrics,'digits':baseline_digits,'static_hold':baseline_hold},
            'selected_sampled_actual_velocity_margin_rad_s':sampled_joint_velocity_margin(trajectory['q_actual'],plan['timestamps'],velocity),
            'replayed_export_exact':True,
            'accepted':bool(selected_report['target_task_success'] and selected_report['full_scene_success'] and selected_hold['static_pass'])}
    jsonwrite(out/'report.json',report)
    manifest.update(shortlist_indices=shortlist,selected_index=index,selected_residual_rad=residual.tolist(),
                    selected_metrics=selected_report,selected_digits=selected_digits,selected_static_hold=selected_hold,
                    selected_command_sha256=sha(out/'final_commands.npz'),
                    accepted=report['accepted'],total_wall_s=time.monotonic()-started)
    jsonwrite(out/'manifest.json',manifest)
    print(json.dumps({'selected_index':index,'static_pass':selected_hold['static_pass'],
        'target_success':selected_report['target_task_success'],'full_scene_success':selected_report['full_scene_success'],
        'total_wall_s':manifest['total_wall_s']},indent=2),flush=True)

if __name__=='__main__':main()

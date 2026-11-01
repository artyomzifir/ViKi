"""Two-arm geometry versus uncalibrated visual-pressure-hint Shadow ablation.

The local optimizer fixes source actual object poses and linearly approximates
robot q_actual under command offsets. Only complete unforced rollouts can test
whether finger force and grasp actually improve.
"""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from experiments.hocap_shadow_polisher import EP,PLAN,TRACK,URDF,source,make_scene,rollout,sha
from experiments.hocap_shadow_multifinger_polisher import (hashes as multifinger_hashes,
    load_palm,kinematic_margins,static_hold,full_digit_trace)
from experiments.hocap_shadow_palm_polisher import build_kinematics,sampled_joint_velocity_margin
from experiments.shadow_multifinger_controls import BOUNDS,multifinger_commands
from experiments.shadow_polisher_controls import command_margins,validate_command_export
from experiments.shadow_pressure_hint_controls import carry_pressure_weights,gap_loss_mm

HINTS=Path('/app/rl-output/pressure_hint_20261026/pressure_hints.npz')
SOURCE_TRAJECTORY=Path('/app/rl-output/pressure_hint_20261026/source_robot_trajectory.npz')
SOURCE_COMMAND=Path('/app/rl-output/shadow_multifinger_followup_20261025/final_commands.npz')
SNAPSHOTS=(60,90,110)
COORDINATES=(1,2,4,5,6)
BACKTRACK=(1.,.75,.5,.25)
SOURCE_RESIDUAL=np.array([.1,.08,0.,0.,0.,0.,0.,0.])
PROTOCOL={'schema':'shadow_pressure_hint_ablation_v1','source':'saved MF+.10 RF+.08 grasp, exact same arm and timestamps',
    'arms':['geometry weights RF/LF [1,1]','same geometry + frozen visual weights 1+normalized median(expected_pressure_proxy*auxiliary_digit_participation)'],
    'visual_semantics':'uncalibrated RGB heuristic, not ground-truth contact force or newtons; hard decoded pressure near zero',
    'parameters':['RF flexion','LF flexion','RFJ4 abduction','LFJ4 abduction','LFJ5 cup'],
    'bounds':'existing shadow_multifinger_controls.BOUNDS for total residual; MF/TH/FF and arm fixed',
    'snapshots':SNAPSHOTS,'local_state_model':'q_actual_candidate≈saved source q_actual+(candidate command−source command); source actual object pose fixed only in geometry surrogate',
    'objective':'RF/LF minimum positive MuJoCo geom gap target0..1mm at each snapshot; extra squared penalty below−2mm; residual regularizer2 normalized by axis span',
    'optimizer':'L-BFGS-B same source start, same bounds, maxiter120, numeric finite-difference gradient',
    'backtrack_scales':BACKTRACK,'selection':'first feasible scale for each arm after unchanged joint/actual-timestamp velocity, passive-finger floor and screened self checks',
    'physics':'one scored full 190-frame rollout and one 2s static hold per feasible arm; separate deterministic physics-step digit-force trace; no changed physics',
    'criteria':'compare RF/LF force>.05N, thumb+two force, static hold and original strict task/full-scene gates; no claim from local geometry score alone'}


def jsonwrite(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def full_residual(x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(5,) or not np.isfinite(x).all():raise ValueError('invalid RF/LF shape')
    r=SOURCE_RESIDUAL.copy();r[list(COORDINATES)]=x
    return r


def geometry_context(model,names,source_commands,snapshot):
    obj=model.body('object_2').id;object_geom=model.geom('object_geom_2').id
    addresses=np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    object_qpos=model.jnt_qposadr[model.body_jntadr[obj]]
    fingers={digit:[i for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name.startswith('rh_'+digit)
        and (model.geom_contype[i] or model.geom_conaffinity[i])]
        for digit in ('rf','lf')}
    if any(not geoms for geoms in fingers.values()):raise RuntimeError('RF/LF geoms missing')
    data=mujoco.MjData(model)
    return data,addresses,object_qpos,object_geom,fingers


def snapshot_gaps(model,context,source_commands,trajectory,candidate_commands):
    data,addresses,object_qpos,object_geom,fingers=context
    gaps=np.empty((len(SNAPSHOTS),2),dtype=float)
    for row,frame in enumerate(SNAPSHOTS):
        data.qpos[addresses]=trajectory['q_actual'][frame]+(
            candidate_commands[frame].astype(float)-source_commands[frame].astype(float))
        data.qpos[object_qpos:object_qpos+3]=trajectory['object_positions'][frame]
        data.qpos[object_qpos+3:object_qpos+7]=Rotation.from_matrix(
            trajectory['object_rotations'][frame]).as_quat(scalar_first=True)
        mujoco.mj_forward(model,data)
        for col,digit in enumerate(('rf','lf')):
            gaps[row,col]=min(float(mujoco.mj_geomDistance(model,data,geom,object_geom,.2,np.zeros(6)))
                              for geom in fingers[digit])*1000
    return gaps


def export(path,commands,plan,names,hashes,residual,mode,pressure_hash,trajectory_hash):
    np.savez_compressed(path,schema='shadow_polisher_commands_v1',commands=commands,
        timestamps=plan['timestamps'],joint_names=np.asarray(names),
        plan_sha256=hashes['plan'],track_sha256=hashes['track'],urdf_sha256=hashes['urdf'],
        mesh_hashes_json=json.dumps(hashes['meshes'],sort_keys=True),carry_timing_factor=1.,
        palm_input_sha256=hashes['palm_input'],multifinger_residual=residual,
        multifinger_candidate_index=-1,pressure_hint_mode=mode,
        pressure_hint_sha256=pressure_hash,source_actual_trajectory_sha256=trajectory_hash)


def load_export(path,plan,names,palm,baseline,hashes,pressure_hash,trajectory_hash):
    expected={'schema':'shadow_polisher_commands_v1','plan_sha256':hashes['plan'],
        'track_sha256':hashes['track'],'urdf_sha256':hashes['urdf'],
        'mesh_hashes_json':json.dumps(hashes['meshes'],sort_keys=True),
        'palm_input_sha256':hashes['palm_input'],
        'pressure_hint_sha256':pressure_hash,'source_actual_trajectory_sha256':trajectory_hash}
    with np.load(path,allow_pickle=False) as saved:
        commands,_,factor=validate_command_export(saved,expected,names,plan['timestamps'],baseline.shape,baseline.dtype)
        residual=np.asarray(saved['multifinger_residual'],dtype=float)
        mode=str(saved['pressure_hint_mode'])
    if factor!=1. or mode not in ('geometry','visual_hint') or not np.array_equal(
        commands,multifinger_commands(palm,names,residual)):
        raise ValueError('export does not reconstruct bounded multifinger command')
    return commands,residual,mode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/app/rl-output/shadow_pressure_hint_ablation_20261026')
    parser.add_argument('--replay',help='replay one saved compatible command independently')
    args=parser.parse_args();out=Path(args.output)
    if any((out/x).exists() for x in ('manifest.json','optimization.jsonl','screening.jsonl','replay_report.json')):
        raise RuntimeError('existing run; choose new output')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    plan,track,names,baseline,lower,upper,velocity=source()
    hashes=multifinger_hashes();palm=load_palm(names,plan,baseline,hashes)
    with np.load(SOURCE_COMMAND,allow_pickle=False) as saved:
        expected={'schema':'shadow_polisher_commands_v1','plan_sha256':hashes['plan'],
            'track_sha256':hashes['track'],'urdf_sha256':hashes['urdf'],
            'mesh_hashes_json':json.dumps(hashes['meshes'],sort_keys=True),
            'palm_input_sha256':hashes['palm_input']}
        source_commands,_,factor=validate_command_export(saved,expected,names,plan['timestamps'],baseline.shape,baseline.dtype)
    if factor!=1. or not np.array_equal(source_commands,multifinger_commands(palm,names,SOURCE_RESIDUAL)):
        raise ValueError('reviewed source command mismatch')
    pressure_hash=sha(HINTS);trajectory_hash=sha(SOURCE_TRAJECTORY)
    model,table_z,xml=make_scene(plan,track,names)
    if args.replay:
        commands,residual,mode=load_export(Path(args.replay),plan,names,palm,baseline,hashes,pressure_hash,trajectory_hash)
        metrics,trajectory=rollout(model,plan,track,names,commands,table_z,full=True)
        digits,digit_trace=full_digit_trace(model,plan,names,commands)
        hold,hold_trace=static_hold(model,plan,names,commands,full=True)
        report={'mode':mode,'residual':residual.tolist(),'metrics':metrics,'digits':digits,
            'static_hold':hold,'command_sha256':sha(Path(args.replay))}
        jsonwrite(out/'replay_report.json',report)
        np.savez_compressed(out/'replay_trajectory.npz',**trajectory,**digit_trace)
        np.savez_compressed(out/'replay_hold_trace.npz',**hold_trace)
        print(json.dumps(report,indent=2),flush=True);return
    with np.load(HINTS,allow_pickle=False) as z:
        if list(map(str,z['digits']))!=['thumb','index','middle','ring','little']:
            raise ValueError('unexpected visual digit order')
        median,visual_weights=carry_pressure_weights(z['fused_expected_pressure_proxy_kpa'],z['fused_digit_participation'])
    with np.load(SOURCE_TRAJECTORY,allow_pickle=False) as z:
        snapshot={k:z[k] for k in ('object_positions','object_rotations','q_actual')}
    source_metrics,source_trajectory=rollout(model,plan,track,names,source_commands,table_z,full=True)
    for key in snapshot:
        if not np.array_equal(snapshot[key],source_trajectory[key]):
            raise ValueError(f'pressure-run source trajectory differs from frozen physics: {key}')
    source_digits,source_digit_trace=full_digit_trace(model,plan,names,source_commands)
    source_hold,source_hold_trace=static_hold(model,plan,names,source_commands,full=True)
    context=geometry_context(model,names,source_commands,snapshot)
    source_gaps=snapshot_gaps(model,context,source_commands,snapshot,source_commands)
    bounds=BOUNDS[list(COORDINATES)];x0=SOURCE_RESIDUAL[list(COORDINATES)]
    span=bounds[:,1]-bounds[:,0]
    protocol={**PROTOCOL,'source_hashes':{**hashes,'source_command':sha(SOURCE_COMMAND),'pressure_hints':pressure_hash,'actual_trajectory':trajectory_hash},
        'source_residual':SOURCE_RESIDUAL.tolist(),'optimized_coordinate_bounds_rad':bounds.tolist(),
        'visual_hint_product_median':median.tolist(),'frozen_visual_weights_rf_lf':visual_weights.tolist(),
        'geometry_weights_rf_lf':[1.,1.],'source_local_gaps_mm':source_gaps.tolist()}
    jsonwrite(out/'protocol.json',protocol)
    (out/'scene.xml').write_text(xml)
    manifest={'schema':'shadow_pressure_hint_ablation_run_v1','protocol':protocol,'accepted':False}
    jsonwrite(out/'manifest.json',manifest)
    jsonwrite(out/'source_control.json',{'metrics':source_metrics,'digits':source_digits,'static_hold':source_hold})
    np.savez_compressed(out/'source_trajectory.npz',**source_trajectory,**source_digit_trace)
    np.savez_compressed(out/'source_hold_trace.npz',**source_hold_trace)
    kin=build_kinematics({**plan,'shadow_q':source_commands[:,6:]})
    outcomes=[]
    for mode,weights in (('geometry',np.ones(2)),('visual_hint',visual_weights)):
        evaluations=0
        def objective(x):
            nonlocal evaluations
            evaluations+=1
            command=multifinger_commands(palm,names,full_residual(x))
            gaps=snapshot_gaps(model,context,source_commands,snapshot,command)
            return gap_loss_mm(gaps,weights,x-x0,span)
        initial_score=objective(x0)
        optimized=minimize(objective,x0,bounds=[tuple(v) for v in bounds],method='L-BFGS-B',
            options={'maxiter':120,'maxfun':900,'ftol':1e-9})
        x=np.asarray(optimized.x,dtype=float)
        opt_gaps=snapshot_gaps(model,context,source_commands,snapshot,
            multifinger_commands(palm,names,full_residual(x)))
        opt_row={'mode':mode,'weights':weights.tolist(),'x0':x0.tolist(),'x_opt':x.tolist(),
            'initial_score':initial_score,'optimized_score':float(optimized.fun),
            'optimizer_success':bool(optimized.success),'optimizer_message':str(optimized.message),
            'iterations':int(optimized.nit),'function_evaluations':evaluations,
            'optimized_local_gaps_mm':opt_gaps.tolist()}
        with (out/'optimization.jsonl').open('a') as file:file.write(json.dumps(opt_row)+'\n')
        print(json.dumps({'phase':'optimization','mode':mode,'iterations':optimized.nit,
            'score':float(optimized.fun),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
        selected=None
        for scale in BACKTRACK:
            proposal=x0+scale*(x-x0)
            residual=full_residual(proposal)
            command=multifinger_commands(palm,names,residual)
            pm,vm=command_margins(command,plan['timestamps'],lower,upper,velocity)
            row={'mode':mode,'scale':scale,'five_parameters_rad':proposal.tolist(),
                'total_residual_rad':residual.tolist(),'position_margin_rad':pm,
                'command_velocity_margin_rad_s':vm}
            if pm< -1e-6 or vm< -1e-6:row['status']='rejected_limits'
            else:
                floor,self_margin,bad_frame=kinematic_margins(kin,names,command,range(10,152))
                row.update(floor_margin_m=floor,screened_self_margin_m=self_margin,bad_frame=bad_frame)
                row['status']='feasible' if bad_frame is None else 'rejected_floor_or_self'
            with (out/'screening.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            if row['status']=='feasible':selected=(scale,residual,command,row);break
        if selected is None:
            outcomes.append({'mode':mode,'status':'no_feasible_backtrack','optimization':opt_row})
            continue
        scale,residual,command,screen=selected
        path=out/f'{mode}_commands.npz'
        export(path,command,plan,names,hashes,residual,mode,pressure_hash,trajectory_hash)
        metrics,trajectory=rollout(model,plan,track,names,command,table_z,full=True)
        digits,digit_trace=full_digit_trace(model,plan,names,command)
        hold,hold_trace=static_hold(model,plan,names,command,full=True)
        actual_margin=sampled_joint_velocity_margin(trajectory['q_actual'],plan['timestamps'],velocity)
        gaps=snapshot_gaps(model,context,source_commands,snapshot,command)
        np.savez_compressed(out/f'{mode}_trajectory.npz',**trajectory,**digit_trace)
        np.savez_compressed(out/f'{mode}_hold_trace.npz',**hold_trace)
        outcome={'mode':mode,'status':'evaluated','scale':scale,'residual':residual.tolist(),
            'optimization':opt_row,'screen':screen,'local_surrogate_gaps_mm':gaps.tolist(),
            'metrics':metrics,'digits':digits,'static_hold':hold,
            'sampled_actual_velocity_margin_rad_s':actual_margin,
            'command_sha256':sha(path),'accepted':False}
        outcomes.append(outcome)
        jsonwrite(out/f'{mode}_report.json',outcome)
        print(json.dumps({'phase':'physics','mode':mode,'rf_lf_contact':digits['carry_digit_active_fraction'][3:5],
            'triad':digits['carry_thumb_plus_two_fraction'],'static_pass':hold['static_pass'],
            'target_pass':metrics['target_task_success'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    report={'schema':'shadow_pressure_hint_ablation_report_v1','source':{'metrics':source_metrics,'digits':source_digits,'static_hold':source_hold,'gaps_mm':source_gaps.tolist()},
        'arms':outcomes,'visual_weights_rf_lf':visual_weights.tolist(),
        'accepted':False,'scope':'two fixed optimizer arms and first-feasible backtrack only; no production insertion'}
    jsonwrite(out/'report.json',report)
    manifest.update(outcomes=[{'mode':x['mode'],'status':x['status']} for x in outcomes],
        wall_s=time.monotonic()-started,accepted=False)
    jsonwrite(out/'manifest.json',manifest)
    print(json.dumps({'wall_s':manifest['wall_s'],'outcomes':manifest['outcomes']},indent=2),flush=True)

if __name__=='__main__':main()

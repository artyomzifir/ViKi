"""Bounded object-frame palm-pose hypothesis on the frozen G10_2 Shadow plan.

The arm alone is re-solved by FK/IK. Finger commands, timestamps, source scene,
physics, score and acceptance gates come from hocap_shadow_polisher.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from experiments.hocap_shadow_polisher import EP, PLAN, TRACK, URDF, source, make_scene, rollout, sha
from experiments.shadow_palm_controls import PALM_BOUNDS, palm_gate, object_frame_palm_target, solve_arm_pose, validate_ik_errors, zero_palm_commands, sampled_joint_velocity_margin
from experiments.shadow_polisher_controls import command_margins
from viki.retarget.frames import base_rotation
from viki.retarget.robots import normalize_robot
from viki.retarget.solver import PinocchioKinematics


PROTOCOL = {
    'schema':'shadow_palm_polisher_protocol_v1', 'seed':20261025,
    'family':'object-frame XYZ translation ±12 mm and XYZ rotation-vector ±8 deg; rotate palm about its own origin',
    'gate':'C1 rise frames 10..23, hold 23..110, fall 110..125; source fingers and timestamps fixed',
    'candidate_count':48, 'axis_probes':'each of 6 axes at +/-half and +/-full bound (24)',
    'random_candidates':'23 Latin-hypercube, fixed seed',
    'position_ik_tol_m':.0015, 'orientation_ik_tol_deg':1.,
    'floor_margin_m':0., 'screened_self_collision_margin_m':0.,
    'ranking':'same frozen score and target/full-scene gates as 2026-10-25 hand-command pilot',
    'held_out_grid':{'mass_kg':[.06,.10],'robot_object_friction':[.9,1.2]},
    'selection':'minimum frozen score across feasible nominal rollouts, including zero',
}


def build_kinematics(plan):
    model = pin.buildModelFromUrdf(str(URDF))
    collision = pin.buildGeomFromUrdf(model, str(URDF), pin.GeometryType.COLLISION)
    robot = SimpleNamespace(model=model, collision_model=collision)
    config = normalize_robot('ur10')
    passive = {str(name):plan['shadow_q'][:,i].astype(float)
               for i,name in enumerate(plan['shadow_joint_names'])}
    kin = PinocchioKinematics(robot, 'rh_palm', collision_pairs=8,
        collision_min_distance_m=.001, actuated_joint_names=config.joint_names,
        actuated_position_limits=config.position_limits, reference_q=np.zeros(6),
        passive_joint_positions=passive, gripper_prefix='rh_',
        floor_normal=(0,0,1), floor_offset_m=0.)
    return kin


def world_fk_sanity(kin, scene, plan, names, baseline, frames=(0,42,90,125,189)):
    """Verify Pin wrist and fixed palm transform against the MuJoCo URDF import."""
    data = mujoco.MjData(scene)
    addresses = np.array([scene.jnt_qposadr[scene.joint(n).id] for n in names])
    base_r = base_rotation(plan['base_rpy_deg'])
    base_p = np.asarray(plan['base_position'])
    wrist_pin_id = kin.model.getFrameId('wrist_3_link', pin.FrameType.BODY)
    wrist_mj_id = scene.body('wrist_3_link').id
    results = []
    for frame in frames:
        data.qpos[addresses] = baseline[frame]
        data.ctrl[:] = baseline[frame]
        mujoco.mj_forward(scene,data)
        kin.set_frame(frame)
        palm_p, palm_r = kin.pose(baseline[frame,:6])
        wrist_pin = kin.data.oMf[wrist_pin_id]
        wrist_world_p = base_r @ np.asarray(wrist_pin.translation) + base_p
        wrist_world_r = base_r @ np.asarray(wrist_pin.rotation)
        wrist_mj_p = data.xpos[wrist_mj_id].copy()
        wrist_mj_r = data.xmat[wrist_mj_id].reshape(3,3).copy()
        # Transport Pin's fixed wrist->palm transform through the MuJoCo wrist.
        relative_p = np.asarray(wrist_pin.rotation).T @ (palm_p-np.asarray(wrist_pin.translation))
        relative_r = np.asarray(wrist_pin.rotation).T @ palm_r
        palm_from_mj_p = wrist_mj_p + wrist_mj_r @ relative_p
        palm_from_mj_r = wrist_mj_r @ relative_r
        expected_p = base_r @ palm_p + base_p
        expected_r = base_r @ palm_r
        results.append({'frame':frame,
            'wrist_position_mm':float(np.linalg.norm(wrist_world_p-wrist_mj_p)*1000),
            'wrist_angle_deg':float(np.rad2deg(Rotation.from_matrix(wrist_world_r@wrist_mj_r.T).magnitude())),
            'palm_position_mm':float(np.linalg.norm(palm_from_mj_p-expected_p)*1000),
            'palm_angle_deg':float(np.rad2deg(Rotation.from_matrix(palm_from_mj_r@expected_r.T).magnitude()))})
    if max(max(x['wrist_position_mm'],x['palm_position_mm']) for x in results) > 2. or max(
        max(x['wrist_angle_deg'],x['palm_angle_deg']) for x in results) > 1.:
        raise RuntimeError(f'Pinocchio/MuJoCo world-frame mismatch: {results}')
    return results


def candidate_parameters():
    candidates = [np.zeros(6)]
    upper = PALM_BOUNDS[:,1]
    for axis in range(6):
        for scale in (-1.,-.5,.5,1.):
            residual = np.zeros(6)
            residual[axis] = scale*upper[axis]
            candidates.append(residual)
    sampler = qmc.LatinHypercube(d=6, seed=20261025)
    candidates.extend(qmc.scale(sampler.random(23), PALM_BOUNDS[:,0], PALM_BOUNDS[:,1]))
    assert len(candidates)==48
    return candidates


def baseline_fk_and_margins(kin, baseline):
    palms_p, palms_r, floors, collisions = [], [], [], []
    for t,q in enumerate(baseline):
        kin.set_frame(t)
        p,r = kin.pose(q[:6])
        palms_p.append(p);palms_r.append(r)
        floors.append(float(np.min(kin.floor_margins(q[:6]))))
        collisions.append(float(np.min(kin.collision_margins(q[:6]))))
    return np.asarray(palms_p),np.asarray(palms_r),np.asarray(floors),np.asarray(collisions)


def construct_candidate(kin, baseline, timestamps, lower, upper, velocity,
                        palm_positions, palm_rotations, object_rotations_robot,
                        base_floors, base_collisions, residual):
    residual = np.asarray(residual,dtype=float)
    if residual.shape!=(6,) or not np.isfinite(residual).all() or np.any(residual<PALM_BOUNDS[:,0]) or np.any(residual>PALM_BOUNDS[:,1]):
        raise ValueError('invalid palm residual')
    commands = zero_palm_commands(baseline)  # original float32 dtype is intentional
    if np.all(residual==0):
        pm,vm = command_margins(commands,timestamps,lower,upper,velocity)
        details = {'max_ik_position_mm':0.,'max_ik_orientation_deg':0.,
                   'min_floor_margin_m':float(base_floors.min()),
                   'min_screened_self_collision_margin_m':float(base_collisions.min()),
                   'position_margin_rad':pm,'velocity_margin_rad_s':vm}
        valid = pm >= -1e-6 and vm >= -1e-6 and base_floors.min() >= -1e-7 and base_collisions.min() >= -1e-7
        return (commands if valid else None), {'status':'feasible' if valid else 'rejected_baseline_limits',**details}
    gate = palm_gate(len(commands))
    position_errors = []
    orientation_errors = []
    floors = base_floors.copy()
    collisions = base_collisions.copy()
    for t in np.flatnonzero(gate>0):
        target_p,target_r = object_frame_palm_target(palm_positions[t],palm_rotations[t],
            object_rotations_robot[t],residual,float(gate[t]))
        q,_,_ = solve_arm_pose(kin,int(t),baseline[t,:6],target_p,target_r)
        commands[t,:6] = q.astype(np.float32)
        kin.set_frame(int(t))
        achieved_p,achieved_r = kin.pose(commands[t,:6])
        position_errors.append(float(np.linalg.norm(achieved_p-target_p)))
        orientation_errors.append(float(Rotation.from_matrix(target_r@achieved_r.T).magnitude()))
        if not validate_ik_errors(position_errors[-1:],orientation_errors[-1:]):
            return None,{'status':'rejected_ik','frame':int(t),
                'position_error_mm':position_errors[-1]*1000,
                'orientation_error_deg':float(np.rad2deg(orientation_errors[-1]))}
        floors[t] = float(np.min(kin.floor_margins(commands[t,:6])))
        collisions[t] = float(np.min(kin.collision_margins(commands[t,:6])))
        if floors[t] < -1e-7 or collisions[t] < -1e-7:
            return None,{'status':'rejected_collision_or_floor','frame':int(t),
                'floor_margin_m':floors[t], 'screened_self_collision_margin_m':collisions[t]}
    pm,vm = command_margins(commands,timestamps,lower,upper,velocity)
    details = {'max_ik_position_mm':float(max(position_errors)*1000),
               'max_ik_orientation_deg':float(np.rad2deg(max(orientation_errors))),
               'min_floor_margin_m':float(floors.min()),
               'min_screened_self_collision_margin_m':float(collisions.min()),
               'position_margin_rad':pm,'velocity_margin_rad_s':vm}
    if pm < -1e-6 or vm < -1e-6:
        return None,{'status':'rejected_limits',**details}
    if not np.array_equal(commands[:,6:],baseline[:,6:]) or not np.array_equal(commands[:11],baseline[:11]) or not np.array_equal(commands[125:],baseline[125:]):
        raise RuntimeError('finger or outside-gate commands changed')
    return commands,{'status':'feasible',**details}


def export_commands(path, commands, timestamps, names, hashes, residual, candidate_index):
    np.savez_compressed(path,schema='shadow_polisher_commands_v1',
        commands=commands,timestamps=timestamps,joint_names=np.asarray(names),
        plan_sha256=hashes['plan'],track_sha256=hashes['track'],
        urdf_sha256=hashes['urdf'],mesh_hashes_json=json.dumps(hashes['meshes'],sort_keys=True),
        carry_timing_factor=1., palm_residual=np.asarray(residual,dtype=float),
        palm_candidate_index=int(candidate_index))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/app/rl-output/shadow_palm_polisher_20261025')
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    out=Path(args.output)
    if not args.probe_only and any((out/name).exists() for name in ('manifest.json','feasibility.jsonl','rollout_ledger.jsonl')):
        raise RuntimeError(f'output already contains a palm run; choose a new directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    plan,track,names,baseline,lower,upper,velocity=source()
    hashes={'plan':sha(PLAN),'track':sha(TRACK),'urdf':sha(URDF),
            'meshes':{str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)}}
    scene,table_z,xml=make_scene(plan,track,names)
    kin=build_kinematics(plan)
    sanity=world_fk_sanity(kin,scene,plan,names,baseline)
    palms_p,palms_r,base_floors,base_collisions=baseline_fk_and_margins(kin,baseline)
    probe={'world_fk_sanity':sanity,'baseline_floor_min_m':float(base_floors.min()),
           'baseline_screened_collision_min_m':float(base_collisions.min()),
           'baseline_floor_min_frame':int(base_floors.argmin()),
           'baseline_collision_min_frame':int(base_collisions.argmin())}
    (out/'probe.json').write_text(json.dumps(probe,indent=2)+'\n')
    print(json.dumps(probe,indent=2),flush=True)
    if args.probe_only:return
    if base_floors.min() < -1e-7 or base_collisions.min() < -1e-7:
        raise RuntimeError('saved baseline violates the chosen floor/self-collision screen')
    candidates=candidate_parameters()
    col=list(map(int,track['object_ids'])).index(2)
    object_rotations_robot=np.einsum('ij,tjk->tik',base_rotation(plan['base_rpy_deg']).T,
                                   track['rotation_world_object'][:,col])
    manifest={'schema':'shadow_palm_polisher_pilot_v1','protocol':PROTOCOL,
              'parameter_bounds':PALM_BOUNDS.tolist(),
              'candidate_parameters':[x.tolist() for x in candidates],
              'source_hashes':hashes,'baseline_command_sha256':hashlib.sha256(baseline.tobytes()).hexdigest(),
              'physics_reused':'experiments/hocap_shadow_polisher.py make_scene and rollout; unchanged gates/score',
              'scene_collision_screen':'IK checks floor and nearest eight retained URDF self-collision pairs per frame; adjacent and neutral-contact pairs excluded; tracked objects checked only in MuJoCo rollout',
              'probe':probe}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'scene.xml').write_text(xml)
    candidates_dir=out/'candidates';candidates_dir.mkdir(exist_ok=True)
    feasibility=[]
    for i,residual in enumerate(candidates):
        command,details=construct_candidate(kin,baseline,plan['timestamps'],lower,upper,velocity,
            palms_p,palms_r,object_rotations_robot,base_floors,base_collisions,residual)
        entry={'index':i,'residual':residual.tolist(),**details}
        if command is not None:
            file=candidates_dir/f'{i:02d}.npz'
            export_commands(file,command,plan['timestamps'],names,hashes,residual,i)
            entry['command_file']=str(file.relative_to(out))
            entry['command_sha256']=sha(file)
        feasibility.append(entry)
        (out/'feasibility.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in feasibility))
        print(json.dumps({'phase':'ik','index':i,'status':entry['status'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    manifest['feasibility_wall_s']=time.monotonic()-started
    manifest['feasible_count']=sum(x['status']=='feasible' for x in feasibility)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if feasibility[0]['status']!='feasible':raise RuntimeError('zero residual rejected')
    ledger=[];selected_index=None;selected_score=float('inf')
    for entry in feasibility:
        if entry['status']!='feasible':continue
        with np.load(out/entry['command_file'],allow_pickle=False) as saved:
            command=saved['commands'].copy()
        metrics=rollout(scene,plan,track,names,command,table_z)
        row={'index':entry['index'],'residual':entry['residual'],'metrics':metrics}
        ledger.append(row)
        with (out/'rollout_ledger.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if metrics['score']<selected_score:
            selected_score=metrics['score'];selected_index=entry['index']
        print(json.dumps({'phase':'physics','index':entry['index'],'score':metrics['score'],
                          'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    selected=candidates[selected_index]
    command_path=out/feasibility[selected_index]['command_file']
    with np.load(command_path,allow_pickle=False) as saved: selected_command=saved['commands'].copy()
    export_commands(out/'final_commands.npz',selected_command,plan['timestamps'],names,hashes,selected,selected_index)
    selected_report,selected_trajectory=rollout(scene,plan,track,names,selected_command,table_z,full=True)
    baseline_report,baseline_trajectory=rollout(scene,plan,track,names,baseline,table_z,full=True)
    np.savez_compressed(out/'trajectory.npz',**selected_trajectory)
    np.savez_compressed(out/'baseline_trajectory.npz',**baseline_trajectory)
    baseline_actual_speed_margin = sampled_joint_velocity_margin(baseline_trajectory['q_actual'],plan['timestamps'],velocity)
    selected_actual_speed_margin = sampled_joint_velocity_margin(selected_trajectory['q_actual'],plan['timestamps'],velocity)
    sensitivity=[]
    for mass in (.06,.10):
        for friction in (.9,1.2):
            varied,varied_z,_=make_scene(plan,track,names,mass=mass,friction=friction)
            for label,command in (('baseline',baseline),('selected_diagnostic',selected_command)):
                sensitivity.append({'mass_kg':mass,'robot_object_friction':friction,'trajectory':label,
                                    'metrics':rollout(varied,plan,track,names,command,varied_z)})
    (out/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    manifest.update(selected_index=selected_index, selected_residual=selected.tolist(),
        selected_metrics=selected_report,baseline_metrics=baseline_report,
        selected_command_sha256=sha(out/'final_commands.npz'),
        baseline_sampled_actual_velocity_margin_rad_s=baseline_actual_speed_margin,
        selected_sampled_actual_velocity_margin_rad_s=selected_actual_speed_margin,
        accepted=bool(selected_report['target_task_success'] and selected_report['full_scene_success']),
        total_wall_s=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'selected_index':selected_index,'selected_metrics':selected_report,
                      'baseline_metrics':baseline_report,'feasible_count':manifest['feasible_count'],
                      'total_wall_s':manifest['total_wall_s']},indent=2),flush=True)


if __name__=='__main__':main()

"""Small, opt-in, open-loop physics search on the frozen G10_2 Shadow scene.

No object forcing, weld, contact attachment, feedback policy, or model tuning.
The exported commands are independently replayed with the same simulator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from experiments.shadow_polisher_controls import PARAMETER_BOUNDS, command_margins, polished_commands, retime_carry, terminal_placement_gate, validate_command_export
from viki.retarget.robots import normalize_robot


EP = Path('/app/data/datasets/hocap_demo_object_v4/g10_object2_190')
PLAN = EP / 'shadow_object_plan.npz'
TRACK = EP / 'intermediates/segmentation/hocap-oracle-v1/object_models.npz'
URDF = Path('/app/data/_hocap_demo_source/ur10_shadow.urdf')
PREDECLARED = {
    'carry_frames': [42, 126], 'grasp_gate_rise': [43, 58], 'grasp_gate_fall': [110, 125],
    'candidate_count': 32, 'seed': 20261025,
    'held_out_grid': {'mass_kg':[.06,.10], 'robot_object_friction':[.9,1.2]},
    'nominal_mass_kg': .08, 'nominal_robot_object_friction': 1.2,
    'task_gate': {'lift_mm': 20, 'carry_p95_mm': 30, 'final_error_mm': 30,
                  'carry_height_deficit_tolerance_mm': 50, 'neighbor_displacement_mm': 5,
                  'settled_speed_m_s': .03, 'terminal_support_fraction': 1.,
                  'released_last_frames': 15,
                  'no_robot_table_or_other_object_contacts': True},
    'ranking': 'carry p95 error mm + 60*(1 - simultaneous opposing contact fraction) + '
               '100*carry drop fraction + 0.1*final error mm + 1000*(robot-table contact exists)',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source():
    with np.load(PLAN, allow_pickle=False) as p:
        plan = {k: p[k] for k in p.files}
    with np.load(TRACK, allow_pickle=False) as p:
        track = {k: p[k] for k in p.files}
    if len(plan['timestamps']) != 190 or json.loads(str(plan['report_json'])).get('object_relative_window') != [42, 126]:
        raise ValueError('unsupported plan length or carry phase for this frozen pilot')
    names = list(normalize_robot('ur10').joint_names) + list(map(str, plan['shadow_joint_names']))
    commands = np.c_[plan['arm_q'], plan['shadow_q']]
    root = ET.parse(URDF).getroot()
    joints = {j.get('name'): j.find('limit') for j in root.findall('joint')}
    lower = np.array([float(joints[n].get('lower')) for n in names])
    upper = np.array([float(joints[n].get('upper')) for n in names])
    velocity = np.array([float(joints[n].get('velocity')) for n in names])
    return plan, track, names, commands, lower, upper, velocity


def make_scene(plan, track, names, mass=.08, friction=1.2):
    # Match hocap_shadow_replay.py's XML construction byte-for-byte in physics fields.
    root = ET.fromstring(mujoco.MjSpec.from_file(str(URDF)).to_xml())
    root.find('compiler').set('angle', 'radian')
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', '0.0005')
    option.set('integrator', 'implicitfast')
    world = root.find('worldbody')
    robot = ET.Element('body', name='robot_base', pos=' '.join(map(str, plan['base_position'])),
                       euler=' '.join(map(str, np.deg2rad(plan['base_rpy_deg']))))
    for child in list(world):
        world.remove(child)
        robot.append(child)
    for body in robot.iter('body'):
        body.set('gravcomp', '1')
    for geom in robot.iter('geom'):
        if geom.get('contype') != '0' or geom.get('conaffinity') != '0':
            geom.set('friction', f'{friction} .005 .0001')
    world.append(robot)
    assets = root.find('asset')
    if assets is None:
        assets = ET.SubElement(root, 'asset')
    table_z = 0.
    for i, oid in enumerate(track['object_ids']):
        mesh = EP / f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{oid}.obj'
        verts = np.array([[float(x) for x in line.split()[1:4]] for line in mesh.read_text().splitlines() if line.startswith('v ')])
        pos = track['translation_world'][0, i]
        rot = track['rotation_world_object'][0, i]
        table_z = min(table_z, float((verts @ rot.T + pos)[:, 2].min()))
        ET.SubElement(assets, 'mesh', name=f'object_mesh_{oid}', file=str(mesh))
        body = ET.SubElement(world, 'body', name=f'object_{oid}', pos=' '.join(map(str, pos)),
                             quat=' '.join(map(str, Rotation.from_matrix(rot).as_quat(scalar_first=True))))
        ET.SubElement(body, 'freejoint', name=f'object_free_{oid}')
        colors = ['.078 .722 .651 1', '.655 .545 .98 1', '.639 .902 .208 1', '.984 .573 .235 1']
        ET.SubElement(body, 'geom', name=f'object_geom_{oid}', type='mesh', mesh=f'object_mesh_{oid}', mass='.08' if mass == .08 else str(mass),
                      friction=f'{friction} .005 .0001', rgba=colors[i])
    ET.SubElement(world, 'geom', name='table', type='box', pos=f'0 0 {table_z-.025}',
                  size='.7 .7 .025', rgba='.25 .28 .32 1', friction='1.0 .005 .0001', priority='1')
    ET.SubElement(world, 'light', pos='0 -1 2', dir='0 .3 -1', diffuse='.8 .8 .8')
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    ET.SubElement(contact, 'exclude', body1='forearm_link', body2='wrist_2_link')
    act = ET.SubElement(root, 'actuator')
    efforts = {j.get('name'): float(j.find('limit').get('effort')) for j in ET.parse(URDF).getroot().findall('joint') if j.find('limit') is not None}
    for i, n in enumerate(names):
        ET.SubElement(act, 'position', name='cmd_' + n, joint=n, kp='1800' if i < 6 else '16',
                      kv='120' if i < 6 else '.15', forcerange=f'{-efforts[n]} {efforts[n]}')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='.45 .45 .45', diffuse='.65 .65 .65', specular='.1 .1 .1')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode')), table_z, ET.tostring(root, encoding='unicode')


def rollout(model, plan, track, names, commands, table_z, full=False):
    data = mujoco.MjData(model)
    addresses = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    data.qpos[addresses] = commands[0]
    data.ctrl[:] = commands[0]
    mujoco.mj_forward(model, data)
    obj = model.body('object_2').id
    obj_ids = [model.body(f'object_{i}').id for i in (1, 2, 3, 4)]
    for _ in range(round(.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    start = data.xpos[obj].copy()
    others_start = np.array([data.xpos[i].copy() for i in obj_ids if i != obj])
    positions, rotations, speeds, qactual, contacts = [], [], [], [], []
    force_frames, simultaneous_frames, support_frames, relative_positions = [], [], [], []
    max_step_speeds = []
    table_steps = robot_other_steps = target_table_steps = 0
    max_neighbor_displacement = 0.
    simultaneous_steps = carry_steps = drop_steps = 0
    longest_contact_gap = contact_gap = 0
    min_contact_distance = 0.
    max_thumb_force = max_finger_force = 0.
    force_saturation_steps = 0
    col = list(map(int, track['object_ids'])).index(2)
    ref = track['translation_world'][:, col]
    for t, command in enumerate(commands):
        duration = (plan['timestamps'][t] - plan['timestamps'][t-1]) / 1e6 if t else 0.
        steps = max(0, round(duration / model.opt.timestep))
        previous = commands[max(t-1, 0)]
        thumb_seen = finger_seen = False
        frame_thumb_peak = frame_finger_peak = 0.
        frame_simultaneous_steps = frame_support_steps = 0
        frame_max_speed = 0.
        for step in range(steps):
            data.ctrl[:] = previous + (command - previous) * (step + 1) / steps
            mujoco.mj_step(model, data)
            frame_max_speed = max(frame_max_speed, float(np.linalg.norm(data.cvel[obj, 3:])))
            thumb_force = finger_force = 0.
            robot_table = robot_other = target_table = False
            for ci, c in enumerate(data.contact):
                b1, b2 = [model.geom_bodyid[g] for g in (c.geom1, c.geom2)]
                n1, n2 = model.body(b1).name, model.body(b2).name
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                normal_force = max(0., float(force[0]))
                if obj in (b1, b2):
                    other = n2 if b1 == obj else n1
                    if normal_force > 1e-6:
                        if other.startswith('rh_th'):
                            thumb_force += normal_force
                            thumb_seen = True
                        elif other.startswith(('rh_ff', 'rh_mf', 'rh_rf', 'rh_lf')):
                            finger_force += normal_force
                            finger_seen = True
                    min_contact_distance = min(min_contact_distance, float(c.dist))
                    if other == 'world' and normal_force > 1e-6:
                        target_table = True
                if 'world' in (n1, n2) and ('link' in n1 or 'link' in n2 or n1.startswith('rh_') or n2.startswith('rh_')):
                    robot_table = True
                if (b1 in obj_ids or b2 in obj_ids) and ('link' in n1 or 'link' in n2 or n1.startswith('rh_') or n2.startswith('rh_')):
                    if obj not in (b1, b2):
                        robot_other = True
            table_steps += int(robot_table)
            robot_other_steps += int(robot_other)
            target_table_steps += int(target_table)
            max_thumb_force = max(max_thumb_force, thumb_force)
            max_finger_force = max(max_finger_force, finger_force)
            frame_thumb_peak = max(frame_thumb_peak, thumb_force)
            frame_finger_peak = max(frame_finger_peak, finger_force)
            frame_simultaneous_steps += int(thumb_force > 1e-6 and finger_force > 1e-6)
            frame_support_steps += int(target_table)
            max_neighbor_displacement = max(max_neighbor_displacement,
                max(np.linalg.norm(data.xpos[bid] - p0) for bid, p0 in zip((i for i in obj_ids if i != obj), others_start)))
            saturation = np.abs(data.actuator_force) >= (model.actuator_forcerange[:, 1] - 1e-4)
            force_saturation_steps += int(bool(saturation.any()))
            if 42 <= t < 126:
                carry_steps += 1
                simultaneous = thumb_force > 1e-6 and finger_force > 1e-6
                simultaneous_steps += int(simultaneous)
                contact_gap = 0 if simultaneous else contact_gap + 1
                longest_contact_gap = max(longest_contact_gap, contact_gap)
                if ref[t, 2] - start[2] > .05 and data.xpos[obj, 2] < ref[t, 2] - .05:
                    drop_steps += 1
        positions.append(data.xpos[obj].copy())
        rotations.append(data.xmat[obj].reshape(3, 3).copy())
        speeds.append(np.linalg.norm(data.cvel[obj, 3:]))
        qactual.append(data.qpos[addresses].copy())
        contacts.append((thumb_seen, finger_seen))
        force_frames.append((frame_thumb_peak, frame_finger_peak))
        simultaneous_frames.append(frame_simultaneous_steps / max(1, steps))
        support_frames.append(frame_support_steps / max(1, steps))
        max_step_speeds.append(frame_max_speed)
        wrist = model.body('wrist_3_link').id
        relative_positions.append(data.xmat[wrist].reshape(3, 3).T @ (data.xpos[obj] - data.xpos[wrist]))
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)
    qactual = np.asarray(qactual)
    contacts = np.asarray(contacts)
    force_frames = np.asarray(force_frames)
    simultaneous_frames = np.asarray(simultaneous_frames)
    support_frames = np.asarray(support_frames)
    relative_positions = np.asarray(relative_positions)
    error = np.linalg.norm(positions-ref, axis=1) * 1000
    neighbor_displacement = np.linalg.norm(np.array([data.xpos[i] for i in obj_ids if i != obj]) - others_start, axis=1) * 1000
    final_speed = float(np.linalg.norm(data.cvel[obj, 3:]))
    carry_fraction = simultaneous_steps / max(1, carry_steps)
    drop_fraction = drop_steps / max(1, carry_steps)
    carry_p95 = float(np.percentile(error[42:126], 95))
    final_error = float(error[-1])
    max_lift = float((positions[:, 2] - start[2]).max() * 1000)
    released, terminal_supported, terminal_settled = terminal_placement_gate(contacts, support_frames, max_step_speeds)
    target_success = bool(max_lift > 20 and carry_p95 < 30 and final_error < 30 and drop_steps == 0 and
                          released and terminal_settled and terminal_supported and table_steps == 0)
    full_scene_success = bool(target_success and robot_other_steps == 0 and max_neighbor_displacement*1000 < 5)
    score = float(carry_p95 + 60*(1-carry_fraction) + 100*drop_fraction + .1*final_error + 1000*(table_steps > 0))
    report = dict(score=score, target_task_success=target_success, full_scene_success=full_scene_success, frames=len(commands),
                  object_position_mean_mm=float(error.mean()), object_position_median_mm=float(np.median(error)),
                  object_position_p95_mm=float(np.percentile(error, 95)), final_object_error_mm=final_error,
                  max_lift_mm=max_lift, carry_object_error_p95_mm=carry_p95,
                  thumb_contact_frames=int(contacts[:, 0].sum()), finger_contact_frames=int(contacts[:, 1].sum()),
                  opposing_contact_frames=int(contacts.all(axis=1).sum()),
                  carry_opposing_contact_fraction=float(contacts[42:126].all(axis=1).mean()),
                  simultaneous_positive_force_carry_fraction=carry_fraction,
                  longest_carry_opposing_contact_gap_s=longest_contact_gap*model.opt.timestep,
                  carry_height_deficit_step_fraction=drop_fraction,
                  carry_drop_step_fraction=drop_fraction,  # historical score alias; includes lift lag
                  robot_table_contact_steps=table_steps,
                  robot_other_object_contact_steps=robot_other_steps, target_table_contact_steps=target_table_steps,
                  max_neighbor_displacement_mm=float(max_neighbor_displacement*1000),
                  final_neighbor_displacement_mm=float(max(neighbor_displacement)),
                  final_target_speed_m_s=final_speed, release_no_hand_contact_last_15_frames=released,
                  terminal_supported_last_15_frames=terminal_supported,
                  terminal_settled_last_15_frames=terminal_settled,
                  terminal_max_step_speed_m_s=float(max(max_step_speeds[-15:])),
                  wrist_relative_object_slip_60_to_98_mm=float(np.linalg.norm(relative_positions[98]-relative_positions[60])*1000),
                  max_thumb_normal_force_n=max_thumb_force, max_finger_normal_force_n=max_finger_force,
                  min_target_contact_distance_mm=min_contact_distance*1000,
                  any_actuator_effort_saturation_steps=force_saturation_steps,
                  arm_tracking_median_rad=float(np.median(np.abs(qactual[:, :6]-commands[:, :6]))),
                  hand_tracking_median_rad=float(np.median(np.abs(qactual[:, 6:]-commands[:, 6:]))))
    if full:
        return report, dict(object_positions=positions, object_rotations=rotations,
                            target_speed=np.asarray(speeds), q_actual=qactual, contacts=contacts,
                            force_frame_peak_n=force_frames, simultaneous_force_step_fraction_by_frame=simultaneous_frames,
                            target_table_support_step_fraction_by_frame=support_frames,
                            max_step_speed_by_frame=np.asarray(max_step_speeds),
                            wrist_relative_object_position=relative_positions)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/app/rl-output/shadow_polisher_20261025')
    parser.add_argument('--candidates', type=int, default=32)
    parser.add_argument('--replay', help='independently replay an exported commands.npz')
    args = parser.parse_args()
    if not 0 <= args.candidates <= 64:
        parser.error('candidates must be 0..64')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plan, track, names, baseline, lower, upper, velocity = source()
    base_hash = sha(PLAN)
    track_hash = sha(TRACK)
    urdf_hash = sha(URDF)
    mesh_hashes = {str(i):sha(EP/f'intermediates/segmentation/hocap-oracle-v1/meshes/object_{i}.obj') for i in (1,2,3,4)}
    model, table_z, scene_xml = make_scene(plan, track, names)
    (out/'scene.xml').write_text(scene_xml)
    if args.replay:
        with np.load(args.replay, allow_pickle=False) as saved:
            expected = {'schema':'shadow_polisher_commands_v1', 'plan_sha256':base_hash,
                        'track_sha256':track_hash, 'urdf_sha256':urdf_hash,
                        'mesh_hashes_json':json.dumps(mesh_hashes,sort_keys=True)}
            commands, timestamps, factor = validate_command_export(
                saved, expected, names, plan['timestamps'], baseline.shape, baseline.dtype)
        pm, vm = command_margins(commands, timestamps, lower, upper, velocity)
        if pm < -1e-6 or vm < -1e-6:
            raise ValueError(f'exported command violates limits: {pm}, {vm}')
        replay_plan = {**plan, 'timestamps':timestamps}
        report, trajectory = rollout(model, replay_plan, track, names, commands, table_z, full=True)
        report.update(carry_timing_factor=factor, schema='shadow_polisher_independent_replay_v1', command_sha256=sha(Path(args.replay)),
                      plan_sha256=base_hash, position_margin_rad=pm, velocity_margin_rad_s=vm)
        (out/'replay_report.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed(out/'replay_trajectory.npz', **trajectory)
        print(json.dumps(report, indent=2), flush=True)
        return
    started = time.monotonic()
    manifest = dict(schema='shadow_polisher_pilot_v1', source_hashes={'plan':base_hash, 'track':track_hash, 'urdf':urdf_hash, 'meshes':mesh_hashes},
                    predeclared_search={k:PREDECLARED[k] for k in ('carry_frames','grasp_gate_rise','grasp_gate_fall','candidate_count','seed','nominal_mass_kg','nominal_robot_object_friction','ranking')},
                    reviewed_acceptance_gate=PREDECLARED['task_gate'],
                    acceptance_gate_chronology='terminal support/speed and 5 mm four-object displacement were tightened in review after first 32-candidate run; ranking and candidates unchanged',
                    held_out_grid=PREDECLARED['held_out_grid'], actual_candidate_count=args.candidates,
                    parameter_bounds=PARAMETER_BOUNDS.tolist(),
                    physics={'dt_s': .0005, 'mass_kg':.08, 'robot_object_friction':1.2,'table_friction':1.,
                             'arm_kp':1800,'arm_kv':120,'hand_kp':16,'hand_kv':.15,
                             'gravity_compensation':True,'weld':False,'object_pose_forcing':False,
                             'table_z':table_z},
                    baseline_command_sha256=hashlib.sha256(baseline.tobytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    ledger = out/'ledger.jsonl'
    ledger.write_text('')
    best_score = float('inf')
    best = None
    best_index = None
    sampler = qmc.LatinHypercube(d=4, seed=20261025)
    samples = qmc.scale(sampler.random(args.candidates), PARAMETER_BOUNDS[:,0], PARAMETER_BOUNDS[:,1]) if args.candidates else []
    for i, params in enumerate([np.zeros(4)] + list(samples)):
        command = polished_commands(baseline, names, params)
        pm, vm = command_margins(command, plan['timestamps'], lower, upper, velocity)
        entry = {'index':i, 'parameters_rad':list(map(float,params)), 'position_margin_rad':pm,
                 'velocity_margin_rad_s':vm}
        if pm < -1e-6 or vm < -1e-6:
            entry.update(status='rejected_limits', reason='position' if pm < -1e-6 else 'velocity')
        else:
            entry['status'] = 'evaluated'
            entry['metrics'] = rollout(model, plan, track, names, command, table_z)
            if entry['metrics']['score'] < best_score:
                best_score = entry['metrics']['score']
                best = command
                best_index = i
        with ledger.open('a') as f:
            f.write(json.dumps(entry)+'\n')
        print(json.dumps({'index':i,'status':entry['status'],'score':entry.get('metrics',{}).get('score'),
                          'elapsed_s':round(time.monotonic()-started,1)}), flush=True)
    if best is None:
        raise RuntimeError('even baseline was rejected')
    np.savez_compressed(out/'commands.npz', schema='shadow_polisher_commands_v1',
                        timestamps=plan['timestamps'], joint_names=np.asarray(names),
                        commands=best, plan_sha256=base_hash, track_sha256=track_hash,
                        urdf_sha256=urdf_hash, mesh_hashes_json=json.dumps(mesh_hashes,sort_keys=True),
                        carry_timing_factor=1.)
    report, trajectory = rollout(model, plan, track, names, best, table_z, full=True)
    np.savez_compressed(out/'trajectory.npz', **trajectory)
    manifest.update(best_index=best_index, best_metrics=report, search_wall_seconds=time.monotonic()-started,
                    command_sha256=sha(out/'commands.npz'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    followup_declaration = {
        'schema':'shadow_polisher_timing_followup_v1',
        'reason':'fixed ablation after 32 hand-only candidates failed the carry gate',
        'trajectories':['baseline','hand_search_best'], 'carry_timing_factors':[1.5,2.0],
        'retimed_intervals':'intervals ending at frames 42..125; all 190 poses preserved',
        'ranking':PREDECLARED['ranking'], 'new_search':False}
    (out/'timing_followup_declaration.json').write_text(json.dumps(followup_declaration,indent=2)+'\n')
    followup = []
    final_commands, final_timestamps, final_label, final_metrics = best, plan['timestamps'], 'hand_search_best', report
    for label, candidate in (('baseline', baseline), ('hand_search_best', best)):
        for factor in (1.5, 2.):
            timed = retime_carry(plan['timestamps'], factor)
            pm, vm = command_margins(candidate, timed, lower, upper, velocity)
            if pm < -1e-6 or vm < -1e-6:
                raise RuntimeError(f'retimed command infeasible: {label} {factor}')
            timed_plan = {**plan, 'timestamps':timed}
            metrics = rollout(model, timed_plan, track, names, candidate, table_z)
            item = {'source':label, 'carry_timing_factor':factor, 'position_margin_rad':pm,
                    'velocity_margin_rad_s':vm, 'duration_s':float((timed[-1]-timed[0])/1e6),
                    'metrics':metrics}
            command_path = out/f'timing_{label}_{str(factor).replace(".", "p")}.npz'
            np.savez_compressed(command_path, schema='shadow_polisher_commands_v1',
                                timestamps=timed, joint_names=np.asarray(names), commands=candidate,
                                plan_sha256=base_hash, track_sha256=track_hash, urdf_sha256=urdf_hash,
                                mesh_hashes_json=json.dumps(mesh_hashes,sort_keys=True),
                                carry_timing_factor=factor)
            item['command_file'] = command_path.name
            item['command_sha256'] = sha(command_path)
            followup.append(item)
            if metrics['score'] < final_metrics['score']:
                final_commands, final_timestamps, final_label, final_metrics = candidate, timed, command_path.name, metrics
    (out/'timing_followup.json').write_text(json.dumps(followup,indent=2)+'\n')
    final_factor = 1. if final_label == 'hand_search_best' else float(next(x['carry_timing_factor'] for x in followup if x['command_file']==final_label))
    np.savez_compressed(out/'final_commands.npz', schema='shadow_polisher_commands_v1',
                        timestamps=final_timestamps, joint_names=np.asarray(names), commands=final_commands,
                        plan_sha256=base_hash, track_sha256=track_hash, urdf_sha256=urdf_hash,
                        mesh_hashes_json=json.dumps(mesh_hashes,sort_keys=True), carry_timing_factor=final_factor)
    manifest.update(final_source=final_label, final_metrics=final_metrics,
                    final_command_sha256=sha(out/'final_commands.npz'),
                    accepted=bool(final_metrics['target_task_success'] and final_metrics['full_scene_success']))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sensitivity = []
    for mass in (.06, .10):
        for friction in (.9, 1.2):
            varied, varied_z, _ = make_scene(plan, track, names, mass=mass, friction=friction)
            for label, candidate, timestamps in (('baseline', baseline, plan['timestamps']),
                                                  ('selected_diagnostic', final_commands, final_timestamps)):
                metrics = rollout(varied, {**plan, 'timestamps':timestamps}, track, names, candidate, varied_z)
                sensitivity.append({'mass_kg':mass, 'robot_object_friction':friction,
                                    'trajectory':label, 'metrics':metrics})
    (out/'sensitivity.json').write_text(json.dumps(sensitivity, indent=2)+'\n')
    print(json.dumps({'best_index':best_index,'hand_best_metrics':report,'final_source':final_label,
                      'final_metrics':final_metrics,'search_wall_seconds':manifest['search_wall_seconds']},indent=2),flush=True)

if __name__ == '__main__':
    main()

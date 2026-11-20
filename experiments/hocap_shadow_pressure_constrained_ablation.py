"""Predeclared constrained RF/LF pressure ablation, separate from original run."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from experiments.hocap_shadow_polisher import make_scene, rollout, sha, source
from experiments.hocap_shadow_multifinger_polisher import (
    full_digit_trace, hashes, kinematic_margins, load_palm, set_passive_fingers,
    static_hold,
)
from experiments.hocap_shadow_palm_polisher import build_kinematics, sampled_joint_velocity_margin
from experiments.hocap_shadow_pressure_hint_ablation import (
    BACKTRACK, COORDINATES, HINTS, SNAPSHOTS, SOURCE_COMMAND, SOURCE_RESIDUAL,
    SOURCE_TRAJECTORY, export, full_residual, geometry_context, jsonwrite,
    snapshot_gaps,
)
from experiments.shadow_multifinger_controls import BOUNDS, multifinger_commands
from experiments.shadow_polisher_controls import command_margins, validate_command_export
from experiments.shadow_pressure_hint_controls import carry_pressure_weights, gap_loss_mm


OUT = Path('/app/rl-output/shadow_pressure_constrained_20261026_retry03')
CONSTRAINT_FRAMES = (25, 27, 28, 29, 30, 31, 60, 90, 110)
PROTOCOL = {
    'schema': 'shadow_pressure_constrained_v1',
    'parent_ablation': 'shadow_pressure_hint_ablation_20261026',
    'motivation': 'Both frozen gap-only endpoints penetrate retained RF/LF pairs even at carry snapshots; impose original 1mm self clearance inside optimizer.',
    'five_variables': ['RF flexion', 'LF flexion', 'RFJ4', 'LFJ4', 'LFJ5'],
    'source': 'exact saved MF+.10 RF+.08 command and actual physics snapshots60/90/110',
    'two_arms': ['same geometry objective, RF/LF weights 1.5/1.5 for equal total contact weight',
                 'same geometry objective, frozen visual RF/LF weights'],
    'optimizer': 'SLSQP maxiter80, finite-difference eps1e-4; source feasible start; unchanged five-variable bounds and objective; project only out-of-bounds intermediate SLSQP callbacks for evaluation, final proposals screened without projection',
    'constraints': 'minimum original screened self margin and floor margin >=0 at frames25,27,28,29,30,31,60,90,110; final full-trajectory screen unchanged',
    'backtrack_scales': BACKTRACK,
    'physics_budget': 'first whole-trajectory feasible scale per arm, then one full190 and one static2s rollout; otherwise no physics',
    'pressure_semantics': 'uncalibrated visual proxy, not measured contact force or newtons',
    'accepted_gate': 'original target, full-scene, static-hold diagnostics unchanged',
}


def checked_margin_vector(kin, names, command, frames=CONSTRAINT_FRAMES):
    """Original floor and retained-pair screen at the frozen sparse frames."""
    set_passive_fingers(kin, names, command)
    margins = []
    for frame in frames:
        kin.set_frame(frame)
        q = command[frame, :6].astype(float)
        margins.append(float(np.min(kin.floor_margins(q))))
        margins.append(float(np.min(kin.collision_margins(q))))
    return np.asarray(margins, dtype=float)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('existing constrained output; choose a fresh run')
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    plan, track, names, baseline, lower, upper, velocity = source()
    provenance = hashes()
    palm = load_palm(names, plan, baseline, provenance)
    with np.load(SOURCE_COMMAND, allow_pickle=False) as saved:
        expected = {'schema': 'shadow_polisher_commands_v1',
                    'plan_sha256': provenance['plan'], 'track_sha256': provenance['track'],
                    'urdf_sha256': provenance['urdf'],
                    'mesh_hashes_json': json.dumps(provenance['meshes'], sort_keys=True),
                    'palm_input_sha256': provenance['palm_input']}
        commands, _, factor = validate_command_export(
            saved, expected, names, plan['timestamps'], baseline.shape, baseline.dtype)
    if factor != 1. or not np.array_equal(commands,
                                          multifinger_commands(palm, names, SOURCE_RESIDUAL)):
        raise ValueError('source commands differ from reviewed source')
    with np.load(HINTS, allow_pickle=False) as saved:
        if list(map(str, saved['digits'])) != ['thumb', 'index', 'middle', 'ring', 'little']:
            raise ValueError('unexpected hint digit order')
        median, visual_weights = carry_pressure_weights(
            saved['fused_expected_pressure_proxy_kpa'], saved['fused_digit_participation'])
    with np.load(SOURCE_TRAJECTORY, allow_pickle=False) as saved:
        trajectory = {key: saved[key] for key in ('q_actual', 'object_positions',
                                                  'object_rotations')}
    scene, table_z, _ = make_scene(plan, track, names)
    source_metrics, reproduced = rollout(scene, plan, track, names, commands, table_z, full=True)
    if any(not np.array_equal(trajectory[key], reproduced[key]) for key in trajectory):
        raise ValueError('frozen source actual physics trajectory mismatch')
    context = geometry_context(scene, names, commands, trajectory)
    kin = build_kinematics({**plan, 'shadow_q': commands[:, 6:]})
    bounds = BOUNDS[list(COORDINATES)]
    x0 = SOURCE_RESIDUAL[list(COORDINATES)]
    span = bounds[:, 1] - bounds[:, 0]
    protocol = {**PROTOCOL, 'constraint_frames': list(CONSTRAINT_FRAMES),
                'parameter_bounds_rad': bounds.tolist(),
                'visual_product_median': median.tolist(),
                'visual_weights_rf_lf': visual_weights.tolist(),
                'source_hashes': {**provenance, 'source_command': sha(SOURCE_COMMAND),
                                  'pressure_hints': sha(HINTS),
                                  'actual_trajectory': sha(SOURCE_TRAJECTORY)}}
    jsonwrite(OUT / 'protocol.json', protocol)
    jsonwrite(OUT / 'manifest.json', {'schema': 'shadow_pressure_constrained_run_v1',
                                      'protocol': protocol, 'accepted': False})
    outcomes = []
    for mode, weights in (('geometry', np.full(2, 1.5)), ('visual_hint', visual_weights)):
        counts = {'objective': 0, 'constraint': 0}
        cached_x = None
        cached_command = None
        def command_for(x):
            nonlocal cached_x, cached_command
            # SLSQP may call constraints just outside Bounds before its own wrapper.
            # Project this intermediate evaluation only, never the saved command.
            evaluated = np.clip(np.asarray(x, dtype=float), bounds[:, 0], bounds[:, 1])
            if cached_x is None or not np.array_equal(evaluated, cached_x):
                cached_x = evaluated.copy()
                cached_command = multifinger_commands(palm, names, full_residual(evaluated))
            return cached_command
        def objective(x):
            counts['objective'] += 1
            gaps = snapshot_gaps(scene, context, commands, trajectory, command_for(x))
            return gap_loss_mm(gaps, weights, np.asarray(x) - x0, span)
        def constraints(x):
            counts['constraint'] += 1
            return checked_margin_vector(kin, names, command_for(x))
        source_margins = constraints(x0)
        if np.min(source_margins) < -1e-7:
            raise RuntimeError('frozen source is not a feasible starting point')
        result = minimize(objective, x0, method='SLSQP',
                          bounds=[tuple(v) for v in bounds],
                          constraints=[{'type': 'ineq', 'fun': constraints}],
                          options={'maxiter': 80, 'eps': 1e-4, 'ftol': 1e-6, 'disp': False})
        endpoint = np.asarray(result.x, dtype=float)
        row = {'mode': mode, 'weights': weights.tolist(), 'x0': x0.tolist(),
               'x_opt': endpoint.tolist(), 'source_score': float(objective(x0)),
               'optimized_score': float(objective(endpoint)),
               'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
               'iterations': int(result.nit), 'evaluations': counts,
               'source_constraint_min_m': float(np.min(source_margins)),
               'endpoint_constraints_m': constraints(endpoint).tolist(),
               'endpoint_local_gaps_mm': snapshot_gaps(
                   scene, context, commands, trajectory, command_for(endpoint)).tolist()}
        with (OUT / 'optimization.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')
        print(json.dumps({'stage': 'optimized', 'mode': mode, 'iterations': result.nit,
                          'score': row['optimized_score'], 'elapsed_s': round(time.monotonic()-started, 1)}), flush=True)
        selected = None
        for scale in BACKTRACK:
            proposed = x0 + scale * (endpoint - x0)
            residual = full_residual(proposed)
            candidate = multifinger_commands(palm, names, residual)
            pm, vm = command_margins(candidate, plan['timestamps'], lower, upper, velocity)
            screen = {'mode': mode, 'scale': scale, 'five_parameters_rad': proposed.tolist(),
                      'total_residual_rad': residual.tolist(),
                      'position_margin_rad': pm, 'command_velocity_margin_rad_s': vm}
            if pm < -1e-6 or vm < -1e-6:
                screen['status'] = 'rejected_limits'
            else:
                floor, self_margin, bad = kinematic_margins(
                    kin, names, candidate, range(10, 152))
                screen.update(floor_margin_m=floor, screened_self_margin_m=self_margin,
                              bad_frame=bad)
                screen['status'] = 'feasible' if bad is None else 'rejected_floor_or_self'
            with (OUT / 'screening.jsonl').open('a') as file:
                file.write(json.dumps(screen) + '\n')
            if screen['status'] == 'feasible':
                selected = scale, residual, candidate, screen
                break
        if selected is None:
            outcomes.append({'mode': mode, 'status': 'no_feasible_backtrack',
                             'optimization': row})
            continue
        scale, residual, candidate, screen = selected
        output = OUT / f'{mode}_commands.npz'
        export(output, candidate, plan, names, provenance, residual, mode,
               sha(HINTS), sha(SOURCE_TRAJECTORY))
        metrics, actual = rollout(scene, plan, track, names, candidate, table_z, full=True)
        digits, digit_trace = full_digit_trace(scene, plan, names, candidate)
        hold, hold_trace = static_hold(scene, plan, names, candidate, full=True)
        np.savez_compressed(OUT / f'{mode}_trajectory.npz', **actual, **digit_trace)
        np.savez_compressed(OUT / f'{mode}_hold_trace.npz', **hold_trace)
        evaluated = {'mode': mode, 'status': 'evaluated', 'scale': scale,
                     'residual': residual.tolist(), 'optimization': row, 'screen': screen,
                     'metrics': metrics, 'digits': digits, 'static_hold': hold,
                     'sampled_actual_velocity_margin_rad_s': sampled_joint_velocity_margin(
                         actual['q_actual'], plan['timestamps'], velocity),
                     'command_sha256': sha(output), 'accepted': False}
        jsonwrite(OUT / f'{mode}_report.json', evaluated)
        outcomes.append(evaluated)
        print(json.dumps({'stage': 'physics', 'mode': mode,
                          'RF_LF_carry': digits['carry_digit_active_fraction'][3:5],
                          'target_success': metrics['target_task_success'],
                          'elapsed_s': round(time.monotonic()-started, 1)}), flush=True)
    report = {'schema': 'shadow_pressure_constrained_report_v1',
              'source_metrics': source_metrics, 'arms': outcomes,
              'accepted': False, 'scope': 'two constrained arms only; source is a single frozen scene'}
    jsonwrite(OUT / 'report.json', report)
    jsonwrite(OUT / 'manifest.json', {'schema': 'shadow_pressure_constrained_run_v1',
                                      'protocol': protocol, 'accepted': False,
                                      'outcomes': [{'mode': item['mode'], 'status': item['status']}
                                                   for item in outcomes],
                                      'wall_s': time.monotonic() - started})
    print(json.dumps({'wall_s': time.monotonic() - started,
                      'outcomes': [(r['mode'], r['status']) for r in outcomes]}), flush=True)


if __name__ == '__main__':
    main()

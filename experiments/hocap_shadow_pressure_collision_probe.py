"""Read-only retained-pair audit for the frozen pressure-hint endpoints."""
import json
from pathlib import Path

import numpy as np

from experiments.hocap_shadow_polisher import source
from experiments.hocap_shadow_multifinger_polisher import hashes, load_palm, set_passive_fingers
from experiments.hocap_shadow_palm_polisher import build_kinematics
from experiments.hocap_shadow_pressure_hint_ablation import BACKTRACK, SNAPSHOTS, full_residual
from experiments.shadow_multifinger_controls import multifinger_commands


def pair_rows(kin, q):
    configuration = kin._collision_configuration(q)
    # The same retained pair model and joint configuration as collision_margins.
    kin.pin.computeDistances(
        kin.model, configuration.data, kin.collision_model,
        configuration.collision_data, configuration.q,
    )
    rows = []
    for pair, distance in zip(kin.collision_model.collisionPairs,
                              configuration.collision_data.distanceResults):
        first = kin.collision_model.geometryObjects[pair.first]
        second = kin.collision_model.geometryObjects[pair.second]
        rows.append({'first': first.name, 'second': second.name,
                     'distance_m': float(distance.min_distance),
                     'margin_m': float(distance.min_distance) - .001})
    return sorted(rows, key=lambda item: item['margin_m'])[:8]


def main():
    out = Path('/app/rl-output/shadow_pressure_hint_ablation_20261026')
    optima = [json.loads(line) for line in (out / 'optimization.jsonl').read_text().splitlines()]
    plan, _, names, baseline, *_ = source()
    palm = load_palm(names, plan, baseline, hashes())
    kin = build_kinematics({**plan, 'shadow_q': palm[:, 6:]})
    rows = []
    for optimum in optima:
        start = np.asarray(optimum['x0'], dtype=float)
        endpoint = np.asarray(optimum['x_opt'], dtype=float)
        for scale in BACKTRACK:
            commands = multifinger_commands(palm, names,
                full_residual(start + scale * (endpoint - start)))
            set_passive_fingers(kin, names, commands)
            for frame in SNAPSHOTS:
                kin.set_frame(frame)
                q = commands[frame, :6].astype(float)
                barrier_min = float(np.min(kin.collision_margins(q)))
                closest = pair_rows(kin, q)
                rows.append({'mode': optimum['mode'], 'scale': scale,
                             'frame': frame, 'barrier_min_m': barrier_min,
                             'closest_retained_pairs': closest})
    (out / 'collision_pair_probe.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

# HO-Cap G10_4 object-grasp pilot evidence

This is a one-sequence, in-sample development experiment, **not** a successful
pick-and-place result. The protocol, caveats, tables, and replay instructions
are in `docs/2026-10-14-hocap-object-grasp-pilot.md`.

- `provenance.json` identifies the selected sequence, intervals, object and
  clipped track hash.
- `hand_ik.json` and `object_ik.json` preserve the complete saved plan metrics,
  configuration, source plan path and SHA-256 of each ignored `plan.h5`.
- `hand_physics.json` and `object_physics.json` are the matched primary MuJoCo
  outputs, including the sampled object trajectory and contact diagnostics.
- `no_standoff_physics.json`, `low_grasp_physics.json`,
  `pressure_010_physics.json`, `light_mass_physics.json`,
  `friction_20_physics.json`, `dwell_07_physics.json`, and
  `slower_4x_physics.json` are in-sample sensitivity/ablation outputs. The
  first two were generated while the diagnostic simulator still used its
  earlier whole-airborne contact field names; their lift and contact counts
  remain the reported evidence. Do not pool these runs as independent trials.
- `SHA256SUMS` covers every JSON file above.

- `contact_trace_physics.json` repeats the unchanged baseline with sampled
  object orientation and pad-centre positions for diagnosing the early drop.
- `overlap_4mm_*`, `overlap_8mm_*`, `overlap8_lower15_*`, and
  `overlap8_early4_*` preserve the full IK and MuJoCo outputs of the bounded
  geometry/timing sensitivity runs. `overlap8_gain100_physics.json` and
  `overlap8_gain150_physics.json` change only the simulator gripper position
  gain; they are not calibrated-force experiments.
- All runs use the same 5–95% fitted box from the original object point cloud.
  Two-sided contact counts distal-phalanx collision geoms, not verified rubber
  pad surface contact. None passes the sustained-grasp or placement gate.

The original HO-Cap files, clipped hand/object arrays, robot plans and cached
URDFs remain under gitignored `data/` and `models/`, not in this snapshot.
The numeric snapshot permits an audit of the reported numbers but is not a
replacement for those inputs when reproducing IK or physics.

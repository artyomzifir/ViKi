# HO-Cap G10_4: deeper insertion is not a reliable grasp (15 October 2026)

This is an in-sample diagnostic on the **same** HO-Cap crop used in
[`2026-10-14-hocap-object-grasp-pilot.md`](2026-10-14-hocap-object-grasp-pilot.md),
not an independent validation set or a real-robot test. The source is
`subject_1/20231025_170105`, original frames `[420, 600)`, object ID 4
(G10_4, body wash). Existing hand-only and other HO-Cap plans are untouched.

## Intervention

The tracked-grasp planner now exposes an opt-in, track-only
`--object-insertion-depth-mm` in `[0, 15]`; default 0 reproduces the previous
behaviour. After choosing the object-aligned tool direction, it moves the
object-fixed grasp point along that direction by the requested distance. The
open-jaw pregrasp waypoint, jaw-axis selection, nominal 6 mm contact overlap,
closing schedule, robot base and table plane are unchanged. The local offset is
fixed to the tracked object during carry rather than recomputed in world
coordinates per frame. Thus this tests insertion depth, not harder squeezing.

Three isolated plans use UR10/Robotiq 2F-85, `hocap_tag_1`, base `[0,-1,0]`,
hand-to-EE RPY `[4.325,15.45,-47.922]` degrees, `--approach-sec 0`, and
`--table-plane-z-m 0`. Their IK solvers converged. The prepared copy requests
`hand_fit` but has no `hand_fit_*` arrays, so these pilot plans actually use
landmark poses; they must not be compared as if they were production hand-fit
plans. The 0 mm control is the original saved pilot plan.

Each plan is replayed over crop frames `[20,180)` with the same MuJoCo timestep
0.5 ms, command time scale 2, inferred box proxy (about
81.7 × 59.3 × 157.5 mm), assumed 0.20 kg mass, and inferred tabletop top
`z=0.03987 m`. `--rubber-pads` substitutes explicit distal contact boxes for
the URDF's coarse distal collision meshes because the cached Robotiq URDF has
its actual rubber-pad links disabled. These boxes are **not** a calibrated
rubber/contact model. No extra gripper pressure or friction tuning is applied.

| Insertion depth | IK position RMSE | Peak simulated lift | Max carry slip | Bilateral pad contact while lifted | Longest lost contact | Final placement error | Strict grasp / transfer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 mm | 28.93 mm | 132.44 mm | 164.93 mm | 85.57% | 134.5 ms | 244.36 mm | fail / fail |
| 10 mm | 29.56 mm | 119.51 mm | 158.21 mm | 84.41% | 120.0 ms | 223.21 mm | fail / fail |
| 15 mm | 29.84 mm | 100.72 mm | 141.10 mm | 82.04% | 104.0 ms | 223.86 mm | fail / fail |

Slip decreases slightly as the jaws move deeper, but lift and sustained
bilateral contact also decrease. Neither variant holds the object through
carry or places it within the strict 50 mm proxy gate. The 10 mm and 15 mm
changes have no numerical IK limit violation in the saved plans; minimum
velocity margins remain 1.349 and 1.343 rad/s, and the nominal `z=0` floor
margins remain positive. Those are planner self-checks, **not** physical
collision guarantees. In particular, the inferred simulator tabletop is
about 40 mm above the nominal IK plane.

## Contact-friction sensitivity

MuJoCo exposes per-geom sliding/torsional/rolling friction, contact solver
compliance and actuator settings (see the [official XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)).
The replay already has `--pad-friction` (sliding coefficient on the two explicit
pad boxes, 0.5–3.0) and `--pressure-rad` (simulator-only extra closing-angle
command, not a measured force). Holding the 15 mm plan, 0.20 kg proxy, geometry
and timing fixed gives:

| Pad friction setting | Extra closing angle | Peak lift | Max carry slip | Final placement error | Strict grasp / transfer |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 rad | 100.72 mm | 141.10 mm | 223.86 mm | fail / fail |
| 2 | 0 rad | 68.55 mm | 105.19 mm | 84.89 mm | fail / fail |
| 3 | 0 rad | 14.15 mm | no sustained two-pad carry | 97.32 mm | fail / fail |
| 2 | 0.05 rad | 103.04 mm | 147.99 mm | 235.43 mm | fail / fail |

Increasing friction does not monotonically improve this grasp. At setting 3
the object is disturbed before a bilateral pickup; with setting 2 plus extra
closure, simulated robot-object penetration reaches 10.71 mm. Thus friction
contributes to the dynamics, but these runs **do not establish friction as the
sole cause**. Contact geometry, approach orientation, normal force and timing
remain entangled. No coefficient or force has been measured on the actual
HO-Cap object or Robotiq pads. Tuning the simulator until it reports success
would not by itself validate a real grasp.

The alternate wide-face/jaw-axis trial was also rejected: its IK position
RMSE was 40.50 mm and the same pad proxy lifted only 8.95 mm, with no
bilateral carry contact. Its temporary override was removed from the code;
its ignored diagnostic plan and JSON remain under
`data/_retarget_diagnostics/hocap_g10_4_jaw_x/`.

**Decision:** keep insertion-depth control opt-in and the default at 0 mm.
Do not label these demonstrations as successful object-level retargeting or
training ground truth. Deeper insertion alone does not fix the corner-contact,
pose-alignment and force-control limitations diagnosed in the original pilot.

## Reproduction and raw results

Start with the source crop from `scripts/prepare_hocap_grasp_pilot.py --first
420 --last 600`. For each depth, copy its `cln.npz`, `meta.json`, and cropped
`object_track.npz` into a fresh episode directory. Run, changing the two path
occurrences and depth together:

```bash
docker compose run --rm cli retarget data/_retarget_diagnostics/hocap_g10_4_depth10 \
  --robot ur10 --gripper robotiq_2f85 --target-anchor pinch_center \
  --base-position 0 -1 0 --reference-policy zero --table-plane-z-m 0 \
  --approach-sec 0 --hand-to-ee-rpy-deg 4.325 15.45 -47.922 \
  --object-grasp --object-track-path data/_retarget_diagnostics/hocap_g10_4_depth10/object_track.npz \
  --object-id 4 --object-insertion-depth-mm 10

docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_grasp_replay.py \
  data/_retarget_diagnostics/hocap_g10_4_depth10/plan.h5 \
  data/_retarget_diagnostics/hocap_g10_4_depth10/object_track.npz \
  --object-id 4 --frame-start 20 --frame-end 180 --rubber-pads
```

Full, unrounded plan and simulation JSON for 0, 10 and 15 mm, plus the three
15 mm friction/pressure sensitivity replays, with SHA-256 checksums, are preserved in
[`paper/results/hocap/object_grasp_insertion_study/`](../paper/results/hocap/object_grasp_insertion_study/).
Source recordings, cropped arrays and plan HDF5s remain gitignored in `data/`;
these JSON files alone do not reproduce the run without those inputs and the
URDF assets in `models/`.

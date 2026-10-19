# HO-Cap G10_4 object-guided physical search (15 October 2026)

This is an **in-sample, one-seed simulator diagnostic**, not a successful
robot-ready trajectory. It uses the same G10_4 180-frame crop, official mesh
aligned at frame 420, UR10/Robotiq 2F-85, and randomized MuJoCo scene as the
[object-trajectory RL pilot](2026-10-15-hocap-object-track-reward.md). The mesh
collision is its convex hull; rubber pads, object mass/friction, table and
actuators remain uncalibrated assumptions.

`experiments/hocap_object_guided_replay.py` checks the source report's SHA-256
inputs before constructing the scene. For each of 27 position/closing-angle
variants on seed 1004, it aligns the open jaws, closes, waits for bilateral
preload, lifts vertically until at least 60 mm lift and 13 stable control steps,
then commands a pad-midpoint trajectory derived from the recorded object
translation at 4x human time. It aborts after eight steps without bilateral
contact. Release and withdrawal occur only after carry. No object pose is
teleported, and no policy is trained.

| Result | Count / value |
|---|---:|
| Physically transferred and released | 0/27 |
| Lost bilateral grasp during carry | 12/27 |
| Failed during settling | 5/27 |
| No stable lift | 5/27 |
| No bilateral preload | 5/27 |
| Best observed peak lift | 135.5 mm |
| Same trial's maximum wrist-relative object slip | 186.3 mm |

Seed 1004 sampled an assumed object mass of 0.10018 kg and object sliding
friction coefficient 0.80412; the pad coefficient was 1.0. These are simulator
settings, not measured HO-Cap material properties.

The full unrounded per-trial JSON is
`data/experiments/hocap-rl/object_guided_search/report_final_default.json`.
That file is local and gitignored; the source recordings, object poses and
robot models remain local as well. The run is reproduced with:

```bash
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/object_guided_search/report_final_default.json \
  --seed 1004
```

The baseline achieved genuine two-pad lift in several variants but the object
slid out on ascent or early carry. Adjusting XY by ±5 mm, initial closing joint
target 0.50–0.60 rad, lower contact height, slower lift, and simple normal-force
feedback did not produce a complete transfer in this one scene. In particular,
feedback that squeezed harder during ground contact often prevented lift. This
does **not** establish friction as the sole cause: curved contact geometry,
normal direction, model assumptions and robot/object acceleration interact.

An opt-in two-view EGL clip of the best-lift candidate is saved at
`data/experiments/hocap-rl/object_guided_search/best_trial.mp4` with its raw
`report_video.json`. Visually, the jaws pinch near the bottle's narrow upper
region; the object then moves downward relative to the pads. This is a visual
observation, not a calibrated contact-area measurement. An X11 GUI trial wrote
the same failure metrics but the viewer process exited with code 139 after
writing the report; use the EGL clip for repeatable inspection.

This diagnostic tracks **translation only** after a vertical lift and retains
the original wrist orientation. It is not the proposed full 6D grasp-candidate
and contact-aware trajectory optimizer. The recorded G10_4 orientation changes
by about 15 degrees during carry, which this controller does not follow.
Future comparison should test independently generated 6D grasp candidates and
an independently validated 2F-85 contact model before another RL run. The
available [GraspGenX implementation](https://github.com/NVlabs/GraspGenX)
supports 2F-85 mesh grasps, but GitHub downloads from this host timed out during
this pilot; it was **not** run or validated here.

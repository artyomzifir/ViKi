# Eight two-Kinect cube takes: offline pilot

Measured on 2026-10-02, branch `v0.0.2`, from
`data/datasets/40mm-new-2kinects/` (1,868 frames across eight takes). These
recordings are **free play / different tasks**, as confirmed by the operator;
they are not eight repetitions of one labelled pick-and-place task.

## What was run

- Preserved each original 50 mm Viewer cloud under
  `intermediates/cloud_bg50/`, then rebuilt `cloud/` with
  `build_cloud(ep, bg_tol_mm=20.0)`. Each new cloud has a
  `processing_override.json` provenance file. The standard cloud `meta.json`
  currently does **not** contain the background tolerance; the sidecar records
  the override until that schema is fixed.
- Generated three-object SAM prompt manifests with 20 mm background tolerance
  under `intermediates/cube_prompts_20mm.json`. All eight selected three compact
  two-view components on seed frame 0. Prompt labels are provisional:
  **all three candidates were called `manipulated_object`**, although only one
  moves in each take. Do not use those labels as verified task roles.
- Ran full SAM 2.1 masks (20-frame chunks, no overlay), lifted every episode's
  masks to semantic 3-D at 20 mm background tolerance, and built all three
  `object_models.npz` per episode. The semantic-cloud metadata records 20 mm.
- Ran the immutable `stable-fused-hand-v4` profile to create `rec.npz` and
  `cln.npz` for the other seven takes, without rebuilding their corrected
  clouds. All eight now have hand trajectories.
- Ran default UR10 + Robotiq 2F-85 retarget on all eight takes. The resulting
  `plan.h5` files are **offline joint plans, not robot or simulator rollouts**.
  On the first take, alternative orientation-weight and robot-base experiments
  are archived under `intermediates/`; the default plan was restored as the
  canonical `plan.h5` and `status.json` was brought back into agreement.

No raw RGB-D recording was changed. No hardware replay, MuJoCo execution,
object-relative retarget, LeRobot training export, or SmolVLA evaluation was
performed.

## Cloud A/B

At frame 0 of the first take, 5 cm-voxel Viewer clouds contained 24,100 points
with the old 50 mm threshold and 27,509 with 20 mm. Coloured points within
60 mm of the three automatically found cubes rose from **68/72/78** to
**197/189/217**. Across all eight takes the same local increase occurs, while
total foreground grows only about 13–16%. These counts are a repeatable
self-comparison, **not** a segmentation-accuracy measurement. The saved 50 mm
clouds remain available for A/B inspection. The display ChArUco plane is also
about 9–14 mm above the measured tabletop; hiding that Viewer layer separates
visual occlusion from actual missing points. See
`docs/2026-10-02-cube-cloud-calibration-diagnostic.md` for the calibration and
raw-depth measurements.

## Object/hand observations

The moving object is selected *diagnostically* by the largest displacement
from frame 0; a movement frame means estimated centre step >1.5 mm. Pinch
distance is the Euclidean distance from the smoothed thumb/index midpoint to
that object's estimated **visible-surface** model origin, not to a verified
solid-cube centre. `bad sync` counts moving frames for which the two saved
device timestamps differ by >10 ms. `tips observed` requires both finger-tip
landmarks to be directly observed rather than filled on those movement frames.

| Take suffix | Moving colour | Max displacement | Pinch distance median | Bad sync / moving | Tips observed |
|---|---|---:|---:|---:|---:|
| 11-30-58 | green | 232 mm | 15.3 mm | 0 / 93 | 100% |
| 11-31-13 | red | 247 mm | 13.7 mm | 0 / 79 | 100% |
| 11-31-26 | blue | 215 mm | 15.0 mm | 0 / 81 | 100% |
| 11-31-40 | green | 310 mm | 14.1 mm | 12 / 131 | 92% |
| 11-31-55 | red | 273 mm | 9.2 mm | 9 / 80 | 99% |
| 11-32-07 | blue | 276 mm | 10.0 mm | 13 / 69 | 100% |
| 11-32-21 | green | 238 mm | 16.9 mm | 0 / 65 | **45%** |
| 11-32-31 | red | 287 mm | 12.0 mm | **33 / 72** | 100% |

This supports a plausible hand/object association in all eight takes, but
does not establish grasp success or 3-D accuracy. The synchronized moving
subset is strongest in takes `11-30-58`, `11-31-13`, and `11-31-26`.
Take `11-32-21` has extensive finger-tip interpolation during motion.
Take `11-32-31` has a full-frame cross-camera timing slip on almost half its
moving frames; its per-camera cube-centroid disagreement rises from 14.1 mm
median on well-paired moving frames to 22.1 mm on mispaired ones. Its current
fused object track should not be treated as a high-quality training label.

The object model is a partial visible shell, not a measured 40 mm solid cube.
Estimated net rotations of the active object are 7°, 20°, 11°, 144°, 22°, 29°,
19°, and 29° across the eight takes. There is no independent orientation ground
truth, and the single 144° estimate needs visual or marked-cube validation.
Most takes do not excite enough rotation to prove full 3-D cube orientation.

## UR10 retarget is not yet a grasp proof

The default first-take plan converges at **17.2 mm TCP position RMSE**, but
**100.9° orientation RMSE**. During cube movement its median tool-axis tilt
error is 69.6°. Raising the orientation weight from 0.01 to 0.03 at the same
base improves total orientation RMSE to 31.0° but worsens position RMSE to
29.3 mm, with 83.4 mm p90 position error on moving frames. Weight 0.1 made the
approach-plus-demonstration trajectory fail a collision-feasibility check.
Moving the simulated robot base from X=-0.70 m to -0.55 m with weight 0.03
made position RMSE worse (93.9 mm). These trials are archived, and the default
plan is restored. Convergence means the optimiser stopped, **not** that a cube
could be grasped and moved.

The other seven default plans were generated. Their position RMSE values are
7.9, 13.8, 11.3, 9.1, 9.7, 38.7, and 18.3 mm respectively, in take order.
These are kinematic target errors, not physical tracking errors. No plan has
passed a dynamics/contact/closed-loop success test. `viki.replay` remains a
dry-run stub, and the current environment has no MuJoCo or LeRobot runtime.

## Decision before more capture

1. Use this set as a **pipeline diagnostic**, not a SmolVLA training/evaluation
   set with a single task label. Segment and describe intended sub-tasks and
   outcomes before assigning policy targets; the operator says these takes are
   free play / different tasks.
2. Repair or explicitly exclude one-frame camera-pair slips during object
   motion, especially `11-32-31`; do not let a smooth fused track hide them.
3. Build and validate a table-grounded solid-cube scene and physical UR10 +
   Robotiq 2F-85 MuJoCo rollout with contact, achieved robot state, rendered
   robot camera images, and clear success criteria. The present `plan.h5` is
   only a kinematic input to that test.
4. Solve the hand-to-tool orientation / reach / floor trade-off on at least one
   take *in simulation* before bulk export or another 50 recordings. Object
   pose and object-relative motion are not yet wired into production retarget.

All numeric claims above are internal consistency or optimizer diagnostics;
none is an external geometric-accuracy or policy-success score.

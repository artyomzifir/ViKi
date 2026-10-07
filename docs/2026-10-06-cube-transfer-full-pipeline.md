# Blue-cube transfer: full offline pilot (2026-10-06)

## Source and processing

- Episode: `data/datasets/40mm-pnp/2026-10-06_08-55-49`, 209 frames at 30 fps
  from two wired-synchronised Kinect cameras. The recorded preset is
  `my_preset`: 8×10 ChArUco, 50 mm squares, 35 mm markers, 18 capture sets.
  Its empty-scene validator is green, but warns that its 3.54 mm nearest-
  neighbour median is not a calibrated accuracy estimate.
- Existing Extract/Prepare outputs use `stable-fused-hand-v4`; the clean
  hand arrays and protected baseline were not regenerated or replaced.
- Automatic foreground prompts selected regions substantially larger than
  40 mm cubes. The saved proposal was rejected; manually reviewed frame-0
  prompts in `data/2026-10-06_08-55-49-cube-prompts.json` label the operator,
  blue manipulated cube, red support, and green goal separately.
- SAM2 masks, semantic RGB-D cloud, and three rigid object tracks were built
  from those prompts. The RGB overlays were inspected at frames 0, 100,
  150 and 208 from both cameras; the blue mask follows the transfer.

## Internal consistency (not external accuracy)

| Check | Result | Meaning |
|---|---:|---|
| Red–green tabletop separation in display XY, frame 0 | 395.3 mm | Consistent with the stated ~400 mm layout |
| Blue model translation, frame 0 → 208 | 409.8 mm | Consistent with red-to-green transfer |
| Blue model origin above red / green origin | 32.2 / 27.2 mm | Partial visible-surface origins, **not** measured cube heights |
| Blue track confidence median / 10th percentile | 0.897 / 0.848 | Model score, not calibrated probability |
| Blue tracking median / p90 residual | 2.0 / 2.4 mm | Fit to its own partial cloud, not ground truth |
| Blue contact-flag frames | 133 / 209 | Proximity heuristic, not physical grasp confirmation |

The blue canonical model spans roughly 55×55×59 mm despite a nominal 40 mm
cube, so its pose should not be mistaken for a calibrated solid-cube label.
The red model has only 47 supported points because the blue cube initially
occludes it. Cube symmetry and partial surfaces make yaw unverified.

## Robot and object-relative artifacts

- `plan.h5`: UR10, Robotiq 2F-85 two-finger gripper, pinch-centre target,
  hand-fit palm orientation, calibration-frame robot base `[-0.7, 0, 0]` m,
  flange-to-gripper adapter `[0, 0, 11]` mm, home-branch reference.
- Whole-trajectory IK converged in 17 iterations. Target TCP position RMSE
  is 13.0 mm (p95 31.4 mm), orientation RMSE 7.27°. Minimum modeled floor
  clearance is 22.6 mm and collision margin 39.8 mm. These are solver/model
  measures, **not** cube-placement or simulation success.
- `intermediates/object_relative.npz`: for all three tracked objects, saves
  `T_object_hand`, `T_object_target_tcp`, `T_object_achieved_tcp`, object IDs,
  contact flags, confidence/rotation information, opening and a validity
  mask. The observed hand is valid in 184/209 frames; invalid hand/target
  transforms are NaN. These estimates are not used to drive the IK.
- The experimental trajectory bundle is
  `data/exports/2026-10-06-cube-transfer-pilot/` (schema v3), including
  `trajectory.npz`, `object_models.npz`, `object_relative.npz` and provenance.
  The task is labelled; outcome remains `unrated`.

Extract and Retarget now request the existing object-model layers in their
shared 3-D scene; a server restart is needed to load the updated API. Replay
is still a stub. No MuJoCo simulation, contact dynamics, object-relative IK,
policy training, or physical robot test was run. This is a successful **offline
artifact pass**, not proof that the UR10 can pick and place the cube.

An opt-in object-aware follow-up subsequently replaced the episode's active
`plan.h5`. This hand-only pilot plan is preserved at
`intermediates/retarget_comparison/hand_only_plan.h5`; its export remains at
`data/exports/2026-10-06-cube-transfer-pilot/`. See
`docs/2026-10-06-cube-object-grasp.md` for the new plan and comparison.

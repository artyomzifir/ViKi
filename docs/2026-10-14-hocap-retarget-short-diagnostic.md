# HO-Cap retarget: short-window speed and handedness diagnostic (2026-10-14)

## Why the full smoke stopped

The 676-frame `subject_1_20231025_165502` smoke wrote a `plan.h5`, but the
numerical gate rejected it. Its position RMSE was **304.50 mm**, position p95
**843.78 mm**, orientation RMSE **83.98°**, and `converged=false` after 40
iterations. The final trajectory obeyed the joint-speed bound (minimum margin
`0 rad/s` after a `8.21e-7 rad` numerical repair), the infinite `z=0` table
half-space, and the configured self-collision margin. Thus this attempt was
not stopped by a remaining speed-limit violation; it was stopped by large
pose errors. The 676-frame solve took about 34 minutes because it linearised
and checked 676 robot states over all 40 whole-trajectory GN/QP iterations.

Frames 330–390 contain one severe error region in the full plan: position
RMSE **615.19 mm** and p95 **1083.79 mm** over those 60 frames. They are the
fixed diagnostic window below. Every variant uses the same `cln.npz` slice,
UR10/Robotiq 2F-85, `hocap_tag_1`, base `[0,-1,0]`, infinite table plane
`z=0`, no synthetic home approach, and no cube-specific grasp. These are
offline geometric tests, not hardware or simulation replay.

| 60-frame experiment | Position RMSE / p95 (mm) | Orientation RMSE (°) | Peak joint speed (rad/s) | Minimum speed margin (rad/s) | GN iterations / converged |
|---|---:|---:|---:|---:|---:|
| 330–390, unchanged settings | 92.03 / 143.12 | 59.92 | 0.361 | 1.818 | 19 / yes |
| 330–390, 2× URDF speed bound | 92.03 / 143.12 | 59.92 | 0.361 | 3.912 (1.818 vs unscaled URDF) | 19 / yes |
| 330–390, orientation weight 0 | 20.70 / 31.24 | 147.82 | 0.557 | 1.535 | 11 / yes |
| 330–390, orientation weight 0.001 | 31.82 / 49.50 | 123.70 | 0.462 | 1.630 | 11 / yes |
| 330–390, left-hand TCP axis flip, default weight 0.01 | 21.17 / 30.81 | 19.54 | 0.540 | 1.554 | 13 / yes |
| Other left-hand episode, frames 200–260, unchanged | 76.41 / 107.09 | 90.13 | 0.34 | 1.88 | 18 / yes |
| Same other left-hand window, TCP axis flip | 16.88 / 30.10 | 28.98 | 0.53 | 1.78 | 16 / yes |
| Right-hand control, frames 100–160, unchanged | 30.97 / 75.01 | 26.70 | 0.36 | 1.93 | 18 / yes |

The TCP-axis-flip runs compose the existing hand-to-EE rotation
`[4.325,15.45,-47.922]°` with a local `Rx(180°)`, equivalent to
`[-175.675,15.45,-47.922]°` in the CLI's extrinsic XYZ convention. It is an
explicit diagnostic variant, not a calibrated hand-to-gripper transform.
All short plans kept non-negative final table and self-collision margins.

Across the 28 HO-Cap episodes, the default target tool axis points downward
for **0–5.9%** of valid frames in the four left-hand recordings, but for
**68.4–96.0%** in the 24 right-hand recordings. A single right-hand
hand-to-EE offset is therefore not a justified mapping for left hands. The
two left-hand windows improved with the local axis flip, while a right-hand
control already tracked reasonably with the unchanged offset. The fixed
right-hand frame is not perfect either; it needs its own task/hand calibration.

## Interpretation and limitations

- Doubling allowed speed made the first short solution **numerically
  identical**; the standard URDF limits had 1.818 rad/s of slack. It does not
  explain that window's pose error. The full solve did saturate a joint-speed
  bound at some frames, so speed can still matter elsewhere, but cannot be
  treated as the sole cause of the failed gate.
- Cropping the same 60 frames changed the solution from 615.19 to 92.03 mm
  RMSE even before any tuning. The whole-horizon optimization also entered a
  poor solution basin; no full-sequence fix has yet been demonstrated.
- Removing orientation pressure improves position but destroys orientation.
  Correcting the left-hand tool-axis mapping improves **both** metrics on two
  short windows. This is evidence for a handedness/frame mismatch, not yet a
  validated mapping for all sequences or gripper contact geometry.
- These HO-Cap `cln.npz` files request `hand_fit` through the local default
  but carry no `hand_fit_*` arrays. Retarget logs explicitly say it fell back
  to landmark poses. All comparisons above use that same fallback; do not
  describe them as fitted-hand evaluations.
- `--velocity-limit-scale 2` is now an explicit offline diagnostic flag. The
  plan archives its scale and its margin against the **unscaled** URDF. A plan
  made with a scale above 1 must not be called physically feasible for UR10.
  The default remains 1. No speed limit was changed for normal runs.
- The tabletop is an unverified-height infinite plane. Collision with
  objects or the human is not constrained. No short-window result is an
  independent full-episode, contact, physics, or real-robot success result.

The diagnostic clips and `plan.h5` files remain under the ignored local
`data/_retarget_diagnostics/`; the source HO-Cap episodes were not modified.
Before any 28-scene rerun, make the left/right orientation policy explicit,
validate it on longer independent sequences, and retain the numerical gate.

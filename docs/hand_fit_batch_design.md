# Batch hand fitting design

This note supersedes the per-frame solver described in `hand_fitting_plan.md`.

## Decisions

- **Batch extent:** one batch per complete episode, including invalid or
  low-observation frames as empty data blocks. This lets velocity and true
  second-difference edges interpolate a gap from both sides. There is no
  per-frame skip/reset or accept/reject path.
- **Solver:** `scipy.optimize.least_squares(method="trf", tr_solver="lsmr")`
  over all `T * nv` tangent increments. A callable loss applies Huber only to
  point-data rows and leaves temporal/prior blocks quadratic, matching the
  written functional (a global string `loss="huber"` incorrectly weakens long
  gap constraints). Its per-row breakpoint accounts for `w/sum(w)`, so Huber
  remains physically fixed at 10 mm after frame normalization. The analytic Jacobian is
  returned as CSR; data rows touch one frame, velocity rows two, acceleration
  rows three. Point-to-capsule identities are frozen for each inner solve and
  recomputed by an outer ICP loop.
- **Landmark decay:** the confidence-weighted landmark anchor is multiplied by
  `0.35 ** outer_iteration`. It is strong enough to pick the first basin, then
  becomes secondary to depth and temporal consistency.
- **Palm proxy:** one wrist-to-middle-MCP capsule with a calibrated broad radius
  replaces the five intersecting wrist-to-MCP capsules. It preserves the
  tested capsule distance/Jacobian while ensuring dense palm samples are counted
  once. The radius is deliberately narrower than half the palm width so it does
  not steal proximal-finger correspondences.
- **ROI and segmentation:** the ROI is the union of current capsules padded by
  3 cm. A plane normal to palm-forward, 1 cm proximal to the wrist, removes the
  forearm. Recorded episodes do not currently persist a detector mask, so the
  optional mask cannot be applied offline; calibrated background subtraction is
  retained.
- **Sampling:** deterministic 4 mm voxel representatives, capped at 400 points
  per frame. Frames below 40 points get an empty data block.
- **Output contract:** landmark `positions` and `rotations` are immutable.
  Results use `hand_fit_*`; `PERCEPTION_HAND_POSE_SOURCE` selects the consumer
  source and falls back to landmarks for historical/unfitted episodes.

## Target-host measurement

Measured 2026-09-01 inside the project Docker image on episode
`data/datasets/new-dataset/2026-09-01_10-51-08`:

| quantity | measured |
|---|---:|
| frames | 599 (about 20 s at 30 fps) |
| hand variables | 599 × 26 = 15,574 |
| full stage wall time | 85.95 s |
| batch ICP time (excludes depth ROI extraction) | 47.19 s |
| peak process RSS | 1,052,528 KiB (about 1.00 GiB) |
| outer ICP iterations / total function evaluations | 4 / 140 |
| empty data-block fraction | 35.39% (212 frames after near-empty filtering) |
| median / p90 absolute point-surface residual | 8.60 / 25.70 mm |
| wrist position jerk norm, landmark warm start → fit | 0.00989190 → 0.00050950 m |
| median fitted PIP/DIP bend | thumb 9.1°/15.8°, index 8.9°/8.0°, middle 5.6°/8.6° |

Calibration frames are first screened against the episode-median palm width and
15 phalanx lengths, then ranked by fingertip spread. This matters on the target
recording: the old maximum-spread rule selected 112–150 mm palm-width landmark
outliers, while the robust sample spans 74–87 mm.

The measured peak is under 8% of 16 GiB, so a sliding-window fallback is not
implemented. Reconsider only if a representative episode exceeds 12 GiB peak
RSS or the target episode length grows far beyond this 599-frame measurement;
that change should include overlap blending and a new A/B validation.

This recording is a useful stress case: its prepared spline diverges to tens of
metres over a long invalid region. Invalid batch references are therefore
initialized by manifold interpolation between valid neighbours (nearest-pose
hold at episode ends), while remaining free variables in the one episode-wide
solve. Without that initialization, bounded ICP increments cannot recover the
reference in 3–5 outer iterations. The final batch reduces jerk about 26× while
retaining a sub-centimetre median point-surface residual.

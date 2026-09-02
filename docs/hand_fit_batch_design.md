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
  gap constraints). Residual and loss share one `alpha = w_data · scale² ·
  w/sum(w)` (`hand_fit.data_alpha`), so Huber's breakpoint stays physically
  fixed at 10 mm for interior and exterior samples alike. The analytic Jacobian is
  returned as CSR; data rows touch one frame, velocity rows two, acceleration
  rows three. Point-to-capsule identities are frozen for each inner solve and
  recomputed by an outer ICP loop.
- **Landmark decay:** the confidence-weighted landmark anchor is multiplied by
  `0.35 ** outer_iteration`. It is strong enough to pick the first basin, then
  becomes secondary to depth and temporal consistency.
- **Palm proxy:** one wrist-to-middle-MCP capsule with a calibrated broad
  radius replaces the five intersecting wrist-to-MCP capsules. It preserves the
  tested capsule distance/Jacobian while ensuring dense palm samples are counted
  once. A thin two-capsule plate (longitudinal + knuckle line, ~12 mm) was
  built and measured against it on the target episode and **rejected**: median
  residual 5.46 vs 5.77 mm, p90 17.53 vs 17.22 mm, and identical fitted flexion
  (index_pip 18.5° median, 42° span, both ways). It moves which capsule absorbs
  a curled fingertip without resolving the ambiguity, so it is not worth the
  extra primitive.
- **ROI and segmentation:** the ROI is the union of current capsules padded by
  3 cm. A plane normal to palm-forward, 1 cm proximal to the wrist, removes the
  forearm. Recorded episodes do not currently persist a detector mask, so the
  optional mask cannot be applied offline; calibrated background subtraction is
  retained.
- **Sampling:** deterministic 4 mm voxel representatives, capped at 400 points
  per frame. Frames below 40 points get an empty data block.
- **Data term scale:** dividing by `sum(w)` makes the block invariant to how
  many depth pixels survived the ROI, but it also turns the data block into a
  per-frame *mean* while every other term is a *sum*. `w_data` restores the
  magnitude; without it the depth cloud contributed 0.1% of the functional on
  the target episode (see below) and the fit was decided entirely by smoothness
  and the posture prior.
- **Posture prior:** weighted per joint by `1/(1 + support)`, where `support`
  counts the frozen correspondences on the capsules distal to that joint
  (`hand_model.joint_capsule_support`). A quadratic prior grows with the very
  bend it is meant to regularise, so on well-measured joints it used to undo a
  correct flexion; it now only holds joints the depth cloud does not see.
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

## Functional balance

Weights were tuned against a measured energy split, not by eye. Every fit
reports one — `fit_trajectory` returns `energy_<term>` / `energy_frac_<term>`
in `info`, they are persisted in `hand_fit_metrics_json` and logged per episode.

Measured on `data/datasets/new-dataset/2026-09-01_10-51-08` (599 frames,
389 valid, 35.4% empty data blocks, 256 ROI points/frame after voxel+cap):

| term | before | after |
|---|---:|---:|
| data (depth cloud) | **0.1%** | **38.6%** |
| vel_rotation | 36.0% | 26.3% |
| vel_joints | 24.2% | 14.0% |
| posture | 29.0% | 10.1% |
| vel_translation | 3.6% | 5.6% |
| acc_joints | 4.0% | 1.9% |
| acc_rotation | 2.3% | 1.4% |
| landmark | 0.6% | 1.4% |
| acc_translation | 0.3% | 0.6% |

| quantity | before | after |
|---|---:|---:|
| median point→surface residual | 8.59 mm | **5.82 mm** |
| p90 point→surface residual | 25.75 mm | **17.43 mm** |
| wrist jerk norm, warm start → fit | 0.00989 → 0.00051 | 0.00989 → 0.00108 |
| batch ICP wall time | 47.19 s | 47.00 s |

The jerk rises slightly on purpose: the fingers now move instead of being held
straight, and the trajectory is still 8.8x smoother than the landmark warm start.

Weights changed: `w_data` 500.0 (new), `w_posture` 0.02 → 0.004,
`w_vel_rotation` 8 → 4, `w_vel_joints` 2 → 0.8, `w_acc_rotation` 20 → 10,
`w_acc_joints` 4 → 1.6.

## Warm-start flexion sign

Interphalangeal joints are flexion-only hinges, so the sign of a PIP/DIP angle
is anatomy, not a measurement. `q_from_landmarks` used to read it off
`axis · cross(u, v)`, which is a coin flip at the angles that actually occur:
with ~3 mm landmark noise on a ~25 mm phalanx the cross product's direction is
ambiguous below ~15°. On the target episode 40–58% of frames came out
hyperextended and 10–25% were clamped at the lower joint limit, collapsing a
real 7–20° median bend to 0–5°. The magnitude is now taken directly; the MCP/CMC
angle is measured against the calibrated rest direction, where hyperextension is
real, so it keeps its sign.

| joint | warm start before | warm start after | fitted |
|---|---:|---:|---:|
| index_pip | 4.8° | 15.3° | 18.6° (p5–p95 4.3–46.1) |
| middle_pip | 2.2° | 10.0° | 16.3° (3.3–40.0) |
| thumb_ip | −7.0° | 20.1° | 21.0° (9.8–51.7) |
| index_dip | −0.9° | 8.9° | 12.5° (3.7–51.2) |

The fitted median exceeds the warm start on every joint, i.e. the depth cloud
adds flexion rather than merely following the landmarks.

## Known limit: the correspondence basin

The fit preserves and sharpens a flexion the warm start indicates, but it cannot
*discover* one. Sampling a synthetic hand bent 50° at the PIPs and warm-starting
from a straight hand, the solve converges to 0° with the data term holding 99.9%
of the energy — dominant, and still blind to the bend. Nearest-surface
assignment is why: in the straight pose the curled phalanges' points are closest
to the *proximal* capsules (index_prox takes 400 of them, index_mid 16,
index_dist none), so the distal joints never see a gradient. Thinning the palm
into a two-capsule plate only moves the absorption to the proximal capsules
(index_prox 400, index_mid 16, index_dist 0) and changes no fitted angle. From the
true pose the same solve holds 55° (it used to fall back to 31° and drag the
wrist 16 mm), and from half the bend it closes to 29°.

Escaping that basin needs per-point part labels from the detector or a
multi-hypothesis search, not a reweighting. The warm start therefore gates
finger fidelity, which is why its sign bug mattered so much.

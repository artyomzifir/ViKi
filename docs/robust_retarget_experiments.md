# Robust perception and retarget experiments

This is a lab notebook for accuracy-affecting changes between camera observations
and robot IK. Its purpose is to keep a fixed reference, change one assumption at
a time, and distinguish measured data from trajectories invented by the pipeline.

## Experimental rules

1. A protected baseline is immutable. Never overwrite its artifact or manifest.
2. One experiment changes one algorithmic factor. Detector, triangulation,
   interpolation, smoothing, pose construction, confidence, and IK are separate
   factors.
3. Every comparison uses the same raw episode and records the input hashes,
   profile/parameters, code revision, output path, and metrics.
4. Metrics are split by evidence class:
   - **observed**: both required landmarks were triangulated in that frame;
   - **partially observed**: only part of the target-defining geometry exists;
   - **fabricated**: the value was filled or extrapolated;
   - **active for IK**: the final solver weight is greater than zero.
5. An interpolated finite number is not an observation. Coverage must never be
   reported from `isfinite(filled_points)` alone.
6. Internal consistency is not ground-truth accuracy. Reprojection residuals,
   smoothness, and bone-length consistency are useful diagnostics, but true metric
   accuracy requires an independent reference trajectory or known fiducial.
7. A candidate is promoted to the default only if its acceptance gates were
   stated before the run and it improves them without silently changing upstream
   observations.

## Metrics to record

| Boundary | Primary metrics |
|---|---|
| Camera/model | detections per camera, landmark confidence, timestamp coverage |
| Multi-view geometry | solved-joint fraction, two-view fraction, reprojection median/p95, ray angle, quality |
| Hand geometry | thumb-index gap and bone lengths, reported separately on observed and fabricated frames |
| Temporal processing | missing-run lengths, per-frame displacement, second difference, discontinuities at gap edges |
| Pose/IK | valid-pose fraction, positive-weight fraction, position/orientation error on active frames, joint/velocity/floor/collision violations |

For this two-finger task, the canonical translation target is the midpoint of
thumb tip and index tip. Their separation drives continuous gripper opening.
Consequently, a frame is directly supported for both quantities only when both
tips are observed in that frame.

## E0 — protected V2 baseline, 2026-09-09

Episode:
`data/datasets/new-dataset/2026-09-09_11-54-10` (`move-shipok`, right hand,
897 frames at 30 fps, two hardware-synchronised Kinect cameras, no recorded
frame drops).

Protected artifact:
`intermediates/baselines/stable-fused-hand-v2/cln.npz`

Artifact SHA-256:
`1d820cf7be39d654c2d0ffa04810b1b724e026ce7dadd9ee2c85598803ee2ed1`

The manifest beside the artifact owns the exact raw-input hashes and V2 recipe.
V2 uses MediaPipe at confidence 0.5, strict two-view triangulation, unlimited
coordinate-wise natural-cubic gap filling, Savitzky-Golay 7/2, articulated hand
post-processing, episode-normalised confidence, and a continuous gripper.

### Acquisition and triangulation

| Metric | Result |
|---|---:|
| Camera 0 detections | 870 / 897 (97.0%) |
| Camera 1 detections | 849 / 897 (94.6%) |
| Triangulated joints | 13,002 / 18,816 (69.1%) |
| Reprojection residual | median 1.55 px, p95 3.07 px |
| Triangulation quality | median 0.807 |
| Thumb-tip solved | 591 / 896 (66.0%) |
| Index-tip solved | 667 / 896 (74.4%) |
| Both tips solved in the same frame | 490 / 897 (54.6%) |

The cameras and the two-view solve are not the first obvious failure: when both
tips are actually observed, their separation is physically plausible for this
episode (median 78.4 mm, p95 136.6 mm, p99 145.1 mm, maximum 150.9 mm).

### Prepared trajectory and solver evidence

| Metric | Result |
|---|---:|
| Valid end-effector pose | 828 / 897 (92.3%) |
| All four palm landmarks observed together | 450 / 897 (50.2%) |
| Positive pose weight `omega` | 772 / 897 (86.1%) |
| Positive final IK evidence weight | 469 / 897 (52.3%) |
| Median positive final IK weight | 0.669 |
| Longest run without an observed tip pair | 93 frames (3.1 s) |

The difference between 92.3% finite poses and 52.3% evidence-backed IK frames is
important: numerical gap filling makes the trajectory look complete but does not
create measurements.

### Failure localised to independent cubic gap filling

For the thumb-index separation:

| Data slice | median | p95 | p99 | maximum | frames >250 mm |
|---|---:|---:|---:|---:|---:|
| Directly observed pair | 78.4 mm | 136.6 mm | 145.1 mm | 150.9 mm | 0 |
| Cubic filled | 79.8 mm | 434.9 mm | 7,519.6 mm | 25,219.1 mm | 96 |
| Cubic + SG 7/2 | 79.5 mm | 434.5 mm | 7,519.6 mm | 25,211.4 mm | 96 |

The largest error is at frame 0, inside a 27-frame leading gap. Natural cubic
extrapolation is therefore a proven failure mode here. Long interior gaps are
also unsafe because every landmark coordinate is interpolated independently;
the result does not preserve a hand configuration.

## E1 — read-only interpolation probe

This probe did not change or write a pipeline artifact. It replaced only the
fused natural-cubic fill with the existing component-wise linear fill, then
applied the same SG 7/2 smoothing to the same `10_fused_observed.npz` points.

| Metric | V2 cubic + SG | Linear + same SG | Change |
|---|---:|---:|---:|
| Thumb-index maximum | 25,211.4 mm | 256.7 mm | -99.0% |
| Thumb-index p95 | 434.5 mm | 156.7 mm | -63.9% |
| Pinch-centre step p95 | 12.27 mm/frame | 8.36 mm/frame | -31.8% |
| Pinch-centre second-difference RMS | 5.43 mm | 0.98 mm | -82.0% |
| Difference on directly observed tip coordinates | — | 0.186 mm RMS | diagnostic only |

Linear filling removes the catastrophic spline extrapolation, but it is not yet
a sufficient solution. It still produces nine frames above 250 mm in the
93-frame missing-pair run (frames 48–140). During that run the two tips are seen
at different times, so independently interpolating each landmark combines
mutually unsupported hand states.

**Initial technical decision:** linear alone does not solve evidence and hand
geometry across long asynchronous gaps.

**Product decision, 2026-09-09:** promote linear to the stable V2 path because
it removes the catastrophic cubic failure and is a strictly safer interpolation
baseline. The remaining 256.7 mm asynchronous-gap failure stays open and must
not be interpreted as solved by this promotion.

V1 and its protected artifacts retain cubic behaviour. Updated V2 artifacts
record `fused_interpolation=linear`, and their checkpoint directory includes
`interp-linear` so it cannot overwrite the cubic control.

## E2 — linear promoted to V2 and regenerated

The same episode was prepared with the updated V2. Extraction inputs and the
triangulation artifact retained the exact hashes from E0. Array comparison also
confirmed exact equality for timestamps, landmark IDs, `observed_points`,
`observed_mask`, and `landmark_confidence` between the cubic and linear runs.

New protected V2 artifact SHA-256:
`5582be962b2158e14cdba139eedebc9e400670ebdfeed7426d1dd31f2fc10295`

The previous immutable cubic artifact remains at
`intermediates/archive/stable-fused-hand-v2-cubic-pre-linear/baseline/cln.npz`
with its original SHA-256
`1d820cf7be39d654c2d0ffa04810b1b724e026ce7dadd9ee2c85598803ee2ed1`.

| Metric | Cubic V2 | Linear V2 | Change |
|---|---:|---:|---:|
| Valid pose frames | 828 | 858 | +30 |
| Positive final IK evidence frames | 469 | 477 | +8 |
| Thumb-index p95 | 434.5 mm | 156.7 mm | -63.9% |
| Thumb-index maximum | 25,211.4 mm | 256.7 mm | -99.0% |
| Frames above 250 mm | 96 | 9 | -87 |
| Pinch-centre step p95 | 12.27 mm/frame | 8.36 mm/frame | -31.8% |
| Pinch-centre second-difference RMS | 5.43 mm | 0.98 mm | -82.0% |

The articulated overlay accepted all quality gates after regeneration. Its
optimized median anchor residual improved from 10.31 to 9.44 mm; p95 changed
from 35.43 to 37.18 mm. Source joint jerk fell from 2.02 to 1.14 mm and fitted
joint jerk from 1.21 to 1.13 mm. The p95 regression is retained as a warning,
not hidden by the better aggregate results.

## E3 — prohibit interpolation outside observation support

V2 still uses linear interpolation for an interior gap bracketed by two
observations. It now leaves leading and trailing gaps as NaN. A finite
articulated/robot state may hold the nearest model pose there, but its sensor
confidence remains zero and the fused landmark layer stays missing.

The previous linear edge-hold baseline is archived at
`intermediates/archive/stable-fused-hand-v2-linear-edge-hold/baseline/cln.npz`
with SHA-256
`5582be962b2158e14cdba139eedebc9e400670ebdfeed7426d1dd31f2fc10295`.
The new protected `fused-hand-no-extrap-v1` candidate is stored at
`intermediates/baselines/fused-hand-no-extrap-v1/cln.npz` and has SHA-256
`8037027140a4dea9bac41be4f27b68e9e771f23ba98f4a9b2b0c30736b57818b`.

Upstream timestamps, landmark IDs, observed points/masks, and confidence arrays
remain exactly equal.

| Metric | Linear edge hold | Linear no extrapolation | Change |
|---|---:|---:|---:|
| Finite thumb-index target frames | 897 | 853 | -44 unsupported edge frames |
| First/last finite pair frame | 0 / 896 | 27 / 879 | matches observation support |
| Clean valid pose frames | 858 | 821 | -37 |
| Retarget-valid pinch frames | 858 | 805 | -53 |
| Positive final IK evidence frames | 477 | 471 | -6 |
| Thumb-index maximum | 256.7 mm | 256.7 mm | unchanged |
| Frames above 250 mm | 9 | 9 | unchanged |
| Pinch-centre second-difference RMS | 0.978 mm | 0.983 mm | +0.005 mm |

The p95 thumb-index gap changes from 156.7 to 162.4 mm only because 44 benign
edge-hold values were removed from the distribution; the overlapping interior
trajectory is effectively unchanged. Directly observed tip coordinates move by
only 0.079 mm RMS from the different SG boundary condition.

The first downstream articulated run exposed a real integration bug: missing
sensor edges caused a root-state discontinuity and failed the temporal quality
gate (fitted jerk 2.39 mm). The failed report and CLN are preserved under
`intermediates/archive/no-extrap-failed-before-model-bridge/`. The final model
bridge holds only the zero-confidence articulated root state, never the fused
measurement. After that correction all gates pass: median anchor residual is
9.43 mm, p95 37.18 mm, fitted jerk 1.25 mm, and wrist preservation 0 mm.

A full production-config UR10 + Robotiq IK A/B also converged for both inputs:

| IK metric | Linear edge hold | Linear no extrapolation |
|---|---:|---:|
| Iterations | 17 | 16 |
| Objective | 0.6053 | 0.6010 |
| Position RMSE | 42.53 mm | 44.32 mm |
| Position p95 | 99.29 mm | 106.23 mm |
| Orientation RMSE | 24.11 deg | 19.14 deg |
| Maximum joint velocity | 1.390 rad/s | 1.502 rad/s |
| Maximum joint acceleration | 2.129 rad/s^2 | 2.301 rad/s^2 |
| Positive-weight frames | 477 | 471 |

Both plans retain positive collision, floor, joint-limit, and velocity margins.
Removing unsupported edge targets therefore improves orientation error and the
objective slightly, but it is not an across-the-board accuracy gain: position
error and peak joint motion increase modestly.

**Decision:** retain no-extrapolation as the separate
`fused-hand-no-extrap-v1` candidate for epistemic correctness, not as a numerical
accuracy boost. Stable V2 remains the linear edge-hold control until this
candidate is tested across several scenes. It prevents 44 unsupported edge
measurements but does not affect the 93-frame interior asynchronous-pair failure.

## Next controlled experiment

After the linear V2 promotion, the next implementation candidate should change
only pair-aware evidence handling:

- do not construct a thumb-index target from two independently filled landmarks
  when the pair was absent for a long interval;
- preserve `observed_points`, timestamps, triangulation quality, and all observed
  coordinates byte-for-byte;
- keep missing/unsupported target frames at zero solver evidence; let the robot
  trajectory regulariser bridge them explicitly rather than disguising them as
  camera measurements.

Before promotion, run it on this exact episode and require at least:

1. unchanged camera and triangulation metrics;
2. zero extrapolated pose/gripper measurements outside observation support;
3. no physically explosive thumb-index gaps in fabricated frames;
4. no reduction in the 469 directly evidence-backed IK frames;
5. separate IK error and constraint reports for evidence-backed and bridged frames.

The precise maximum bridge duration is deliberately not chosen yet. A fixed
frame count would depend on FPS; any threshold must be expressed in time and
validated across more than one episode before it becomes a default.

## Verification of the rollback checkpoint

- Backend default: `stable-fused-hand-v2`.
- Frontend default: `stable-fused-hand-v2`; `fused-hand-no-extrap-v1` is exposed
  as a separate locked candidate for multi-scene A/B runs.
- Focused perception tests: 5 passed.
- Full test suite: 278 passed, 3 skipped.
- Frontend module syntax check: passed.
- `git diff --check`: passed.

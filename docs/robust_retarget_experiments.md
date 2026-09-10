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

## E4 — no-extrapolation rejected on five scenes, 2026-09-09

E3 measured `fused-hand-no-extrap-v1` on one scene and deferred the promotion
decision pending a multi-scene run. That run is done. Both profiles were
prepared from the identical extraction and triangulation artifacts of each
episode, so only the fused fill differs; `pair observed` is therefore equal
within every scene by construction and confirms the comparison was clean.

`pair observed` counts frames where thumb tip and index tip were both actually
triangulated. `pair finite` counts frames where the target could be formed at
all, including fabricated values.

| Scene | pair observed | pair finite | valid | omega > 0 | gap p95 | gap max | frames > 250 mm | pinch 2nd-diff RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| move-shipok | 490 / 897 | 897 → 853 | 858 → 821 | 774 → 767 | 156.7 → 162.4 mm | 256.7 → 256.7 mm | 9 → 9 | 1.0 → 1.0 mm |
| cup_grab | 249 / 898 | 898 → 898 | 884 → 836 | 632 → 614 | 171.2 → 171.2 mm | 221.8 → 221.8 mm | 0 → 0 | 1.9 → 1.9 mm |
| block-push | 489 / 897 | 897 → 896 | 884 → 879 | 845 → 840 | 132.2 → 132.2 mm | 142.9 → 142.9 mm | 0 → 0 | 2.2 → 2.2 mm |
| pyramid | 403 / 898 | 898 → 891 | 871 → 869 | 801 → 799 | 148.7 → 148.9 mm | 228.6 → 228.6 mm | 0 → 0 | 1.8 → 1.7 mm |
| pick_up_u | 556 / 898 | 898 → 898 | 889 → 877 | 781 → 771 | 122.3 → 122.3 mm | 130.4 → 130.4 mm | 0 → 0 | 2.0 → 2.0 mm |

**Decision: reject `fused-hand-no-extrap-v1`; `stable-fused-hand-v2` remains the
default.**

Every geometry metric is identical on every scene: maximum thumb-index
separation, p95, the count of frames above 250 mm, and pinch-centre smoothness
do not move. The candidate's stated purpose was epistemic — never represent an
unbracketed value as a measurement — and it does achieve that. It does not
improve any measured quantity, and it consistently removes valid frames: −37,
−48, −5, −2 and −12 across the five scenes. On this evidence it is cost without
measured benefit, so it is not promoted.

Two observations are recorded rather than acted on here:

- On `cup_grab` the thumb-index pair stays finite in all 898 frames while
  `valid` drops by 48. The removed edges therefore belong to other landmarks,
  whose NaN edges then fail the geometric plausibility filter. Suppressing edge
  extrapolation has a side effect on frames whose target pair was never
  affected. If the no-extrapolation idea is revisited, it should be applied per
  landmark group rather than to the whole fused array.
- `move-shipok` is the hardest of the five scenes, and the only one with any
  frame above 250 mm. E0–E3 were all calibrated on it. Conclusions drawn there
  should not be assumed to transfer; this run is the first check of that, and
  it did not transfer.

### What the numbers actually point at

The `pair observed` column is the finding that outlives this experiment:

| Scene | frames with both tips triangulated |
|---|---:|
| cup_grab | 249 / 898 (27.7%) |
| pyramid | 403 / 898 (44.9%) |
| block-push | 489 / 897 (54.5%) |
| move-shipok | 490 / 897 (54.6%) |
| pick_up_u | 556 / 898 (61.9%) |

Between 28% and 62% of frames carry a directly observed thumb-index pair. In the
remainder, both the translation target and the gripper opening are fabricated by
interpolation regardless of which of the five profiles is selected. Every
experiment so far — cubic to linear, edge hold to no extrapolation, and the
pending gap-cap work — argues about how to fill that hole. None of them reduces
it.

Coverage was never examined by the batch-solver audit either: that review was
scoped to `viki/retarget/`, and its confidence-related points (`interpolated_mask`
unread, episode-normalised `omega`, the 0.05 confidence floor) concern how
fabricated data is weighted downstream, not how much real data is acquired
upstream.

The acquisition-side levers that already exist in code and are set by no profile:
`tracking_confidence` and `strict_handedness`, both plumbed from
`PerceptionProfile` through `perceive_episode` and `extract_episode` into the
MediaPipe backend. `interp_max_gap` also remains 0 (unlimited) in all five
profiles. The next experiment should move coverage, not interpolation.

## E5 — handedness gating removed, 2026-09-09

E4 showed that every fill strategy argues over a hole covering 28–62% of frames.
This experiment attacks the hole itself.

### What the gate did

`MediaPipeHandBackend._extract` asked the detector for a hand whose label
matched the episode's declared side and, on a mismatch, returned `None` — the
whole detection, all 21 landmarks and their depth samples, for a frame whose
geometry was fine. The rejection was symmetric: asking for `left` discarded a
hand labelled `Right` the same way.

MediaPipe infers handedness from the appearance of the hand in one 2-D image,
so a hand seen palm-on and the same hand seen back-on are mirror images. In a
two-camera rig the two views legitimately disagree. Measured by running the
detector directly over the recordings:

| Scene / camera | no hand | labelled as the declared hand | labelled as the other hand | median score of the mislabel |
|---|---:|---:|---:|---:|
| cup_grab k0 | 13 (1.4%) | 884 (98.4%) | 1 (0.1%) | — |
| cup_grab k1 | 104 (11.6%) | 688 (76.6%) | **106 (11.8%)** | **0.967** |
| move-shipok k1 | 0 | 849 (94.6%) | 48 (5.4%) | 0.614 |
| pyramid k1 | 36 (4.0%) | 842 (93.8%) | 20 (2.2%) | 0.735 |

On `cup_grab` the second camera is confidently wrong on 11.8% of frames for the
same physical right hand that the first camera labels correctly 98.4% of the
time. That is not detector noise; it is a viewpoint-dependent classification of
a property we already know.

The anatomical side is a recording parameter: one hand per episode, declared by
the operator and stored in `meta["hand"]`. Re-deriving it per camera per frame
and then discarding evidence when the guess disagrees is unsound. The gate and
its `strict_handedness` knob were removed rather than made configurable, and the
graph still runs with `num_hands=1`, so the returned hand is used as-is.

The handedness score was also the only "confidence" the detection path produced,
broadcast identically to all 21 landmarks (`lm_score_per_pt=False`, measured
constant at 0.872 and 0.762 on two episodes). It measured certainty about a
label, never landmark quality, so it is no longer propagated; triangulation
geometry owns rejection.

### Effect on acquisition

Predicted from the mislabel counts and confirmed exactly:

| Scene | kinect_1 detections before | after |
|---|---:|---:|
| cup_grab | 76.6% | **88.4%** |
| move-shipok | 94.6% | **100.0%** |

### Effect on the pipeline, five scenes

Same profile (`stable-fused-hand-v2`), same episodes, extraction re-run. The
strict artifacts are archived at
`intermediates/archive/stable-fused-hand-v2-strict-handedness/`.

| Scene | observed landmarks | both tips observed | frames with IK evidence | gap p95 | gap max | frames > 250 mm |
|---|---:|---:|---:|---:|---:|---:|
| cup_grab | 8 963 → 9 556 (+593) | 249 → 266 | 632 → 698 | 171.2 → 162.7 mm | 221.8 → 218.4 mm | 0 → 0 |
| pyramid | 13 848 → 14 058 (+210) | 403 → 410 | 801 → 818 | 148.7 → 139.4 mm | 228.6 → 228.6 mm | 0 → 0 |
| block-push | 13 968 → 14 277 (+309) | 489 → 495 | 845 → 874 | 132.2 → 132.3 mm | 142.9 → 142.9 mm | 0 → 0 |
| move-shipok | 13 002 → 13 636 (+634) | 490 → 517 | 774 → 820 | 156.7 → 156.8 mm | 256.7 → 256.7 mm | 9 → 9 |
| pick_up_u | 13 556 → 14 012 (+456) | 556 → 574 | 781 → 817 | 122.3 → 122.5 mm | 130.4 → 130.4 mm | 0 → 0 |

**Decision: keep. The gate is removed permanently.**

Directly observed landmarks rise on every scene (+2.3% to +6.6%) and the number
of frames carrying positive solver evidence rises on every scene (+17 to +66).
No geometry metric degrades: maximum thumb-index separation and the count of
frames above 250 mm are unchanged everywhere, and the p95 improves on the two
scenes with the worst mislabelling. Unlike E1–E3, this adds measurements rather
than redistributing the consequences of missing ones.

Two effects are recorded rather than smoothed over:

- On `cup_grab` the geometric plausibility filter passes 31 fewer frames
  (884 → 853) while evidence-backed frames rise by 66. The share of valid frames
  that are actually evidence-backed goes 71% → 82% there, and rises on all five
  scenes (92→94, 96→99, 90→95, 88→92). Recovered frames come from the harder
  viewpoint, so a few reconstruct implausibly and are correctly dropped; what
  survives is better supported than before.
- `omega` is still normalised by the episode maximum, so changing the
  observation set rescales every weight and the `omega > 0` counts are not
  perfectly comparable between runs. The unnormalised observed-landmark count is
  the metric to trust here. This is the open audit point on confidence
  calibration, unchanged by this experiment.

Not addressed: with `num_hands=1` the detector returns its most confident hand.
If a second hand entered frame, the pipeline would now silently track whichever
one the detector preferred. The single-hand recording protocol is the assumption
that makes this safe, and it is now load-bearing rather than advisory.

## E6 — why joints are rejected, five scenes, 2026-09-09 (measurement only)

E5 raised acquisition. This is a read-only census of what happens to a joint
that *was* acquired by both cameras: `triangulate_joint`'s accept/reject logic
was replayed per frame per landmark without writing anything.

### Rejection causes

| Scene | solved | < 2 views | cheirality | ray angle | rays disagree |
|---|---:|---:|---:|---:|---:|
| cup_grab | 50.8% | 12.4% | **0%** | **0%** | 36.8% |
| move-shipok | 72.4% | 3.0% | **0%** | **0%** | 24.6% |
| block-push | 75.8% | 0.0% | **0%** | **0%** | 24.2% |
| pyramid | 74.5% | 4.0% | **0%** | **0%** | 21.4% |
| pick_up_u | 74.3% | 5.8% | **0%** | **0%** | 19.9% |

The cheirality and minimum-ray-angle gates never fire on any of the five
episodes. Every rejection of an acquired joint comes from one place: the DLT
point reprojects beyond `TRI_REPROJ_INLIER_PX = 4`, so the hypothesis collects
fewer than two inliers and the joint is deleted.

### How badly do the rays actually disagree

| Scene | reproj med | 4–8 px | 8–16 px | 16–40 px | 40–100 px | > 100 px | reproj max |
|---|---:|---:|---:|---:|---:|---:|---:|
| cup_grab | 8.0 px | 50.2% | 36.8% | 12.9% | 0.1% | **0%** | 52 px |
| move-shipok | 5.8 px | 76.4% | 18.8% | 4.7% | 0.2% | **0%** | 42 px |
| block-push | 5.8 px | 73.2% | 24.4% | 2.4% | 0% | **0%** | 24 px |
| pyramid | 5.8 px | 72.5% | 25.1% | 2.4% | 0% | **0%** | 36 px |
| pick_up_u | 5.9 px | 76.8% | 20.3% | 2.7% | 0.1% | **0%** | 49 px |

The same population expressed physically — the shortest distance between the two
lines of sight:

| Scene | median | p90 | p99 | max | tips median | tips p90 |
|---|---:|---:|---:|---:|---:|---:|
| cup_grab | 17.8 mm | 40.2 | 65.0 | 103 | 20.2 mm | 45.3 |
| move-shipok | 11.4 mm | 23.4 | 55.8 | 76 | 12.8 mm | 39.8 |
| block-push | 12.5 mm | 24.9 | 42.1 | 56 | 14.9 mm | 31.6 |
| pyramid | 11.1 mm | 22.8 | 40.7 | 71 | 12.8 mm | 25.4 |
| pick_up_u | 12.2 mm | 22.0 | 44.6 | 122 | 13.2 mm | 28.0 |

### What this settles

The design question before this measurement was whether a rejected joint is a
noisy measurement (worth keeping at low weight) or a hallucination by one camera
(worth deleting). If occluded fingertips were being invented, the rejected
population would be bimodal with a tail at hundreds of pixels and tens of
centimetres.

It is not. Across roughly 23,000 rejections on five episodes there is **not one**
above 100 px, and 0.0–0.2% above 40 px. The distribution is smooth and pressed
against the gate: the median rejected joint misses by 5.8–8.0 px, which is
11–18 mm of ray separation — about one finger width, the distance between a
fingernail and a finger pad. Two detectors placing the same tip slightly
differently on the same finger, not one of them inventing it.

Meanwhile E0–E2 measured what replaces a deleted joint: coordinate-wise
interpolation, with a thumb-index error up to 25,219 mm before the linear
promotion and still 256.7 mm after it. The pipeline is discarding 11–18 mm
measurements and substituting values that can be wrong by 256 mm.

### The gate contradicts the weighting function it feeds

```python
quality = inlier_frac
        * clip(1 - mean_err / (2 * cfg.reproj_inlier_px), 0, 1)
        * clip(ray / cfg.ray_ref_deg, 0, 1)
```

The reprojection term is written to grade linearly to zero at **twice** the
inlier gate. With the gate at 4 px it is designed to express confidence for
errors up to 8 px — but nothing above 4 px ever reaches it. Half of the
designed grading range is unreachable by construction.

So the graded-confidence mechanism this experiment was going to add already
exists; a threshold is preventing it from operating over its own range.
`TRI_REPROJ_INLIER_PX` is already a per-profile knob (`profile.triangulation`)
and already doubles as the `f_scale` of the robust refinement loss, so widening
it relaxes the gate and the robust loss consistently, with no code change.

**No decision is taken here.** The candidate and its acceptance gates are stated
under "Next controlled experiment".

## E7 — widening the reprojection gate, pre-registered 2026-09-09

**Registered before the run**, per rule 7. E6 established the mechanism; this is
the change it implies, with its acceptance gates fixed in advance.

### Change

One number, in the profile, no code change:
`triangulation.reproj_inlier_px` 4 px → **8 px** (`fused-hand-reproj8-v1`) and
→ **16 px** (`fused-hand-reproj16-v1`). Everything else is V2 verbatim.

That number is used in exactly three places, and moving it moves all three
consistently:

| use | at 4 px | at 16 px |
|---|---|---|
| inlier gate, `e <= reproj_inlier_px` | accept ≤ 4 px | accept ≤ 16 px |
| `f_scale` of the `soft_l1` refinement loss | robust past 4 px | robust past 16 px |
| `quality` falloff, `1 - err/(2·reproj_inlier_px)` | zero at 8 px | zero at 32 px |

So a joint is never admitted at a confidence the recipe could not already
express: the gate and the confidence ramp widen together. Two doses rather than
one, so the result is a response curve — if 16 px is too far, 8 px says whether
the direction was right.

### Hypothesis

Rejected joints are not hallucinations (E6: not one rejection above 100 px in
~23,000; median 5.8–8.0 px ≈ 11–18 mm of ray separation). They are the same
finger seen slightly differently by two detectors. Deleting them and filling the
hole by interpolation substitutes a value that E0–E2 measured wrong by up to
256.7 mm. Admitting them at a weight proportional to their agreement should
therefore raise the observed fraction without degrading geometry.

### The counter-hypothesis this must rule out

Widening `f_scale` makes `soft_l1` nearly quadratic across the whole working
range, so a genuinely bad view is no longer down-weighted during refinement and
can drag the refined point. If that dominates, the trajectory gets noisier and
hand geometry stops closing. Gates 2–4 are there to catch exactly this, and a
failure on them is a reason to reject 16 px, or both doses, not to reinterpret
them.

### Acceptance gates

Measured against the protected `stable-fused-hand-v2` artifact on the same five
episodes (`move-shipok`, `cup_grab`, `block-push`, `pyramid`, `pick_up_u`).

1. **Coverage must rise.** `pair_observed` — frames where thumb tip *and* index
   tip were both triangulated, not filled — strictly increases on **all five**
   scenes. This is the point of the change; no improvement anywhere means reject.
2. **No new fabricated explosions.** `gap_over250` (thumb-index separation above
   250 mm) does not increase on any scene, and `gap_max` does not increase.
3. **No new jitter.** `pinch_d2_rms`, the RMS second difference of the
   thumb-index midpoint, rises by no more than **10%** on any scene. This is the
   primary detector for the counter-hypothesis.
4. **Hand geometry must not stretch.** Bone-length dispersion on *observed*
   frames — median across the 20 hand bones of the robust coefficient of
   variation, and the p95 absolute deviation in mm — does not degrade by more
   than **10%** on any scene. An admitted bad joint shows up here as a bone that
   changes length.
5. **Upstream untouched.** `frames` and per-camera detection counts are
   identical to V2; only triangulation may differ.

A dose is promoted only if it passes all five. If 8 px passes and 16 px fails,
8 px is the answer and the falloff-range coupling is the reason. If both pass,
the one with the better coverage/jitter trade is promoted and the other is kept
as a locked candidate.

### Result

All ten runs completed (two doses × five scenes). Compared against each scene's
protected `stable-fused-hand-v2` artifact.

| scene | profile | pair_obs | obs LM % | gap_max | gap>250 | d2_rms | bone_cv |
|---|---|---:|---:|---:|---:|---:|---:|
| move-shipok | v2 (4 px) | 517 | 72.4 | 256.7 | 9 | 0.98 | 6.64 |
| | 8 px | 696 | 91.2 | 256.9 | 9 | 1.05 | 7.11 |
| | 16 px | 783 | 95.8 | 166.4 | **0** | 1.15 | 7.37 |
| cup_grab | v2 (4 px) | 266 | 50.7 | 218.4 | 0 | 2.13 | 8.76 |
| | 8 px | 469 | 69.1 | 126.8 | 0 | 2.46 | 10.80 |
| | 16 px | 655 | 82.5 | 126.7 | 0 | 2.80 | 12.02 |
| block-push | v2 (4 px) | 495 | 75.8 | 142.9 | 0 | 2.23 | 6.19 |
| | 8 px | 730 | 93.5 | 135.8 | 0 | 2.48 | 6.55 |
| | 16 px | 855 | 99.4 | 135.8 | 0 | 2.66 | 6.96 |
| pyramid | v2 (4 px) | 410 | 74.6 | 228.6 | 0 | 1.76 | 9.99 |
| | 8 px | 616 | 90.1 | 193.0 | 0 | 2.13 | 10.16 |
| | 16 px | 839 | 95.5 | 122.7 | 0 | 2.46 | 9.86 |
| pick_up_u | v2 (4 px) | 574 | 74.3 | 130.4 | 0 | 2.02 | 7.62 |
| | 8 px | 769 | 89.6 | 130.3 | 0 | 2.39 | 9.47 |
| | 16 px | 834 | 93.6 | 130.3 | 0 | 2.53 | 9.19 |

Against the gates as registered:

| gate | 8 px | 16 px |
|---|---|---|
| 1 coverage rises on all five | **pass** (+195…+235) | **pass** (+260…+429) |
| 2 no new fabricated explosions | **fail** — move-shipok `gap_max` +0.19 mm | **pass**, and move-shipok 9 → 0 |
| 3 jitter ≤ +10% | **fail** on 4/5 (up to +21.0%) | **fail** on 5/5 (up to +39.5%) |
| 4 bone dispersion ≤ +10% | **fail** on 2/5 (up to +24.3%) | **fail** on 4/5 (up to +37.2%) |
| 5 upstream unchanged | pass | pass |

**Neither dose is promoted.** Rule 7 is not satisfied and the profiles stay
locked candidates.

### Gates 3 and 4 were specified wrong

This is a defect in the experiment, not a finding about the change.

`d2_rms` and `bone_cv` were each computed over whatever frames the recipe
produced. At 4 px only 50–76% of landmarks are triangulated and the remainder is
interpolation, which is smoother than any measurement by construction. Raising
coverage replaces an invented smooth curve with real data, so both statistics
must rise even if every recruited joint is perfect. As written, gates 3 and 4
cannot separate "more real noise" from "worse geometry" — which is precisely the
distinction they were introduced to make.

### E7b — the same artifacts, conditioned on population

Re-measured over the existing outputs, no re-run. `common` restricts both
recipes to the frames/landmarks **both** observed; `new` measures only the joints
the candidate added.

| scene | dose | jitter on common | bone_cv on common | bone_cv of new joints | v2's own bone_cv | new LM % |
|---|---|---|---|---:|---:|---:|
| move-shipok | 8 px | 1.04 → 1.05 (+1.5%) | 6.64 → 6.41 (−3.5%) | **6.01** | 6.64 | 18.8 |
| | 16 px | 1.04 → 1.08 (+4.0%) | 6.64 → 6.53 (−1.7%) | 7.02 | 6.64 | 23.4 |
| cup_grab | 8 px | 2.26 → 2.21 (−2.2%) | 8.76 → 9.18 (+4.8%) | **14.09** | 8.76 | 18.4 |
| | 16 px | 2.26 → 2.35 (+3.9%) | 8.76 → 9.36 (+6.8%) | **17.78** | 8.76 | 31.8 |
| block-push | 8 px | 2.13 → 2.26 (+6.0%) | 6.19 → 6.48 (+4.6%) | 6.17 | 6.19 | 17.7 |
| | 16 px | 2.13 → 2.36 (+10.4%) | 6.19 → 6.56 (+5.8%) | 7.93 | 6.19 | 23.6 |
| pyramid | 8 px | 1.60 → 1.69 (+5.6%) | 9.99 → 9.92 (−0.7%) | **7.81** | 9.99 | 15.5 |
| | 16 px | 1.60 → 1.72 (+7.3%) | 9.99 → 10.17 (+1.8%) | **7.57** | 9.99 | 20.9 |
| pick_up_u | 8 px | 1.74 → 1.83 (+5.2%) | 7.62 → 7.54 (−1.0%) | 9.98 | 7.62 | 15.3 |
| | 16 px | 1.74 → 1.89 (+8.1%) | 7.62 → 7.56 (−0.8%) | 9.90 | 7.62 | 19.3 |

Two things follow.

**The widened gate does not disturb the joints V2 already had.** On the common
population jitter moves by −2.2% to +10.4% and bone dispersion by −3.5% to
+6.8%; several scenes improve. The +21…+39% figures under gate 3 were the
coverage confound, not the counter-hypothesis. The counter-hypothesis — a
non-robust `f_scale` letting a bad view drag the refined point — is not
supported at 8 px and is visible only as block-push's +10.4% at 16 px.

**Recruit quality is scene-dependent, and that is the real result.** Judged by
its own bone-length consistency, the joints 8 px adds are *better than V2's own
joints* on move-shipok (6.01 vs 6.64) and pyramid (7.81 vs 9.99), equal on
block-push (6.17 vs 6.19), mediocre on pick_up_u (9.98 vs 7.62) and clearly
worse on cup_grab (14.09 vs 8.76). 16 px degrades the recruits on three of five.

Caveat that must be kept attached to those numbers: recruited joints are by
definition the harder ones, so some dispersion increase is expected and a fair
comparison against V2's easier population is not available. That makes
move-shipok and pyramid — where the harder subpopulation is *more* consistent
than V2's easier one — the strong evidence, and pick_up_u's +31% the weak
evidence.

cup_grab being the failure is consistent with E6, which already singled it out:
the only scene with 12.4% of joint-slots below two views, the worst rejected
reprojection median (8.0 px vs 5.8 elsewhere) and the largest ray separation
(17.8 mm median vs 11–12 mm). Its camera geometry is the poorest of the five, so
widening the agreement gate there admits genuinely disagreeing views.

### Status and what a promotion would require

8 px is a promising candidate; 16 px is not supported. But re-adjudicating this
run under the conditioned metrics is exactly what rule 7 forbids — the gates
must precede the run. So 8 px is **not** promoted here.

A promotion run must register, in advance: jitter and bone dispersion on the
**common** population (≤ +10%), recruit bone dispersion no worse than 1.25× the
recipe's own on at least four of five scenes, coverage strictly up on all five,
and `gap>250` non-increasing — evaluated on scenes not used to form this
hypothesis. cup_grab suggests the threshold may need to depend on the episode's
measured camera geometry rather than being a constant, which is a larger change
than one number and should not be smuggled into this one.

## E8 — the default profile gives fabricated palm poses a real IK weight

Found while wiring perception profiles to retarget. This is a defect in the
shipped default, not a candidate under test, so it is recorded before the
retarget comparison it was found during.

### What omega is supposed to be

`omega` is the per-frame scalar that weights the whole SE(3) pose in the IK data
term. It is built in `viki/prepare/run.py::_confidence_arrays` as the **mean of
the landmark confidences of the four landmarks that define the palm frame**
(wrist, index MCP, middle MCP, pinky MCP).

`landmark_confidence` is correctly zero on any landmark that was not
triangulated — verified on move-shipok V2: 5,201 fabricated cells, **all
exactly 0.0**, against mean 0.799 on the 13,636 observed ones. The filling stage
does not invent confidence.

### The defect

A mean over four numbers is not zero when only some of them are. A frame whose
palm was one-quarter observed and three-quarters interpolated therefore gets
roughly a quarter of full weight, and pulls the solver toward a pose whose
orientation was mostly invented.

Measured on the five protected V2 baselines — frames that pull the IK
(`omega > 0`), grouped by how many of the four palm landmarks were actually
triangulated, with the mean omega each group receives:

| scene | 4/4 | 3/4 | 2/4 | 1/4 | 0/4 |
|---|---|---|---|---|---|
| move-shipok | 450 @ 0.79 | 181 @ 0.57 | 134 @ 0.39 | 55 @ 0.20 | 0 |
| cup_grab | 275 @ 0.81 | 152 @ 0.58 | 144 @ 0.38 | 127 @ 0.20 | 0 |
| block-push | 396 @ 0.80 | 347 @ 0.61 | 93 @ 0.39 | 38 @ 0.18 | 0 |
| pyramid | 448 @ 0.80 | 235 @ 0.60 | 100 @ 0.38 | 35 @ 0.18 | 0 |
| pick_up_u | 323 @ 0.81 | 230 @ 0.59 | 172 @ 0.38 | 92 @ 0.20 | 0 |

So on every scene **45–61% of the frames that drive the IK have an incompletely
observed palm**, and 131–271 frames per scene are driven by a palm frame with
only one or two real landmarks out of four. A palm *orientation* fitted through
one measured point and three interpolated ones is not a partial measurement —
it is a fabrication that the weight presents as 20% of a measurement.

This also contradicts a comment in `viki/retarget/run.py::load_targets`, which
states that confidence stays exactly zero so that "no fabricated target row
contributes to the objective". That holds only for rows where the whole frame is
invalid, not for partially observed palms.

### The fix already exists, attached to the wrong profile

`_confidence_arrays` has the guard, and its comment states the reasoning:

```python
if calibration == "absolute" and columns:
    # One scalar weights the complete SE(3) pose. If any landmark that
    # defines the palm frame was fabricated, treating the remaining three
    # as a partially observed orientation would overstate what the sensor
    # actually measured. Smoothness, not the data term, owns that frame.
    palm_observed = np.asarray(observed_mask, dtype=bool)[:, columns].all(axis=1)
    omega = np.where(palm_observed, omega, 0.0)
```

It runs only under `confidence_calibration == "absolute"`, which is
`stable-fused-hand-v3` alone. The `episode_max` branch — V1, V2, the no-extrap
candidate and both E7 reproj candidates — skips both this and the
`observed_mask` gate on `landmark_confidence` above it. The default profile is
`stable-fused-hand-v2`, so **the product ships the unguarded path** and the
guarded one is on a profile that has never been measured on any scene.

Whether that was deliberate (V1/V2 manifests are immutable) or an oversight, the
consequence is the same and is a fact about the default, not about a candidate.

### Interaction with E7

The widened reprojection gate removes most of this contamination as a side
effect, by leaving almost nothing to fabricate. Frames with `omega > 0` and an
incompletely observed palm, V2 (4 px) → `fused-hand-reproj16-v1`:

| scene | V2 | 16 px |
|---|---|---|
| move-shipok | 370 | **0** |
| cup_grab | 423 | 62 |
| block-push | 478 | **0** |
| pyramid | 370 | 6 |
| pick_up_u | 494 | 6 |

This is an argument for the widened gate that does not depend on E7's jitter or
bone-dispersion gates at all, and it is about the IK objective rather than about
trajectory smoothness. It is also independent of the omega fix: one removes the
fabrication, the other stops mis-weighting whatever fabrication remains. They
compose, and cup_grab needs both.

### Not yet answered

Whether any of this changes the IK solution measurably. The retarget comparison
across the fifteen (scene, profile) pairs was running when this was recorded;
its result goes below. Note in advance that `position_rmse_mm` is measured
against the target trajectory, which *includes* the fabricated rows — so a
recipe that fabricates more has a smoother target and can score a *better* RMSE
while tracking a worse trajectory. The comparison must therefore split the
position error by evidence class, per rule 4, and the headline RMSE must not be
read on its own.

## E9 — episode-max confidence makes a bad recording look as certain as a good one

Second defect found while wiring perception to retarget. Audit point 9 was on the
open list as a suspicion; this measures it.

### The mechanism

Under `confidence_calibration == "episode_max"` — V1, V2, the no-extrap candidate
and both E7 reproj candidates, i.e. everything except V3 —
`viki/prepare/run.py::_confidence_arrays` normalises the per-joint triangulation
quality by the episode's own maximum:

```python
scale = float(np.max(evidence)) if evidence.size else 1.0
landmark_confidence = np.clip(evidence / (scale or 1.0), 0.0, 1.0)
```

Every episode is therefore rescaled so its best joint scores exactly 1.0,
whatever that joint's absolute quality was.

### Measured on the five protected V2 baselines

| scene | max | mean over observed | p05 | p50 | observed LM % |
|---|---:|---:|---:|---:|---:|
| move-shipok | 1.0000 | 0.7985 | 0.618 | 0.809 | 72.4 |
| cup_grab | 1.0000 | 0.7980 | 0.604 | 0.808 | 50.7 |
| block-push | 1.0000 | 0.8014 | 0.605 | 0.816 | 75.8 |
| pyramid | 1.0000 | 0.8134 | 0.621 | 0.826 | 74.5 |
| pick_up_u | 1.0000 | 0.8202 | 0.621 | 0.839 | 74.3 |

The five means span 2.2 percentage points. But E6 measured these episodes and
they are not comparable: cup_grab has 50.7% of landmarks observed against
pick_up_u's 74.3%, a rejected-reprojection median of 8.0 px against 5.9, a
ray-ray separation median of 17.8 mm against 12.2, and it is the only scene with
12.4% of joint-slots below two views. It is by every geometric measure the worst
of the five — and it reports a mean confidence of 0.798 against move-shipok's
0.799.

The normalisation has erased the difference it exists to express.

### Why it matters here specifically

`omega` derives from these values and becomes the IK data weight. So the weight
a frame receives depends on how good the *best* frame of its own episode
happened to be. Within one episode the ordering is preserved and the weights are
meaningful. Across episodes they are not — and a ViKi dataset is many episodes
concatenated, which is the entire product. A frame from a poor recording can
outweigh a frame from a good one purely because its episode had no better frame
to be divided by.

Note also the perverse direction: the worse an episode's best joint, the larger
the multiplier applied to every joint in it. Degrading a recording raises its
reported confidence.

### The fix already exists, on the same unused profile as E8's

The `absolute` branch takes `triangulate_joint`'s quality directly — it is
already defined on [0, 1] and already comparable between episodes — and its
docstring says so. It runs only under `stable-fused-hand-v3`.

So both defects recorded today, E8 and E9, are fixed in `stable-fused-hand-v3`
and live in `stable-fused-hand-v2`, which is the default. V3 has never been run
on any scene, has no protected artifact anywhere, and was described in
`profiles.py` as awaiting evidence that it "improves fixed-episode metrics
without reducing usable observations". That framing now looks like the wrong
test: V3's value is not better fixed-episode metrics, it is weights that mean
the same thing in every episode and that stop crediting fabricated palms.

### Two open items resolved while looking

- **Audit 10 — the IK confidence floor — is correctly implemented, not a
  defect.** `solver.py` applies
  `np.where(omega > 0.0, np.maximum(omega, options.confidence_floor), 0.0)`, so
  the 0.05 floor lifts only weights that are already positive and never promotes
  a zero-weight fabricated row into data. The earlier note claiming otherwise was
  wrong.
- **Audit 2 — Huber mixing metres and radians — is a tuning choice, not a
  dimensional error.** `_pose_error` returns
  `[sqrt(w_pos)·Δp, sqrt(w_ori)·rotvec]`, so the norm the Huber sees is a
  properly weighted one and the weights are the unit conversion. With
  `position=1.0` and `orientation=0.01` the exchange rate is 1 rad ≈ 0.1 m, i.e.
  57° of palm rotation costs as much as 100 mm of position error. That is worth
  revisiting for a gripper task, but it is a parameter, not a bug.
- **Audit 8 — `interpolated_mask` still has no consumer.** Written in
  `prepare/run.py` and declared in `contracts.py`; nothing reads it. Harmless,
  and it is the array a pair-aware evidence stage would need, so it should stay.

### Correction, same day — E9's causal claim was wrong

E9 above observed that mean confidence is 0.798–0.820 on five episodes of very
different geometric quality, and concluded that the episode-max division had
erased the difference. Running `stable-fused-hand-v3`, which takes quality
directly and performs no division, refutes that:

| scene | v2 max | v3 max | v2 mean (observed) | v3 mean (observed) |
|---|---:|---:|---:|---:|
| move-shipok | 1.00000 | 0.99961 | 0.79853 | 0.79821 |
| cup_grab | 1.00000 | 0.99943 | 0.79799 | 0.79753 |
| block-push | 1.00000 | 0.99966 | 0.80143 | 0.80116 |
| pyramid | 1.00000 | 0.99973 | 0.81336 | 0.81314 |
| pick_up_u | 1.00000 | 0.99972 | 0.82017 | 0.81994 |

Every episode's best joint already scores 0.9994–0.9997, so dividing by it moves
confidence by about 0.04%. **The normalisation is a no-op on this data.** It did
not erase the difference; the difference was never there to erase.

The structural hazard E9 describes is still real — if an episode's best joint
were genuinely poor, every joint in it would be inflated, and nothing in the code
prevents that. But it does not fire on these five recordings, and no claim about
their measured confidence can be attributed to it.

**What is actually true, and is the more useful finding:** `quality` carries
almost no cross-episode information, for a reason that has nothing to do with
calibration. The hard reprojection gate has already deleted every joint that
would have scored low, so the survivors are uniformly good by construction —
cup_grab's surviving joints are as clean as move-shipok's. Episode quality shows
up in **how many** joints survive, not in how good the survivors look. Confidence
is high everywhere because the evidence that would have been low was thrown away
before it could be measured. This is the same survivorship that E6 documented
from the rejection side, seen from the acceptance side.

That also means `landmark_confidence` must not be compared between profiles with
different `reproj_inlier_px`: the quality falloff is `1 − err/(2·reproj_inlier_px)`,
so the same 3 px residual scores 0.63 at a 4 px threshold and 0.91 at 16 px. The
rise from 0.799 to 0.914 between v2 and the 16 px candidate is a change of ruler,
not of measured quality. This coupling was flagged when E7 was registered; it is
recorded here because the number is now on the page and would otherwise be
misread.

### Consequence for v3

`stable-fused-hand-v2.1` and `stable-fused-hand-v3` produce the same weighted
frame set on all five scenes (450/450, 275/275, 396/396, 448/448, 323/323) and
total evidence agreeing to three digits (356.3 vs 356.1, 223.5 vs 223.4, and so
on). **v3's only distinguishing factor is inert here.** Its entire practical
effect is the palm gate, which v2.1 now carries on its own.

So v3 is not junk, but it is redundant: it is v2.1 plus a recalibration that
changes nothing measurable on this data. If a new default is promoted it should
be promoted from v2.1, which changes one thing, not from v3, which changes two
and can only be told apart from v2.1 on an episode whose best joint is bad.

Separating the two factors was what made this visible. Bundled, the palm gate's
effect would have been credited to the recalibration.

### What the palm gate costs

Not free, and the cost must be stated before the retarget comparison reads on it:

| scene | frames with weight, v2 → v2.1 | total evidence, v2 → v2.1 |
|---|---|---|
| move-shipok | 820 → 450 (−45%) | 524.1 → 356.3 (−32%) |
| cup_grab | 698 → 275 (−61%) | 391.8 → 223.5 (−43%) |
| block-push | 874 → 396 (−55%) | 571.4 → 318.0 (−44%) |
| pyramid | 818 → 448 (−45%) | 543.9 → 358.8 (−34%) |
| pick_up_u | 817 → 323 (−60%) | 480.8 → 261.9 (−46%) |

v2.1 removes a fabrication, and it removes a lot of data-term support with it.
Whether the IK is better for it is exactly what the retarget grid is running to
find out — and it is why the widened reprojection gate matters: it is the only
lever measured so far that raises the number of *honest* weighted frames rather
than lowering the dishonest ones. On move-shipok the 16 px candidate reaches 870
weighted frames with zero incomplete palms, against v2's 820 with 370 incomplete
and v2.1's 450.

## E10 — the floor constraint rejects whole episodes, and honest perception makes it worse

Found by running the retarget grid: **block-push fails on every profile**, so the
scene is unusable downstream regardless of perception recipe.

```
RuntimeError: retarget could not find a floor-feasible trajectory;
remaining penetration is 0.0 mm   (v2)   2.5 mm (8 px)   4.9 mm (16 px)
```

### Where the violation comes from

The full target transform chain was replayed without solving — `load_targets` →
`rig_pose_to_calibration` → `_apply_hand_to_ee` → `calibration_to_robot` — to
attribute the violation to the targets rather than to the solver. Target TCP
height in the robot frame, floor at z = 0:

| scene | profile | frames z<0 | of those, weighted | min z | median z |
|---|---|---:|---:|---:|---:|
| block-push | v2 / v2.1 / v3 | 24 | 13 / 12 / 12 | −12.9 mm | 36.1 mm |
| | 8 px | 41 | 38 | −12.8 mm | 36.6 mm |
| | 16 px | 93 | 58 | −12.8 mm | 36.6 mm |
| move-shipok | all five | 0 | 0 | +122.8…128.9 mm | 211 mm |
| pick_up_u | all five | 0 | 0 | +9.7…9.9 mm | 189 mm |

The **depth** of the violation is the same for every recipe — about 12.9 mm. Only
the **count** changes: 24 frames under v2, 93 under the 16 px candidate. The hand
genuinely passes ~13 mm below the calibrated board plane while pushing a block
across the table, and better perception simply reveals more of the frames where
it does.

### Three separate problems, in order of importance

**1. The constraint is tighter than the calibration that defines it.** The floor
is the board plane at exactly z ≥ 0, and the violation is 12.9 mm. The board
solve's own accuracy on this dataset was measured at about 14 mm (a table landing
at world Z ≈ +0.014 m). So the trajectory is rejected for a penetration smaller
than the uncertainty in where the floor is. That comparison uses a plane error
measured on a *different* episode and should be re-measured on block-push before
being relied on, but the order of magnitude is the point.

**2. It is all-or-nothing.** 13 to 58 offending frames abort a 897-frame episode
with an exception. There is no per-frame reporting, no clamp, no option to drop
the offending frames and keep the rest. An operator gets a scene that simply
cannot be processed and a message that names one number.

**3. It punishes honest perception.** v2 fabricated its way past the constraint:
its interpolation smoothed the hand's lowest excursions upward, leaving 24
below-floor frames of which only 13 carried weight. The 16 px candidate exposes
93 and 58. The pipeline is therefore *more* likely to fail the better perception
gets — the exact inversion of what a quality gate should do, and the same shape
as E8, where fabricated data flattered the IK weighting.

### Also: the error message hides sub-0.05 mm penetrations

`solver.py` rejects at `min_floor < -1e-7` m (0.0001 mm) but formats the message
with `.1f` in millimetres, so everything from 0.0001 mm to 0.05 mm prints as
"remaining penetration is 0.0 mm". That is what v2 reported, and it reads as a
contradiction — infeasible by nothing. Cosmetic, one format specifier, but it
cost time here and would cost an operator more.

### Consequence for the grid

The retarget comparison runs on **four scenes**, not five. block-push is excluded
for a reason that has nothing to do with the perception profiles under test, and
excluding it silently would have hidden the most interesting failure in the set.

## E11 — the extract×retarget grid: perception is not the limiting factor

Twenty-two of twenty-five (scene × profile) pairs retargeted; block-push fails on
v2, 8 px and 16 px per E10. Every plan converged.

### The level, before any comparison

| | range across all 22 plans |
|---|---|
| position RMSE | **72 – 141 mm** |
| orientation RMSE | **64 – 93°** |
| min joint-limit margin | 2.68 – 3.07 rad |

A retarget that misses by 100 mm and 80° is not tracking the hand. Whatever the
perception recipe contributes is a correction to a number that is already an
order of magnitude too large, and no profile choice moves it into a usable range.
The joint margins are far from any limit, so the arm is not joint-blocked.

### Cause A — a large part of every trajectory is out of reach

The base is pinned at x = −0.7 m in the calibration frame while the hand works
around the board origin, so target TCP distance from the base is **525–911 mm**,
median 694–760. Across all 22 plans the achieved TCP never exceeds **739 mm**,
which pins the assembly's actual reach without appealing to a datasheet. So
12–61% of the frames of every episode ask for a pose the arm cannot occupy.

Per frame, split at that 739 mm radius:

| scene / profile | RMSE, target in reach | RMSE, target beyond | ori in | ori beyond |
|---|---:|---:|---:|---:|
| move-shipok / v2 | 71.4 mm | 94.9 mm | 59.7° | 71.9° |
| move-shipok / 16px | 38.1 mm | 95.6 mm | 52.9° | 72.0° |
| cup_grab / v2 | 72.6 mm | 119.5 mm | 80.7° | 92.1° |
| cup_grab / 16px | 54.1 mm | 98.4 mm | 61.6° | 75.3° |
| pyramid / v2 | 81.1 mm | 171.9 mm | 85.8° | 109.3° |
| pyramid / 8px | 53.9 mm | 136.7 mm | 73.7° | 108.2° |
| pick_up_u / v2 | 45.3 mm | 112.2 mm | 78.8° | 87.9° |
| pick_up_u / 16px | 39.5 mm | 109.2 mm | 63.1° | 80.2° |

Error on unreachable frames is **2–3× that on reachable ones** in every one of
the 22 plans, with no exception.

Either the base offset is wrong or the demonstrations are recorded too far from
where the robot will stand. Both are rig decisions, not code, and nothing
downstream can compensate: the arm is 740 mm long and the work is at 700–900 mm.

### Cause B — orientation is effectively unweighted

`BatchWeights` has `position=1.0`, `orientation=0.01`, and `_pose_error` scales
by their square roots, so the exchange rate is **1 rad ≈ 0.1 m**: 57° of palm
rotation costs the solver as much as 100 mm of position. With position errors
already around 100 mm, trading orientation away is the cheaper option every time,
and it does — 64–93° RMSE on every plan, even on reachable frames.

E9 recorded this as "a tuning choice, not a bug". At these magnitudes that
understates it: the retarget is not solving for palm orientation at all. For a
two-finger grasp, orientation is not a secondary objective.

### Methodological note — the aggregate correlation is a trap

Across the 22 plans, the Pearson correlation between the fraction of
out-of-reach frames and position RMSE is **−0.131**: absent, and slightly
backwards. The per-frame split above shows a factor of 2–3 with no exceptions.
The aggregate washes out because each plan's mix of scene difficulty, target
geometry and weighting differs; the effect only survives when the comparison is
made *within* a trajectory. This is the same population error as E7's gates 3–4,
in a new place: an average over inhomogeneous frames answers no question.

### What the profiles do, on the frames the robot can actually reach

**v2 → v2.1 is a perfectly controlled comparison.** Verified byte-identical
targets (`max|Δtarget| = 0.0000 mm` on all four solved scenes); only `omega`
differs. So any difference is caused by the palm gate alone.

| scene | v2 | v2.1 | v3 | 8 px | 16 px |
|---|---:|---:|---:|---:|---:|
| move-shipok | 71.4 | **88.9** | 88.7 | 68.8 | **38.1** |
| cup_grab | 72.6 | **86.0** | 86.1 | 56.1 | **54.1** |
| pyramid | 81.1 | **101.1** | 101.1 | **53.9** | 102.5 |
| pick_up_u | 45.3 | **68.3** | 68.3 | 38.6 | **39.5** |

**The E8 fix, applied on its own, makes the IK worse — on all four scenes, on
identical targets.** It removes 45–61% of the weighted frames (E9 correction
table), and the solver, left with less data-term support, drifts further from the
target it is still being asked to follow. Removing a fabrication is correct; doing
it without replacing the support it was providing is a net loss here.

That is a negative result for promoting v2.1 as-is, and it is the reason the two
factors were separated: bundled inside v3, this cost would have been invisible
and v3's inert recalibration would have been blamed or credited for it.

The widened reprojection gate goes the other way, roughly halving the reachable
frame error on three scenes of four, because it adds honest measurements rather
than removing dishonest weight. pyramid is the exception, and 8 px beats 16 px
there by a factor of two — consistent with E7, where 16 px was the dose that
degraded recruits on three of five scenes.

### What this grid does and does not license

It does **not** license promoting any profile. Position error is measured against
each profile's own target, so for 8 px and 16 px, whose target geometry differs
from v2's, part of the improvement may be an easier target rather than a better
one. Only v2 → v2.1 is fully controlled.

It does establish the order of work. Perception is not the binding constraint at
the IK stage: reach and orientation weighting are, and both are larger than every
perception effect measured in E4 through E9 combined. Fixing the palm gate's
lost support (rather than reverting the gate) is worth doing, but only after a
trajectory the arm can physically follow exists to measure it on.

### Retraction — E11 was run on the wrong robot

**The harness hardcoded `robot="ur3"`. The project default is
`RETARGET_DEFAULT_ROBOT = "ur10"`, and every historical plan on disk used it.**
That is a mistake in the experiment, not a finding about the pipeline.

Consequences:

- **Cause A above is void.** A UR10 reaches 1300 mm; the targets sit at
  525–911 mm. Nothing was out of reach. The measured 739 mm ceiling was the UR3
  saturating, and the 2–3× error on "unreachable" frames was that arm stretched
  straight. With the configured robot there is no reach problem to report.
- The absolute error level (72–141 mm, 64–93°) is not a property of the pipeline
  either. It is what a UR3 does when asked to follow a UR10-sized workspace.
- **Cause B may still stand** — the orientation weight is 1% of position
  regardless of arm — but its measured magnitude was taken from the same invalid
  runs and must be re-measured before being quoted.
- **The profile comparison must be redone.** The v2 → v2.1 result was internally
  controlled (byte-identical targets, only `omega` differing), so its *direction*
  may survive; its magnitudes may not. Nothing from E11 should be cited until the
  rerun replaces it.

### What the historical plans actually show

Four plans from 2026-09-08 survive on disk, and they are the "it used to be good"
reference:

| scene | robot | base | pose source | position RMSE | orientation RMSE |
|---|---|---|---|---:|---:|
| cup_grab | ur10 | **−0.7** | hand_fit | **33.8 mm** | 33.2° |
| pyramid | ur10 | +0.7 | hand_fit | 37.0 mm | 68.5° |
| pick_up_u | ur10 | +0.7 | hand_fit | 202.9 mm | 115.6° |
| block-push | ur10 | +0.7 | hand_fit | 379.0 mm | 91.8° |

Two things follow, and both are about the rig rather than perception.

**The base sign matters enormously and −0.7 is the right one.** The single run at
−0.7 scores 33.8 mm; the three at +0.7 score 37, 203 and 379 mm. Pinning the
default to −0.7 was correct, and the two poor historical scenes were mirrored to
the wrong side of the workspace, not badly perceived.

**block-push retargeted successfully on 2026-09-08** at 379 mm. Its E10 floor
failure is therefore not intrinsic to the scene either — it appears under a UR3,
whose shorter arm must dive to reach the same targets. E10 needs the same rerun
before its "the floor constraint rejects whole episodes" claim can stand; what
survives unconditionally is the formatting defect that prints sub-0.05 mm
penetrations as "0.0 mm".

### A real divergence, found while checking this

The historical plans record `source_pose = hand_fit`. The retarget config asks
for `pose_source: "hand_fit"` and `load_targets` honours it through
`cln_pose_keys`. The **active** `cln.npz` carries `hand_fit_positions` and
`hand_fit_rotations`, so production retargets the articulated fit.

The **protected baseline artifacts carry no `hand_fit_*` keys at all** — verified
on both episodes checked. So every experiment run off a baseline silently falls
back to landmark poses, while production uses the articulated fit.

That is not a harness bug; it is a gap between the reproducible-baseline
machinery and the shipped path, and it affects the single field that defines the
IK target. Every profile comparison in this notebook that ran off a baseline has
therefore been comparing landmark-pose pipelines, while the product ships
hand_fit poses. Either baselines must carry the overlay, or profile comparisons
must be stated as landmark-pose-only and never generalised to production.


## E13 — the first profile comparison measured on what production actually uses

Every comparison before this one ran off `intermediates/baselines/<profile>/cln.npz`,
which never receives the `hand_fit_*` overlay (see the E11 retraction: `protect_baseline`
is write-once and runs before `install_articulated_overlay`). Retarget is configured with
`pose_source: "hand_fit"`, so production follows the articulated fit, not the landmarks.

These artifacts are the **active** `cln.npz` snapshotted after each `prepare_episode`,
so they carry the overlay. `prepare_episode` is sufficient and cheap (~35 s): triangulation
runs there, and all five profiles share detector settings, so MediaPipe need not re-run.

### Metric definitions (`viki/perception/articulated.py`)

| metric | definition | gate |
|---|---|---|
| `anchor_joint_fraction` | share of (frame, joint) slots with `confidence > 0`, i.e. a genuinely triangulated landmark for the model to anchor to | — |
| `supported_frame_fraction` | share of frames with **≥ 8 reliable joints and ≥ 4 reliable palm joints**. Unsupported frames are not solved — their pose is filled from neighbouring supported frames | — |
| `anchor_residual_median/p95_mm` | ‖fitted − observed‖ over anchored slots only: how far the rigid model lands from the measurement it was fitted to | median ≤ 20 mm, p95 ≤ 100 mm |
| `bone_length_cv_max` | largest coefficient of variation of any fitted bone length over the episode | < 1e-3 |
| `wrist_preservation_max_mm` | largest wrist displacement between source and fit | < 0.01 mm |
| `palm_outlier_frames` | frames where a palm edge deviates >20% from its reference length | must be empty |
| `source/fitted_joint_jerk_rms_mm` | RMS third difference before and after the fit | fitted ≤ max(1.5×source, source + 0.75) |

`anchor_residual` is internal consistency — model against its own input — not ground truth.
Its value is that the model is **rigid**: bone lengths are held to a CV of ~1e-5 and the
wrist to 0.00 mm. Observations that are geometrically impossible for a human hand cannot be
absorbed by it and must surface as residual.

### Results

| scene | profile | anchor % | **supp %** | resMed | **resP95** | boneCV | srcJerk | fitJerk | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| move-shipok | v2 | 72.3 | 50.2 | 9.87 | 37.39 | 8.4e-06 | 1.17 | 1.16 | ok |
| | v2.1 | 72.3 | 50.2 | 9.87 | 37.39 | 8.4e-06 | 1.17 | 1.16 | ok |
| | v3 | 72.3 | 50.2 | 9.85 | 37.23 | 8.4e-06 | 1.17 | 1.16 | ok |
| | 8px | 91.1 | **89.6** | 11.84 | 27.15 | 1.1e-05 | 1.51 | 1.29 | ok |
| | 16px | 95.7 | **97.0** | 11.74 | 26.30 | 7.1e-06 | 1.71 | 1.32 | ok |
| cup_grab | v2 / v2.1 / v3 | 50.3 | 30.3 | 8.53 | 33.76 | 1.1e-05 | 1.99 | 1.48 | ok |
| | 8px | 68.5 | **60.2** | 8.16 | 20.44 | 1.1e-05 | 2.62 | 1.97 | ok |
| | 16px | 81.7 | **80.2** | 10.32 | 24.73 | 1.1e-05 | 3.48 | 2.15 | ok |
| block-push | v2 / v2.1 / v3 | 75.7 | 43.6 | 7.10 | 33.72 | 1.0e-05 | 2.02 | 1.56 | ok |
| | 8px | 93.5 | **91.6** | 7.77 | 17.73 | 9.9e-06 | 2.43 | 1.98 | ok |
| | 16px | 99.4 | **100.0** | 7.96 | 17.47 | 1.1e-05 | 2.59 | 2.17 | ok |
| pyramid | v2 / v2.1 / v3 | 74.3 | 49.4 | 8.25 | 19.37 | 1.3e-05 | 2.22 | 1.57 | ok |
| | 8px | 90.0 | **88.4** | 7.92 | 16.51 | 1.2e-05 | 2.71 | 2.08 | ok |
| | 16px | 95.4 | **95.0** | 8.14 | 16.70 | 1.0e-05 | 2.94 | 2.15 | ok |
| pick_up_u | v2 / v2.1 / v3 | 74.1 | 36.0 | 8.47 | 25.79 | 1.0e-05 | 2.02 | 1.39 | ok |
| | 8px | 89.1 | **80.8** | 7.53 | 18.92 | 1.1e-05 | 2.56 | 1.88 | ok |
| | 16px | 92.7 | **93.5** | 8.62 | 19.55 | 1.1e-05 | 2.95 | 2.10 | ok |

`wrist_preservation_max_mm = 0.00` and `palm_outlier_frames = []` on all 25 runs; every
quality gate accepted.

### Findings

**1. v2, v2.1 and v3 are the same artifact.** `hand_fit_positions` are byte-identical
between v2 and v2.1, and differ from v3 by 0.0000 mm (rotations by at most 7.9° on one
scene, ≤1.6° elsewhere). The three differ only in `omega`, and `omega` is not an input to
the fit — it reaches only the IK data term. **Everything E8 and E9 changed is invisible in
the artifact that goes downstream.** v2.1 as built is indistinguishable from the default at
the output of extract.

**2. The reprojection gate roughly doubles the frames that are genuinely solved.**
`supported_frame_fraction` rises **+30 to +58 points**: 50→97, 30→80, 44→100, 49→95, 36→94.
On the default profile, **half to two thirds of every episode reaches the robot as a pose
filled from neighbouring frames rather than solved from measurements.** That is the single
most consequential number in this notebook, and no earlier experiment reported it because
none of them looked at this artifact.

**3. The added observations are anatomically admissible, and the rigid model says so.**
`anchor_residual_p95` **falls** on every scene: 37.4→26.3, 33.8→20.4, 33.7→17.5, 25.8→18.9,
19.4→16.5. Bone CV stays at ~1e-5 and the palm-outlier list stays empty. If the widened gate
were admitting geometric garbage, a model that holds bone lengths to five decimal places
could not absorb it — the tail residual would rise. It falls. This is the strongest evidence
produced so far that the extra coverage is real, and it is not my judgement but the model's.

**4. 8 px is better on agreement, 16 px on coverage.** Median residual improves under 8 px on
three scenes (−0.37, −0.34, −0.94 mm) and degrades under 16 px on two of those (+1.79, +0.14).
16 px wins `supported_frame_fraction` everywhere. cup_grab prefers 8 px on both residual
measures, consistent with E6 singling it out for the worst camera geometry.

**5. Two E7 gates were measuring quantities production discards.** Gate 4 rejected profiles
on raw-landmark bone dispersion of 6–12%; the fit drives bone CV to ~1e-5 regardless. Gate 3
measured raw-landmark jitter; `fitJerk` shows the fit absorbing most of the increase
(source 1.17→1.71 while fitted only 1.16→1.32). E7's rejection of both candidates rested
substantially on metrics that never reach the robot.

### What is still not established

Residual is model-vs-observation, not accuracy. Nothing here measures whether the fitted hand
is where the real hand was. And the retarget grid must be redone on these artifacts, with the
configured UR10, before any promotion — the E11 grid used baselines and therefore landmark
poses.
## Decision — the adapter is 11 mm; the 2026-09-08 plans are not a reference

The four plans on disk from 2026-09-08 record `adapter.translation_m = [0, 0, 0.05]`.
Every default in the codebase says 11 mm: `config.py`, both configuration JSONs, the
`/api/pipeline` route default (`adapter_z_mm: float = 11.0`) and the Retarget tab's form
(`S.adapterTranslationMm || [0, 0, 11]`). `data/user_configuration.json` is clean and holds
11 mm, so the 50 mm was typed into the UI by hand for that session and survived only inside
the plans it produced.

**Confirmed with the operator: the physical adapter is 11 mm.**

Consequences:

- The 50 mm plans placed the TCP 39 mm off along the tool axis. **cup_grab's
  33.8 mm / 33.2° is not a valid benchmark** and must not be quoted as "it used to be
  better"; it was measured against the wrong tool geometry. The same applies to the other
  three historical plans, which additionally used the mirrored `+0.7` base.
- There is therefore **no historical reference for retarget quality**. The E14 grid — ur10,
  base −0.7, adapter 11 mm, `pose_source: hand_fit` — is the first correctly configured
  measurement and becomes the reference itself.
- No code or configuration change is needed: 11 mm is already what every default and the
  running grid use.

What remains unexplained is not the adapter. Between 2026-09-08 and now the perception
generation changed twice — the handedness gate was removed (`d7466a0`) and fused filling
became linear (`0d51a50`) — so the landmarks, and hence `hand_fit`, differ. Any comparison
across that boundary confounds tool geometry with perception, which is why the reference is
being re-established rather than repaired.

## E14 — the profile comparison, correctly configured at last

First measurement with every known confound removed: inputs are the snapshotted
**active** `cln.npz`, so `pose_source: "hand_fit"` resolves to the articulated
fit that production follows (E13); the robot is the configured `ur10` rather
than a hardcoded `ur3` (E11 retraction); base −0.7, adapter 11 mm.

`supp%` is `geometry_support_mask`: frames whose hand pose was **solved** from
≥8 reliable joints including ≥4 palm joints, rather than filled from neighbours.
`err|supported` is the tracking error restricted to those frames — the only
column that compares like with like, since a recipe that fabricates more has a
smoother target and flatters the headline RMSE.

| scene | profile | supp% | posRMSE | p95 | ori | **err\|supported** | err\|filled |
|---|---|---:|---:|---:|---:|---:|---:|
| cup_grab | v2 | 30.3 | 64.1 | 138.5 | 37.5° | **46.0** | 70.6 |
| | 8 px | 60.2 | 32.5 | 67.7 | 32.4° | **23.5** | 42.7 |
| | 16 px | 80.2 | 23.8 | 46.6 | 26.8° | **21.9** | 30.3 |
| move-shipok | v2 | 50.2 | 47.4 | 110.7 | 25.3° | **50.0** | 44.6 |
| | 8 px | 89.6 | 33.9 | 82.1 | 22.8° | **33.3** | 38.3 |
| | 16 px | 97.0 | 19.3 | 57.8 | 22.2° | **16.0** | 63.5 |
| block-push | v2 | 43.6 | 63.2 | 113.1 | 51.8° | **73.8** | 53.6 |
| | 8 px | 91.6 | 122.8 | 171.0 | 48.1° | **119.2** | 156.9 |
| | 16 px | 100.0 | 99.6 | 139.2 | 45.3° | **99.6** | — |

### What holds

**On the two scenes where the table is not an active constraint, the widened
gate halves to thirds the error on evidence-backed frames.** cup_grab
46.0 → 23.5 → 21.9 mm; move-shipok 50.0 → 33.3 → 16.0 mm. Monotone in the gate
width, on the frames that carry real measurements.

**Orientation improves monotonically on all three scenes**, block-push included:
37.5 → 32.4 → 26.8, 25.3 → 22.8 → 22.2, 51.8 → 48.1 → 45.3. Orientation is not
blocked by the floor the way height is, so this is the one column free of the
block-push confound, and it agrees with the coverage ordering everywhere.

**The two widened profiles converge on each other and both depart from the
default.** Achieved-trajectory divergence: 16 px vs 8 px is 2.0 mm (move-shipok)
and 6.1 mm (cup_grab) median, while each sits 14–31 mm from v2. Two different
thresholds, two different triangulations, two independent articulated fits, one
trajectory — and the default somewhere else. Agreement between independent
methods is evidence that no single-recipe metric can supply.

**The default tracks its fabricated frames better than its measured ones.** On
move-shipok v2 scores 50.0 mm on supported frames against 44.6 on filled: the
invented frames are smoother and easier to follow. At 16 px the relation
inverts hard, 16.0 against 63.5, which is the healthy shape — accurate where
there is evidence, poor where there is none. That inversion is also why the
headline RMSE must never be read alone.

### What reverses, and why it is not perception

**block-push gets worse with better perception**, 73.8 → 119.2 → 99.6 mm on
supported frames. Diagnosed rather than accepted: 68–73% of the squared error
there is vertical, and the robot sits systematically *above* the target — median
+48 mm under v2, +103.6 mm under 8 px.

The floor constraint applies to the lowest point of every moving collision
geometry, not to the tool point. A Robotiq 2F-85 interposes its own body between
the flange and the grasp centre, so keeping the assembly above the plane holds
the tool point high. Targets sit a median 36 mm above the table; the achieved
tool point is held at 87–145 mm. The floor margin is ≈0 throughout: the arm
rides the constraint for the whole episode.

The human hand *is* the gripper; the robot's is not. **block-push is not
executable by this end-effector at this target definition**, and honest
perception makes that more binding rather than less, because interpolation had
been smoothing the hand's lowest excursions upward. This is a task-definition
problem, not a perception one, and it is now recorded as an assumption in the
thesis.

### Floor tolerance

The re-run also validates the tolerance change. Penetrations are 36 µm (8 px)
and 31 µm (16 px) against the new 0.5 mm bound, and v2 clears by
5.1 × 10⁻⁷ mm — half a micron. Under the old 10⁻⁷ m bound, whether an 897-frame
episode survived was decided by the last bits of the solve. The fix did not mask
a real penetration: at 14 mm calibration accuracy, 31 µm is nothing.

### Status

No promotion. Position error is still measured against each profile's own
target, and for 8 px and 16 px that target differs from v2's, so part of the
gain could be an easier target rather than a better one — the convergence
argument and the `err|supported` split narrow that, but do not close it. What is
established is the ordering, its consistency across scenes and metrics, and that
the one reversal has a diagnosed non-perception cause.

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

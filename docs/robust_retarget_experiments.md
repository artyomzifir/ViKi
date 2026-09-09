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

**Result: recorded below once the run completes.**

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

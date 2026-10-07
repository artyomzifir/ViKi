# Repository transfer audit for object-aware retargeting

Date: 2026-09-20

## Purpose

This document records the repository and paper review performed after the
first production object-perception work in ViKi.  Its purpose is to preserve
the reasoning about what can be reused directly, what should be independently
reimplemented, and what should not be brought into the project.

The immediate problem is not generic policy learning.  ViKi already produces:

- calibrated multi-view RGB-D clouds;
- SAM 2.1 instance tracks;
- compact rigid object core/shell models;
- per-frame object position, orientation, confidence, residual, coverage and
  operator-proximity contact estimates;
- a whole-trajectory robot IK solution with floor and self-collision terms;
- a conservative upper-workspace retarget branch.

What is still missing is the connection between those two branches.  Retarget
does not yet consume the object artifact.  The provisional top-side
hand-to-tool mapping is useful for pick-and-place, but it behaves as a fixed
orientation prior and therefore cannot represent pouring or a carry that
deliberately rotates the object.

The main conclusion of this audit is that ViKi should **not import another
project's complete pipeline**.  The useful result is a combination of four
narrow ideas:

1. phase and object/reference relations from ORION and OKAMI;
2. the tool/object-to-EEF transform from Tool-as-Interface;
3. contact-onset and multiple-grasp hypotheses from HOWTransfer;
4. explicit tracking-quality gates from DexWild and RoboPaint.

These ideas should be implemented against ViKi's existing metric multi-view
artifacts rather than against the original projects' single-view, hardware- or
dataset-specific representations.

## Reproducibility of the audit

The following public repository snapshots were inspected locally.  Commit
identifiers make this document independent of later upstream changes.

| project | repository | audited commit | last commit date in snapshot |
|---|---|---:|---:|
| EgoMimic | <https://github.com/SimarKareer/EgoMimic> | `6d63e9a` | 2024-11-10 |
| EgoMimic Eve | <https://github.com/SimarKareer/EgoMimic-Eve> | `b0e45c3` | 2025-05-04 |
| DexWild | <https://github.com/dexwild/dexwild> | `5fa34af` | 2025-08-13 |
| DexWild training | <https://github.com/dexwild/dexwild-training> | `efc1643` | 2025-06-14 |
| Tool-as-Interface | <https://github.com/Tool-as-Interface/Tool_as_Interface> | `ec1c459` | 2025-04-03 |
| OKAMI | <https://github.com/UT-Austin-RPL/OKAMI> | `de4edba` | 2025-06-18 |
| DexCanvas loader | <https://github.com/DexRobot/dexcanvas> | `e9e7d08` | 2025-10-21 |
| RoboPaint/PX toolkit | <https://github.com/px-DataCollection/px_omnisharing_dataprocess_kit> | `99dd06e` | 2026-04-22 |

No ViKi implementation was changed as part of the audit.

## Paper timeline and publication status

The distinction below is between the first public preprint and an archival
conference or journal publication.  It is intentionally explicit because the
paper date, conference year and proceedings year differ for some works.

| work | first preprint | archival status as of 2026-09-20 | public implementation |
|---|---:|---|---|
| [ORION](https://arxiv.org/abs/2405.20321) | 2024 | Journal article in *Autonomous Robots*, published 2026: <https://doi.org/10.1007/s10514-026-10253-8> | No official implementation found on the project page. |
| [OKAMI](https://arxiv.org/abs/2410.11792) | 2024 | CoRL 2024; the archival [PMLR volume 270](https://proceedings.mlr.press/v270/li25a.html) is dated 2025. | Large public repository, but no root licence. |
| [EgoMimic](https://arxiv.org/abs/2410.24221) | 2024 | ICRA 2025, DOI [10.1109/ICRA55743.2025.11127989](https://doi.org/10.1109/ICRA55743.2025.11127989). | Processing, training, sample data and separate hardware repository. MIT. |
| [DexWild](https://arxiv.org/abs/2505.07813) | 2025 | [RSS 2025](https://www.roboticsproceedings.org/rss21/p075.html). | Capture, processing, deployment and separate training repository. MIT. |
| [Tool-as-Interface](https://arxiv.org/abs/2504.04612) | 2025 | [CoRL 2025, PMLR 305](https://proceedings.mlr.press/v305/chen25d.html). | Human-play processing, training and UR5 evaluation. MIT root repository; dependencies have separate licences. |
| [DexCanvas](https://arxiv.org/abs/2510.15786) | 2025 | The public manuscript says it was under review at ICLR 2026. It does not appear under this title in the final ICLR 2026 paper list, so it must still be treated as a preprint. | MIT dataset loader and examples, not the full reconstruction/force pipeline. |
| [Vi-RoIL](https://doi.org/10.1145/3788731.3788736) | 2025 conference work | EILM 2025 proceedings, pages 30--38; the bibliographic record appeared in 2026. | No official repository found. |
| [RoboPaint](https://arxiv.org/abs/2602.05325) | 2026 | arXiv preprint only. | Public documentation/sample toolkit; core runtime is a proprietary binary distribution. |
| [HOWTransfer](https://arxiv.org/abs/2606.10743) | 2026 | arXiv preprint only. | No official repository found. |

## Licence consequences

The paper licence and the code licence are separate.  A permissively licensed
paper does not make an unlicensed repository reusable, and a restrictive paper
licence does not override an MIT repository licence.

### Code that can legally be reused with attribution

The root repositories for EgoMimic, EgoMimic Eve, DexWild, DexWild training,
Tool-as-Interface and the DexCanvas loader use MIT licences.  Small portions can
be incorporated into an Apache-2.0 project if the relevant copyright and MIT
licence notices are preserved.

In practice, most high-value portions are short transforms or design patterns.
Independent ViKi-native implementations are preferable because they avoid
large dependency trees and make frame conventions and quality gates explicit.

### Code that must not be copied

The OKAMI repository has no top-level licence.  Its visibility on GitHub does
not grant redistribution or modification rights.  Its algorithms may be
studied and independently reimplemented from the paper, but its source should
not be copied into ViKi.

RoboPaint's PX toolkit has mixed terms:

- documentation and any exposed Python source are MIT;
- sample datasets are CC-BY-NC-SA 4.0;
- the wheel, model weights and runtime components are proprietary and limited
  to non-commercial research/education unless separately licensed;
- the source of the binary runtime is explicitly not public.

Consequently RoboPaint is a source of schema and validation ideas, not a
production dependency for ViKi.

Tool-as-Interface's root repository is MIT, but its FoundationPose submodule is
covered by the separate NVIDIA Source Code License.  The root MIT file does not
relicense that dependency.  FoundationPose can be evaluated as an optional
research baseline, but it should not silently enter ViKi's distributable core.

## Findings by repository

## ORION and OKAMI

### Useful ideas

ORION represents a demonstration as a sequence of Open-World Object Graphs.
Object nodes describe task-relevant objects, point nodes describe tracked
features belonging to an object, and edges record hand--object and
object--object contact.  The plan is a sequence of object-centric keyframes
rather than one undifferentiated wrist trajectory.

The public OKAMI repository contains a closely related implementation in:

- `okami/plan_generation/algos/human_video_plan.py`;
- `okami/plan_generation/algos/hoig.py`;
- `okami/plan_generation/algos/tap_segmentation.py`;
- `okami/oar/algos/trajectory_generator.py`.

The implementation provides several valuable concepts:

1. split a demonstration into `reach`, `manipulation` and `release` segments;
2. identify the manipulated object from object keypoint velocity;
3. identify a reference object from changing contact relations;
4. store the desired hand--object offset for reach;
5. store the desired manipulated--reference-object offset for manipulation;
6. warp a reference trajectory between new start and end poses;
7. interpolate endpoint orientation corrections with quaternion SLERP.

This is substantially closer to ViKi's actual need than fixing EEF axes.  It
allows an orientation prior to apply during approach while the carried-object
motion remains meaningful during manipulation.

### What should not be ported literally

The code uses several assumptions that do not belong in ViKi:

- per-object motion thresholds are in pixels rather than metric, timestamped
  velocity;
- object contact is decided by a fixed point-count threshold and a 5 cm
  distance threshold;
- object centres are arithmetic means of partial single-view clouds;
- orientation observability and symmetry are not represented;
- VLM calls resolve ambiguous manipulated/reference objects;
- trajectory rotation and scale warping contains singular cases for parallel,
  opposite or zero-length displacement vectors;
- outlier thresholds are fixed per frame and therefore rate-dependent;
- the implementation is tied to SMPL-H, GR-1 and two simulation environments.

### ViKi-native form

The phase/object graph idea should be rebuilt on top of:

- metric `T_world_object[t]` tracks;
- object-track confidence, residual and coverage;
- operator proximity contact with temporal hysteresis;
- hand position and orientation validity;
- timestamp-normalised object and hand twists;
- a per-axis orientation observability mask;
- explicit `unknown` states rather than forced VLM guesses.

The graph need not be a general graph library.  For the first implementation a
small durable artifact containing phase intervals, manipulated object id,
optional reference object id, contact confidence and the recorded relative
transforms is sufficient.

## Tool-as-Interface

### The central transform

The most directly reusable idea is the conversion from a predicted tool pose
to the required end-effector pose.  The implementation in
`ti/common/data_utils.py::policy_action_to_env_action` computes the equivalent
of:

\[
T^B_E(t) = T^B_T(t) \left(T^E_T\right)^{-1}.
\]

For ViKi, with an object observed in the world, a human hand frame and the
robot adapter, the corresponding form is:

\[
T^W_E(t) = T^W_O(t)\,T^O_H\,T^H_E.
\]

This is the crucial correction to the fixed-axis prototype.  During reliable
contact, the robot preserves the demonstrated grasp transform relative to the
object.  A pouring rotation of the object therefore rotates the EEF.  During
approach, before the relative transform is valid, the conservative upper-side
prior can still select a safe IK branch.

### Other useful components

`ti/common/pose_trajectory_interpolator.py` provides translation interpolation,
quaternion SLERP and scheduling under independent maximum positional and
rotational speeds.  It is relevant to the future replay/controller layer, not
to the current whole-trajectory retarget objective.  Adding it inside retarget
would duplicate optimisation and obscure the original timestamps.

The repository also demonstrates a useful perception chain:

1. text-conditioned GroundingDINO box;
2. SAM-HQ mask;
3. FoundationPose registration in the first RGB-D frame;
4. FoundationPose tracking in later frames.

This could serve as a comparison baseline for object orientation when a mesh is
available.  It is not a drop-in replacement for ViKi's unknown-object model.

### Reasons not to copy the preprocessing pipeline

- only the first view is used in the inspected human-play pose path;
- a mesh is required;
- depth cleanup is performed by slow per-pixel neighbourhood replacement;
- absent calibration falls back to `np.ones((4, 4))`, which is not an identity
  transform and can silently corrupt every pose;
- most paths and examples are tied to a UR5 deployment;
- FoundationPose and bundled third-party components require their own licence
  audit.

## HOWTransfer

HOWTransfer is useful conceptually despite having no public implementation.
It avoids committing immediately to one gripper pose:

1. recover a temporally consistent hand trajectory;
2. localise contact intervals and especially contact onset;
3. convert grasp intent into several parallel-jaw grasp hypotheses;
4. propagate each hypothesis along the wrist trajectory;
5. edit and score the resulting robot-executable trajectories.

This directly addresses ViKi's current failure mode.  Instead of one global
hand-to-tool rotation, ViKi can generate candidates such as:

- top pinch;
- side pinch;
- an orientation that preserves the observed hand--object transform;
- symmetry-equivalent object-frame orientations.

Candidates can then be compared using the same solver and objective terms:

- pose residual;
- joint limits and continuity;
- floor/table clearance;
- self- and scene-collision margin;
- deviation from the recorded relative grasp;
- object-track confidence and orientation observability.

The result is not a learned force controller.  It is a better discrete search
over kinematically and semantically meaningful grasp frames.

## DexWild

### Useful patterns

DexWild exposes its capture, preprocessing, deployment and policy-training
stacks.  The most useful parts for ViKi are quality-control patterns:

- record a per-episode tracking-loss fraction;
- reject an episode above a tracking-loss threshold;
- interpolate bounded lost blocks with linear translation and quaternion
  SLERP;
- convert absolute poses to relative actions before measuring outliers;
- derive dataset-level translation and rotation thresholds;
- report how much of every episode had to be clipped;
- quarantine episodes for which too much data was repaired;
- use current-frame-relative action chunks and 6D rotation representations
  for policy training.

These patterns are more valuable than the specific filters.  ViKi already has
better access to observation validity, timestamps and solver residuals, so its
quality report can be more explicit.

### Unsafe implementation details

The inspected code also contains shortcuts that must not propagate:

- a hard-coded 0.46 m left/right end-effector frame shift;
- a coordinate stream named `world` that is derived relative to the first
  valid tracker pose;
- nearest-timestamp matching without a maximum allowed time error;
- per-frame hard caps that change meaning with recording frequency;
- quaternion rotation clipping based directly on `qw`, without first making
  `q` and `-q` equivalent;
- hardware- and hand-specific retarget constants embedded in processing code.

The desired ViKi adaptation is therefore a **quality report and quarantine
policy**, not a port of DexWild's trajectory mutation functions.

## EgoMimic

EgoMimic is the most complete public cross-embodiment training release in this
set.  It provides Aria processing, ALOHA processing, policy code, configs,
sample datasets and a separate Eve hardware repository.

Important design patterns are:

- represent a future action chunk in the current observation-camera frame;
- treat human and robot data as separate modalities of one shared model;
- use modality-specific state projections and action heads around a common
  transformer;
- keep separate normalisation statistics for human and robot datasets;
- mask the human hand and robot arm to reduce the appearance gap;
- provide a matched visual cue showing the action-bearing region.

These patterns belong to a later ViKi policy-learning/export milestone.  They
do not improve the current geometric retarget by themselves.  Importing the
ACT/Robomimic/PyTorch Lightning stack now would increase complexity without
fixing object-relative pose transfer.

An eventual ViKi exporter could expose both:

- workspace/robot-base action chunks;
- current-observation-frame or current-EEF-frame action chunks;

while retaining the raw absolute poses needed to audit either representation.

## RoboPaint / PX Pipeline

The public repository is mainly documentation, sample HDF5 data, container
entry scripts and large sample assets.  The processing implementation is
inside a proprietary Docker image and wheel, so claims in the paper cannot be
verified line by line from public source.

The useful public design is its explicit staged schema:

```text
DF-1 raw capture
  -> DF-2 pose/tactile-enriched human data
  -> DF-2R retargeted hand data
  -> DF-3 LeRobot dataset
```

Additional useful practices are:

- retain observations and actions separately;
- make object pose optional but explicit;
- validate pose discontinuities and bad tactile streams before retargeting;
- collision-check finger--object penetration after retargeting;
- discard an episode when repair cannot restore it safely;
- store error/validation information as dataset attributes;
- use different RGB-D views for hand and object pose estimation when one view
  is heavily occluded.

ViKi already follows the durable-stage-artifact principle.  The remaining
lesson is to make phase/contact/object-retarget validity just as explicit in
`status.json`, the export manifest and future replay eligibility.

## DexCanvas

The public DexCanvas repository is a dataset loader, visualiser and training
example collection.  It is not the full real-to-sim reconstruction or contact
force recovery system described by the paper.

Useful schema elements are:

- explicit active manipulation intervals;
- trajectory quality ratings;
- operator, object and manipulation-type metadata;
- object 6D pose beside hand/MANO state;
- attention masks for variable-length episodes;
- inexpensive loading of metadata and pose fields without loading dense hand
  meshes.

These are applicable to ViKi labelling and export.  The data may also be useful
for offline experiments on object-relative representations, but it is not a
ground-truth benchmark for ViKi's multi-view RGB-D reconstruction.

## Vi-RoIL

No official source repository was found.  The paper is a published conference
work, but it currently contributes no auditable implementation that ViKi can
reuse.  It should remain a related-work citation unless source or a sufficiently
detailed project page appears.

## Proposed ViKi design

## 1. Phase artifact

Add an experimental, non-production phase/object-relation artifact derived
from `cln.npz` and `object_models.npz`.  The initial state set should be:

```text
unknown
approach
contact_candidate
manipulation
release
```

Each interval should store:

- start/end frame and timestamps;
- manipulated object id;
- optional reference object id;
- hand and object observation validity;
- contact confidence and evidence;
- object pose confidence/residual/coverage statistics;
- orientation observability mask;
- reason for every state transition.

Transitions should use hysteresis and minimum duration.  A single proximity
frame must not establish a grasp, and a single missing frame must not end one.

## 2. Robust grasp transform

At confirmed contact onset, estimate the object-relative hand transform over a
short window:

\[
T^O_H(t) = \left(T^W_O(t)\right)^{-1} T^W_H(t).
\]

Do not take one arbitrary frame.  Estimate translation robustly and average
rotation on SO(3), weighted by:

- hand orientation validity;
- object-track confidence;
- model coverage and residual;
- contact confidence;
- per-axis object orientation observability.

The window's residual is itself a rigidity/quality metric.  A large spread
means the grasp frame is not trustworthy and object-relative retarget should
remain disabled for that phase.

## 3. Per-DOF observability

A scalar object confidence is insufficient.  A partial cuboid, cup or
rotationally symmetric object may constrain translation while leaving one or
more rotation axes ambiguous.

The object artifact should eventually expose an information/observability
measure for each rotational degree of freedom.  Until a principled information
matrix is available, a conservative categorical mask is acceptable:

- full SE(3);
- position plus table yaw;
- position only;
- unusable.

Only observed DOFs should enter the retarget target.  Unobserved DOFs should be
continued from the recorded hand trajectory or the existing safe prior, not
filled with a noisy PCA/ICP orientation.

## 4. Phase-dependent target construction

### Approach

Use the current upper-workspace reference policy and top/side grasp candidate
selection.  Object position can move the approach endpoint, but object
orientation must only participate in observable axes.

### Manipulation

When contact and pose confidence are sufficient, construct the EEF target from
the moving object and the recorded grasp:

\[
T^W_E(t) = T^W_O(t)\,\widehat{T^O_H}\,T^H_E.
\]

This preserves meaningful carry rotation and pouring.  It does not freeze EEF
axes in the world.

### Release

Blend out the object-relative factor over a bounded interval and return to the
workspace trajectory.  The release must not cause an orientation jump when the
object track stops or becomes occluded.

### Low confidence

Fall back smoothly to the workspace target.  Record the fallback; do not claim
the resulting phase was object-relative.

## 5. Multiple grasp hypotheses

Generate a small candidate set rather than a single hand-to-tool rotation.
Candidate sources should include:

- demonstrated relative grasp;
- top-side parallel-jaw grasp;
- side grasp;
- symmetry-equivalent rotations of the object model;
- previous-frame continuation.

Run the same short-horizon or whole-trajectory evaluation for every candidate.
The candidate score should contain:

- observed position and orientation residual;
- change from the previous candidate/pose;
- joint-limit and reference posture cost;
- minimum self-collision margin;
- minimum scene/object/floor clearance;
- relative-grasp drift during confirmed contact;
- use of unobservable object rotation;
- number and duration of confidence fallbacks.

This allows a top grasp to win for pick-and-place without making it the only
orientation available to a pouring task.

## 6. Time and smoothing policy

All thresholds must be stated in metric units and seconds, not pixels or
frames.  Translation and rotation must be treated separately:

- m/s and m/s² for translation;
- rad/s and rad/s² for rotation;
- SLERP or Lie-group interpolation for missing orientations;
- maximum admissible gap duration;
- no interpolation across an unknown phase or a contact transition without an
  explicit rule.

The original frame timestamps remain authoritative.  Replay may later impose
speed and acceleration limits, but retarget should not silently resample away
the recorded timing.

## Validation protocol

Existing full-rate scenes are useful for regression but do not establish
transfer correctness:

- `move-shipok` is the best current grasp/carry candidate;
- `pyramid` exposes sparse support and orientation ambiguity;
- `cup_grab` exposes the failure of a fixed top-side rotation for a task where
  object orientation matters;
- `block-push` tests a non-carry manipulation and should not be forced into a
  rigid grasp relation.

The next decisive recording set remains:

1. one task and one rigid, visually distinctive object;
2. 5--6 takes without changing camera calibration;
3. translated object placements;
4. several deliberate yaw rotations, including approximately 90 degrees;
5. at least one task with deliberate carried-object rotation or pouring;
6. if possible, an independently measured object pose for a subset of frames.

Measurements should include:

- spread of `T_object_hand` across repeated takes;
- workspace versus object-relative endpoint and path consistency;
- orientation error separately on observable and unobservable axes;
- grasp-transform drift during manipulation;
- phase transition timing and contact precision/recall on manually inspected
  intervals;
- fallback fraction and longest fallback duration;
- position/orientation IK residual;
- minimum floor, self-collision and scene-cloud clearance;
- solve time and iteration count;
- number of frames whose chosen grasp hypothesis changed.

All millimetre and degree values must be labelled as self-consistency,
tracking, solver or ground-truth error.  These quantities are not
interchangeable.

## Priorities

### P0: next experimental branch

1. Produce confidence-gated phase intervals from existing hand/object tracks.
2. Estimate and report `T_object_hand` stability without changing IK.
3. Add per-axis orientation observability or an explicit conservative proxy.
4. Generate object-relative EEF targets offline and visualise them beside the
   workspace targets.
5. Evaluate multiple grasp-frame candidates using existing retarget metrics.
6. Only then add an opt-in retarget factor.

### P1: robustness and operational semantics

1. Reference-object relations for placement/pouring targets.
2. Scene-cloud collision terms excluding the operator and the manipulated
   object according to phase.
3. Replay-time translation/rotation speed and acceleration scheduling.
4. Dataset eligibility rules based on repaired/fallback fractions.

### P2: policy learning

1. Export current-frame-relative future action chunks.
2. Add active manipulation ranges and quality ratings.
3. Co-train human and robot modalities using EgoMimic/DexWild-style separate
   heads and normalisation.
4. Evaluate embodiment masking and action-region visual cues.

## Explicit non-goals

The first object-aware retarget implementation should not:

- enable a full object-relative constraint unconditionally;
- assume every SAM instance has a well-defined 6D orientation;
- infer grip force from proximity;
- depend on a VLM to make a safety-critical phase decision;
- import ROS2, Robomimic, FoundationPose or a proprietary runtime into the
  normal ViKi pipeline;
- replace ViKi calibration or cloud fusion with single-view code from another
  project;
- fix EEF axes for the entire trajectory;
- call solver convergence proof of physical task correctness.

## Decision

The next object-centric experiment should be **phase-dependent and
confidence-gated**:

- safe upper-workspace priors before contact;
- recorded object-relative grasp during reliable manipulation;
- observable object rotation only;
- multiple grasp hypotheses rather than one permanent axis fix;
- smooth, recorded fallback when perception becomes unreliable.

This design preserves the part of the current retarget that improved
pick-and-place while removing the assumption that every task should keep the
same world-frame tool orientation.

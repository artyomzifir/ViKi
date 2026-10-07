# SAM + RGB-D segmentation plan — 2026-09-14

## Goal

Produce a confidence-bearing scene decomposition for object-relative retargeting
and environment avoidance:

- `operator` — removed from replay obstacles because the robot replaces them;
- `manipulated_object` — one or more tracked rigid or partially rigid task
  objects, with the active instance allowed to change by task phase;
- `other_object` — retained as an obstacle;
- `other_dynamic` — people or moving bodies other than the demonstrated
  operator, retained as obstacles;
- `static_environment` — retained as an obstacle;
- `ambiguous_contact` / `unknown` — never silently treated as free space.

This began as a design and experiment plan.  A first non-production probe now
lives in `viki/perception/segmentation.py`: prompted SAM 2.1 masks and a
calibrated RGB-D lift under `intermediates/segmentation/`.  Retarget does not
consume the result yet; the remaining sections describe the intended path from
that probe to a confidence-bearing object track.

## Decision

Start with **video SAM masks lifted through the recorded RGB-D calibration**, then
perform lightweight 3-D voting and temporal association.  Do not start by
integrating a monolithic 3-D foundation model.

The practical first backend should be **SAM 2.1 Hiera Small** with visual
prompts.  Meta reports 46 M parameters and 84.8 FPS on an A100 (compiled), close
to Tiny's 38.9 M / 91.2 FPS but with better MOSE validation quality; these are
reference numbers, not a prediction for the ViKi host.  The released SAM 2 code
and checkpoints are Apache-2.0 licensed.  SAM 3.1 is a
useful second backend when text/exemplar prompting materially reduces manual
seeding, but it belongs in a separate inference image: its official environment
requires Python 3.12, PyTorch 2.7+, and CUDA 12.6+, while ViKi currently targets
Python 3.10+ and does not otherwise depend on PyTorch.  SAM 3.1 has 848 M
parameters and gated checkpoints under the SAM License, so it should earn that
extra deployment and licensing cost in the comparison.  Meta SAM 3D Objects is a
single-image 3-D reconstruction model, not a point-cloud segmenter, and requires
at least 32 GB of GPU memory.  It may later provide a shape prior, but it should
not be the source of metric pose, scale, or collision geometry.

The 2023 Pointcept `SegmentAnything3D` project contains the relevant algorithmic
idea — project 2-D SAM masks into 3-D and merge overlapping regions — but its
released pipeline targets static ScanNet scenes, Python 3.8-era PyTorch 1.11,
custom CUDA point operations, and a default 50 mm voxel.  ViKi should implement
the small calibrated RGB-D fusion needed for manipulation rather than inherit
that environment wholesale.

## Why the current cloud is not enough

`viki.perception.cloud` already does the exact geometry needed:

1. depth pixel to colour-camera 3-D using the stored Kinect or RealSense
   calibration;
2. colour-camera 3-D to colour pixel;
3. colour-camera 3-D to the calibrated world frame;
4. multi-camera concatenation, crop, voxelisation, and point-budget sampling.

However, `cloud/<frame>.bin` retains only world XYZ and RGB.  It discards the
source camera, source depth pixel, projected colour pixel, and pre-voxel mask
votes.  A 2-D segmentation cannot therefore be attached reliably after this
artifact is written.  The semantic lift must run before multi-view fusion (or
the cloud builder must expose a reusable deprojection primitive).

## Proposed pipeline

### 1. Per-camera video masks

Run one independent video predictor per recorded colour stream, with the same
frame indices and timestamps already used by extraction.

Process cameras sequentially and bound memory by overlapping temporal chunks
(initially 150–300 frames, with the last high-confidence mask re-seeded into the
next chunk).  The official SAM 2 predictor can offload frames/state to CPU, but
its inference state is still episode-oriented; 900 frames resized to its
1024-square working resolution can consume the current 11 GB container budget
before masks and model state.  Chunk boundaries must be included in the ID-
switch/flicker evaluation rather than assumed harmless.

- Operator seed: existing 2-D hand landmarks as positive points, static
  background as negative evidence, and a visual box/mask refinement on one or a
  few clear frames.  With SAM 3, `person`, `arm`, and `hand` concept masks can be
  candidates, but their union still needs geometry checks because recordings
  often show only a partial operator.  Never classify every `person` instance as
  the operator: real ViKi frames contain other people in the background.  Only
  the instance connected to and co-moving with the demonstrated hand is
  removable; other people remain `other_dynamic` obstacles.
- Manipulated-object seed: choose a frame immediately before contact, select the
  foreground component nearest the 3-D pinch centre after excluding the
  operator, then project that component into every camera as a box or mask
  prompt.  Text class alone is insufficient: `cup_grab` visibly contains two
  cups, only one of which is manipulated.  Use an exemplar/click plus proximity
  and later co-motion to retain the correct instance.  The Viewer should permit
  one corrective click/box when this seed is ambiguous.
- Track both directions from a clear seed frame.  Add a new correction prompt
  after an occlusion or identity switch instead of trusting unbounded mask
  propagation.

SAM masks are instance proposals, not final semantic truth.  Store prompt
provenance, prediction score, and whether each frame is propagated or manually
corrected.

Both calibration presets used by the four current evaluation scenes already
contain per-camera empty-scene depth maps.  Those maps provide a strong,
model-independent foreground proposal and a static-environment source; SAM
should refine instance boundaries/identity rather than rediscover all motion
from RGB alone.

### 2. Mask-to-depth lift

For every valid depth pixel `(u_d, v_d, z)`:

1. reuse `color_deproject_maps` to obtain the corresponding colour-camera point;
2. project it with recorded colour intrinsics to `(u_c, v_c)`;
3. sample each 2-D mask at `(u_c, v_c)` (eroded/core and boundary probabilities
   should remain distinguishable);
4. transform the point to the calibrated world frame;
5. carry label probabilities, instance id, camera id, and depth validity into
   voxel fusion.

This is more reliable than mapping a colour prompt to a depth pixel with a fixed
depth hint, especially at object/hand boundaries.

### 3. Multi-view 3-D fusion

Voxelise at approximately 3–5 mm for manipulation objects.  Within each voxel,
combine per-camera class votes using mask score, distance from the 2-D mask
boundary, depth-edge spread, and view support.  Do not resolve disagreement by
an unconditional priority order.

Apply semantic masks before voxel downsampling and point-budget capping.  The
four present evaluation scenes each contain about 897–898 frames at 29.94 Hz
from two cameras.  The Viewer cloud's 40k-point cap is reasonable for display
but must not discard the small manipulated object before pose estimation.
Retain dense masked depth for object fitting, while the environment can use a
coarser occupancy/SDF representation.

Useful confidence signals are:

- number of agreeing cameras and camera bit mask;
- winner/runner-up probability margin;
- signed distance to the nearest 2-D mask boundary;
- static-background agreement;
- temporal label persistence and 3-D component continuity.

Uncertain contact-boundary voxels remain `ambiguous_contact`.  The previous
experiment showed why a geometric hand-capsule subtraction is unsafe: at a
20 mm margin it removed about half of the colour-labelled red object.

### 4. Instance and contact association

Associate components across frames with 3-D overlap, centroid displacement,
visible dimensions, colour/appearance embedding, and mask-track identity.  The
manipulated object is the component that remains near the grasp and develops
co-motion with the hand; gripper opening is supporting evidence, not the sole
contact detector.

Do not assume one manipulated object per episode.  The real `pyramid` scene
contains several separate blocks, so the durable contract needs stable
`object_instance_id` values plus `active_object_id(t)` / phase intervals.  A
composite assembly can be represented by parent/child rigid relations once two
blocks move together.  In `block-push`, the palm covers much of the object's top
surface during contact; keep propagating the pre-contact object model and use
its reprojection/depth support instead of asking a single contact frame to
separate the boundary from scratch.

Build the initial object shell from high-confidence pre-contact frames.  During
tracking, estimate only supported pose degrees of freedom:

- translation from robust depth/point alignment;
- yaw or full rotation only when the registration Hessian and observed shape
  make it identifiable;
- a per-axis observable-DOF mask and covariance, rather than one confidence for
  all SE(3).

Symmetric or partial objects should fall back to translation or tabletop
`(x, y, z, yaw)`.  Generated single-view shape must not manufacture confidence
in an unobserved rotation.

### 5. Downstream safety semantics

- Remove only high-confidence `operator` points from the replay obstacle field.
- Keep `unknown`, `other_object`, `other_dynamic`, and `static_environment`
  occupied, with a conservative dilation for calibration/depth error.
- Keep the manipulated object collision-enabled except at explicit gripper
  contact patches.
- Build a static environment field once from the saved empty-scene background;
  do not make retarget collide against every raw point every solver iteration.
- Feed object pose and observable-DOF confidence into the contact-phase
  object-relative factor.  Unsupported axes retain the recorded hand motion.

## Artifact boundary

A first non-destructive artifact should live beside, not replace, `cloud/` and
`cln.npz`.  Suggested logical contents:

- compressed per-camera masks with frame index, instance id, class, score, and
  prompt provenance;
- fused semantic point labels/confidence aligned to a reproducible cloud
  variant, or concatenated points plus per-frame offsets;
- per-instance object model points and robust dimensions;
- per-instance `T_world_object(t)`, covariance, and observable-DOF mask;
- `active_object_id(t)` and optional assembly parent/child relations;
- contact intervals and their confidence;
- a static environment voxel/SDF artifact with calibration provenance.

Only the compact object track and contact metadata should be copied into
`cln.npz`; the full scene field remains a perception artifact supplied to
retarget explicitly.

## Experiment before production

Compare three variants on `move-shipok`, `pyramid`, `cup_grab`, and
`block-push`:

1. SAM 2.1 video tracker with visual seeds;
2. SAM 3.1 text/exemplar detection plus video tracking, if the available GPU can
   run it;
3. the existing colour/foreground lab proxy as a lower-cost reference, never as
   the production semantic contract.

Hand-label a small stratified set per camera: clear pre-contact, first contact,
carry/rotation, strongest occlusion, and release.  Measure:

- 2-D mask IoU and boundary F score for operator and manipulated object;
- operator leakage into retained obstacles, and non-operator points incorrectly
  removed (the safety-critical pair);
- multi-view 3-D disagreement and reprojection consistency;
- object identity switches and temporal mask/volume flicker;
- static-pose translation/orientation repeatability;
- pose residual and observable-axis stability through contact;
- runtime, peak RAM, and peak VRAM at native frame rate.

Acceptance for retarget should be confidence-gated.  A failed or ambiguous
segmenter must produce `unknown`/`object_pose_unavailable`, leaving the existing
workspace-anchored retarget intact, rather than creating free space or a false
object-relative correction.

## Primary sources consulted

- Meta SAM 2 repository: <https://github.com/facebookresearch/sam2>
- Meta SAM 3 repository: <https://github.com/facebookresearch/sam3>
- Meta SAM 3 research page: <https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/>
- Pointcept SegmentAnything3D: <https://github.com/Pointcept/SegmentAnything3D>
- `SAM3D: Segment Anything in 3D Scenes`: <https://arxiv.org/abs/2306.03908>
- Meta SAM 3D Objects: <https://github.com/facebookresearch/sam-3d-objects>
- `SAM 3D: 3Dfy Anything in Images`: <https://arxiv.org/abs/2511.16624>

# SAM 2.1 RGB-D segmentation probe — 2026-09-14

## Scope

This is an experimental, non-production evaluation of prompted SAM 2.1 video
masks lifted through ViKi's recorded RGB-D calibration.  It does not feed
retargeting.  All millimetre results below are self-consistency measurements;
there is no segmentation or object-pose ground truth for these recordings.

## Reproducibility

- GPU: NVIDIA RTX 5060 Ti, 16 GB VRAM.
- Docker memory ceiling: 11 GiB; no container swap.
- PyTorch 2.8.0 + torchvision 0.23.0, CUDA 12.8 wheels.
- SAM 2 upstream commit:
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Model: `sam2.1_hiera_small`, checkpoint SHA-256
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.
- One frame-0 box plus positive/negative clicks per instance and camera; no
  later correction prompts were used in these two probes.
- 2-D masks were propagated independently per camera, carried between bounded
  chunks, thresholded at logit zero, and bit-packed.
- Depth was sampled at stride 2, projected into the colour mask with the saved
  Kinect calibration, cropped to the configured workspace AABB, fused by
  instance, and voxelised at 4 mm.

## Runs

| scene | frames × cameras | instances | chunk | SAM wall time | non-empty object tracks |
|---|---:|---:|---:|---:|---:|
| `move-shipok` (`2026-09-09_11-54-10`) | 897 × 2 | operator + 1 object | 150 | 163.30 s | 897 / 897 |
| `cup_grab` (`2026-09-04_13-41-37`) | 898 × 2 | operator + 2 objects | 100 | 204.62 s | 898 / 898 for both |

The observed per-camera inference rates were approximately 11.0 FPS with two
instances and 8.8 FPS with three instances.  These are measurements on this
host, not the upstream A100 benchmark.

### 3-D track self-consistency

World-centroid step is the Euclidean displacement between consecutive median
centroids of the mask-supported, fused object points.

| scene / instance | median points | median step | p99 step | max step | median visible world-axis dimensions |
|---|---:|---:|---:|---:|---:|
| `move-shipok` object 2 | 1,849 | 1.43 mm | 8.66 mm | 10.12 mm | 146.6 × 145.7 × 179.0 mm |
| `cup_grab` object 2 | 1,568 | 1.48 mm | 13.40 mm | 15.36 mm | 139.7 × 107.1 × 129.7 mm |
| `cup_grab` object 3 | 1,565 | 0.98 mm | 13.85 mm | 16.87 mm | 151.8 × 98.1 × 158.6 mm |

The visible dimensions are percentiles of the observed surface in fixed world
axes.  They are not an oriented bounding box and must not be interpreted as a
recovered object model.

### Visual and chunk-boundary inspection

Twelve frames spanning the beginning, end, and both sides of several chunk
boundaries were inspected for each camera.  The blue object in `move-shipok`
remained attached to the same instance through grasp, transport, and placement.
The two cups in `cup_grab` retained separate IDs even when one passed near the
other.  No visible identity switch occurred in these samples.

For `cup_grab`, the largest relative area jump at any 100-frame boundary was
2.4% on Kinect 1 and 0.8% on Kinect 0.  Mask overlap peaked at 581 pixels on
Kinect 0 and 163 pixels on Kinect 1, concentrated near hand/object contact; the
3-D artifact marks multi-mask support as `ambiguous_contact` instead of choosing
an unconditional class priority.

## Failures found and retained

The first full three-instance run used 150-frame chunks and was killed under
the 11 GiB container memory ceiling after two chunks.  The async loader's local
`images` reference kept the previous CPU-offloaded 1024-pixel frame tensor alive
while the next tensor was allocated.  Explicitly dropping that reference and
using 100-frame chunks allowed the full run to finish.  Its recorded post-cleanup
RSS ranged from 3,639 to 9,737 MiB; peak allocated CUDA memory was 936 MiB.  The
implementation now also trims the process heap between chunks, but the extra
reduction from that final safeguard has not been separately measured.  More
simultaneous instances may still require a smaller chunk or a process-per-chunk
runner.

The initial 3-D lift also retained a few invalid far-field depth projections,
stretching diagnostic plots several metres away from the task.  It now applies
the same configured workspace crop as the ordinary Viewer cloud before
centroid/dimension estimation.  This changed `move-shipok` object's median
visible x extent from 198.2 mm to 146.6 mm; the earlier number is retracted.

## What this establishes—and what it does not

The probe establishes that one visual seed can preserve useful instance
identity through these two recordings and that the calibrated depth lift can
produce dense, continuous object point tracks.  It does not establish mask IoU,
metric pose accuracy, object orientation, complete shape, or safe collision
geometry.

In particular:

- A few isolated object-labelled points remain attached to the hand in the
  `move-shipok` 3-D sample.  Pose fitting needs a temporally associated connected
  component, multi-view agreement, depth-boundary weighting, and robust loss.
- `cup_grab` moves both cups at different times.  Stable instance identity must
  therefore be separate from the time-varying `active_object_id(t)` role; an
  episode-wide `manipulated_object` versus `other_object` flag is insufficient.
- Orientation must come from registration of a pre-contact object model, with
  an observable-DOF mask/covariance.  Symmetric or partially visible objects
  must not be assigned a confident full SO(3) pose.
- Unknown/unlabelled space is not free space, and the semantic cloud is not yet
  a collision field.

The next justified experiment is high-confidence component/model accumulation
before contact, followed by robust SE(3) or reduced-DOF registration and
phase-wise hand-to-object transform residuals.  Only after those checks should
object-relative factors be connected to retargeting.

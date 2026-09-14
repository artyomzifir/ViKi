# Pyramid automatic prompt experiment — 2026-09-14

## Question

Can the stored empty-scene depth plate and camera calibration replace manual
SAM frame-zero prompts for a multi-object manipulation scene?

This is an experimental self-consistency check. There is no labelled mask or
pose ground truth, so the millimetre results below are model-to-observation and
model-to-model distances, not physical accuracy claims.

## Scene and protocol

- episode: `data/datasets/new-dataset/2026-09-04_13-37-41` (`pyramid`)
- cameras: `kinect_0`, `kinect_1`, 1280×720 RGB, raw 640×576 depth
- frames: 898 at 30 fps; every frame processed
- calibration/background: `calib-v1-checkpoint`
- foreground: absolute depth difference greater than 50 mm
- seed lift stride: 2; fused voxel: 6 mm; component radius: 18 mm
- object proposal: at least 100 voxels, two-camera support, median RGB
  saturation at least 0.25, maximum visible extent at most 250 mm
- segmentation: pinned SAM 2.1 Hiera Small, 60-frame chunks
- lift: stride 2, 4 mm semantic-cloud voxel
- filtering: default `ObjectModelConfig`, one core and final track per instance

Commands:

```bash
docker compose run --rm cli auto-prompts \
  data/datasets/new-dataset/2026-09-04_13-37-41 \
  --objects 3 --out data/_lab/sam2_prompts/pyramid_auto.json

docker compose run --rm sam2 segment \
  data/datasets/new-dataset/2026-09-04_13-37-41 \
  --prompts data/_lab/sam2_prompts/pyramid_auto.json \
  --chunk-frames 60 --lift-3d

docker compose run --rm cli object-model \
  data/datasets/new-dataset/2026-09-04_13-37-41
```

## Automatic seed result

Background subtraction produced 41,524 foreground samples and 24,352 fused
6 mm voxels. Five components passed the minimum size. The largest component was
the operator (20,979 voxels, visible in both cameras). Exactly three compact,
colour-saturated, two-camera components passed the object proposal thresholds:

| id | voxels | visible dimensions in rig AABB (mm) | median RGB saturation |
|---:|---:|---:|---:|
| 2 | 504 | 97 × 87 × 153 | 0.729 |
| 3 | 676 | 89 × 90 × 114 | 0.488 |
| 4 | 743 | 94 × 76 × 145 | 0.758 |

The same fused component generated a robust image-space box and one observed
positive pixel in each camera. The other three component pixels were used as
negative clicks. A 30-frame preflight had zero median inter-mask overlap in
both cameras; maximum overlap was 17 pixels in `kinect_0` and 108 pixels in
`kinect_1`.

The generator labels all three candidates `other_dynamic`. It deliberately does
not infer `manipulated_object` from frame zero; task role must be assigned from
contact/motion phases later.

## Complete run

- SAM masks: 898/898 frames in both cameras
- SAM elapsed: 300.56 s (two camera streams, four instances)
- peak allocated CUDA memory per 60-frame chunk: approximately 0.90 GiB
- semantic-cloud frames: 898
- object-model elapsed: 234.16 s
  - load: 8.64 s
  - core/final modelling: 197.01 s
  - point classification: 28.51 s
- candidate points classified: 4,590,379
  - accepted: 4,036,921
  - rejected: 420,389
  - reassigned between objects: 60,393
  - ambiguous: 72,676

Track self-consistency percentiles:

| id | model points (core + shell) | confidence p10 / p50 / p90 | median residual p10 / p50 / p90 (mm) | retained p50 | contact frames |
|---:|---:|---:|---:|---:|---:|
| 2 | 1,908 (1,272 + 636) | 0.808 / 0.838 / 0.953 | 1.31 / 2.52 / 2.77 | 0.878 | 17.7% |
| 3 | 3,206 (2,137 + 1,069) | 0.696 / 0.897 / 0.906 | 1.89 / 1.98 / 2.97 | 0.875 | 19.0% |
| 4 | 3,205 (2,629 + 576) | 0.951 / 0.960 / 0.970 | 1.04 / 1.22 / 1.34 | 0.964 | 0.0% |

At frame 800 the closest canonical-surface distances were 2.2 mm between the
upper and middle objects and 0.4 mm between the middle and lower objects. The
apparent sparse-cloud gap in some Viewer angles is therefore partial surface
coverage, not a separated pose estimate.

## Verdict and limitations

For this colourful, rigid tower scene, calibrated 3-D foreground components are
a sufficient automatic source of consistent multi-camera SAM prompts. They
remove the large table/shadow leakage seen with the first loose manual prompt in
`kinect_1` and preserve three distinct instances through stacking.

This result does not establish a general object detector. The current proposal
ranking assumes compact objects with enough colour saturation, a valid empty
background plate, separation between components at the seed frame, and visibility
in two cameras. White/grey objects, deformables, initial hand contact, or touching
objects need geometric/temporal proposal alternatives. Semantic task role and
physical orientation accuracy also remain unvalidated without ground truth.

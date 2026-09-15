# Depth-edge thickness diagnostic — 2026-09-15

## Question

The fused Viewer clouds made rigid blocks look thicker than the physical
objects. The known reference in `move-shipok` is the Г-shaped block with an
external envelope of **145 × 95 × 50 mm**. The investigation separated three
possible causes: depth-edge samples, colour/depth projection, and inter-camera
extrinsics/fusion.

## Protocol

- Episodes: `2026-09-09_11-54-10` (`move-shipok`) and
  `2026-09-04_13-37-41` (`pyramid`).
- Points were lifted with the recorded raw libk4a calibration and classified
  with the existing SAM masks using exact K4A colour projection.
- The reported sizes are robust minimum-volume OBB extents of the largest
  connected mask-supported component (1% tails removed). They are
  self-consistency/visible-surface measurements, not a calibrated metrology
  result; occlusion can make an axis too short.
- Candidate filtering rejects a depth sample when the local depth span exceeds

  `max(30 mm, 0.02 * depth_mm)`.

  Its pixel radius is derived from an angular radius rather than hard-coded:

  `radius_px = ceil(0.004 rad * sqrt(fx_depth * fy_depth))`.

  This is two pixels for the recorded 640 × 576 Kinect depth stream and scales
  with focal length when resolution changes.

## Result

### `move-shipok`, object 2

| Frame | Baseline OBB, mm | Filtered OBB, mm | Reference, mm | Points kept |
|---:|---:|---:|---:|---:|
| 100 | 133.79 × 111.71 × 70.27 | 132.30 × 112.61 × 49.78 | 145 × 95 × 50 | 1233 / 1358 (90.8%) |
| 300 | 131.13 × 111.10 × 66.97 | 134.87 × 110.92 × 52.26 | 145 × 95 × 50 | 1618 / 1776 (91.1%) |
| 800 | 132.21 × 110.52 × 61.52 | 132.15 × 112.06 × 47.99 | 145 × 95 × 50 | 1380 / 1639 (84.2%) |

The axis affected most strongly by the mixed/flying pixels moved from
61.5–70.3 mm to 48.0–52.3 mm, matching the known 50 mm thickness. A larger
8–10 mrad neighbourhood reduced the same axis to about 36 mm and was rejected
as over-filtering.

### `pyramid`, frame 0

There is no independently recorded physical-size reference for all three
visible instances, so these numbers are only an A/B stability check.

| Object | Baseline OBB, mm | Filtered OBB, mm | Points kept |
|---:|---:|---:|---:|
| 2 | 100.07 × 78.98 × 55.06 | 96.57 × 50.32 × 48.96 | 825 / 998 (82.7%) |
| 3 | 140.74 × 71.88 × 57.96 | 133.96 × 72.01 × 57.86 | 1093 / 1323 (82.6%) |
| 4 | 145.21 × 75.55 × 59.88 | 135.60 × 75.99 × 59.78 | 1105 / 1329 (83.1%) |

The filter removes the discontinuity halo but does not force every shape toward
a preferred box size.

## What was and was not confirmed

- **Confirmed:** mixed/flying depth pixels at foreground discontinuities were
  the main cause of the Г-block's false thickness.
- **Not a geometry cause:** the old pinhole colour lookup differs from exact
  K4A projection by roughly 2–8 colour pixels, but colour association does not
  move XYZ. Exact libk4a colour-to-depth warp was implemented as an opt-in path
  and measured at about 0.29–0.47 seconds per camera/frame, so it remains off by
  default. It is more appropriate for a later exact SAM-mask lift.
- **Still present:** the two background clouds differ by about 5.1 mm in table
  height and 2.34 degrees in fitted table normal on one measured frame. A global
  background ICP correction (6.36 mm, 0.65 degrees) did not improve the object
  envelope consistently and was therefore not installed.

## Installed behavior and real-scene run

The angular/metric edge filter is enabled for Viewer cloud building, automatic
prompt foreground samples, and semantic-cloud lifting. Its parameters and
whether exact colour projection was used are recorded in artifact metadata.

Both complete Viewer clouds were rebuilt with every frame:

| Episode | Frames | Runtime |
|---|---:|---:|
| `move-shipok` | 897 | 52.63 s |
| `pyramid` | 898 | 53.35 s |

The production default keeps exact colour warp disabled; both rebuilds used the
new depth-edge filter and the existing calibrated XYZ lift.

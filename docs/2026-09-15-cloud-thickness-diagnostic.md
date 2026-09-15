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

## Full diagnosis (supersedes the first single-frame conclusion)

The initial conclusion that mixed/flying depth pixels were the *main* cause was
incomplete: frame 100 exposed the depth-edge failure, but the object was visible
to essentially one camera and could not test fusion.  Frame 574 was selected as
the balanced check (954 object points from Kinect 0 and 962 from Kinect 1).

Two independent errors were confirmed:

1. Mixed depth-edge pixels inflated the physical 50 mm axis to 61–70 mm.  The
   angular/metric edge filter returns it to about 48–52 mm.
2. ChArUco extrinsics were solved with a zero-distortion pinhole approximation,
   while the depth cloud was reconstructed with the full factory K4A lens
   model.  This model mismatch displaced Kinect 1 by 22.2 mm / 0.46 degrees.
   It inflated the fused middle extent on frame 574 from the physical 95 mm to
   119.3 mm.

The extrinsics solve now unprojects every ChArUco colour pixel to an exact SDK
camera ray and feeds those rays to the joint bundle adjustment.  It is metric
and resolution-independent: the output resolution only changes the SDK pixel
mapping, not the recovered camera ray.  On the same 1,127 stored corners,
reprojection RMS changed from 1.93 / 1.32 px to 1.09 / 1.22 px for Kinect 0 / 1.
The independent stereo check agrees with the bundle within 2.28 mm / 0.15
degrees.

| Frame 574 | Robust fused OBB, mm |
|---|---:|
| old extrinsics | 150.93 × 119.30 × 51.04 |
| exact K4A-ray extrinsics (rebuilt artifact) | 150.63 × 98.47 × 51.52 |
| physical envelope | 145 × 95 × 50 |

The static-background check remains green.  Its symmetric NN p90 improves from
32.56 to 25.62 mm and fitted residual from 4.81 to 4.69 mm.  Per-frame object
ICP was explicitly rejected: because the cameras see complementary surfaces,
its proposed correction varied by roughly 70–190 mm and 3.5–13 degrees across
the recording and was not a calibration signal.

The independent `pyramid` preset was re-solved from its own 22 capture sets
(2,594 corners; rectified RMS 0.92 / 1.01 px), producing a 21.5 mm / 0.56 degree
Kinect-1 correction.  On balanced-visibility frames, replaying the old matrix
against the exact same semantic points gives the following A/B result (still no
physical ground truth for these three blocks):

| Object / frame | Camera points | Old fused OBB, mm | Exact-ray fused OBB, mm |
|---|---:|---:|---:|
| #2 / 552 | 964 + 1070 | 167.76 × 109.48 × 54.17 | 149.52 × 109.20 × 54.25 |
| #3 / 262 | 951 + 956 | 163.57 × 116.83 × 55.58 | 142.26 × 119.33 × 56.43 |
| #4 / 144 | 478 + 434 | 135.04 × 76.15 × 59.31 | 137.62 × 66.00 × 52.86 |

The exact K4A colour-to-depth image warp is a separate matter: it changes RGB or
SAM-mask association, not XYZ.  The first call includes SDK setup and measured
0.31–0.48 s/camera, but warmed calls take about 4.9–6.7 ms/camera/frame.  It is
therefore enabled by default.  Semantic lift also subtracts the saved empty
background before assigning mask-supported 3-D points.

## Installed behavior and real-scene run

The angular/metric edge filter is enabled for Viewer cloud building, automatic
prompt foreground samples, and semantic-cloud lifting. Its parameters and
whether exact colour projection was used are recorded in artifact metadata.

Both complete Viewer clouds were rebuilt with every frame:

| Episode | Frames | Runtime |
|---|---:|---:|
| `move-shipok` | 897 | 52.63 s |
| `pyramid` | 898 | 53.35 s |

Those timings predate the exact-ray extrinsics correction.  Both artifacts are
rebuilt again below before demonstration; the earlier files must not be used to
judge fusion.

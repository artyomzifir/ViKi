# Static cubes and hand: two-Kinect depth control (2026-10-05)

**2026-10-06 follow-up:** Recalibration with the 400×500 mm ChArUco board
produced visually much better cloud alignment than the small-board `table-v1`
calibration. The operational recommendation below predates that test and is
qualified by `2026-10-06-large-board-calibration.md`. The measurements below
remain valid for their recorded small-board preset; they do not certify the
new preset or prove that depth range was the only cause of the doubled hand.

## Purpose and immutable inputs

The moving `40mm-pnp/2026-10-05_14-15-39` pilot showed a doubled finger.
Previous ChArUco corner tests found 7–13 mm depth-vs-colour residuals and
suggested a depth-range correction. This control checks that interpretation
against rigid, known-size objects **without moving the cameras**.

- Empty/static scene: `data/datasets/rig-static-cubes-2026-10-05/2026-10-05_16-43-26`,
  85 synced RGB-D frames. Green cube on table, blue 40 mm cube on red 40 mm
  cube, ChArUco board on table.
- Static hand: `data/datasets/rig-static-hand-2026-10-05/2026-10-05_16-57-28`,
  272 synced frames, index finger extended above the same scene. The hand
  drifts slightly; the comparison is not a precision still-life benchmark.
- The existing pilot and both new recordings use `table-v1`, colour
  1920×1080, depth `NFOV_UNBINNED` 640×576 at 30 fps, and the recorded K4A
  calibration blobs. No preset, raw depth, or canonical cloud was altered.

All figures below are **internal consistency** measurements, not independent
metrology. Colour masks/regions are diagnostics and not semantic ground truth.

## Rigid controls

At seven static-board frames, corresponding RGB ChArUco rays have a median
closest-approach gap of **0.182 mm** (per-frame range 0.147–0.215 mm). The raw
depth points sampled at the printed corners have a median pair gap of
**7.933 mm**, falling to 1.805 mm if a per-camera corner-derived offset is
subtracted. That alone cannot validate a global depth correction: a printed
corner is a colour feature, not guaranteed to be the ToF-measured surface.

At frame 45, the two RGB views triangulate a board plane with 0.33 mm corner
scatter. Relative to that plane, cube-top Z p95 is:

| Cube top | Kinect 0 | Kinect 1 | Between-camera difference |
|---|---:|---:|---:|
| Green (40 mm above table) | 31.3 mm | 31.3 mm | <0.1 mm |
| Blue on red (80 mm above table) | 69.5 mm | 70.8 mm | 1.3 mm |

The board face is above the neighbouring table in this setup. Measured local
table depth is -7.2/-7.6 mm below the board plane near green (Kinect 0/1),
giving green heights of **38.5/38.9 mm**. Near the stack, separate small table
patches give -13.9/-9.1 mm, so the blue top is **83.4/79.9 mm** above those
patches; that local patch mismatch is an uncertainty, not a cube calibration.
Across 109 common 25 mm tabletop cells, excluding board and cubes, Kinect 0
minus Kinect 1 table height has median **-1.11 mm** and p10–p90
**[-2.27, +0.02] mm**. There is no uniform 10–20 mm rigid cloud offset in
this still scene.

Depth from the ChArUco *interior* lies 5.62/4.06 mm below its RGB plane
(Kinect 0/1), while surrounding-table depth is lower and spatially mixed.
Black and white printed regions have similar offsets. Whether this is board
material/thickness, local ToF bias, or another board-specific effect remains
unresolved. **Retraction/qualification:** the earlier board-corner 7–13 mm
residual is real at those sampled pixels, but it must not be described as a
calibrated camera-wide range bias or used to shift hand/cube points. The rigid
controls do not show such a global bias; a correction that improves board
corners has already worsened some moving-cube checks.

## Hand-range control

For reproducibility, the simple hand-region count uses points in the task
workspace 0.10–0.50 m above the display plane whose aligned RGB is reddish
skin-like (`R > 1.3 G`, `R > 1.1 B`, `R > 28`, `G > 10`, `G < 0.9 R`). It is a
rough diagnostic mask; it excludes invalid depth because such points have no
XYZ to count. `stride=2`, existing angular depth-edge filter, no background
subtraction. Fractions are **of retained skin-like depth points**, not all
skin pixels.

| Take/frame | Camera | Median raw depth | Retained skin points below 0.5 m |
|---|---|---:|---:|
| Moving / 90 | Kinect 0 | 466 mm | 88.1% |
| Moving / 90 | Kinect 1 | 645 mm | 0% |
| Moving / 120 | Kinect 0 | 453 mm | 97.3% |
| Moving / 120 | Kinect 1 | 541 mm | 0% |
| Moving / 160 | Kinect 0 | 692 mm | 0% |
| Moving / 160 | Kinect 1 | 390 mm | 100% |
| Moving / 180 | Kinect 0 | 723 mm | 0% |
| Moving / 180 | Kinect 1 | 432 mm | 100% |
| Static / 20, 130, 250 | Kinect 0 | 593, 590, 587 mm | 0%, 0%, 0% |
| Static / 20, 130, 250 | Kinect 1 | 437, 433, 433 mm | 91.6%, 92.6%, 92.7% |

[Microsoft's Azure Kinect hardware specification](https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/hardware-specification)
gives **0.5–3.86 m** as the supported operating range for `NFOV_UNBINNED`;
values may still be emitted outside it, without the stated error guarantees.
It lists **0.25–2.88 m** for `WFOV_2X2BINNED` at 30 fps. The hand alternates
between cameras' unsupported near fields during transfer, precisely when the
visible doubled-hand problem is most pronounced. The static hand confirms this
is not just a frame-pair slip or motion blur.

This is not proof that every return below 0.5 m is wrong: the green and blue
cube tops also have sub-0.5 m raw depths in one view and still agree within
about 1 mm between cameras. Range, surface reflectance, thin finger edges,
and view-dependent occlusion interact. The test establishes an unsupported
working condition, not a universal centimetre correction for near points.

The two RGB fingertip pixels manually matched at static frame 130 have a
2.1 mm stereo-ray gap, but the projected fingertip depth pixel is zero in
**both** raw depth maps, with only partial returns at the edge of its 9×9
neighbourhood. A manual fingertip match carries annotation uncertainty; it is
not a millimetre benchmark. It does show that a colour-visible fingertip need
not exist in either depth cloud. At a distal finger section, the two raw
surface sets differ in median height by roughly 8–17 mm depending on slice;
the proximal arm has larger separation. Those are opposite visible surfaces,
not matched skin correspondences, so neither value by itself is an extrinsic
error estimate. The Viewer concatenates them without a hand-surface model.

Turning the angular depth-edge filter off at static frame 130 increases the
retained distal-finger sample count from 1,305 to 1,448 (Kinect 0) and
2,184 to 2,320 (Kinect 1), but changes the distal X 95th percentile by
less than 1.2 mm. Edge rejection is not the main cause of the missing tip.

## RGB silhouette cross-check (static frame 130)

SAM2 single-frame hand masks were generated independently for both RGB views.
Only raw-depth-derived, skin-coloured points projecting inside or at most 5 px
outside their **own** camera's mask were tested against the **other** camera's
mask. Each projection uses the recorded `T_world_display`, camera extrinsics,
and the exact K4A factory colour-lens model. The quick pinhole approximation
was rejected: it differs from the SDK projection by 5–7 px near the tip.

| Display-frame X section | Kinect 0 points inside Kinect 1 RGB mask | Kinect 1 points inside Kinect 0 RGB mask | >10 px outside, respectively |
|---|---:|---:|---:|
| Middle, -180 to -120 mm | 98.6% of 3,752 | 99.0% of 4,541 | 1.1%, 0.0% |
| Distal, -120 to -40 mm | 70.0% of 961 | 76.1% of 1,323 | 16.3%, 12.2% |

An alternative SAM2 mask changed the distal >10 px fractions only to
17.4%/13.1%. The outlying points form a coherent fringe below or beyond the
RGB fingertip in the other view, while middle-finger points stay inside.
At roughly 0.5 m range, 10 px in RGB represents about 5 mm transverse to
the colour ray; this is only a scale estimate, not a measured 3-D correction.
The mask is model-generated and the hand drifted slightly, so this test is a
strong **local inconsistency flag**, not proof that one specific sensor's
range is biased by a fixed amount. It independently rejects a global rigid
extrinsic shift: such a shift would also displace the well-matched middle
section and rigid cubes. The two visible skin sides may remain separated in a
perfect raw cloud even after invalid edge points are removed.

The same exact-SDK test was repeated from raw depth at frames 20 and 250,
using separately generated RGB masks and stride-2 source points. For the
middle section, 98.2–98.9% of points project inside the other hand silhouette.
For the distal section, only 69.7–81.8% do; 8.6–16.3% lie more than 10 px
outside. The pattern therefore persists across three samples separated by
seconds and cannot be explained solely by one bad RGB pair or one hand pose.
It still does not distinguish out-of-range ToF returns from mixed edge pixels,
within-camera colour/depth exposure offset, or imperfect masks. A controlled
depth-mode or camera-distance A/B test is required for that distinction.

At frame 130, the distal points projecting **inside** the other RGB silhouette
have median source raw depths of 566 mm (Kinect 0) and 488.5 mm (Kinect 1).
The points projecting more than 10 px **outside** have medians of 583 mm and
548 mm, respectively: these outliers are *farther away* than the coherent
finger points. For Kinect 1, 46.3% of outliers are within 3 depth pixels of
a zero-depth hole, versus 18.7% of inside points; medians of distance to a
hole are 3.2 versus 6.6 pixels. Kinect 0 also has a smaller hole-distance
contrast (4.4 versus 8.8 pixels median). This supports a thin-object edge /
occlusion component, not a single camera transform. The outliers do **not**
match the depth of the earlier empty scene, so calling them proven static
background leakage would be wrong. Nor does a simple `depth >= 500 mm` gate
solve this: it would discard many valid-looking Kinect 1 finger points while
keeping these farther outliers. The exact physical failure within the sensor
remains to be isolated by the near-range A/B test.

## Decision and next discriminating test

The old camera transform and a global board-derived range shift are not
supported. The current `NFOV_UNBINNED` geometry **violates the sensor's
published range for much of the task**, especially the camera nearest the
hand. This is a demonstrated invalid operating condition and a strong cause
of unstable/missing finger depth; it does not exclude an additional depth
timing, interference, or fusion issue.

Run an A/B static hand test with the same cameras and RGB extrinsics but
`WFOV_2X2BINNED` depth (minimum 0.25 m), then restore the original mode.
Alternatively move both cameras until the entire hand workspace measures at
least 0.5 m from **both** depth sensors with margin, then recapture
calibration/background. Compare RGB-ray gaps, static cube heights, valid hand
depth fraction, and cross-camera finger surface sections before making any
default change or resuming trusted dataset collection. The MediaPipe hand
landmarker detected no hand in five inspected static frames; that is a
separate perception concern, not evidence about cloud XYZ.

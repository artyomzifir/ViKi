# Two-Kinect cloud handedness check

**Latest check:** The transferred range correction improves `test1` only
slightly; it does not make its two-view hand geometry trustworthy. The
measurements and remaining uncertainties are detailed below.

**Follow-up:** The first interpretation below was too dismissive. The finger
offset is visible in 3-D, and the fused cloud is not an accurate hand surface.
The controlled checks and revised conclusion are at the end of this note.

Measured 2026-10-04 on branch `v0.0.2`, using the newest episode available at
the time: `data/datasets/40-mm-cube-pnp/2026-10-04_09-54-36`. The user
identified frames 180–200 as the clearest apparent left/right finger reversal.
The recording and its saved cloud were read without modification. Diagnostic
figures and per-camera samples are in gitignored
`data/diagnostics/cloud-mirror/`.

## What the recording contains

The episode has 220 frames from two Kinect cameras. Only `record` and `cloud`
are complete; it has no extracted hand landmarks or downstream robot plan.
The cameras are on opposite sides of the workspace. In the display frame,
`kinect_0` is at approximately `(0.282, 0.507, 0.451)` m and `kinect_1` at
`(-0.442, 0.372, 0.500)` m. Their colour images therefore show the hand and
stacked cubes from opposing horizontal directions. The saved rig-to-display
rotation has determinant +1, not a reflection. The colour-to-depth aligned
images retain the same orientation as their raw colour videos.

The device timestamp differences are 156, 145 and 144 microseconds at frames
180, 190 and 200 respectively. No frame of this take differs by over 1 ms,
so the old 33 ms cross-camera pairing fault is not present in this episode.

## Geometry check

Both depth maps were independently deprojected with the recorded Kinect SDK
calibrations, placed with the recorded extrinsics and display anchor, and
filtered with the cloud builder's empty-scene subtraction and depth-edge
filter. The following are directional *nearest-point distances between visible
surfaces*, not physical accuracy or point correspondences:

| Frame | Region | `kinect_0` to `kinect_1` median / p90 | Reverse median / p90 |
|---:|---|---:|---:|
| 180 | arm/hand area | 11.4 / 44.1 mm | 7.5 / 29.6 mm |
| 190 | arm/hand area | 13.5 / 52.5 mm | 10.0 / 36.1 mm |
| 200 | arm/hand area | 13.5 / 47.3 mm | 11.9 / 48.9 mm |
| 180 | stacked cubes | 12.4 / 23.3 mm | 11.7 / 21.1 mm |
| 190 | stacked cubes | 7.1 / 15.8 mm | 6.0 / 16.3 mm |
| 200 | stacked cubes | 7.2 / 15.4 mm | 5.9 / 16.5 mm |

At frame 190 the yellow cube's per-view visible-colour centroids differ by
approximately 9 mm. That is **not** a centre-to-centre calibration error:
the cameras see different faces of a 40 mm solid. The calibration preset's
separate empty-scene validation reported a 4.11 mm nearest-neighbour median
and a 3.34 mm ICP residual; these are self-consistency measures, not ground
truth. The saved fused `cloud/000190.bin`, transformed for display, lies within
1.9–2.1 mm median (2.9–3.2 mm p90) of the independently reconstructed
per-camera samples; the remaining difference is expected from the 5 mm voxel
and point budget. Thus no evidence points to a mirror/swap in cloud writing or
Viewer matrix application.

The arm and finger contours are not supposed to coincide point-for-point:
each depth camera returns its *nearest visible skin and clothing surface* from
its own side. Some near-finger contours appear to cross in the XY projection,
but a 2-D projection of distinct 3-D surfaces cannot establish that the
physical finger sides have been exchanged. The hand-region nearest-point
numbers also include clothing and unshared/occluded surfaces. They do show
that this fused cloud is too coarse to treat as a millimetre-accurate solid
hand mesh. The demonstrator's torso remains foreground because the background
plate was recorded empty; it further obscures the Viewer cloud.

## Decision

Do **not** flip an axis, exchange camera IDs, or alter extrinsics based on the
apparent finger reversal. That would break the currently aligned static cubes
and background. Preserve the raw synchronized views. If hand geometry itself
becomes an input to modelling, inspect camera-specific coloured surfaces and
landmark tracks, use semantic hand/operator separation and measured surface
fusion, and validate against a known 3-D object or tracked target. The Viewer
cloud alone is a visualization artifact, not proof of hand-surface accuracy.

## Follow-up: isolate the source of the apparent shift

The first report established that there was no obvious axis reflection, but
incorrectly treated the finger split as sufficiently explained by opposite
viewpoints. A more focused check of the index-finger area at frames 180, 190
and 200 finds a roughly 9, 14 and 15 mm separation in X between the two
camera-specific depth samples. This is a real feature of the saved raw depth,
not introduced by the Viewer binary or its matrix. It is **not** an accuracy
measurement of corresponding skin points: the two cameras observe different
surface patches, and nearest-neighbour pairing may match different parts of
the finger.

Several independent controls narrow the cause:

* Re-solving all 13 saved ChArUco sets in memory with the exact Kinect colour
  ray models reproduces the saved camera transform to numerical precision.
  Reprojection RMS is 1.17 / 0.97 px. An independent stereo solve differs by
  0.24 mm / 0.024 degrees. The saved solve metadata says `legacy-migrated`
  because it lost statistics, not because the stored transform was arbitrary.
* The yellow 40 mm cube, lifted to approximately 0.18 m in frame 100, has
  coloured-depth nearest-surface medians of 6.9 / 5.7 mm. Its top-surface
  height p90 is 0.1880 / 0.1887 m. At frame 190, when stacked near the table,
  the analogous medians are 6.7 / 5.8 mm. Thus there is no evidence of a
  large calibration error that appears only at hand height.
* A one-frame camera shift is not a valid fix. The moving finger's visible
  patches can be made to overlap better by comparing different times, but the
  wrist/arm region has its best bidirectional agreement at zero-frame lag
  (6.3 / 4.6 mm median for the chosen region). The hardware colour timestamps
  are already within 1 ms throughout the take. Per-depth-image timestamps
  were not saved, so a subframe RGB/depth phase error cannot be measured
  directly from this recording.
* MediaPipe's image-mode and video-mode index-tip estimates disagree by about
  31 colour pixels in `kinect_0` at frame 190; one apparent depth hole arose
  from that incorrect 2-D landmark. Hand landmarks are therefore not ground
  truth for diagnosing millimetre depth bias here.

The proximate software cause is in `viki/perception/cloud.py`: it concatenates
the depth returns from both views and voxel-decimates them. It does not
establish corresponding finger points, reason about occluded surfaces, model
sensor uncertainty, or reconstruct one watertight hand. A flat table makes
this look registered; a thin finger viewed from opposite sides exposes
different patches and edge/occlusion effects, which can project in reversed
order. This is a limitation of the current cloud representation, **not** proof
that one camera should be translated or mirrored. The precise contribution of
Kinect depth edge bias, RGB/depth parallax and hand motion remains unmeasured
without a known rigid target in the same volume or per-depth-frame timing.

For subsequent work, retain camera provenance in cloud diagnostics and
separate the hand from clothing/background before attempting a visibility-
aware surface reconstruction. Do not use this fused Viewer cloud as a
hand-geometry label or correct it with an arbitrary rigid offset.

## Static ChArUco board control, same calibration

The user recorded `data/datasets/40-mm-cube-pnp/2026-10-04_11-04-59` with the
unchanged `new-arch` preset. The episode contains 264 paired RGB-D frames. Its
recorded camera transforms exactly match the preset after converting the stored
Rodrigues board-to-camera pairs to camera-to-reference transforms. The board is
stationary on the table; this does **not** test the hand-height volume or moving
finger boundaries. A visual check of frame 128 is saved at
`data/diagnostics/board-static/frame128_board_crop.jpg`.

The paired colour device-timestamp difference has median 155 µs, p10/p90
137/166 µs, and maximum absolute value 167 µs. For ChArUco corners visible in
both colour views, rays were undistorted with each recorded K4A SDK calibration,
placed with the recorded extrinsics and triangulated. A rigid fit of the known
board corner geometry to those triangulated points gives:

| Frame | Shared corners | Stereo-ray gap p50/p90 | Rigid-board fit error p50/p90 |
|---:|---:|---:|---:|
| 16 | 30 | 0.69 / 1.09 mm | 0.38 / 0.50 mm |
| 128 | 29 | 0.68 / 1.14 mm | 0.40 / 0.50 mm |
| 240 | 31 | 0.73 / 1.09 mm | 0.40 / 0.49 mm |

The board interior was conservatively masked in each colour image, warped to
the corresponding depth image with the SDK and lifted to the reference frame.
Signed depth-point distance to the colour-corner board plane is about +5 mm in
both cameras: frame 16 medians 6.19 / 4.79 mm, frame 128 medians 5.63 / 4.94 mm,
and frame 240 medians 5.18 / 5.05 mm for `kinect_0` / `kinect_1`. The within-view
robust scatter is roughly 1–1.5 mm. This is a **colour-vs-depth surface offset**,
not an inter-camera translation estimate: both cameras show the same sign and
the printed board's physical/IR response is not independently known. Do not
correct either camera by 5 mm on this evidence alone.

**Conclusion:** The old extrinsics are strongly supported by an independent
static RGB target at table height. A gross camera swap, reflection or large
table-height stereo misregistration is unlikely. The result does not explain
the 9–15 mm finger-region split: a flat, static board has no reliable lateral
depth correspondences and is not at hand height. The next discriminating test
is the same rigid target held stationary in the hand workspace, at multiple
heights and tilts, with both cameras seeing the printed face and its depth
surface. Save raw RGB and depth for each pose; compare matched colour corners,
depth planes and board edges before any extrinsic adjustment or fusion change.

## Hand-test follow-up: the static board really does split in depth

The subsequent `data/datasets/40-mm-cube-pnp/2026-10-04_11-15-23` take has
474 paired RGB-D frames with a moving hand but the ChArUco board left fixed in
the background. The board is still near table height, not in the finger working
volume. Its colour-derived pose differs from the earlier static board take by
only 1.0 mm / 0.09°. The board in the old calibration photos reached roughly
0.27–0.43 m display height, but those photos contain **no depth** and therefore
cannot calibrate colour-to-depth or depth range.

This take changes the diagnosis: the static printed board has measurable
per-camera depth disagreement too, so the earlier explanation based only on
opposite views and naive cloud concatenation was **incomplete**. Across frames
sampled every 30 frames, corresponding ChArUco **colour** rays have a
0.43–0.51 mm median closest-approach gap. At frames 0, 128, 240 and 400, the
same corner's raw depth point was sampled at the SDK-projected depth pixel and
compared with its two-view RGB triangulation:

| Frame | `kinect_0` depth-vs-RGB median | `kinect_1` depth-vs-RGB median | Matched depth-point gap median |
|---:|---:|---:|---:|
| 0 | 13.1 mm | 6.8 mm | 11.2 mm |
| 128 | 13.4 mm | 7.0 mm | 11.6 mm |
| 240 | 10.0 mm | 7.1 mm | 9.1 mm |
| 400 | 12.1 mm | 7.2 mm | 10.6 mm |

The depth-vs-RGB errors are almost entirely **along the individual colour
ray**: perpendicular medians are about 0.8–1.1 mm. Therefore the major static
split is not an axis mirror, mistaken camera ID, or bad 2-D colour projection;
the recorded depth ranges place the two independently viewed board textures
at different 3-D locations. These are same-*texture-corner* comparisons, not
nearest-neighbour surface distances or external accuracy measurements. Pixel
quantisation and the fact that depth cannot resolve printed corners still
contribute around a millimetre.

The independent board-only take, at almost the same physical pose, had depth-
to-RGB medians around 7.4 / 6.7 mm for `kinect_0` / `kinect_1` at frame 128.
At the same depth pixels in frame 128 of the new take, `kinect_0` raw depth is
7 mm farther (p10/p90 5/9 mm), while `kinect_1` is only 1 mm farther
(p10/p90 -1/3 mm). Nearby `kinect_0` pixels also shift by about 6 mm median,
versus 2 mm for the wider background. The colour board did not move enough to
explain this. Thus **a fixed 10 mm camera translation would be an unsafe fix**:
the discrepancy changes between takes and within the new take. The current
data do not separate depth-sensor drift, local multipath/interference, and
other scene-dependent raw-range bias.

The Viewer adds a separate visual ambiguity. It draws a synthetic reference
board at display Z=0, while the *recorded* board centre is around Z=+22 mm and
tilted; hide the `ChArUco board` layer when judging the measured cloud. The
50 mm background subtraction also removes most actual board-depth pixels:
at frame 128 only 792/2911 (`kinect_0`) and 3312/9185 (`kinect_1`) valid
interior board pixels pass the raw depth-difference test. This produces a
patchy cloud but **does not create** the raw depth split above.

**Required discriminating capture:** With the board fixed in a rigid holder,
record one continuous RGB-D take that begins with no person in view, then
brings the hand near the board without moving it, then removes the hand. Repeat
with the board at 0.15–0.35 m display height in several tilts visible to both
depth sensors. Keeping the stream running within each comparison avoids a
restart/warm-up confound. This yields colour-corner ground geometry plus actual
depth at multiple positions; only then fit and validate a depth-range or
depth-to-colour correction. Do not train on the current merged hand cloud.

## Offline range-bias experiment on the recordings already available

The preceding request for another capture is **not a prerequisite** for an
initial correction experiment. We used the two existing board-containing
episodes without moving or re-recording anything. This does not retract the
need for independent validation at hand height before treating corrected
clouds as training ground truth.

The Kinect SDK's own depth-to-colour transformation reproduces the observed
board discrepancy: at frame 128 it gives approximately 7.1 / 6.8 mm
depth-to-RGB median in the board-only take and 13.2 / 7.1 mm in the hand-test
take (`kinect_0` / `kinect_1`). Therefore the difference does not originate in
ViKi's vectorised depth deprojection. The colour stereo rays intersect within
approximately 0.5 mm in the hand-test; the main residual is depth *range*.

For each sampled frame, `scripts/diagnose_board_depth.py` triangulates common
ChArUco colour corners using the saved extrinsics and exact SDK ray model. At
each corner's projected depth pixel, it computes the depth range expected from
the colour triangulation. Even-numbered corner IDs estimate one scalar range
bias per camera; odd IDs are held out. A gate requires at least 12 shared RGB
corners, sufficient held-out observations, a corrected pair median below 5 mm
and a strict improvement over raw. Samples are 15 frames apart and linearly
interpolated. The correction subtracts range bias only *after* raw-depth
edge/background filtering; colour warping still uses raw depth. This is an
empirical, per-episode/per-frame **diagnostic**, not a new factory depth
calibration or a rigid camera transform.

| Episode | Sampled frames passing gate | Median held-out board pair distance, raw → corrected | Bias p10–p90, `kinect_0` / `kinect_1` |
|---|---:|---:|---:|
| Board only `11-04-59` | 18/18 | 6.83 → 1.90 mm | 6.50–8.08 / 6.30–7.05 mm |
| Board + hand `11-15-23` | 32/32 | 9.57 → 2.25 mm | 9.92–12.67 / 6.18–7.88 mm |

No sampled frame's held-out median worsened. At hand-test frame 128, the
board-corner pair median/p90 changed from 11.90/13.89 to 2.52/3.69 mm. An
independent white-tabletop ROI at that frame, paired by XY within 5 mm,
changed from 5.83/8.48 to 1.80/3.67 mm median/p90. This is useful evidence
that the range shift is not confined to printed board corners. However the
same tabletop test in the *board-only* take did **not** improve materially:
2.08/3.68 to 1.98/4.61 mm. The correction can therefore over-adjust some
surfaces. For the moving hand, directional nearest-surface median at frame
128 changed from 8.2 to 5.3 mm for `kinect_0`→`kinect_1`, but the p90 remains
28.3 mm, and opposite views are not point correspondences. A lifted yellow
cube's top heights already agreed to about 0.65 mm in another take; applying
a blanket board correction there would be unjustified.

The hand-test has a full 474-frame corrected artifact. The offline script
normally writes it separately as `cloud_board_corrected/`, without changing
`raw/`, canonical `cloud/`, or stage status. At the user's request, the two
cloud directories were then manually swapped for visual review: **Viewer now
serves corrected `cloud/`**, and the untouched original is backed up as
`cloud_original_before_board_correction/` beside it. Corrected `cloud/meta.json`
records the interpolated bias per camera and frame. Reproduce the separate
artifact offline with:

```bash
docker compose --profile tools run --rm terminal python3 /app/scripts/diagnose_board_depth.py \
  data/datasets/40-mm-cube-pnp/2026-10-04_11-15-23 \
  --frames $(seq 0 15 465) \
  --out data/diagnostics/board-static/handtest_depth_bias_sweep.json \
  --corrected-cloud
```

**Decision:** Keep the canonical pipeline unchanged. The range-bias diagnosis
is supported by the stored board and by one independent tabletop check, but
it is not yet validated over the whole hand/cube workspace. Do not use the
experimental cloud as ground truth for object-centric labels, retargeting,
or robot-policy training. It also does not solve occlusion, mixed depth-edge
pixels, or background-subtraction holes.
## Review of the still-misaligned `test1` hand cloud

The user inspected the experimental cloud for the 220-frame
`2026-10-04_09-54-36` episode and found it only slightly better. This take
has no detected ChArUco corners at frames 0, 100, 190 or 200. Its current
Viewer `cloud/` uses **borrowed** constant range adjustments of 11.2 mm
(`kinect_0`) and 6.9 mm (`kinect_1`) from the *later* board + hand take;
`cloud/transfer_provenance.json` identifies this as experimental, and the
unmodified cloud is backed up in `cloud_original_before_bias_transfer/`.
It is not a validated calibration of `test1`.

Approximately XY-matched tabletop points in `test1` already have a
2.7–3.0 mm median absolute inter-camera height difference across frames
0/100/190/200 before transfer. With the borrowed correction this is
2.4–2.7 mm: modest improvement, not a repair. The lifted yellow cube's
view-dependent upper-surface heights at frame 100 are 188.03 / 188.68 mm
before and approximately 192.05 / 191.74 mm after; at frame 190 they
are 63.36 / 65.74 mm before and approximately 70.47 / 70.49 mm after.
These are not matched surface points or external accuracy measurements.
A global centimetre-scale camera translation would spoil static surfaces
that are already fairly close.

The plot `data/diagnostics/cloud-mirror/test1_hand_roi_views_190_200.png`
shows the *raw* depth samples from each camera separately, in broad
colour-hand ROIs, after the recorded extrinsics and the existing depth-edge
filter. The ROIs include clothing/background and the points are not
anatomical correspondences. After the transferred range correction,
directional nearest-neighbour medians in similar broad ROIs remain around
7–12 mm at frame 190 and 8–13 mm at frame 200; the reverse direction
sometimes worsens. The two cameras see opposite visible sides of thin
fingers. The current cloud builder concatenates the two views, then uses
5 mm voxels and a 40k-point frame budget; it does not infer correspondence,
visibility, a shared skin surface or a hand model. Empty-scene background
subtraction also keeps the operator's torso as foreground. These effects
explain why a small range adjustment cannot make the Viewer hand look solid.

Colour-only stereo offers a separate control: in frame 190, 13/21
automatically detected hand landmarks have colour-ray closest-approach
below 5 mm (index MCP 1.2 mm, index tip 1.3 mm); frame 200 has 12/21.
This supports the colour-camera geometry at hand height, but automatic
landmarks are not ground-truth physical surface matches. Using the
official Kinect SDK depth-to-colour transform on saved raw depth, the
projected `kinect_0` index DIP/tip pixels in frame 190 are zero; many
other landmarks differ from colour triangulation by over 10 mm.
The RGB overlay puts the automatic tip slightly below the visible
physical fingertip, so those zeros cannot isolate sensor dropout from
tracker error and depth-edge occlusion. This does **not** validate hand
surface accuracy.

A disabled live `align_depth_to_color` path has a wrong/unallocated
SDK output handle; a correctly allocated offline transform succeeds.
That live path is **not used** when recording these raw depth maps or
building the Viewer cloud, so it is not the cause here. The saved
timestamps are for colour images; subframe RGB/depth or depth/depth
phase cannot be verified from this take alone.

There is a distinct downstream issue: `perception/geometry.py` and
`perception/observations.py` project colour hand landmarks to depth at
an assumed 1.0 m. In frame 190 the colour-triangulated hand is roughly
0.57–0.68 m away. For the 13 low-gap landmarks, that assumption shifts
the depth pixel by median 9.8 / 10.3 pixels in the two cameras
(frame 200: 6.3 / 9.1 pixels). It does **not** affect the Viewer cloud,
but it can corrupt extracted 3-D hand poses and must be fixed before
using these episodes for orientation or retargeting.

**Revised conclusion:** The board recordings prove a scene/time-dependent
raw-depth range discrepancy; `test1` has no own board estimate. Separately,
naive fusion of opposing, incomplete hand surfaces leaves the finger
geometry visually split. Neither observation supports mirroring a camera
or adding a fixed extrinsic shift. The physical source of the range
discrepancy (drift, interference, multipath or target response) remains
unidentified. Do not bake borrowed biases into future calibration presets
or training labels.

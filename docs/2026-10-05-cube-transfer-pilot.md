# Blue-cube transfer: capture protocol and first cloud audit

Recorded 2026-10-05 on `v0.0.2`. The first pilot is
`data/datasets/40mm-pnp/2026-10-05_14-15-39` (`sample-1`, 267 frames at
approximately 30 fps). Raw RGB, depth, camera timestamps, and the `table-v1`
calibration snapshot were preserved. All distances below are internal
consistency checks, not externally measured 3-D accuracy.

## Operator's intended task and layout

- Table: 1200 × 700 mm; cube edge: 40 mm.
- Start: blue cube on the red cube. Goal: move blue onto the green cube.
- Red and green cubes are nominally 400 mm apart. The coloured-cube X
  positions in the first static frame differ by about 398–403 mm, consistent
  with this layout; that is an image/depth estimate, not a tape measurement.
- The reference for the 250 mm and 400 mm edge offsets is the **camera-view
  upper-right corner of the red cube**, not the two cube centres. The 250 mm
  offset is from the operator's left short side of the table. The 400 mm
  offset is from a perpendicular long side; whether this is the near or far
  side remains unconfirmed. The operator's left is the cameras' right.

Do not silently turn either unconfirmed edge convention into a robot/simulator
coordinate. Record the chosen table corner and near/far edge explicitly before
using this layout as metric simulator ground truth.

## What the new recording actually checks

The two Kinect colour device timestamps are paired within 122–167 µs
(median 155 µs) on all 267 saved groups. Both streams have 267 video frames
and 267 depth maps. Their only repeated startup frame and two skipped capture
cycles occur in **both** streams at the same indices. No camera-specific
one-frame slip appears during the transfer. `raw/sync_stats.json` reports
host-arrival drift; that field is not an inter-camera wired-sync measure.

The `table-v1` pre-recording validator reported green, with a 3.22 mm
nearest-neighbour median and 2.67 mm ICP translation. Its own report warns
that the NN median is below the presumed sensor noise and may be a metric
bug. More importantly, this is an **empty-scene/table-overlap** check. It does
not certify the moving hand, elevated objects, colour-to-depth alignment, or
solid-cube geometry. On the pilot's shared tabletop bins, the signed height
difference has medians about -1.5 to -2.8 mm across sampled frames, with a
tail near -10 mm and sparse regions that differ more. A single rigid camera
offset would spoil the better-aligned parts of the scene.

For corresponding MediaPipe hand landmarks, the two **colour** camera rays
have median closest-approach gaps of 2.2 mm at frame 90, 4.9 mm at frame 160,
and 3.4 mm at frame 180. This supports the colour extrinsics when the detector
finds the same anatomical landmarks; it is not an accuracy benchmark. Around
frames 131–150, detection is intermittent and, when present in both views,
median gaps jump to roughly 70–180 mm. This is a separate hand-landmark
association/occlusion failure during transfer, not evidence of a 33 ms
camera-pair slip. It will affect hand extraction even though Viewer cloud
construction does not use MediaPipe landmarks.

At frame 180, colour-ray-compatible index and middle knuckle landmarks give
raw-depth-minus-colour-triangulation range residuals of 23.3/33.2 mm in
`kinect_0`, versus 13.3/22.8 mm in `kinect_1`. Their between-camera
differences are about 10 mm, comparable to the operator's visible 10–20 mm
hand-region split. These are *landmark-pixel depth residuals*, not the error
of a matched physical skin point: opposite views see different skin patches,
and fingers have depth edges/occlusions. The data establishes that the split
exists upstream of cloud merging; it does not identify one universal depth
bias or prove which camera is physically correct.

The cloud builder uses the saved Kinect depth and calibration, then simply
concatenates and voxel-decimates the two visible surfaces. It neither matches
finger surfaces nor models occlusions or sensor uncertainty. The default
50 mm background tolerance can also remove parts of a 40 mm cube; that is a
separate object-completeness problem, not the cause of the finger offset.

## Decision and next control

Preserve the raw pilot and any new takes. Do **not** flip axes, swap cameras,
apply the previous episode's scalar depth correction, or claim the green
empty-scene verdict proves hand-height accuracy. No correction was applied
to `sample-1` here. The immediate blocker for metric hand/object labels is a
rigid, known target visible to both RGB and depth cameras **at hand height**,
sampled at several poses/tilts. Its colour corners and depth surfaces can
separate extrinsic error, per-camera range error, and occlusion. Until that
test, treat the two-view hand cloud as visualization, gate out badly matched
hand-landmark frames, and do not export the pilot as trusted robot-training
geometry.

## End-to-end cloud trace

Read-only follow-up on frame 180:

1. Both RGB videos and both depth folders contain 267 indexed frames. The
   colour device timestamps remain paired within 122–167 µs (median 155 µs).
   There is no camera-specific one-frame slip during transfer.
2. For 1,000 sampled depth pixels per camera, the vectorised cloud XYZ differs
   from direct `libk4a` depth-to-colour 3-D conversion by at most 0.00051 mm
   (`kinect_0`) / 0.00031 mm (`kinect_1`) after saved extrinsics. The
   factory projection is not applied with a wrong axis or scale.
3. Rebuilding frame 180 with the recorded settings reproduces all 35,176
   points and the saved `cloud/000180.bin` **byte for byte**. The HTTP
   endpoint serves the identical bytes. The Viewer receives the unchanged
   `T_world_display` (rotation determinant +1) and applies it to both
   cameras together. No extra relative shift appears in cloud writing or UI.

The blue 40 mm cube is a rigid check during the same transfer. The two
cameras' colour-selected top-surface Z p90 values differ by 1, 1, 4, 0,
and 3 mm at frames 120, 135, 145, 155, and 165 (rounded to mm). At frame
0, blue-on-red measures 81.8 / 79.8 mm above the local table in
`kinect_0` / `kinect_1`; at frame 260, blue-on-green measures 81.4 /
81.1 mm. Nominal height is 80 mm. These are self-consistency checks of
visible surfaces, not external ground truth. They rule out a *global*
10–20 mm camera transform or depth-range offset as the repair for this take.
The earlier board-derived Kinect range bias belonged to other sessions.

The cameras view the frame-180 hand from directions about 55° apart.
For 5 mm XY cells within a broad skin-coloured hand ROI, signed height
difference `kinect_0 - kinect_1` has median -0.3 mm, p10 -23.4 mm,
p90 +16.5 mm. A smaller shared dorsal-palm patch has median -4.6 mm.
The difference changes sign across the hand, not like one rigid offset.
These cells are not physical point correspondences: fingers, occlusions
and view-dependent skin patches are mixed. Turning off the depth-edge
filter restores only 2–4% of skin-coloured points in this frame. The
cloud builder concatenates two partial visible surfaces, then applies
5 mm voxel thinning; it does not model visibility or reconstruct one
hand surface. That is the demonstrated proximate cause of the doubled
or locally reversed hand contours. Local Kinect depth/edge errors may
contribute, but this pilot cannot assign their exact fraction.

Two separate cloud/display faults make the cubes look worse:

- `CLOUD_BG_TOLERANCE_MM = 50` is larger than the cube edge. On frame 0,
  the red-cube ROI retains 0 / 10 stride-2 points from `kinect_0` /
  `kinect_1`; a read-only 20 mm test retains 99 / 108. The green ROI
  changes from 14 / 0 to 44 / 86. This deletes real object surfaces,
  not just background.
- The synthetic ChArUco display plane sits at Z=0, while local table
  samples are Z=-12 to -18 mm. It can visually intersect a lower cube;
  it does not cause the hand split.

No raw recording, calibration, canonical cloud or runtime configuration
was changed. A fixed camera shift or borrowed depth correction is
contradicted by the rigid-cube controls. Making one useful hand surface
requires visibility-aware reconstruction or a hand model, not another
coordinate transform. A known rigid target held still at hand height in
both RGB and depth views would isolate any remaining physical Kinect
range contribution.

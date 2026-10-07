# Calibration/cloud Git-history audit — 2026-10-05

## Question and scope

Could a prior version of the calibration or point-cloud path correctly align
the current two-Kinect recordings? This audit compares Git changes with the
saved `table-v1` calibration (16 ChArUco capture sets) and two new RGB-D board
episodes. No recorded depth, active preset, or canonical Viewer cloud was
changed. All millimetre values below are **internal consistency** measurements,
not externally validated geometric accuracy.

## Relevant history

| Commit | Change | Effect on XYZ |
|---|---|---|
| `0fb387b` | Initial cloud stage | Raw depth pixel through K4A SDK depth-to-colour 3-D, then recorded camera extrinsics |
| `31cb733` | Cached affine depth-to-colour ray maps | Intended same SDK XYZ, faster; current direct-SDK spot check differs by at most 0.00051 mm for 1,000 sampled pixels per camera in the pilot |
| `24a247a` | Empty-scene subtraction, default 50 mm | Removes points; cannot translate remaining points, but can delete 40 mm cubes |
| `298b1b0` | Global planar PnP | Replaced an earlier board-pose ambiguity that had caused roughly 50 mm of rigid two-cloud offset |
| `39c65ce` | Joint multi-pose bundle | Couples camera and board poses instead of two independent solves |
| `d2f8b0c` | Separate display/world anchor | A common Viewer transform, not a per-camera cloud transform |
| `ffc7cd9` | Mixed-depth-edge rejection and exact colour warp | Removes edge points; exact warp affects RGB assignment, not XYZ |
| `1b399d3` | Exact K4A colour rays in the bundle | Removes the pinhole-vs-factory-lens mismatch; cloud XYZ deprojection is otherwise unchanged |

There are no committed `cloud.py`, `bundle.py`, or `k4a_offline.py` geometry
changes after `1b399d3` through the current HEAD. The current uncommitted
`cloud.py` additions permit an **opt-in** scalar depth-bias comparison in a
separate directory; the default bias is zero. None of those changes introduces
a per-camera translation in the normal cloud build. Its final fusion remains
concatenation followed by cropping, voxel thinning, and a point-count cap.

## Historical calibration replay

The current joint bundle solver was run on the *same* 16 stored `table-v1`
capture sets **without** K4A ray projectors. That reproduces the pre-`1b399d3`
zero-distortion pinhole observation model while keeping the other bundle logic
and input fixed. The resulting `kinect_1` pose differs from the active exact-ray
pose by **16.90 mm / 0.506°**. The pinhole solve's internal reprojection RMS
is 1.20 / 2.31 px for cameras 0/1; its independent stereo check disagrees by
4.17 mm / 0.529°. This is a model replay, not a checkout of the entire old app.

On held-out RGB-D board recordings, each visible ChArUco corner was triangulated
from the two exact SDK colour rays. The RGB-ray gap and independently sampled
raw-depth point-pair distance were evaluated with each candidate extrinsic
matrix. Frames were sampled every 15 indices: 6 with the board on the table,
25 with the board raised/moved. Values are medians of per-frame medians, in mm.

| Board location | Calibration | RGB-ray gap | Raw depth-pair gap |
|---|---|---:|---:|
| Table, 6 frames | Old pinhole replay | 1.679 | 14.443 |
| Table, 6 frames | Current exact-ray | **0.161** | **7.444** |
| Raised/moving, 25 frames | Old pinhole replay | 0.722 | **8.604** |
| Raised/moving, 25 frames | Current exact-ray | **0.111** | 12.360 |

The old pose can make a *raised depth surface* look less split by compensating
some of the actual depth-range error with a wrong camera pose. It does **not**
solve it: 8.6 mm remains, RGB agreement is 6.5 times worse on those frames,
and the same pose doubles the raw depth-pair error on the table. Reverting it
would exchange one inconsistent workspace region for another. The older
single-pose PnP and pre-bundle paths are less constrained still; their known
offsets motivated the 2026-09-03 fixes.

## What remains unexplained by Git

With the current exact-ray calibration, the two RGB views of the board agree
within a median **0.11–0.16 mm** ray gap, yet the raw depth surfaces disagree
by **7.4 mm** on the table and **12.4 mm** for the raised board. Depth-vs-RGB
range residuals change with board position: about +8.5/+6.7 mm (Kinect 0/1)
on the table and +12.6/+9.7 mm raised. These are direct measurements from the
recorded SDK depth values before cloud concatenation; switching cloud or bundle
versions does not repair the depth measurements.

The pilot transfer has camera-specific missing/biased hand-depth patches and
opposite-view occlusions. Board-derived range offsets cannot be applied to
skin as a proven correction: the previous scalar experiment and a spatial
board-fit experiment improve some rigid patches while worsening others. The
current 50 mm background tolerance also deletes real 40 mm cube surfaces,
which is a separate point-completeness fault. Neither a Git rollback nor a
rigid ICP per frame is a defensible general fix for the doubled finger.

## Decision

Do **not** restore an older calibration/cloud implementation or activate the
old pinhole pose. Keep the original raw recordings and the current exact-ray
calibration. Diagnose depth validity in the actual hand volume, with a rigid
target at several positions and both cameras at a reliable working range;
retain per-camera depth/timing evidence and assess point visibility before
merging. Any depth correction must beat independent held-out measurements
across table, raised target, cube, and hand regions before becoming a default.

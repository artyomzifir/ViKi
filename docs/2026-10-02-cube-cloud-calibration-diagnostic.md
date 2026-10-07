# Two-Kinect cube clouds: table plane, background subtraction, and frame pairing

Measured 2026-10-02 on branch `v0.0.2` using the eight 2026-09-29 takes in
`data/datasets/40mm-new-2kinects/`. Detailed spatial measurements below use
`2026-09-29_11-30-58`, frames 0 and 100. The raw recordings and saved cloud
artifacts were read without modification.

## What the stored calibration supports

All eight takes snapshot preset `2-usb` and the same extrinsics and world anchor.
Its pre-recording empty-scene validation was green: symmetric nearest-neighbour
median 4.84 mm, ICP residual 3.45 mm. This is an agreement check, not physical
ground truth.

The split preset's `solve` metadata says `legacy-migrated`, `n_points: 0` because
save-as discarded the bundle statistics. A read-only in-memory re-solve of its
19 stored ChArUco sets, with the exact Kinect colour-ray models, gave RMS
0.86 px (`kinect_0`) and 1.07 px (`kinect_1`) over 1,872 observations. The
independent stereo check differed by 0.81 mm / 0.083°. The new extrinsic for
`kinect_1` differs from the stored one by only 0.11 mm / 0.008°. The original
solve did override the frame-coverage readiness criterion: measured 0.25 versus
required 0.45. That weakens confidence away from the observed board positions,
but the current evidence does not show a gross rigid transform error.

The recorded tabletop points corroborate local agreement. At frame 0, in the
shared central 0.4 × 0.3 m area, 6,387 point pairs within 10 mm in XY have a
median absolute height difference of 2.7 mm and p90 of 5.4 mm. At frame 100,
the respective figures are 3.8 and 8.7 mm, with more occlusion.

## Why the floor cuts the cubes

The saved `world_anchor.json` puts the ChArUco board at display Z=0. On raw
depth reconstructed with the episode's exact Kinect calibrations, the table
centre is at Z=-12.6 mm from `kinect_0` and -11.2 mm from `kinect_1`. The
estimated planes tilt about 1.4° relative to the board plane; their X slopes
differ by about 1.1°. At the red, green, and blue cube locations the measured
table lies roughly 9–14 mm below Z=0. The red cube top is near Z=+28 mm, so
the observed height above the table is about 41 mm. This is consistent with the
cube's nominal 40 mm height. The Viewer renders a semitransparent ChArUco board
mesh at Z=0; where the cube lies within that board footprint, the mesh visibly
intersects its lower centimetre. The world-space crop starts at Z=-0.8 m, so
the crop is not responsible.

The larger loss occurs earlier, during foreground cloud construction.
`_camera_samples` drops a depth pixel when its depth differs from the saved
empty-scene background by at most `CLOUD_BG_TOLERANCE_MM = 50`. This tolerance
is larger than the cube height. With exact colour/depth association, at frame 0:

| Camera / cube | Coloured raw points | Kept at 50 mm | Kept at 20 mm |
|---|---:|---:|---:|
| `kinect_0` red | 924 | 72 | 803 |
| `kinect_0` green | 651 | 264 | 565 |
| `kinect_0` blue | 574 | 283 | 519 |
| `kinect_1` red | 592 | 275 | 520 |
| `kinect_1` blue | 796 | 0 | 637 |

Colours were selected by RGB dominance and a 45 mm XY radius around each cube,
with 0 < Z < 80 mm. These counts are a diagnostic, not full object segmentation.
For example, the red cube's median depth difference from the empty plate is
46 mm in `kinect_0`, so the 50 mm threshold deletes most of it.

Reducing the tolerance has a cost. Among points selected as tabletop in the
central area of frame 0, a 20 mm threshold would retain 2.7% (`kinect_0`) and
3.4% (`kinect_1`) as false foreground; 10 mm would retain about 14.5% from
each camera. Thus 20 mm is the first sensible A/B setting for these 40 mm cubes,
not a universal replacement for the default. The stored `cloud/*.bin` were
built at 50 mm and must be regenerated to show a different result.

## Temporal disagreement in the recordings

Hardware sync itself is close: well-paired frames have a median device timestamp
separation of about 145 µs. The recorder occasionally pairs frames a full
33 ms apart despite that. Device timestamps in `raw/timestamps.json` give:

| Take suffix | Pairs more than 10 ms apart |
|---|---:|
| `11-30-58` | 16/252 (6.3%) |
| `11-31-13` | 9/250 (3.6%) |
| `11-31-26` | 10/235 (4.3%) |
| `11-31-40` | 42/288 (14.6%) |
| `11-31-55` | 12/216 (5.6%) |
| `11-32-07` | 19/223 (8.5%) |
| `11-32-21` | 7/197 (3.6%) |
| `11-32-31` | 42/207 (20.3%) |

This can double or skew a moving hand/cube in fused frames. It cannot explain
the missing static cube surfaces in frame 0.

## Practical next steps

1. For a visual A/B test, rebuild **one** take's cloud with
   `build_cloud(ep, bg_tol_mm=20)` and compare cube surfaces and table noise
   against the saved 50 mm cloud. Save the old artifact if a reversible
   comparison is wanted; `build_cloud` overwrites `cloud/*.bin`.
2. In the Viewer, hide the `ChArUco board` layer to distinguish visual
   occlusion from missing points. A measured per-episode table plane should
   eventually replace the board plane as the floor reference.
3. Pair or reject frames using device timestamps before relying on fused
   moving geometry. The cameras' hardware sync is already sufficiently close.
4. Re-capture the world anchor with the board flush on the tabletop if Z=0 is
   intended to represent the table. Collect a wider distribution of board
   positions for the next extrinsic calibration; the present forced coverage
   remains a quality caveat, even though observed tabletop agreement is good.

None of these measurements is an external accuracy benchmark. They compare
observations inside the same recording and calibration.

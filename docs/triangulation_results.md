# Multi-view triangulation — A/B on the `pick_up` episode

Episode `new-dataset/2026-09-03_15-52-12` (task `pick_up`, 898 frames, 2×Kinect).
Detector: `rtmpose-m-hand5` on the GPU (~11 ms/frame). Same episode, same
`hand_fit` config, only `PERCEPTION_FUSE_MODE` flipped.

> **Caveat:** this episode was recorded **before** the Kinect wired hardware sync
> was enabled — inter-camera skew P95 ≈ 21 ms (`docs/sync_stage0.md`), ~21 mm of
> spatial error at 1 m/s. Triangulation runs *handicapped* here. A fresh
> hardware-synced take (P95 2.5 ms) should widen the gap.

## Results

| metric | `xyz_mean` (legacy) | `triangulate` | Δ |
|---|---|---|---|
| hand_fit median point→capsule residual | 10.8 mm | **9.9 mm** | −0.9 mm (−8 %) |
| hand_fit p90 residual | 26.7 mm | **24.3 mm** | −2.4 mm (−9 %) |
| warm-start jerk (landmark skeleton) | 9.9e-3 | **9.4e-4** | **−90 %** |
| fitted-trajectory jerk | 8.5e-2 | 5.6e-2 | −35 % |
| hand_fit energy: data (cloud) term | 12.0 % | **19.2 %** | cloud gets more say |
| hand_fit empty-frame fraction | 11.8 % | 50.1 % | +38 pp — see below |
| mean ROI points / frame | 352 | 198 | fewer |
| triangulation reprojection error | — | median **0.94 px** / P95 **2.6 px** | — |
| joints by #views (0 / 1 / 2 / 3+) | — | 42 / 12048 / 4563 / 0 | — |

## Reading it

- **The residual improved (−8 % median, −9 % p90) even handicapped**, and the
  landmark-skeleton jerk dropped **10×**. The triangulated joints are far more
  frame-to-frame consistent than an average of two monocular reconstructions, so
  `hand_fit` stops fighting the anchor and trusts the depth cloud more (data
  term 12 % → 19 % of the energy). This is the mechanism the task predicted.
- **Triangulation geometry is sound**: reprojection P95 2.6 px. Calibration +
  the undistort-once discipline + DLT are correct.
- **`empty_frame_fraction` jumped to 50 %** because triangulation only produces a
  joint where *both* cameras saw the hand, and on this take `kinect_0` detected
  the hand in only ~300 / 898 frames vs `kinect_1`'s ~715 — `rtmpose` has no
  handedness label and mis-locks on that camera's wider framing. So the gain is
  real but covers only ~half the timeline; the rest falls back to gaps. That is
  a **detector** limitation (out of scope — use `mediapipe`, or reframe
  `kinect_0`), not a triangulation one.

## Next

Re-record `pick_up` with the pair hardware-synced and the hand kept inside both
cameras' overlap, then re-run this A/B. Expect a larger residual drop and full
timeline coverage.

# Multi-view triangulation — A/B results

## Clean run (hardware-synced, full coverage) — `2026-09-03_16-58-45` (`pick_up_u`)

898 frames, 30 fps, 2×Kinect **wired hardware sync** (inter-camera gap median
0.5 ms, P95 5.4 ms). Detector **`mediapipe`** (handedness + tracker → the hand is
seen by *both* cameras in 807/898 frames = 90 %). Same `hand_fit` config, only
`PERCEPTION_FUSE_MODE` flipped.

| metric | `xyz_mean` (legacy) | `triangulate` | Δ |
|---|---|---|---|
| valid frames | 86 % | **99 %** | +13 pp |
| gap runs (count / longest) | 23 / 36 fr | **2 / 11 fr** | — |
| hand_fit empty-frame fraction | 14 % | **1 %** | — |
| hand_fit median point→capsule residual | 9.6 mm | **9.0 mm** | −6 % |
| hand_fit p90 residual | 26.1 mm | **22.5 mm** | −14 % |
| **fitted wrist step, median** | 13.9 mm/fr | **6.2 mm/fr** | **−55 %** |
| fitted wrist step, p95 | 59.8 mm/fr | **31.9 mm/fr** | −47 % |
| fitted wrist step, max | 191.5 mm/fr | **77.8 mm/fr** | −59 % |
| **fitted wrist jerk, RMS** | 2601 | **1327** | **−49 %** |
| fitted wrist jerk, p95 | 5441 | 2925 | −46 % |
| warm-start (landmark) jerk | 1.4e-2 | **1.1e-3** | −92 % |
| hand_fit energy: data (cloud) term | 7 % | **20 %** | — |
| triangulation reprojection error | — | median 1.3 px / P95 3.1 px | — |
| joints by #views (0 / 1 / 2 / 3+) | — | 0 / 1911 / 16947 / 0 | — |

### Reading it

An **unambiguous, large win** on the clean setup:

- The fitted **wrist trajectory frame-to-frame step is halved** (median 13.9 →
  6.2 mm/fr; p95 −47 %; max −59 %) and its **jerk RMS is down 49 %**. The
  trajectory is genuinely smoother — 99 % of frames are valid, only two short
  gaps, so this is real, not the interpolation artifact of the earlier run.
- The point→capsule residual improves (−6 % median, −14 % p90) and
  `hand_fit`'s empty-frame fraction collapses 14 % → 1 % — it sits on the depth
  cloud across almost the whole take.
- The cloud (data) term rises from 7 % to 20 % of `hand_fit`'s energy: the
  triangulated landmark anchor is consistent enough that the optimiser stops
  fighting it and lets the cloud decide the pose.
- Triangulation geometry is sound: reprojection P95 3.1 px over 16947 two-view
  joints.

### What it took to get here (all committed)

1. **`8ac8e17`** — GPU inference back on (probe graph was IR-13; onnxruntime
   1.23 rejects it). extract 30 min → ~30–75 s.
2. **`ab252f7`** — Kinect wired sync: master is **`kinect_1`**, subordinate
   `kinect_0` (SDK cable-presence check fails on both when reversed). Inter-camera
   skew 20–31 ms → **<1 ms**.
3. **`cb96a5f`** — `MultiCameraSync.max_offset_us` default was 25 ms, *below* one
   frame period, so `nearest_to`'s legitimate half-period-plus-jitter frames were
   rejected — a 30 s take came out 22 s / 660 frames. Now `1.5 / sync_fps`.
4. **`804becf`** — `triangulate_episode` grouped observations by
   `host_timestamp_us` (carries each camera's per-frame offset → never matches
   across cameras). Now groups by `frame_index`.
5. **Detector**: `rtmpose-m-hand5` has no handedness label and detects the hand
   in only ~13 % of two-view frames on this rig — it was the reason the first A/B
   looked jittery (gap-dominated). `mediapipe` (handedness + tracker) → 90 %
   two-view coverage. **Recommend `POSE_BACKEND = mediapipe` for this rig.**

---

## First run (software-synced, rtmpose) — superseded, kept for context

Episode `2026-09-03_15-52-12`, recorded before the wired sync, detector
`rtmpose-m-hand5`. Inter-camera skew P95 ≈ 21 ms; two-view coverage only ~13 %
of frames (rtmpose one-camera dropout). The scalar `hand_fit` metrics improved
(median resid 10.8 → 9.9 mm, warm-start jerk −90 %) but that was largely a
selection / over-smoothing artifact: `empty_frame_fraction` jumped 12 → 50 %, so
half the trajectory was unconstrained spline fill (one gap ran 111 frames). On
the per-frame graphs it read as *worse* — frozen stretches and discontinuities.
The clean run above is the real comparison.

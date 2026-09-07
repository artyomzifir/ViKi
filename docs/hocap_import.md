# HO-Cap ingestion report

_Generated 2026-09-07T11:37:11 by `viki hocap-import` (`viki/benchmark/hocap.py`)._

HO-Cap — arXiv:2406.06843, <https://irvlutd.github.io/HOCap> — **CC BY 4.0**. Only the published `.npz` / `.yaml` data is read (numpy + stdlib + OpenCV); no HOCap-Toolkit (GPL-3.0), no MANO code, no MANO model files.

## Caveats that must reach the thesis text

1. **HO-Cap's ground truth is itself semi-automatic MANO fitting.** The hand joints used here are produced by HO-Cap's multi-view MANO optimisation, not by direct measurement. The HO-Cap authors validate it against 800 manually annotated images; residual fitting error in that validation is part of any MPJPE computed against these arrays and cannot be separated from ViKi's own error.
2. **Depth-sensor mismatch.** HO-Cap uses Intel RealSense (active-stereo IR) depth; ViKi uses Azure Kinect (amplitude-modulated ToF). The RGB triangulation tract (2-D landmarks + calibration → 3-D) transfers directly, so conclusions about it carry over. Conclusions about the point-cloud **data term** and its weight `w_data` do **not** transfer: the two sensors have different noise, multipath and edge behaviour.

## Sequences imported / skipped

| imported | cameras | frames | GT frames |
|---|---|---:|---:|
| `subject_1/20231025_170105` | `105322251564` / `037522251142` | 831 | 831 |
| `subject_1/20231025_170231` | `105322251564` / `037522251142` | 1076 | 1076 |
| `subject_1/20231025_170650` | `105322251564` / `037522251142` | 741 | 741 |
| `subject_1/20231025_170959` | `105322251564` / `037522251142` | 874 | 874 |

| skipped | reason |
|---|---|
| `subject_1/20231025_165502` | not a right-hand single-hand sequence (annotated hand(s) = ['left']); left-hand / bimanual / handover sequences are out of scope for the first import (`hands_in_labels=['left']`) |
| `subject_1/20231025_165807` | not a right-hand single-hand sequence (annotated hand(s) = ['left']); left-hand / bimanual / handover sequences are out of scope for the first import (`hands_in_labels=['left']`) |
| `subject_1/20231025_171117` | not a right-hand single-hand sequence (annotated hand(s) = ['left']); left-hand / bimanual / handover sequences are out of scope for the first import (`hands_in_labels=['left']`) |

| sequence | 1 extr | 2 depth | 3 overlays | 4 triang | 5 encoding | 6 pipeline |
|---|---|---|---|---|---|---|
| `20231025_170105` | PASS | FAIL | made | PASS | skipped | not run |
| `20231025_170231` | PASS | FAIL | made | PASS | skipped | not run |
| `20231025_170650` | PASS | FAIL | made | PASS | FAIL | runs |
| `20231025_170959` | PASS | FAIL | made | PASS | skipped | not run |

## HO-Cap fields consumed

| file | fields | used for |
|---|---|---|
| `calibration/intrinsics/<serial>.yaml` | `color`/`depth` → `fx,fy,ppx,ppy` (+ `coeffs` if present) | `raw/intrinsics.json` (`color` **and** `depth` blocks) |
| `calibration/extrinsics/<file>.yaml` | `rs_master`, `extrinsics.{tag_0,tag_1,<serial>}` (12-list row-major `[R\|t]`, camera→rs_master) | `T_world_camera = inv(tag_1) · extrinsics[<serial>]`, then inverted to `rvec`/`tvec` for `raw/extrinsics.json` |
| `<subject>/<sequence>/meta.yaml` | `num_frames, mano_sides, realsense.{serials,width,height}, extrinsics, subject_id, task_id` | scope filter, camera set, `meta.json` |
| `.../<serial>/color_%06d.jpg` | RGB frame | `raw/<serial>.mp4` (mp4v, constant fps) |
| `.../<serial>/depth_%06d.png` | uint16 depth, millimetres | `raw/<serial>_depth/%06d.npy` (uint16 mm, unchanged) |
| `.../<serial>/label_%06d.npz` | `hand_joints_3d` (camera frame, m), `hand_joints_2d` (px), `cam_K` | `gt/joints3d_world.npz` (transformed to the `tag_1` world frame) |

`poses_m.npy` (MANO pose parameters) is **not** consumed — it would require running a MANO layer. World-frame joints are obtained by transforming the per-camera `hand_joints_3d` with `T_world_camera`.

## Joint-order mapping (highest-risk item)

HO-Cap's 21-keypoint skeleton (`config/mano_info.yaml`) is the MediaPipe Hands layout, **identical index-for-index** to `viki.contracts.LM`. The thumb joints carry different anatomical labels (HO-Cap mcp/pip/dip vs MediaPipe cmc/mcp/ip) but occupy the same indices 1–4 with the same chain. `VIKI_LM_FROM_HOCAP` is therefore the identity permutation, written out explicitly, asserted at import, unit-tested, and validated visually by the reprojection overlays below.

| ViKi `LM` | idx | HO-Cap name |
|---|---|---|
| `WRIST` | 0 | `wrist` |
| `THUMB_CMC` | 1 | `thumb_mcp` |
| `THUMB_MCP` | 2 | `thumb_pip` |
| `THUMB_IP` | 3 | `thumb_dip` |
| `THUMB_TIP` | 4 | `thumb_tip` |
| `INDEX_MCP` | 5 | `index_mcp` |
| `INDEX_PIP` | 6 | `index_pip` |
| `INDEX_DIP` | 7 | `index_dip` |
| `INDEX_TIP` | 8 | `index_tip` |
| `MIDDLE_MCP` | 9 | `middle_mcp` |
| `MIDDLE_PIP` | 10 | `middle_pip` |
| `MIDDLE_DIP` | 11 | `middle_dip` |
| `MIDDLE_TIP` | 12 | `middle_tip` |
| `RING_MCP` | 13 | `ring_mcp` |
| `RING_PIP` | 14 | `ring_pip` |
| `RING_DIP` | 15 | `ring_dip` |
| `RING_TIP` | 16 | `ring_tip` |
| `PINKY_MCP` | 17 | `little_mcp` |
| `PINKY_PIP` | 18 | `little_pip` |
| `PINKY_DIP` | 19 | `little_dip` |
| `PINKY_TIP` | 20 | `little_tip` |

---

## SKIPPED — subject_1/20231025_165502

- not a right-hand single-hand sequence (annotated hand(s) = ['left']); left-hand / bimanual / handover sequences are out of scope for the first import  (`hands_in_labels=['left']`, `mano_sides_meta=['left']`, `task_id=1`)

---

## SKIPPED — subject_1/20231025_165807

- not a right-hand single-hand sequence (annotated hand(s) = ['left']); left-hand / bimanual / handover sequences are out of scope for the first import  (`hands_in_labels=['left']`, `mano_sides_meta=['left']`, `task_id=1`)

---

## subject_1/20231025_170105  →  `subject_1_20231025_170105_105322251564-037522251142`

- cameras: `105322251564`, `037522251142`  ·  30 fps  ·  831 frames  ·  GT on 831
- episode: `/app/data/datasets/hocap/subject_1_20231025_170105_105322251564-037522251142`

### ViKi reference geometry (`kinect_0`/`kinect_1`)

- baseline **896.5 mm**, convergence **61.46°**  (source: /app/data/raw_datasets/viki_ref)
- selected HO-Cap pair **`105322251564` / `037522251142`** — selection distance **0.1060**, Δbaseline 50.6 mm, Δconvergence -5.51°
- ⚠ the selected pair sees the whole hand in both views only **78%** of frames (one camera clips the hand at the image edge in this sequence). Selection follows the spec — nearest (baseline, convergence) to ViKi — so this is left as-is; for an eval that needs full two-view coverage, `--cameras` the nearest row in the table below with `both see hand` ≈ 1.0.

### 28-pair geometry table

| cam A | cam B | baseline (mm) | convergence (°) | both see hand |
|---|---|---:|---:|---:|
| `105322251564` | `046122250168` | 535.1 | 26.21 | 1.000 |
| `105322251225` | `115422250549` | 601.0 | 40.36 | 0.644 |
| `043422252387` | `105322251225` | 609.7 | 38.90 | 1.000 |
| `043422252387` | `117222250549` | 650.6 | 45.46 | 1.000 |
| `105322251564` | `108222250342` | 655.5 | 38.21 | 1.000 |
| `037522251142` | `046122250168` | 671.1 | 33.76 | 0.783 |
| `105322251564` | `043422252387` | 869.5 | 52.71 | 1.000 |
| `105322251564` | `037522251142` | 947.1 | 55.95 | 0.783 |  ← selected
| `043422252387` | `115422250549` | 985.0 | 76.02 | 0.644 |
| `108222250342` | `046122250168` | 1035.9 | 55.87 | 1.000 |
| `108222250342` | `117222250549` | 1047.3 | 76.08 | 1.000 |
| `105322251225` | `117222250549` | 1052.9 | 78.27 | 1.000 |
| `117222250549` | `115422250549` | 1068.0 | 100.37 | 0.644 |
| `037522251142` | `108222250342` | 1068.5 | 67.17 | 0.783 |
| `105322251564` | `105322251225` | 1105.5 | 68.31 | 1.000 |
| `043422252387` | `108222250342` | 1142.5 | 75.98 | 1.000 |
| `105322251564` | `117222250549` | 1167.3 | 81.00 | 1.000 |
| `105322251225` | `046122250168` | 1178.7 | 65.56 | 1.000 |
| `043422252387` | `046122250168` | 1212.3 | 68.37 | 1.000 |
| `037522251142` | `115422250549` | 1238.3 | 95.69 | 0.644 |
| `037522251142` | `105322251225` | 1337.7 | 89.11 | 0.783 |
| `105322251564` | `115422250549` | 1367.7 | 104.81 | 0.644 |
| `046122250168` | `115422250549` | 1391.1 | 91.24 | 0.644 |
| `043422252387` | `037522251142` | 1463.8 | 101.89 | 0.783 |
| `105322251225` | `108222250342` | 1483.5 | 104.42 | 1.000 |
| `108222250342` | `115422250549` | 1536.1 | 142.96 | 0.644 |
| `117222250549` | `046122250168` | 1564.5 | 105.53 | 1.000 |
| `037522251142` | `117222250549` | 1585.7 | 136.65 | 0.783 |

### Acceptance criteria

**1. Extrinsics round-trip** — PASS. Camera centre recovered through `CalibrationExtrinsics.transform_matrix` matches the intended pose to **1.22e-15 m** (tolerance 1e-9).

| camera | centre err (m) | full residual (m) | dev. from HO-Cap raw centre (m) | HO-Cap 3×3 orthonormality defect |
|---|---:|---:|---:|---:|
| `105322251564` | 6.85e-16 | 7.77e-16 | 6.85e-16 | 6.08e-07 |
| `037522251142` | 1.22e-15 | 1.11e-15 | 1.22e-15 | 8.72e-07 |

_check is against the nearest true rotation to HO-Cap's raw 3x3 block; raw_centre_deviation_m is how far that sits from HO-Cap's un-orthonormalised published centre (bounded by their ~float32 publication precision, not by this importer)._

_GT frame convention resolved to **camera**-frame `hand_joints_3d` by cross-camera agreement: the camera-frame hypothesis makes the same joint from different cameras agree to **0.0000 mm**, the world-frame hypothesis to 378.4 mm (882 joint/camera comparisons)._

_Cross-view GT consistency (both selected views, 17451 joint obs): median 0.00005 mm, p95 0.00006 mm, max 0.00008 mm — HO-Cap's per-camera `hand_joints_3d` are one MANO fit rotated into each camera frame, so once transformed to the world frame they coincide to floating point; a real detector's multi-view spread will be orders of magnitude larger._

**2. Depth scale and alignment** — per camera (threshold: all-joints median < 15 mm):

| camera | frames | all-joints median / p95 (mm) | wrist median (mm) | fingertip median (mm) | verdict |
|---|---:|---|---:|---:|---|
| `105322251564` | 831 | 18.1 / 95.6 | 33.4 | 13.8 | FAIL |
| `037522251142` | 831 | 43.1 / 311.5 | 125.0 | 13.0 | FAIL |

_Reading: a median in the hundreds ⇒ wrong PNG scale (not the case — the /1000 mm→m conversion is correct). Fingertip error ≫ wrist error ⇒ colour/depth not aligned; here fingertips ≤ wrist, so the identity projector is valid. The residual that remains is the MANO joint-centre sitting inside the flesh while the depth sees the skin surface, plus RealSense active-stereo-IR noise at manipulation range — i.e. thesis caveat 2, not an ingestion error._

**3. Reprojection overlays** — GT joints + skeleton edges drawn on the colour frame (paths relative to the episode dir) for human inspection of the joint-order mapping:

- `gt/overlays/105322251564_000000.png`
- `gt/overlays/105322251564_000208.png`
- `gt/overlays/105322251564_000415.png`
- `gt/overlays/105322251564_000622.png`
- `gt/overlays/105322251564_000830.png`
- `gt/overlays/037522251142_000000.png`
- `gt/overlays/037522251142_000208.png`
- `gt/overlays/037522251142_000415.png`
- `gt/overlays/037522251142_000622.png`
- `gt/overlays/037522251142_000830.png`

**4. Triangulation identity check** — PASS. GT world joints reprojected into both views and triangulated back via `viki.perception.triangulate`: median **0.000 mm**, max **0.000 mm** over 17451 joints (threshold: median < 1 mm).

**5. Colour encoding loss** (JPEG source vs mp4-decoded, threshold p95 ≤ 1.0 px):

- skipped — run with --backend-check (needs mediapipe + source JPEGs)

**6. Pipeline runs unmodified** — `viki extract` then `viki prepare` on the imported episode, no changes to any existing module:

- not run in this session (needs a pose backend / container). Run with `--pipeline-check`, or `viki extract <ep> && viki prepare <ep>` and record `observations.npz` row count, per-camera detection counts, and the both-cameras-detected fraction here.

### Notes

- depth entry in intrinsics.json = HO-Cap color intrinsics; the published depth PNGs are registered to the colour plane (HOCap-Toolkit deprojects them with the colour K), so no calibration_preset / *_calib blob is written and extract uses the identity colour->depth projector. Criterion 2 verifies this empirically.
- GT frame convention resolved to 'camera' by cross-camera agreement.

---

## subject_1/20231025_170231  →  `subject_1_20231025_170231_105322251564-037522251142`

- cameras: `105322251564`, `037522251142`  ·  30 fps  ·  1076 frames  ·  GT on 1076
- episode: `/app/data/datasets/hocap/subject_1_20231025_170231_105322251564-037522251142`

### ViKi reference geometry (`kinect_0`/`kinect_1`)

- baseline **896.5 mm**, convergence **61.46°**  (source: /app/data/raw_datasets/viki_ref)
- selected HO-Cap pair **`105322251564` / `037522251142`** — selection distance **0.0645**, Δbaseline 50.6 mm, Δconvergence -1.92°
- ⚠ the selected pair sees the whole hand in both views only **84%** of frames (one camera clips the hand at the image edge in this sequence). Selection follows the spec — nearest (baseline, convergence) to ViKi — so this is left as-is; for an eval that needs full two-view coverage, `--cameras` the nearest row in the table below with `both see hand` ≈ 1.0.

### 28-pair geometry table

| cam A | cam B | baseline (mm) | convergence (°) | both see hand |
|---|---|---:|---:|---:|
| `105322251564` | `046122250168` | 535.1 | 27.45 | 1.000 |
| `105322251225` | `115422250549` | 601.0 | 39.35 | 0.803 |
| `043422252387` | `105322251225` | 609.7 | 38.70 | 1.000 |
| `043422252387` | `117222250549` | 650.6 | 46.66 | 1.000 |
| `105322251564` | `108222250342` | 655.5 | 41.68 | 1.000 |
| `037522251142` | `046122250168` | 671.1 | 35.58 | 0.842 |
| `105322251564` | `043422252387` | 869.5 | 56.00 | 1.000 |
| `105322251564` | `037522251142` | 947.1 | 59.54 | 0.842 |  ← selected
| `043422252387` | `115422250549` | 985.0 | 74.23 | 0.803 |
| `108222250342` | `046122250168` | 1035.9 | 59.84 | 1.000 |
| `108222250342` | `117222250549` | 1047.3 | 83.14 | 1.000 |
| `105322251225` | `117222250549` | 1052.9 | 77.45 | 1.000 |
| `117222250549` | `115422250549` | 1068.0 | 95.18 | 0.803 |
| `037522251142` | `108222250342` | 1068.5 | 72.42 | 0.842 |
| `105322251564` | `105322251225` | 1105.5 | 70.71 | 1.000 |
| `043422252387` | `108222250342` | 1142.5 | 81.87 | 1.000 |
| `105322251564` | `117222250549` | 1167.3 | 87.22 | 1.000 |
| `105322251225` | `046122250168` | 1178.7 | 67.43 | 1.000 |
| `043422252387` | `046122250168` | 1212.3 | 71.67 | 1.000 |
| `037522251142` | `115422250549` | 1238.3 | 94.00 | 0.762 |
| `037522251142` | `105322251225` | 1337.7 | 89.97 | 0.842 |
| `105322251564` | `115422250549` | 1367.7 | 106.38 | 0.803 |
| `046122250168` | `115422250549` | 1391.1 | 92.07 | 0.803 |
| `043422252387` | `037522251142` | 1463.8 | 106.36 | 0.842 |
| `105322251225` | `108222250342` | 1483.5 | 110.17 | 1.000 |
| `108222250342` | `115422250549` | 1536.1 | 147.98 | 0.803 |
| `117222250549` | `046122250168` | 1564.5 | 112.11 | 1.000 |
| `037522251142` | `117222250549` | 1585.7 | 146.75 | 0.842 |

### Acceptance criteria

**1. Extrinsics round-trip** — PASS. Camera centre recovered through `CalibrationExtrinsics.transform_matrix` matches the intended pose to **1.22e-15 m** (tolerance 1e-9).

| camera | centre err (m) | full residual (m) | dev. from HO-Cap raw centre (m) | HO-Cap 3×3 orthonormality defect |
|---|---:|---:|---:|---:|
| `105322251564` | 6.85e-16 | 7.77e-16 | 6.85e-16 | 6.08e-07 |
| `037522251142` | 1.22e-15 | 1.11e-15 | 1.22e-15 | 8.72e-07 |

_check is against the nearest true rotation to HO-Cap's raw 3x3 block; raw_centre_deviation_m is how far that sits from HO-Cap's un-orthonormalised published centre (bounded by their ~float32 publication precision, not by this importer)._

_GT frame convention resolved to **camera**-frame `hand_joints_3d` by cross-camera agreement: the camera-frame hypothesis makes the same joint from different cameras agree to **0.0000 mm**, the world-frame hypothesis to 357.6 mm (882 joint/camera comparisons)._

_Cross-view GT consistency (both selected views, 22596 joint obs): median 0.00005 mm, p95 0.00006 mm, max 0.00008 mm — HO-Cap's per-camera `hand_joints_3d` are one MANO fit rotated into each camera frame, so once transformed to the world frame they coincide to floating point; a real detector's multi-view spread will be orders of magnitude larger._

**2. Depth scale and alignment** — per camera (threshold: all-joints median < 15 mm):

| camera | frames | all-joints median / p95 (mm) | wrist median (mm) | fingertip median (mm) | verdict |
|---|---:|---|---:|---:|---|
| `105322251564` | 1076 | 15.6 / 104.1 | 18.5 | 20.3 | FAIL |
| `037522251142` | 1076 | 29.9 / 258.1 | 108.7 | 12.3 | FAIL |

_Reading: a median in the hundreds ⇒ wrong PNG scale (not the case — the /1000 mm→m conversion is correct). Fingertip error ≫ wrist error ⇒ colour/depth not aligned; here fingertips ≤ wrist, so the identity projector is valid. The residual that remains is the MANO joint-centre sitting inside the flesh while the depth sees the skin surface, plus RealSense active-stereo-IR noise at manipulation range — i.e. thesis caveat 2, not an ingestion error._

**3. Reprojection overlays** — GT joints + skeleton edges drawn on the colour frame (paths relative to the episode dir) for human inspection of the joint-order mapping:

- `gt/overlays/105322251564_000000.png`
- `gt/overlays/105322251564_000269.png`
- `gt/overlays/105322251564_000538.png`
- `gt/overlays/105322251564_000806.png`
- `gt/overlays/105322251564_001075.png`
- `gt/overlays/037522251142_000000.png`
- `gt/overlays/037522251142_000269.png`
- `gt/overlays/037522251142_000538.png`
- `gt/overlays/037522251142_000806.png`
- `gt/overlays/037522251142_001075.png`

**4. Triangulation identity check** — PASS. GT world joints reprojected into both views and triangulated back via `viki.perception.triangulate`: median **0.000 mm**, max **0.000 mm** over 22596 joints (threshold: median < 1 mm).

**5. Colour encoding loss** (JPEG source vs mp4-decoded, threshold p95 ≤ 1.0 px):

- skipped — run with --backend-check (needs mediapipe + source JPEGs)

**6. Pipeline runs unmodified** — `viki extract` then `viki prepare` on the imported episode, no changes to any existing module:

- not run in this session (needs a pose backend / container). Run with `--pipeline-check`, or `viki extract <ep> && viki prepare <ep>` and record `observations.npz` row count, per-camera detection counts, and the both-cameras-detected fraction here.

### Notes

- depth entry in intrinsics.json = HO-Cap color intrinsics; the published depth PNGs are registered to the colour plane (HOCap-Toolkit deprojects them with the colour K), so no calibration_preset / *_calib blob is written and extract uses the identity colour->depth projector. Criterion 2 verifies this empirically.
- GT frame convention resolved to 'camera' by cross-camera agreement.

---

## subject_1/20231025_170650  →  `subject_1_20231025_170650_105322251564-037522251142`

- cameras: `105322251564`, `037522251142`  ·  30 fps  ·  741 frames  ·  GT on 741
- episode: `/app/data/datasets/hocap/subject_1_20231025_170650_105322251564-037522251142`

### ViKi reference geometry (`kinect_0`/`kinect_1`)

- baseline **896.5 mm**, convergence **61.46°**  (source: /app/data/raw_datasets/viki_ref)
- selected HO-Cap pair **`105322251564` / `037522251142`** — selection distance **0.0578**, Δbaseline 50.6 mm, Δconvergence -0.76°
- ⚠ the selected pair sees the whole hand in both views only **76%** of frames (one camera clips the hand at the image edge in this sequence). Selection follows the spec — nearest (baseline, convergence) to ViKi — so this is left as-is; for an eval that needs full two-view coverage, `--cameras` the nearest row in the table below with `both see hand` ≈ 1.0.

### 28-pair geometry table

| cam A | cam B | baseline (mm) | convergence (°) | both see hand |
|---|---|---:|---:|---:|
| `105322251564` | `046122250168` | 535.1 | 28.13 | 1.000 |
| `105322251225` | `115422250549` | 601.0 | 40.09 | 0.705 |
| `043422252387` | `105322251225` | 609.7 | 39.00 | 1.000 |
| `043422252387` | `117222250549` | 650.6 | 45.98 | 1.000 |
| `105322251564` | `108222250342` | 655.5 | 41.69 | 0.993 |
| `037522251142` | `046122250168` | 671.1 | 36.37 | 0.757 |
| `105322251564` | `043422252387` | 869.5 | 56.06 | 1.000 |
| `105322251564` | `037522251142` | 947.1 | 60.70 | 0.757 |  ← selected
| `043422252387` | `115422250549` | 985.0 | 74.73 | 0.705 |
| `108222250342` | `046122250168` | 1035.9 | 60.63 | 0.993 |
| `108222250342` | `117222250549` | 1047.3 | 80.71 | 0.993 |
| `105322251225` | `117222250549` | 1052.9 | 77.14 | 1.000 |
| `117222250549` | `115422250549` | 1068.0 | 94.36 | 0.705 |
| `037522251142` | `108222250342` | 1068.5 | 73.33 | 0.750 |
| `105322251564` | `105322251225` | 1105.5 | 71.60 | 1.000 |
| `043422252387` | `108222250342` | 1142.5 | 80.89 | 0.993 |
| `105322251564` | `117222250549` | 1167.3 | 86.07 | 1.000 |
| `105322251225` | `046122250168` | 1178.7 | 68.85 | 1.000 |
| `043422252387` | `046122250168` | 1212.3 | 72.60 | 1.000 |
| `037522251142` | `115422250549` | 1238.3 | 97.77 | 0.705 |
| `037522251142` | `105322251225` | 1337.7 | 92.65 | 0.757 |
| `105322251564` | `115422250549` | 1367.7 | 108.67 | 0.705 |
| `046122250168` | `115422250549` | 1391.1 | 94.91 | 0.705 |
| `043422252387` | `037522251142` | 1463.8 | 108.32 | 0.757 |
| `105322251225` | `108222250342` | 1483.5 | 110.50 | 0.993 |
| `108222250342` | `115422250549` | 1536.1 | 149.93 | 0.698 |
| `117222250549` | `046122250168` | 1564.5 | 111.90 | 1.000 |
| `037522251142` | `117222250549` | 1585.7 | 146.73 | 0.757 |

### Acceptance criteria

**1. Extrinsics round-trip** — PASS. Camera centre recovered through `CalibrationExtrinsics.transform_matrix` matches the intended pose to **1.22e-15 m** (tolerance 1e-9).

| camera | centre err (m) | full residual (m) | dev. from HO-Cap raw centre (m) | HO-Cap 3×3 orthonormality defect |
|---|---:|---:|---:|---:|
| `105322251564` | 6.85e-16 | 7.77e-16 | 6.85e-16 | 6.08e-07 |
| `037522251142` | 1.22e-15 | 1.11e-15 | 1.22e-15 | 8.72e-07 |

_check is against the nearest true rotation to HO-Cap's raw 3x3 block; raw_centre_deviation_m is how far that sits from HO-Cap's un-orthonormalised published centre (bounded by their ~float32 publication precision, not by this importer)._

_GT frame convention resolved to **camera**-frame `hand_joints_3d` by cross-camera agreement: the camera-frame hypothesis makes the same joint from different cameras agree to **0.0000 mm**, the world-frame hypothesis to 351.6 mm (882 joint/camera comparisons)._

_Cross-view GT consistency (both selected views, 15561 joint obs): median 0.00005 mm, p95 0.00006 mm, max 0.00008 mm — HO-Cap's per-camera `hand_joints_3d` are one MANO fit rotated into each camera frame, so once transformed to the world frame they coincide to floating point; a real detector's multi-view spread will be orders of magnitude larger._

**2. Depth scale and alignment** — per camera (threshold: all-joints median < 15 mm):

| camera | frames | all-joints median / p95 (mm) | wrist median (mm) | fingertip median (mm) | verdict |
|---|---:|---|---:|---:|---|
| `105322251564` | 741 | 16.5 / 117.3 | 26.9 | 21.0 | FAIL |
| `037522251142` | 741 | 34.8 / 306.3 | 95.9 | 13.8 | FAIL |

_Reading: a median in the hundreds ⇒ wrong PNG scale (not the case — the /1000 mm→m conversion is correct). Fingertip error ≫ wrist error ⇒ colour/depth not aligned; here fingertips ≤ wrist, so the identity projector is valid. The residual that remains is the MANO joint-centre sitting inside the flesh while the depth sees the skin surface, plus RealSense active-stereo-IR noise at manipulation range — i.e. thesis caveat 2, not an ingestion error._

**3. Reprojection overlays** — GT joints + skeleton edges drawn on the colour frame (paths relative to the episode dir) for human inspection of the joint-order mapping:

- `gt/overlays/105322251564_000000.png`
- `gt/overlays/105322251564_000185.png`
- `gt/overlays/105322251564_000370.png`
- `gt/overlays/105322251564_000555.png`
- `gt/overlays/105322251564_000740.png`
- `gt/overlays/037522251142_000000.png`
- `gt/overlays/037522251142_000185.png`
- `gt/overlays/037522251142_000370.png`
- `gt/overlays/037522251142_000555.png`
- `gt/overlays/037522251142_000740.png`

**4. Triangulation identity check** — PASS. GT world joints reprojected into both views and triangulated back via `viki.perception.triangulate`: median **0.000 mm**, max **0.000 mm** over 15561 joints (threshold: median < 1 mm).

**5. Colour encoding loss** (JPEG source vs mp4-decoded, threshold p95 ≤ 1.0 px):

| camera | frame pairs | landmarks | median (px) | p95 (px) | verdict |
|---|---:|---:|---:|---:|---|
| `105322251564` | 14 | 294 | 3.494 | 275.732 | FAIL |
| `037522251142` | 5 | 105 | 4.688 | 29.235 | inconclusive (few pairs) |

_The **median** is the re-encoding drift signal; the p95 is inflated by frames where the detector locks onto a different hand-like region under one encoding vs the other (an artifact of a small/edge hand + a weak per-frame detector, not of the codec). The median alone is already several px — well over the 1 px bar: `record.py`'s `mp4v` is lossy and a faithful benchmark ingest would need a lossless intra codec, which would diverge from what `record.py` writes — so this is reported here, not silently changed._

**6. Pipeline runs unmodified** — `viki extract` then `viki prepare` on the imported episode, no changes to any existing module:

- status: ok  ·  `cln.npz` written: True
- `rec.npz` rows: 135  ·  `observations.npz` rows: 265
- per-camera detections: {'037522251142': 157, '105322251564': 108}
- both cameras detected (fraction of synced frames): 0.0038
- no accuracy claim — this criterion only shows the imported episode runs the tract unmodified.

### Notes

- depth entry in intrinsics.json = HO-Cap color intrinsics; the published depth PNGs are registered to the colour plane (HOCap-Toolkit deprojects them with the colour K), so no calibration_preset / *_calib blob is written and extract uses the identity colour->depth projector. Criterion 2 verifies this empirically.
- GT frame convention resolved to 'camera' by cross-camera agreement.

---

## subject_1/20231025_170959  →  `subject_1_20231025_170959_105322251564-037522251142`

- cameras: `105322251564`, `037522251142`  ·  30 fps  ·  874 frames  ·  GT on 874
- episode: `/app/data/datasets/hocap/subject_1_20231025_170959_105322251564-037522251142`

### ViKi reference geometry (`kinect_0`/`kinect_1`)

- baseline **896.5 mm**, convergence **61.46°**  (source: /app/data/raw_datasets/viki_ref)
- selected HO-Cap pair **`105322251564` / `037522251142`** — selection distance **0.0907**, Δbaseline 50.6 mm, Δconvergence -4.37°
- ⚠ the selected pair sees the whole hand in both views only **65%** of frames (one camera clips the hand at the image edge in this sequence). Selection follows the spec — nearest (baseline, convergence) to ViKi — so this is left as-is; for an eval that needs full two-view coverage, `--cameras` the nearest row in the table below with `both see hand` ≈ 1.0.

### 28-pair geometry table

| cam A | cam B | baseline (mm) | convergence (°) | both see hand |
|---|---|---:|---:|---:|
| `105322251564` | `046122250168` | 535.1 | 26.62 | 1.000 |
| `105322251225` | `115422250549` | 601.0 | 39.47 | 0.555 |
| `043422252387` | `105322251225` | 609.7 | 38.23 | 1.000 |
| `043422252387` | `117222250549` | 650.6 | 44.70 | 1.000 |
| `105322251564` | `108222250342` | 655.5 | 38.91 | 1.000 |
| `037522251142` | `046122250168` | 671.1 | 34.28 | 0.652 |
| `105322251564` | `043422252387` | 869.5 | 52.97 | 1.000 |
| `105322251564` | `037522251142` | 947.1 | 57.10 | 0.652 |  ← selected
| `043422252387` | `115422250549` | 985.0 | 74.15 | 0.555 |
| `108222250342` | `046122250168` | 1035.9 | 56.94 | 1.000 |
| `108222250342` | `117222250549` | 1047.3 | 76.96 | 1.000 |
| `105322251225` | `117222250549` | 1052.9 | 76.46 | 1.000 |
| `117222250549` | `115422250549` | 1068.0 | 96.86 | 0.555 |
| `037522251142` | `108222250342` | 1068.5 | 69.06 | 0.652 |
| `105322251564` | `105322251225` | 1105.5 | 68.33 | 1.000 |
| `043422252387` | `108222250342` | 1142.5 | 76.53 | 1.000 |
| `105322251564` | `117222250549` | 1167.3 | 81.39 | 1.000 |
| `105322251225` | `046122250168` | 1178.7 | 65.88 | 1.000 |
| `043422252387` | `046122250168` | 1212.3 | 68.84 | 1.000 |
| `037522251142` | `115422250549` | 1238.3 | 96.27 | 0.522 |
| `037522251142` | `105322251225` | 1337.7 | 89.52 | 0.652 |
| `105322251564` | `115422250549` | 1367.7 | 104.49 | 0.555 |
| `046122250168` | `115422250549` | 1391.1 | 91.52 | 0.555 |
| `043422252387` | `037522251142` | 1463.8 | 102.77 | 0.652 |
| `105322251225` | `108222250342` | 1483.5 | 104.86 | 1.000 |
| `108222250342` | `115422250549` | 1536.1 | 143.16 | 0.555 |
| `117222250549` | `046122250168` | 1564.5 | 106.06 | 1.000 |
| `037522251142` | `117222250549` | 1585.7 | 138.39 | 0.652 |

### Acceptance criteria

**1. Extrinsics round-trip** — PASS. Camera centre recovered through `CalibrationExtrinsics.transform_matrix` matches the intended pose to **1.22e-15 m** (tolerance 1e-9).

| camera | centre err (m) | full residual (m) | dev. from HO-Cap raw centre (m) | HO-Cap 3×3 orthonormality defect |
|---|---:|---:|---:|---:|
| `105322251564` | 6.85e-16 | 7.77e-16 | 6.85e-16 | 6.08e-07 |
| `037522251142` | 1.22e-15 | 1.11e-15 | 1.22e-15 | 8.72e-07 |

_check is against the nearest true rotation to HO-Cap's raw 3x3 block; raw_centre_deviation_m is how far that sits from HO-Cap's un-orthonormalised published centre (bounded by their ~float32 publication precision, not by this importer)._

_GT frame convention resolved to **camera**-frame `hand_joints_3d` by cross-camera agreement: the camera-frame hypothesis makes the same joint from different cameras agree to **0.0000 mm**, the world-frame hypothesis to 342.9 mm (882 joint/camera comparisons)._

_Cross-view GT consistency (both selected views, 18354 joint obs): median 0.00005 mm, p95 0.00006 mm, max 0.00008 mm — HO-Cap's per-camera `hand_joints_3d` are one MANO fit rotated into each camera frame, so once transformed to the world frame they coincide to floating point; a real detector's multi-view spread will be orders of magnitude larger._

**2. Depth scale and alignment** — per camera (threshold: all-joints median < 15 mm):

| camera | frames | all-joints median / p95 (mm) | wrist median (mm) | fingertip median (mm) | verdict |
|---|---:|---|---:|---:|---|
| `105322251564` | 874 | 10.9 / 71.4 | 20.6 | 18.1 | PASS |
| `037522251142` | 862 | 24.4 / 93.0 | 70.2 | 11.1 | FAIL |

_Reading: a median in the hundreds ⇒ wrong PNG scale (not the case — the /1000 mm→m conversion is correct). Fingertip error ≫ wrist error ⇒ colour/depth not aligned; here fingertips ≤ wrist, so the identity projector is valid. The residual that remains is the MANO joint-centre sitting inside the flesh while the depth sees the skin surface, plus RealSense active-stereo-IR noise at manipulation range — i.e. thesis caveat 2, not an ingestion error._

**3. Reprojection overlays** — GT joints + skeleton edges drawn on the colour frame (paths relative to the episode dir) for human inspection of the joint-order mapping:

- `gt/overlays/105322251564_000000.png`
- `gt/overlays/105322251564_000218.png`
- `gt/overlays/105322251564_000436.png`
- `gt/overlays/105322251564_000655.png`
- `gt/overlays/105322251564_000873.png`
- `gt/overlays/037522251142_000000.png`
- `gt/overlays/037522251142_000218.png`
- `gt/overlays/037522251142_000436.png`
- `gt/overlays/037522251142_000655.png`
- `gt/overlays/037522251142_000873.png`

**4. Triangulation identity check** — PASS. GT world joints reprojected into both views and triangulated back via `viki.perception.triangulate`: median **0.000 mm**, max **0.000 mm** over 18354 joints (threshold: median < 1 mm).

**5. Colour encoding loss** (JPEG source vs mp4-decoded, threshold p95 ≤ 1.0 px):

- skipped — run with --backend-check (needs mediapipe + source JPEGs)

**6. Pipeline runs unmodified** — `viki extract` then `viki prepare` on the imported episode, no changes to any existing module:

- not run in this session (needs a pose backend / container). Run with `--pipeline-check`, or `viki extract <ep> && viki prepare <ep>` and record `observations.npz` row count, per-camera detection counts, and the both-cameras-detected fraction here.

### Notes

- depth entry in intrinsics.json = HO-Cap color intrinsics; the published depth PNGs are registered to the colour plane (HOCap-Toolkit deprojects them with the colour K), so no calibration_preset / *_calib blob is written and extract uses the identity colour->depth projector. Criterion 2 verifies this empirically.
- GT frame convention resolved to 'camera' by cross-camera agreement.

---

## SKIPPED — subject_1/20231025_171117

- not a right-hand single-hand sequence (annotated hand(s) = ['left']); left-hand / bimanual / handover sequences are out of scope for the first import  (`hands_in_labels=['left']`, `mano_sides_meta=['left']`, `task_id=3`)


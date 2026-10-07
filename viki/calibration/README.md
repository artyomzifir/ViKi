# viki.calibration — intrinsics + extrinsics

Side input to `perception`. Per-camera **intrinsics** (chessboard / ChArUco) and
rig **extrinsics** (camera pose relative to a reference camera) are kept
separate from the ChArUco-derived world/display anchor. Results persist as JSON
so they survive restarts.

## Files

| file | what |
|---|---|
| `manager.py` | `CalibrationManager` — worker orchestration, sample collection, solve, persistence; `load_all_extrinsics()` at startup |
| `worker.py` | `_CalibrationWorker` base — sample collection loop + solve hooks |
| `chessboard_worker.py` / `aruco_worker.py` | board detection + intrinsics/extrinsics solve for each board type |
| `file.py` | JSON read/write of intrinsics + extrinsics |
| `models.py` | compat re-export of the calibration DTOs from `viki.contracts` + `canonical_board_extrinsics` |

For Azure Kinect, the multi-pose bundle solve does not treat colour pixels as a
zero-distortion pinhole. Each ChArUco pixel is first unprojected to a camera ray
with the factory K4A SDK model, then expressed in an ideal pinhole plane for the
optimizer. Live solves obtain rays from the running backend; offline re-solves
rebuild the same model from the raw calibration blob saved in the preset. This
keeps extrinsics consistent with cloud reconstruction across colour resolutions.

The default large ChArUco board is 8×10 squares, 50 mm per square (400×500 mm),
35 mm markers, `DICT_5X5_50`. These settings must match the physical print.
Changing the default does not convert existing presets or recordings: each saved
solve retains its own board geometry. Capture a new rig calibration before using
the large board for new episodes.

For the current two-Kinect rig, use the large board for new recordings: the
operator observed substantial cloud misalignment with the earlier 11×7,
25 mm-board `table-v1` calibration and visually usable alignment after a new
large-board solve. Both presets passed the automatic validator, so that verdict
alone does not establish task-volume accuracy. See
`docs/2026-10-06-large-board-calibration.md` for the observation and limits.

## Contract

- **out:** `CalibrationExtrinsics.transform_matrix` (4×4 camera→world) — consumed by `perception.lift.camera_landmarks_to_world`.
- Extrinsics JSON is keyed by `device_id`; moving a camera invalidates the rig
  solve, world anchor and validation tied to that solve.

## Not here

Robot-base registration (`T^W_R`) lives in `viki.retarget.frames` — it is a
fixed config constant, not a per-session calibration.

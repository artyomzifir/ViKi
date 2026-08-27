# viki.calibration — intrinsics + extrinsics

Side input to `perception`. Per-camera **intrinsics** (chessboard / ChArUco) and
per-camera **extrinsics** (camera pose in a shared world frame anchored to a
ChArUco board). Results persist as JSON so they survive restarts.

## Files

| file | what |
|---|---|
| `manager.py` | `CalibrationManager` — worker orchestration, sample collection, solve, persistence; `load_all_extrinsics()` at startup |
| `worker.py` | `_CalibrationWorker` base — sample collection loop + solve hooks |
| `chessboard_worker.py` / `aruco_worker.py` | board detection + intrinsics/extrinsics solve for each board type |
| `file.py` | JSON read/write of intrinsics + extrinsics |
| `models.py` | compat re-export of the calibration DTOs from `viki.contracts` + `canonical_board_extrinsics` |

## Contract

- **out:** `CalibrationExtrinsics.transform_matrix` (4×4 camera→world) — consumed by `perception.lift.camera_landmarks_to_world`.
- Extrinsics JSON is a list keyed by `device_id`; adding/moving one camera does not invalidate the others (workspace-anchor calibration).

## Not here

Robot-base registration (`T^W_R`) lives in `viki.retarget.frames` — it is a
fixed config constant, not a per-session calibration.

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

## Contract

- **out:** `CalibrationExtrinsics.transform_matrix` (4×4 camera→world) — consumed by `perception.lift.camera_landmarks_to_world`.
- Extrinsics JSON is keyed by `device_id`; moving a camera invalidates the rig
  solve, world anchor and validation tied to that solve.

## Not here

Robot-base registration (`T^W_R`) lives in `viki.retarget.frames` — it is a
fixed config constant, not a per-session calibration.

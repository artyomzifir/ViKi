# ViKi Calibration

Multi-camera calibration: per-camera **intrinsics** (from a chessboard or ChArUco
board) and per-camera **extrinsics** (the camera's pose relative to a shared
world frame). Results persist as JSON so they survive server restarts.

## Responsibilities

- Detect a calibration board in live camera frames (`chessboard_worker`,
  `aruco_worker`).
- Collect samples across many board poses.
- Solve intrinsics (`cv2.calibrateCamera` / `cv2.calibrateCameraCharuco`) and
  extrinsics (`cv2.solvePnP` / `estimatePoseCharucoBoard`).
- Persist/load results to/from JSON via `file.py`.
- Expose everything through the `/api/calibration/*` router.

## Layout

```text
viki/calibration/
  manager.py          # CalibrationManager: public API, worker orchestration, persistence
  worker.py           # _CalibrationWorker base: sample collection + solve hooks
  chessboard_worker.py# ChessboardWorker: chessboard detection + intrinsics/extrinsics
  aruco_worker.py     # ArucoWorker: ChArUco detection + intrinsics/extrinsics
  models.py           # BoardParameters, ArucoBoardParameters, CalibrationSample,
                      # CalibrationIntrinsics, CalibrationExtrinsics
  file.py             # JSON read/write of intrinsics + extrinsics
  __init__.py
```

## Data model (`models.py`)

- `BoardParameters(board_size, square_size)` — chessboard (internal corners).
- `ArucoBoardParameters(board_size, square_size, marker_size, aruco_dict)` —
  ChArUco.
- `CalibrationSample` — one accepted board observation per camera.
- `CalibrationIntrinsics` — camera matrix `K`, distortion `dist`, resolution;
  serialisable to/from NumPy arrays.
- `CalibrationExtrinsics` — `rvec`/`tvec` (and `rotation_matrix` /
  `transform_matrix`) describing camera pose in the world frame.

## CalibrationManager

Singleton-style coordinator created in `app.py` from the `CameraManager`. Key
methods:

- `start(device_id, mode, board_type, ...)` — spin up a `ChessboardWorker` or
  `ArucoWorker`. `mode="auto"` runs an internal capture thread; `mode="manual"`
  only adds samples on `capture()`.
- `capture(device_id)` / `capture_all()` — collect one sample now.
- `intrinsics_calibration(device_id)` / `extrinsics_calibration(device_id)` —
  solve and **persist** (JSON). Extrinsics require intrinsics to be present
  (cached or loaded from file).
- `get_intrinsics` / `get_extrinsics` / `set_intrinsics` / `set_extrinsics` —
  cache access; fall back to the default JSON files.
- `load_all_extrinsics()` — load every entry from the extrinsics JSON into the
  cache at startup (called by `app.py` lifespan).
- `clear` / `reset` / `status` / `samples_count` / `is_device_active`.

## Persistence

Files (paths from `viki.config`):

```text
data/<INTRINSICS_FILENAME>   # per-device intrinsics JSON
data/<EXTRINSICS_FILENAME>   # list of per-device extrinsics JSON entries
```

Extrinsics are stored as a JSON **list** keyed by `device_id`, so
`load_all_extrinsics()` can repopulate the cache for all cameras at once.

## API (`/api/calibration`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/calibration/reset` | Reset the calibration manager. |
| POST | `/api/calibration/sync` | Trigger a software sync. |
| POST | `/api/calibration/capture/{device_id}` | Capture one sample. |
| POST | `/api/calibration/capture` | Capture a sample on every active worker. |
| POST | `/api/calibration/start/{device_id}` | Start chessboard worker. |
| POST | `/api/calibration/start/aruco/{device_id}` | Start ChArUco worker. |
| GET | `/api/calibration/status/{device_id}` | Samples + started flag. |
| GET | `/api/calibration/samples_count/{device_id}` | Sample count. |
| GET | `/api/calibration/is_device_active/{device_id}` | Worker active? |
| POST | `/api/calibration/clear/{device_id}` | Clear collected samples. |
| POST | `/api/calibration/intrinsics/{device_id}` | Solve + persist intrinsics. |
| GET | `/api/calibration/intrinsics/{device_id}` | Read intrinsics. |
| POST | `/api/calibration/extrinsics` | Solve extrinsics for all. |
| GET | `/api/calibration/extrinsics/{device_id}` | Read extrinsics. |
| GET | `/api/calibration/viz` | Static extrinsics visualisation. |
| GET | `/api/calibration/{device_id}/stream` | MJPEG board-detection preview. |

## Typical workflow

1. Start each camera (`/api/cameras/{id}/start`).
2. Start a calibration worker per camera (`/api/calibration/start/{id}` or the
   ChArUco variant), choosing `board_type` and dimensions.
3. Move the board through varied poses; capture samples (`/api/calibration/capture`
   or auto mode), aiming for a healthy `samples_count`.
4. Solve intrinsics per camera (`POST /api/calibration/intrinsics/{id}`).
5. Solve extrinsics (`POST /api/calibration/extrinsics`) — uses the board sample
   pose in the world frame shared across cameras.
6. Extrinsics are persisted and reloaded automatically on the next server start.

The world frame used for extrinsics is the board's frame at the calibration
sample; `SkeletonPipeline` consumes the resulting per-camera extrinsics to lift
and transform landmarks into that shared world frame.

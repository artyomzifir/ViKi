# ViKi Server

FastAPI application that ties the capture, calibration, skeleton, and
optimisation packages together behind a single HTTP/WebSocket API and serves the
plain-HTML web UI. Everything runs in one container; the UI is a self-contained
page at `viki/server/static/index.html`.

## Responsibilities

- Device discovery, camera start/stop, MJPEG colour + colourised-depth streams.
- Calibration (chessboard / ChArUco) intrinsics and per-camera extrinsics.
- Live skeleton estimation, 3D skeleton WebSocket stream, and recording.
- Smoothing + retargeting job orchestration and robot-trajectory visualisation.
- Configuration load/save and a soft server restart.

## Layout

```text
viki/server/
  app.py              # App assembly ONLY: lifespan, static mount, include_router
  deps.py             # FastAPI DI: resolve managers/workers from app.state
  streams.py          # MJPEG stream generators: poll manager + timing, call viz
  skeleton_worker.py  # Background thread running SkeletonPipeline each sync tick
  robot_viz.py        # Matplotlib robot-trajectory visualisation + MJPEG stream
  routes/
    cameras.py        # /api  device list + per-camera start/stop/info/streams
    calibration.py    # /api/calibration  capture/intrinsics/extrinsics/viz
    skeleton.py       # /api/skeleton  toggle/record/capture_base + /stream websocket
    recording.py      # /api/record  start a recording session
    system.py         # /api  config get/post/reset + restart
    optimization.py   # /api/optimization  raw→prepared: recordings + smooth
    dataset.py        # /api/dataset  prepared→h5: optimize + viz
    models.py         # Shared Pydantic/NumPy response models (intrinsics/extrinsics)
```

**Layering:** `routes/` (thin handlers) → `deps.py` (DI) → `streams.py` (poll +
timing) → `viz/` (pure pixel work) → `config.py` (constants). Handlers delegate;
they do not compute pixels or camera frames themselves.

## App assembly (`app.py`)

`lifespan` constructs every long-lived object and stores it on `app.state`:

```text
app.state.manager            CameraManager()
app.state.calibrator         CalibrationManager(manager); .load_all_extrinsics()
app.state.sync               MultiCameraSync(manager)
app.state.skeleton_pipeline  SkeletonPipeline(calibrator, manager)
app.state.skeleton_recorder  SkeletonRecorder(filter_indices=[WRIST, MIDDLE_MCP, THUMB_CMC])
app.state.skeleton_worker    SkeletonWorker(manager, sync, pipeline, recorder); .start()
app.state.skeleton_processor PreparationPipeline()
```

On shutdown it stops the skeleton worker and all cameras. `GET /` returns
`static/index.html`; `/static/*` is mounted for assets.

## Dependency injection (`deps.py`)

Handlers receive shared objects via `Depends(get_manager)` / `get_calibrator` /
`get_worker` / `get_sync` / `get_pipeline` / `get_recorder` / `get_processor`,
all of which read from `app.state`. Adding a new endpoint: resolve its
dependencies here rather than constructing them in the handler.

## Streaming (`streams.py`)

Stream generators do **not** block the event loop. They poll
`manager.latest_frame()` / `manager.nearest_frame()` at ~30 fps and hand the raw
pixels to the `viz/` helpers (`DepthColorizer`, `Undistorter`, MJPEG encoder).
All camera consumers (colour, depth, calibration preview, skeleton worker) read
the one shared ring buffer per camera — there is no push pub/sub.

## Background skeleton worker (`skeleton_worker.py`)

`SkeletonWorker` runs a daemon thread that, on every `MultiCameraSync` tick,
runs `SkeletonPipeline.process(...)` (detect → lift → world transform per
camera), optionally records `SkeletonFrame`s via `SkeletonRecorder`, and pushes
the latest `PipelineResult` to the `/api/skeleton/stream` WebSocket. Live view
shows detection per-camera; rejected frames are simply not shown (no hold of the
last pose).

## Endpoints

### Cameras — `/api` (router: `cameras`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/cameras/devices` | Detected camera IDs grouped by type (`realsense`, `kinect`, `active`). |
| POST | `/api/cameras/{device_id}/start` | Start a camera backend (fps/res/depth mode). |
| POST | `/api/cameras/{device_id}/stop` | Stop a camera. |
| GET | `/api/cameras/{device_id}/info` | Running state, frame shapes, colour intrinsics. |
| GET | `/api/cameras/{device_id}/stream` | MJPEG colour stream. |
| GET | `/api/cameras/{device_id}/depth` | MJPEG colourised-depth stream. |

### Calibration — `/api/calibration` (router: `calibration`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/calibration/reset` | Reset the calibration manager. |
| POST | `/api/calibration/sync` | Trigger a software sync. |
| POST | `/api/calibration/capture/{device_id}` | Capture one calibration sample. |
| POST | `/api/calibration/capture` | Capture a sample on every active worker. |
| POST | `/api/calibration/start/{device_id}` | Start a chessboard calibration worker. |
| POST | `/api/calibration/start/aruco/{device_id}` | Start a ChArUco calibration worker. |
| GET | `/api/calibration/status/{device_id}` | Samples collected + started flag. |
| GET | `/api/calibration/samples_count/{device_id}` | Number of collected samples. |
| GET | `/api/calibration/is_device_active/{device_id}` | Worker active? |
| POST | `/api/calibration/clear/{device_id}` | Clear collected samples. |
| POST | `/api/calibration/intrinsics/{device_id}` | Run intrinsics calibration + persist. |
| GET | `/api/calibration/intrinsics/{device_id}` | Read intrinsics. |
| POST | `/api/calibration/extrinsics` | Run extrinsics calibration for all. |
| GET | `/api/calibration/extrinsics/{device_id}` | Read extrinsics. |
| GET | `/api/calibration/viz` | Static extrinsics visualisation. |
| GET | `/api/calibration/{device_id}/stream` | MJPEG board-detection preview. |

### Skeleton — `/api/skeleton` (router: `skeleton`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/skeleton/toggle` | Enable/disable live estimation. |
| POST | `/api/skeleton/record` | Start/stop a recording session. |
| POST | `/api/skeleton/depth-debug` | Toggle per-frame depth diagnostics. |
| GET | `/api/skeleton/status` | Worker running/recording state. |
| WS | `/api/skeleton/stream` | Live 3D skeleton frames (canvas-driven). |

### Recording — `/api/record` (router: `recording`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/record/start` | Begin a recording session (filename returned). |

### System — `/api` (router: `system`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/config` | Current configuration (from `data/user_configuration.json`). |
| POST | `/api/config` | Save configuration. |
| POST | `/api/config/reset` | Reset to defaults. |
| POST | `/api/restart` | Soft-restart the server process. |

### Optimisation — `/api/optimization` (router: `optimization`)

Raw recording → prepared data. Smoothing/interpolation/fusion turn raw
`rec-*.npz` captures into prepared `cln-*.npz` (see
[`viki/optimization/README.md`](../optimization/README.md)).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/optimization/recordings` | List raw `rec-*.npz` recordings (paginated). |
| POST | `/api/optimization/smooth` | Run `PreparationPipeline.smooth_recording` → `cln-*.npz`. |
| GET | `/api/optimization/smooth-plot` | Smoothing diagnostic plot for a file. |

### Dataset — `/api/dataset` (router: `dataset`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/dataset/recordings` | List smoothed `cln-*.npz` (paginated). |
| POST | `/api/dataset/optimize` | Retarget a prepared `cln-*.npz` → robot `.h5` dataset. |
| GET | `/api/dataset/optimize/status/{job_id}` | Job status. |
| GET | `/api/dataset/optimize/jobs` | List dataset jobs. |
| GET | `/api/dataset/outputs` | List robot `.h5` outputs. |
| GET | `/api/dataset/debug-viz` | Retarget debug visualisation. |
| GET | `/api/dataset/viz-stream` | MJPEG robot-trajectory stream. |

## Configuration

All tunables live in `data/user_configuration.json` (copied from
`data/default_configuration.json` on first run) and are surfaced through
`viki/config.py`. The `system` router reads/writes this file; `config.py`
reloads it at import time.

## Tests

Logic tests live with their packages and run without the server:

```bash
python -m unittest discover viki.skeleton
python -m unittest discover viki.optimization.preparation
python -m unittest discover viki.optimization.retarget
```

Route-level `TestClient` tests were removed — the `/api/optimization` retargeting
endpoints they covered were unused (the UI drives retargeting via `/api/dataset`).

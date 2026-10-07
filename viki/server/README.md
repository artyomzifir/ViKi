# viki.server — HTTP/WS transport

Thin FastAPI wrapper over the pipeline packages and a static web UI. **No domain
logic and no live pipeline** — there is no skeleton worker, no 3-D WebSocket
stream. Scenes are recorded, then extracted / prepared / retargeted offline.

## Assembly (`app.py`)

`lifespan` builds exactly two long-lived objects: `CameraManager` and
`CalibrationManager` (on `app.state`). `deps.py` hands them to routes via
`Depends(get_manager)` / `Depends(get_calibrator)`.

## Routes

| router | endpoints | drives |
|---|---|---|
| `cameras.py` | device list, start/stop, info, colour/depth MJPEG | `CameraManager`, `render` |
| `calibration.py` | intrinsics/extrinsics capture + solve + board preview | `CalibrationManager` |
| `skeleton.py` | `POST /skeleton/capture_base/{id}` — static background depth for offline scene subtraction | — |
| `recording.py` | `POST /record/start` → `SceneRecorder` in a background job | `cameras.record` |
| `pipeline.py` | episode geometry/cloud/object-model diagnostics, batch `POST /pipeline/{perceive,retarget}`, retarget robot/plan scene APIs, and jobs | `perception`, `prepare`, `retarget`, `episode` |
| `replay.py` | `POST /replay` job | `viki.replay` |
| `label.py` | `GET/POST /label` | `viki.labeling` |
| `export.py` | `POST /export` job | `viki.export` |
| `system.py` | config get/set/reset, restart | `viki.config` |

## Jobs

`jobs.py` — a minimal in-process registry (`submit` / `get` / `all_jobs`). Not
durable; fine for a single-user research tool. `pipeline` / `replay` / `export`
return a `job_id`; poll `…/jobs/{job_id}`.

## Frontend

`static/` — plain ES-module HTML/JS, no build step, served at `/`. Panels:
cameras, calibration, **episodes** (`episodes.js` — list, per-stage job buttons,
label form, record) and **3-D viewer** (`viewer.js` + `scene3d.js` — a three.js
orbit view of point clouds, hand/robot trajectories and object-model filter
diagnostics). The object layers use `/pipeline/episode/{id}/object-model` for
canonical core/shell + pose tracks and the binary `.../object-model/{frame}`
endpoint for accepted/rejected/reassigned/ambiguous observations. No live
skeleton panel. A different UI engine would talk to the same routes.

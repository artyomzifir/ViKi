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
| `pipeline.py` | `/pipeline/extract` + the legacy optimization/dataset routers | `perception.extract`, `prepare`, `retarget` |
| `replay.py` | `POST /replay` job | `viki.replay` |
| `label.py` | `GET/POST /label` | `viki.labeling` |
| `export.py` | `POST /export` job | `viki.export` |
| `system.py` | config get/set/reset, restart | `viki.config` |

## Jobs

`jobs.py` — a minimal in-process registry (`submit` / `get` / `all_jobs`). Not
durable; fine for a single-user research tool. `pipeline` / `replay` / `export`
return a `job_id`; poll `…/jobs/{job_id}`.

## Frontend

`static/` — plain HTML/JS, one module per panel. Served at `/`. (A different UI
engine would talk to the same routes.)

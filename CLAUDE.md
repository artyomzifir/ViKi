# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ViKi turns multi-view RGB-D video of a human doing a manipulation task into a robot-ready
LeRobot demonstration dataset. The pipeline stages, and the file each stage produces, are:

```
camera frames → 3D hand skeleton → rec-*.npz (raw per-camera landmark trajectories)
             → cln-*.npz (smoothed + cross-camera fused + end-effector pose)
             → robot .h5 (IK solution / dataset)
```

`data/skeleton_recs/` (rec), `data/skeleton_smoothed/` (cln), `data/robot_out/` (h5) — all gitignored.

## Running everything in Docker

The app talks to hardware SDKs (`pyrealsense2`, `libk4a.so`) that are installed in the image,
not on the host. Run and test through Docker Compose.

```bash
sudo ./scripts/host_setup.sh              # once per host: Docker, udev rules, groups
docker compose up --build                 # web UI + API on :8000 (first run; then: docker compose up)
docker compose run --rm terminal          # debug shell inside the container
docker compose run --rm test              # full test suite (pytest tests/)
docker compose run --rm cli <verb> ...    # run one pipeline stage (viki <verb> ...)
```

One `docker-compose.yml`, one image (Dockerfile `test` target). Services: `web` (default,
`up`), `test` / `cli` / `terminal` (behind the `tools` profile, meant for `run`). `test` and
`cli` append args to `pytest` / `viki` respectively.

Server serves the UI + API at `http://localhost:8000` (`network_mode: host`). `viki/` is bind-mounted,
so code edits apply on container restart (no rebuild unless `pyproject.toml` changes).

Kinect setup has host-level prerequisites (GRUB `usbcore.usbfs_memory_mb=1000`, `xhost +local:`,
separate 10 Gbps USB hubs per Kinect). See `SETUP_GUIDE.md` before touching camera bring-up.

## Tests

Two conventions coexist:

`tests/unit_tests/**` — pytest, hardware-independent, one dir per package. Run all with
`docker compose run --rm test`; a subset with `docker compose run --rm test
tests/unit_tests/perception/test_geometry.py::test_name` (args append to `pytest`).

`tests/unit_tests/e2e/test_pipeline_smoke.py` walks a synthetic episode
rec.npz → prepare → plan.h5 → replay(dryrun) → label → export.

PINK/Pinocchio ships in the image, so the retarget leg runs in-container.

## Architecture

Five packages under `viki/`, each with its own README (read these — they carry hardware quirks and
API tables not repeated here):

- **`capture/`** — the only package that touches camera SDKs. `CameraManager` owns one daemon
  `_CameraWorker` thread per active camera, each writing frames to a bounded ring buffer under a lock.
  All consumers (MJPEG streams, calibration preview, skeleton worker) *pull* via `latest_frame()` /
  `nearest_frame()` — there is no push pub/sub. `KinectBackend` is raw ctypes over `libk4a.so`;
  its README lists constraints that must not change without hardware (`align_depth_to_color` is
  bugged and disabled, `WFOV_UNBINNED` capped at 15 fps, `stop()` sleeps 2 s).
- **`calibration/`** — per-camera intrinsics + extrinsics (chessboard / ChArUco), persisted as JSON
  in `data/` so they survive restarts. `CalibrationManager.load_all_extrinsics()` runs at startup.
- **`skeleton/`** — MediaPipe hand detection → lift to 3D via measured depth → transform to shared
  world frame → end-effector pose. `SkeletonPipeline.process()` runs cameras **sequentially** (two
  live MediaPipe models contend on the GPU). Capture-time fusion is deliberately *not* done here;
  `SkeletonRecorder` writes per-camera trajectories as-is to `rec-*.npz`.
- **`optimization/`** — `preparation/` (rec → cln: interpolate, cross-camera fuse, smooth, compute
  EE pose) and `retarget/` (cln → h5: PINK/Pinocchio IK). `compute_end_effector_pose` in
  `skeleton/hand_angles.py` is the single site deriving EE pose from landmarks, shared by live
  pipeline and preparation.
- **`viz/`** — pure pixel work (depth colorization, MJPEG encoding, undistortion). Must not import
  FastAPI or the camera layer, so it stays testable without hardware.
- **`server/`** — FastAPI assembly only. `app.py` builds every long-lived object in `lifespan` and
  stores it on `app.state`; handlers get them via `Depends(...)` from `deps.py`.

**Server layering (do not skip levels):** `routes/` (thin handlers) → `deps.py` (DI) →
`streams.py` (poll + timing) → `viz/` (pure pixel work) → `config.py` (constants).
Handlers delegate; they don't compute pixels or camera frames.

**Router ↔ pipeline-stage mapping** (easy to confuse):
- `/api/skeleton/*` — live estimation + start/stop `rec-*.npz` recording
- `/api/optimization/*` — raw → prepared (`rec-*.npz` → `cln-*.npz`, smoothing/fusion)
- `/api/dataset/*` — prepared → robot (`cln-*.npz` → `.h5`, retargeting jobs + viz)

## Configuration

All tunables live in `data/user_configuration.json` (copied from `default_configuration.json` on
first run). `viki/config.py` reads that file **once at import** and injects every key into its module
globals, so code does `from viki.config import RETARGET_IK_SOLVER`. Changing config at runtime means
editing the JSON and restarting (the `/api/config` routes + `/api/restart` do this). Adding a tunable:
add it to both JSON files, declare its type annotation in `config.py`.

## Frontend

`viki/server/static/` — plain HTML/CSS/JS, no build step. `index.html` + one JS module per UI panel
(`cameras.js`, `calibration.js`, `record.js`, …). Served directly by FastAPI.

Exception: the **Viewer tab** (`viewer.js`) renders a per-frame coloured depth
point cloud + skeleton with **three.js** (WebGL). three.js is the one vendored
dep — `static/js/vendor/three.module.js` + `OrbitControls.js`, resolved by an
importmap in `index.html`. A dense cloud (~10^5 points/frame) is not viable on a
hand-rolled 2-D canvas. Everything else stays plain, no-build JS.

**Pipeline viz artifact:** `viki cloud <episode>` (`viki/perception/cloud.py`)
turns `raw/` into `cloud/<i:06d>.bin` (`int32 n` · `float32[n*3]` xyz m ·
`uint8[n*3]` rgb) + `cloud/meta.json`, served by
`GET /api/pipeline/episode/{id}/cloud[/{frame}]`. Nothing downstream reads it;
it is not part of `viki run`.

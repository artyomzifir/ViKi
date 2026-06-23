# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ViKi (Vision-based Kinematic Imitation) turns RGB-D video of a human manipulation task into a robot-ready LeRobot dataset. The pipeline is: multi-view RGB-D capture → 3D skeleton extraction → trajectory optimisation → LeRobot HDF5 dataset. Only Phase 1 (capture) is currently implemented.

## Running the server

Everything runs inside Docker. The `viki` directory is bind-mounted at `/app/viki`, so Python changes take effect on restart without a rebuild.

```bash
# First run (builds image, ~5 min)
docker compose up --build

# Subsequent runs
docker compose up

# Debug terminal inside the container
docker compose run --rm terminal
```

The web UI is at `http://localhost:8000`.

## One-time host setup

```bash
sudo ./scripts/host_setup.sh   # installs Docker, udev rules, groups
# then log out and back in
```

For two Azure Kinects: also add `usbcore.usbfs_memory_mb=1000` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub && sudo reboot`.

For Kinect depth engine (OpenGL via llvmpipe): add `xhost +local: > /dev/null 2>&1` to `~/.bashrc`.

## Architecture

```
viki/
  capture/
    base.py        # CameraBackend ABC + Frame/CameraIntrinsics dataclasses
    realsense.py   # RealSenseBackend (pyrealsense2)
    kinect.py      # KinectBackend (ctypes over libk4a.so — no pyk4a)
    manager.py     # CameraManager: device discovery, start/stop, per-camera worker threads
  server/
    app.py         # FastAPI app: /api/devices, /api/cameras/{id}/start|stop|stream|depth
    static/        # index.html UI
```

**Data flow:** `CameraManager` owns one `_CameraWorker` (daemon thread) per active camera. Each worker calls `backend.get_frame()` in a tight loop and stores the result under a lock. The MJPEG streamers in `app.py` poll `manager.latest_frame()` at 30 fps — never blocking the FastAPI event loop.

**Adding a new camera backend:** subclass `CameraBackend` (implement `start`, `stop`, `get_frame`, `device_id`, `is_running`), then add detection in `CameraManager.list_devices()` and routing in `CameraManager._make_backend()`.

**Frame format:** `Frame.color` is HxWx3 uint8 BGR when materialised; preview-optimised backends may set `Frame.color_jpeg` plus `Frame.color_shape` instead. `Frame.depth` is HxW uint16 in millimetres.

## Key implementation details

- **Kinect backend uses ctypes directly over `libk4a.so`** — no `pyk4a`. All function signatures are declared in `kinect.py`. This was chosen to avoid compilation inside Docker.
- **`KinectBackend.align_depth_to_color`** defaults to disabled for realtime preview; aligned depth is computed on demand for snapshots/debug, or continuously only when explicitly enabled.
- **`WFOV_UNBINNED` depth mode** is capped at 15 fps by hardware; the backend raises `ValueError` at 30 fps.
- **USB release delay:** `KinectBackend.stop()` sleeps 2 seconds after closing to let USB fully release before the next open.
- **Kinect sync startup order:** always start subordinate (`kinect_1`) before master (`kinect_0`). Currently both run in standalone `wired_sync_mode=0`; hardware sync mode is planned.

## Planned phases (not yet implemented)

| Phase | Description |
|---|---|
| 2 — Skeleton | MediaPipe pose estimation, depth-fused 3D keypoints, multi-view fusion |
| 3 — Smoothing | One Euro Filter, outlier rejection |
| 4 — Retargeting | URDF IK via PINK/Pinocchio, object-relative cost, gripper inference |
| 5 — Dataset | LeRobot HDF5 writer (RGB + depth + joints + actions) |
| 6 — Evaluation | ACT and Diffusion Policy on UR3 |

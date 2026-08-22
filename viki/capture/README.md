# ViKi Capture (Cameras)

Camera abstraction layer: device discovery, streaming backends, a per-camera
worker thread, and host-clock synchronisation across heterogeneous devices. This
is the only package that talks to hardware SDKs (`pyrealsense2`, `libk4a.so`).

## Responsibilities

- Detect and open RealSense and Azure Kinect cameras.
- Stream colour (BGR) + depth (uint16 mm) frames in background worker threads.
- Synchronise frames from multiple cameras to a common host-clock tick.
- Provide last-value + nearest-in-time frame access to all consumers.
- Record synchronised RGB-D clips to disk.

## Layout

```text
viki/capture/
  base.py        # CameraBackend ABC + Frame / CameraIntrinsics / SyncedFrameGroup
  manager.py     # CameraManager: discovery, start/stop, worker threads, backend factory
  realsense.py   # RealSenseBackend (pyrealsense2) — serial-addressed D4xx cameras
  kinect.py      # KinectBackend (ctypes over libk4a.so) — Azure Kinect DK
  sync.py        # MultiCameraSync: software host-clock sync across cameras
  recorder.py    # RGBDRecorder: MP4 colour/depth + raw .npy depth to data/videos
  __init__.py
```

## Frame contract (`base.py`)

`Frame` is the universal currency:

```text
color             HxWx3  uint8   BGR (OpenCV convention)
depth             HxW     uint16  millimetres
timestamp_us      int     device monotonic clock
device_id         str     unique id (serial, or "kinect_<n>")
aligned_depth     HxW     uint16  SDK-aligned depth (optional)
host_timestamp_us int     host clock (time.time_ns()//1000) stamped by the worker
color_intrinsics / depth_intrinsics  CameraIntrinsics | None
```

`has_depth()` is the canonical "do we have a usable depth image?" check.

## CameraManager (`manager.py`)

- `list_devices()` — groups detected cameras by type (`realsense`, `kinect`) and
  reports currently `active` ones; SDK errors are returned as `*_error` fields
  rather than raised.
- `start(device_id, ...)` / `stop(device_id)` / `stop_all()` — lifecycle. A
  device is already running if present in `_workers`.
- `_CameraWorker` — a daemon thread that loops `backend.get_frame()`, stamps
  `host_timestamp_us`, and appends to a bounded `deque` ring buffer under a lock.
  `backend.stop()` runs in the worker's `finally` block, never concurrently with
  `get_frame()`, which eliminates the earlier stop/deadlock race.
- `latest_frame(device_id)` / `nearest_frame(device_id, host_timestamp_us)` —
  non-blocking reads used by streams, calibration preview, and the skeleton
  worker.
- `get_info(device_id)` — running state, frame shapes, colour intrinsics.
- `_make_backend(device_id, ...)` — factory: `kinect_*` → `KinectBackend`,
  anything else → `RealSenseBackend`.
- `start_kinect_sync(master_id, subordinate_ids, ...)` — starts Azure Kinects in
  **hardware** wired-sync mode with the correct startup order (subordinate
  before master). Not yet wired to the `/start` UI endpoint — needs the sync
  cable connected.

## Backends

### RealSenseBackend (`realsense.py`)

- Dependency: `pyrealsense2 >= 2.58`.
- Addressed by **serial** (the `device_id` from `list_devices`).
- Provides `project_color_to_depth` for depth-at-pixel queries.

### KinectBackend (`kinect.py`)

- Uses **ctypes directly over `libk4a.so`** — no `pyk4a`, no compilation.
- Addressed by `device_index` (`kinect_0`, `kinect_1`, ...).
- Known constraints (do not change without hardware testing):
  - `align_depth_to_color` is present but **bugged — do not enable**.
  - `WFOV_UNBINNED` depth mode is hardware-capped at **15 fps**; requesting
    30 fps raises `ValueError`.
  - `stop()` sleeps **2 s** to let USB fully release before the next open.
  - Hardware sync: always start **subordinate before master** (handled by
    `start_kinect_sync`).

## MultiCameraSync (`sync.py`)

Software synchronisation by host clock. Each worker keeps a short rolling
buffer (not just the latest frame) so frames landing a few ms before a tick
remain reachable.

- Driven by `sync_fps` (default 15) and `max_offset_us` (default 150 000 µs ≈
  half a 30 fps frame — conservative for USB jitter).
- At each tick it calls `nearest_to(tick_us)` on every required worker and
  accepts the group only if all required cameras are within tolerance; otherwise
  it emits `None`.
- Azure Kinect **hardware** sync (wired master/subordinate) aligns captures to
  ~1 ms; RealSense cameras are host-clock aligned only (software grouping).

## Recording (`recorder.py`)

`RGBDRecorder` records synchronised frames via `MultiCameraSync` to
`data/videos`: colour + colourised depth as MP4, raw depth as `.npy`, plus a
timestamps list. (Distinct from `viki/skeleton/recorder.py`, which saves
landmark trajectories.)

## Adding a backend

Subclass `CameraBackend` (implement `start`, `stop`, `get_frame`, `device_id`,
`is_running`, and optionally `project_color_to_depth` / `deproject_2d_to_3d`),
then add detection in `CameraManager.list_devices()` and routing in
`_make_backend()`.

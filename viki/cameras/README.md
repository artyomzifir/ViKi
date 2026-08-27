# viki.cameras — capture

**Stage 1** · live device → `episodes/<id>/raw/`

The only package that talks to camera SDKs (`pyrealsense2`, `libk4a`). Detects
and opens RealSense / Azure Kinect, streams colour (BGR uint8) + depth (uint16
mm) in per-camera worker threads, aligns frames across devices on a host clock,
and records whole scenes into an episode directory.

## Files

| file | what |
|---|---|
| `base.py` | `CameraBackend` ABC (compat re-export of `Frame` / `SyncedFrameGroup` / `CameraIntrinsics` from `viki.contracts`) |
| `realsense.py` / `kinect.py` | concrete backends, imported lazily by the manager so a missing SDK doesn't break other imports |
| `manager.py` | `CameraManager` — discovery, start/stop, per-camera ring buffer, backend factory |
| `sync.py` | `MultiCameraSync` — software host-clock grouping → `SyncedFrameGroup` |
| `record.py` | `SceneRecorder` — records synced RGB-D into `raw/` (colour `.mp4`, raw depth `.npy`, `timestamps.json`, the intrinsics/extrinsics in force) and marks `status.json` |

## Contract

- **out:** `Frame` (`latest_frame` / `nearest_frame`), `SyncedFrameGroup`, and the `raw/` directory layout.
- Consumers pull frames; there is no push pub/sub.
- Kinect quirks (`align_depth_to_color` disabled, WFOV capped at 15 fps, `stop()` sleeps 2 s) live in `kinect.py` and must not change without hardware.

## Adding a backend

Subclass `CameraBackend`, implement `start` / `stop` / `get_frame` / `device_id`
/ `is_running` / `project_color_to_depth`, then register it in
`CameraManager._make_backend()` and detection in `list_devices()`.

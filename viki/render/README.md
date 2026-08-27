# viki.render — visualisation

All pixel work: depth colourisation, undistortion, MJPEG encoding, and the
matplotlib 3-D views of trajectories / robot / skeleton. Pure numpy / cv2 /
matplotlib — **no FastAPI, no camera, no episode I/O** — so it is reusable from
scripts and testable without hardware.

## Files

| file | what |
|---|---|
| `depth.py` | `DepthColorizer` (uint16 depth → BGR turbo, EMA range, last-good hold), `Undistorter` (cached remap), `DepthStabilizer` |
| `mjpeg.py` | `mjpeg_chunk`, `placeholder` — multipart JPEG framing |
| `robot_viz.py` | `robot_trajectory_stream(h5, cfg)` — 3-D scene from a `plan.h5`: world/board, cameras, FK arm, human + robot EE trails, reach sphere, debug overlay |
| `smooth_viz.py` | `smooth_trajectory_stream(npz)` — raw vs smoothed wrist trajectory |
| `*_viz_shared.py` | pure helpers (coordinate transforms, FK, reach, config dataclasses) used by the two stream generators |

## Contract

- **in:** numpy arrays, or a `plan.h5` / `cln.npz` path.
- **out:** BGR frames, or an iterator of MJPEG chunks.
- Each stateful helper (`DepthColorizer`, `Undistorter`) holds per-stream state — one instance per stream.

*Follow-up:* the four `*_viz*` files could collapse into one `scene3d.py`; kept
split for now (no behaviour change, no test coverage on the 3-D path).

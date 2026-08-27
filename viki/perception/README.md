# viki.perception — video → 3-D hand skeleton

**Stage 2** · `episodes/<id>/raw/` → `rec.npz`

Offline. Decode the recorded colour + depth, run a pluggable hand-pose backend
per camera per frame, lift the 2-D detection to 3-D with measured depth,
transform into the workspace frame with the recorded extrinsics, and write
per-camera landmark trajectories. Cross-camera fusion is **not** done here — it
is deferred to `viki.prepare`.

## Files

| file | what |
|---|---|
| `extract.py` | offline orchestrator: `raw/` → `rec.npz`. Assumes depth aligned to colour (identity colour→depth projector). |
| `backends/` | `HandPoseBackend` ABC + implementations — see `backends/README.md` |
| `camera_prep.py` | `prepare_frame` : `Frame` → `PreparedFrame` (RGB, depth in metres, depth K) |
| `geometry.py` | `lift_to_3d` (2-D + depth → camera-frame 3-D), `camera_landmarks_to_world` (apply extrinsics) |
| `hand_angles.py` | `compute_end_effector_pose` — the single site that derives the wrist SE(3) pose from landmarks (also used by `prepare`) |
| `pipeline.py` | `SkeletonPipeline` — per-`SyncedFrameGroup` orchestration (kept for tooling; the offline path is `extract.py`) |
| `models.py` | compat re-export of the perception DTOs from `viki.contracts` |

## Contract

- **in:** `PreparedFrame`, `DepthProjector` (Protocol), `CalibrationExtrinsics`.
- **out:** `rec.npz` — keys in `contracts.REC_KEYS`:
  `device_ids`, `timestamps`, `points (N,21,3)`, `landmark_ids (21,)`,
  `confidence (N,21)` *(stub: detector visibility only)*.

## Palm frame

`compute_end_effector_pose` builds the wrist rotation from the MCP knuckle
spread (INDEX→PINKY), **not** the thumb: `x = norm(MIDDLE_MCP − WRIST)`,
`z = norm(x × (PINKY_MCP − INDEX_MCP))`, `y = z × x`. Missing a required
landmark falls back to the centroid of the available palm landmarks with an
identity rotation.

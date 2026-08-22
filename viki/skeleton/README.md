# ViKi Skeleton

Phase-2 package: live hand-pose estimation from multi-view RGB-D, depth-fused
3-D keypoints, and per-camera recording of landmark trajectories. This package
stops at **recording** — smoothing, fusion, and end-effector (EE) pose
computation happen later in `viki/optimization/preparation/`.

## Responsibilities

- Per-camera frame preparation (colour/depth conversion, intrinsics).
- MediaPipe hand detection (`detectors/`).
- Lift 2-D detections to 3-D using measured depth (`geometry.py`).
- Transform each camera's landmarks into a shared world frame.
- Compute EE pose (position + palm rotation) per camera for live preview.
- Record per-camera trajectories to `rec-*.npz`.

## Layout

```text
viki/skeleton/
  pipeline.py        # SkeletonPipeline: orchestrates prepare -> detect -> lift -> world
  models.py          # LM enum, PreparedFrame, HandDetection, Landmarks3D, EndEffectorPose, ...
  camera_prep.py     # Frame -> PreparedFrame (no MediaPipe dependency)
  detectors/         # Modular detection: CompositeLandmarkDetector + PartialLandmarkDetector
    base.py          # PartialLandmarkDetector, PartialDetection2D, FusionMode
    mediapipe_base.py# MediaPipeTaskRunner (model loading/inference)
    hand_pose.py     # MediaPipeHand
    arm_pose.py      # MediaPipeArm (reserved; disabled)
    composite.py     # CompositeLandmarkDetector: assembles detectors -> HandDetection
  geometry.py        # lift_to_3d, camera_landmarks_to_world
  hand_angles.py     # compute_end_effector_pose, compute_palm_rotation
  recorder.py        # SkeletonRecorder: per-camera trajectories -> rec-*.npz
  test_hand_angles.py
  __init__.py
```

## Landmarks (`models.LM`)

MediaPipe Hands indices 0–20, plus two reserved arm landmarks kept for schema
compatibility (not produced by the detector):

```text
0  WRIST          1  THUMB_CMC       5  INDEX_MCP       9  MIDDLE_MCP
13 RING_MCP      17  PINKY_MCP      ...  (full 21-hand layout)
21 ELBOW         22 SHOULDER        (reserved, not detected)
N = 23
```

For optimisation, elbow/shoulder are ignored; `WRIST` supplies the position
target and `WRIST`/`THUMB_CMC`/`MIDDLE_MCP` the palm orientation.

## Pipeline (`pipeline.py`)

`SkeletonPipeline.process(group: SyncedFrameGroup) -> PipelineResult` runs, per
SyncedFrameGroup, for **each** camera sequentially (no concurrent MediaPipe
inference — two live models contend on the GPU and produce stale detections):

1. `prepare_frame` → `PreparedFrame` (RGB, depth in metres, depth intrinsics).
2. `CompositeLandmarkDetector` → `HandDetection` (pixel-space, one hand).
3. `lift_to_3d` → `Landmarks3D` (camera-frame 3-D from measured depth).
4. `camera_landmarks_to_world(extrinsics)` → world-frame points.
5. `compute_end_effector_pose` → `SkeletonFrame.end_effector` (position +
   `R_world_palm`).

One `SkeletonFrame` per camera is emitted, tagged with its `device_id`.
**Capture-time fusion is intentionally not performed here** — per-camera
trajectories are recorded as-is and fused later in the preparation stage.

## Detection (`detectors/`)

`CompositeLandmarkDetector` holds a list of `PartialLandmarkDetector`s (currently
`MediaPipeHand`; `MediaPipeArm` is reserved/disabled) and merges their pixel
outputs into a single `HandDetection` per frame, honouring a `FusionMode`.

## Geometry (`geometry.py`)

- `lift_to_3d` — project each landmark to depth space, deproject at its own
  measured depth, and estimate the hand **position** (wrist-only or robust
  palm/knuckle median, with an outlier filter capped at
  `DISCARD_OUTLIERS_MAX_PORTION`).
- `camera_landmarks_to_world` — apply per-camera extrinsics to produce the
  shared world-frame point set consumed downstream.

## End-effector pose (`hand_angles.py`)

`compute_end_effector_pose(world_points, timestamp_us)` returns the full
world-frame wrist pose:

```text
position      : (3,)        WRIST world XYZ (m)
R_world_palm  : (3, 3)      rotation: palm frame -> world
                x_palm = normalise(MIDDLE_MCP - WRIST)
                z_palm = normalise((MIDDLE_MCP - WRIST) x (THUMB_CMC - WRIST))
                y_palm = z_palm x x_palm
```

`compute_palm_rotation` is the palm-only rotation helper; it is the single copy
used by both the live pipeline and `viki/optimization/preparation/`.

## Recording (`recorder.py`)

`SkeletonRecorder` buffers `(device_id, SkeletonFrame)` pairs during a session
and, on `stop()`, writes one compressed NPZ (`rec-<HH.MM-dd.mm.YYYY>.npz`) to
`data/skeleton_recs` with the per-camera `points`, `timestamps`, `landmark_ids`,
and depth-debug columns. When `SKELETON_SAVE_JSON_DEBUG` is set, a JSON mirror is
also written. It records **landmarks only** — no smoothing/fusion here.

## API (`/api/skeleton`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/skeleton/toggle` | Enable/disable live estimation. |
| POST | `/api/skeleton/record` | Start/stop a recording session. |
| POST | `/api/skeleton/depth-debug` | Toggle per-frame depth diagnostics. |
| POST | `/api/skeleton/capture_base/{device_id}` | Snapshot static background depth for a camera. |
| GET | `/api/skeleton/status` | Worker running/recording state. |
| WS | `/api/skeleton/stream` | Live 3-D skeleton frames. |

> Raw `rec-*.npz` → prepared `cln-*.npz` (smoothing / interpolation / fusion)
> lives in the **optimisation** router:
> `/api/optimization/recordings`, `/api/optimization/smooth`,
> `/api/optimization/smooth-plot`. Prepared `cln-*.npz` → robot `.h5` is the
> **dataset** router (`/api/dataset/*`).

## Tests

```bash
python -m unittest viki.skeleton.test_hand_angles
```

`test_hand_angles` covers `compute_end_effector_pose` / `compute_palm_rotation`
and requires no camera hardware.

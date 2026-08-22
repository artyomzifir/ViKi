# Optimisation: Preparation & Retargeting

This directory holds the skeleton → robot optimisation workflow, organised into
`preparation/` (recorded landmarks → end-effector pose) and `retarget/`
(end-effector pose → IK solution `.h5`).

No frontend code or Docker mounts are changed for this workflow.

## Directory Layout

```text
viki/optimization/
  __init__.py
  preparation/                # landmarks -> end-effector rotation + position
    __init__.py
    processor.py              # PreparationPipeline: list + smooth recordings -> cln-*.npz
    smoothing.py              # Savitzky-Golay helpers for landmark sequences
    fusion.py                 # cross-camera trajectory fusion onto a common time grid
    test_processor_orientation.py
  retarget/                   # end-effector pose -> IK solution (.h5)
    __init__.py
    retarget_rgb_only.py      # IK retargeting entry points: retarget / retarget_from_poses
    archive_io.py             # HDF5 trajectory writer/reader + smoothed-target loader
    smoothing.py              # Savitzky-Golay helpers for joint trajectories
    eval_tracking_error.py    # FK-based tracking-error evaluation + plot helpers
    debug.py                  # retarget debug visualisation
    test_*.py                 # Lightweight tests
```

The package is tracked normally under `viki/optimization/` — no force-add needed.

## Input Recording Format

The pipeline consumes ViKi skeleton recordings named `rec-*.npz` produced by the
capture server (`SkeletonRecorder`), stored under:

```text
data/skeleton_recs/          # raw recordings (per-camera landmark trajectories)
```

Each NPZ keeps the **per-camera** trajectories (capture-time fusion is
deliberately not performed at record time) plus depth-debug columns. Layout:

```text
device_ids        : (N,)        object array of camera ids (one per recorded frame)
timestamps       : (N,)        int64 sync timestamps (µs)
points            : (N, L, 3)   world-frame landmark positions (NaN where missing)
landmark_ids      : (L,)        int32 landmark id for the second axis of points
depth_debug_*     : per-frame, per-camera depth diagnostics (valid fraction,
                    median/mean depth, wrist depth, hand-detected flag)
```

Landmark indices follow `viki.skeleton.models.LM` (MediaPipe Hands 0–20 plus
reserved arm landmarks 21 ELBOW, 22 SHOULDER). For optimisation, elbow/shoulder
are ignored. Landmark `0` (WRIST) supplies the wrist position target; landmarks
`0`, `1` (THUMB_CMC), and `9` (MIDDLE_MCP) supply the palm orientation for
`hand_se3`.

## Pipeline

The legacy ViKi2.3 JSON converter (`convert_viki23_json.py`) and its
`/api/optimization/convert` + `/api/optimization/recordings` endpoints have been
removed — there is no legacy format to import.

The current flow is:

```text
skeleton recording (rec-*.npz) -> PreparationPipeline.smooth_recording
  (per-camera interpolate, cross-camera fuse, smooth, compute EE pose)
  -> cln-*.npz (positions, rotations, valid, timestamps)
  -> retarget (PINK IK) -> data/robot_out/*.h5
```

`PreparationPipeline.smooth_recording` reads a raw `rec-*.npz`, interpolates and
smooths per-camera landmark trajectories, fuses the cameras onto a common time
grid, and computes end-effector poses. The resulting `cln-*.npz` is the input to
the retarget endpoints. `compute_end_effector_pose` (in
`viki/skeleton/hand_angles.py`) is the single site that derives the EE pose from
landmarks.

## Running

There is no standalone CLI; the workflow is driven either through the FastAPI
endpoints (see below) or by calling the Python entry points directly:

```python
from viki.optimization.preparation.processor import PreparationPipeline
from viki.optimization.retarget.retarget_rgb_only import retarget

prep = PreparationPipeline()
prep.smooth_recording("data/skeleton_recs/rec-12.00-01.01.2026.npz")  # -> data/skeleton_smoothed/cln-*.npz

retarget(
    sample_path=Path("data/skeleton_smoothed/cln-12.00-01.01.2026.npz"),
    out_path=Path("data/robot_out/boardbase_ur10"),
    cfg=RunConfig(robot=normalize_robot("ur10"), target_mode="hand_se3", ...),
)
```

Full IK execution requires the `viki-fk` (PINK/Pinocchio) environment; the
preparation and retarget *logic* tests do not.

## FastAPI Endpoints

The optimisation router is mounted at `/api/optimization` and handles the
**raw recording → prepared data** stage (smoothing/interpolation/fusion).
Retargeting (prepared → robot `.h5`) is exposed via the `/api/dataset/*`
router, not here.

```text
GET  /api/optimization/recordings    # list raw rec-*.npz (paginated)
POST /api/optimization/smooth        # smooth a rec-*.npz -> cln-*.npz
GET  /api/optimization/smooth-plot   # raw-vs-smoothed wrist trajectory PNG
```

### List Recordings

```text
GET /api/optimization/recordings?page=0&limit=10
```

Returns `{ "recordings": [...] }` — the raw `rec-*.npz` files in the
configured recordings directory.

### Smooth

```text
POST /api/optimization/smooth
```

Request body (`SmoothRequest`):

```json
{
  "filename": "rec-17.20-12.07.2026.npz",
  "window_length": 7,
  "polyorder": 2
}
```

Runs `PreparationPipeline.smooth_recording` and writes a prepared `cln-*.npz`
into `data/skeleton_smoothed`. Returns `{ "status": "success", "path": "..." }`.

### Smoothing Plot

```text
GET /api/optimization/smooth-plot?filename=cln-17.20-12.07.2026.npz
```

Returns a PNG comparing the raw and smoothed wrist trajectories for the named
prepared file.

(The `/api/dataset/*` router in `viki/server/routes/dataset.py` is the
higher-level UI entry point for the prepared → `.h5` step: it lists smoothed
recordings, runs `retarget_from_poses`, and streams robot trajectories. See the
server README.)

## Retargeting Behavior

Use `target_mode="wrist_position"` for position-only retargeting. In this mode
orientation does not drive the robot trajectory.

Use `target_mode="hand_se3"` to use wrist position plus palm orientation. The
palm frame is derived from hand bones:

```text
x_palm = normalize(MIDDLE_MCP - WRIST)
z_palm = normalize((MIDDLE_MCP - WRIST) x (THUMB_CMC - WRIST))
y_palm = z_palm x x_palm
```

`hand_se3` writes `ee_target_rot` and `orientation_valid` into the HDF5
trajectory output. `stable_palm_orientation_mask` (in `preparation/processor.py`)
rejects palm frames with implausible hand geometry or temporal flips before the
EE rotation is trusted.

For uncalibrated debug runs, `align_initial_orientation` maps the first valid
palm orientation onto the robot's neutral end-effector rotation, then tracks
relative hand rotation after that. Leave it off when the sample orientation is
already calibrated to the robot tool frame.

The intended current flow is:

```text
ViKi skeleton recording -> PreparationPipeline -> wrist_position or hand_se3 retargeting
```

Current board-base defaults (see `_retarget_defaults` in the router):

```text
UR10 hand_se3: trajectory-scale-origin auto, scale 0.75, orientation cost 0.6, align-initial-orientation
iiwa14 hand_se3: trajectory-scale-origin auto, scale 0.55, orientation cost 0.3
```

Literal calibrated scale, when the target is reachable:

```text
trajectory-scale-origin robot_base, trajectory-scale 1.0
```

Do not use `recenter_to_neutral` for calibrated robot-base trajectories because
it hides the real spatial relationship between the skeleton and robot.

## Tests

Run from the repo root (the logic tests do not require PINK/Pinocchio):

```bash
python -m unittest discover viki/optimization/preparation
python -m unittest discover viki/optimization/retarget
```

The retarget logic tests do not require PINK/Pinocchio. Full IK execution still
requires the `viki-fk` conda environment.

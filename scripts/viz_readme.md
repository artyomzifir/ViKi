# test_extrinsics_viz.py — Extrinsics & Retargeting Visualisation

## What it does

Loads a skeleton recording and extrinsics calibration, then produces a 3D
Matplotlib plot (`scripts/extrinsics_viz.png`) showing:

- The **world/board origin** (red square) and a reference point 1 m above it
- **Camera positions** computed from extrinsics (rvec/tvec) with gaze direction
  arrows
- The **robot base location** from `ROBOT_BASE_OFFSET` (black square)
- The **robot's neutral end-effector** position (orange star)
- An approximate workspace sphere centred at the robot base
- The **human wrist path** in world coordinates (magenta)
- The **robot EE trajectory** computed by the same transforms used during
  dataset IK (cyan)
- An optional **debug overlay** from a completed retargeting job
  (`data/retarget_debug.json`)

## How to run

```bash
# From the project root, inside the Docker container:
python3 scripts/test_extrinsics_viz.py \
  --sample data/skeleton_smoothed/cln-17.20-12.07.2026.npz \
  --start-frame 50 --end-frame 300
```

### Arguments

| Flag             | Default                  | Description                              |
|------------------|--------------------------|------------------------------------------|
| `--sample`       | — (required)             | Path to a cleaned `.npz` skeleton file   |
| `--frame`        | `0`                      | Single frame to highlight                |
| `--start-frame`  | same as `--frame`        | First frame of the trajectory segment    |
| `--end-frame`    | `--frame` + 1            | Last frame (exclusive)                   |
| `--robot`        | from `RETARGET_DEFAULT_ROBOT` | Robot alias (`ur10`, `iiwa14`, …)   |
| `--debug-file`   | `data/retarget_debug.json` | Optional debug JSON from the IK pipeline |

Output is always written to `scripts/extrinsics_viz.png`.

## How the two repositions work

The script applies two coordinate transforms — the same ones the dataset IK
pipeline uses. They are defined inline at lines 206–216 and 45–48.

### 1. `to_robot_frame(p_world)` — human wrist → robot target

Transforms a world/board-relative position into a robot-base-relative target:

```
p_robot = p_world + TARGET_OFFSET - ROBOT_BASE_OFFSET
```

Then, optionally:

- **Recentre to neutral** — shifts the whole trajectory so that frame-0's
  wrist aligns with the robot's neutral end-effector position
  (`p_neutral_robot`). This means the robot starts at its neutral pose instead
  of wherever the human started.

```
p_robot = p_robot + (p_neutral_robot - p_robot_at_frame_0)
```

- **Trajectory scale** — when `trajectory_scale != 1.0`, the motion is scaled
  relative to an anchor point. If recentering is ON, the anchor is the neutral
  EE position; otherwise it is the first-frame wrist. This keeps the motion
  centred at the anchor rather than flying away from the robot base.

```
p_robot = anchor + (p_robot - anchor) × scale
```

### 2. `get_robot_world_pos(p_robot)` — robot target → plot coordinate

Simply adds `ROBOT_BASE_OFFSET` back so robot-frame targets can be overlaid
with world-frame skeleton data in the same plot:

```
p_world = p_robot + ROBOT_BASE_OFFSET
```

Without this, the human path (world coords) and robot path (robot coords)
would live in different origins and could not be compared visually.

### Visual result

- **Magenta path**: human wrist in world coordinates (raw skeleton data)
- **Cyan path**: robot EE after `to_robot_frame` + `get_robot_world_pos` —
  what the robot would actually do given the current configuration

If the cyan path veers far from the magenta path (especially at
`trajectory_scale > 1`), it indicates the scale origin is wrong — see the
`trajectory_scale_origin` parameter in `RunConfig`.

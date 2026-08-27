# viki.retarget — IK to a robot trajectory

**Stage 4** · `cln.npz` → `plan.h5`

Take the end-effector targets, move them into the robot base frame, and solve
differential IK (PINK on Pinocchio) for a joint trajectory: an approach from
neutral to the first target, then scene tracking, then a Savitzky-Golay pass.

## Files

| file | what |
|---|---|
| `run.py` | `retarget` / `retarget_from_poses` / `retarget_episode(ep, robot)` — the IK loop and archive write |
| `robots.py` | `RobotConfig` (description, EE frame, joint names), `ROBOT_CONFIGS` (`ur3` `ur5` `ur10` `ur5e` `ur10e` `iiwa14`), `ROBOT_ALIASES`, `normalize_robot` |
| `frames.py` | `world_to_robot(cfg)` — `T^W_R` from `RETARGET_BASE_ROTATION` / `RETARGET_BASE_TRANSLATION`. Fixed config constant (a hand-eye procedure, paper §3.3, would produce it). |
| `cost.py` | seam for the eq. 4 cost functional — see **Stubbed** |
| `archive.py` | `.h5` / `.npz` trajectory archive read/write (`Hdf5Archive`, `write_hdf5_archive`) |
| `evaluate.py` | FK-based tracking-error evaluation + debug plots (dev only) |
| smoothing | `viki.dsp` |

## Contract

- **in:** `cln.npz` (`positions`, `rotations`, `valid`) + `RunConfig` built from the `RETARGET_*` config keys.
- **out:** `plan.h5` — `q_approach`, `q_scene_raw`, `q_scene_smooth`,
  `ee_target_pos` / `ee_target_rot`, `pos_err_smooth` / `ori_err_smooth`,
  `fps`, `dt`, `robot`, `ee_frame`, config echo. This is the *synthesised*
  trajectory and its *model* tracking error, not what a robot attained.

## Stubbed

The cost functional in the code is the two working PINK tasks (frame + posture).
`cost.build_tasks` / `huber_residual` / `acceleration_penalty` /
`collision_barriers` raise `NotImplementedError` — the Huber robustifier, the
explicit `λ_a‖q_t − 2q_{t−1} + q_{t−2}‖²` term, and the collision barriers of
paper eq. 4 are not wired in; joint smoothing is a post-hoc Savitzky-Golay pass.

Full IK needs Pinocchio / PINK (`require_ik_dependencies()` gives the message).

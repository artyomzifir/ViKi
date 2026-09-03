# viki.prepare — fuse, smooth, represent

**Stage 3** · `rec.npz` → `cln.npz`

Turn the per-camera landmark trajectories into one clean end-effector
trajectory: interpolate gaps, fuse across cameras onto a common time grid,
smooth, derive the wrist pose and gripper state, and — when an object-pose
track exists — the object-relative form.

## Files

| file | what |
|---|---|
| `run.py` | `PreparationPipeline` + `prepare_episode(ep)` — orchestrates the below, writes `cln.npz`, marks status |
| `fuse.py` | `fuse_trajectories(trajs, ts, ids, weights=None)` — resample to a common grid, then `Σ w·x / Σ w` per landmark per step (paper eq. 2). Plain mean when `weights` is `None`. |
| `interpolate.py` | `fill_linear` (working) · `fill_se3_spline` *(stub, paper §3.7 — falls back to linear)* |
| `represent.py` | `object_relative(wrist_world, object_world)` = `inv(O)·H` *(stub: no object tracker → returns `None`, paper §3.6)* |
| Savitzky-Golay | in `viki.dsp` (`smooth_landmark_sequence`), shared with `retarget` |
| gripper | `viki.gripper.BinaryGripper` over the fused frames |

## Contract

- **out:** `cln.npz` — keys in `contracts.CLN_KEYS`:
  `timestamps`, `positions (T,3)`, `rotations (T,3,3)`, `valid (T,)`,
  `omega (T,)` (aggregated confidence), `gripper (T,)`, `coordinate_frame`,
  `raw_points`, `smoothed_points`, `landmark_ids`.
  Optional `landmark_confidence (T,L)`, `T_world_obj` / `T_obj_hand`, and the
  non-destructive `hand_fit_*` trajectory arrays when those stages run.

## Stubbed

- fusion weights: the caller passes detector visibility only — the range and
  incidence factors of eq. 2 are not computed.
- object-relative representation, SE(3) spline interpolation.

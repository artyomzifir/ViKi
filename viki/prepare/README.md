# viki.prepare — fuse, smooth, represent

**Stage 3** · `rec.npz` → `cln.npz`

Turn the per-camera landmark trajectories into one clean end-effector
trajectory: interpolate gaps, fuse across cameras onto a common time grid,
smooth, derive the wrist pose and gripper state, and — when an object-pose
track exists — the object-relative form.

## Files

| file | what |
|---|---|
| `run.py` | `PreparationPipeline` + `prepare_episode(ep)` — orchestrates the below, writes `cln.npz`, marks status; `generate_stage_checkpoints(ep)` makes non-destructive A/B runs |
| `fuse.py` | `fuse_trajectories(trajs, ts, ids, weights=None)` — resample to a common grid, then `Σ w·x / Σ w` per landmark per step (paper eq. 2). Plain mean when `weights` is `None`. |
| `interpolate.py` | per-coordinate linear and natural-cubic gap filling; `max_gap` prevents fabrication across long occlusions. This is **not** a geometry-preserving SE(3) spline. |
| `checkpoints.py` | atomic NPZ/JSON checkpoint persistence plus motion/anatomy diagnostics |
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

Every episode prepare run also keeps its material boundaries under
`intermediates/prepare/<fusion>__gap-<N>__sg-<window>-<polyorder>/`:

| checkpoint | exact boundary |
|---|---|
| `00_per_camera_observed.npz` | detector/lift output separated by camera |
| `05_per_camera_filled.npz` | per-camera linear gap fill |
| `10_fused_observed.npz` | fusion/triangulation output, before fabrication |
| `20_fused_filled.npz` | fused coordinate-wise cubic gap fill |
| `30_smoothed.npz` | Savitzky–Golay output |
| `40_hand_fit.npz` | optional capsule trajectory fit |

The last four are complete viewer-compatible artifacts. `manifest.json` records
the knobs, while `comparison.json` reports finite/direct fractions, motion
derivatives, anatomical outlier frames, and palm-collapse frame indices.

Generate A/B variants without replacing the active `cln.npz`:

```bash
viki checkpoints <episode> --fusion triangulate xyz_mean --interp-max-gap 6
```

Running the command again with another gap/window adds a parameter-named run;
the episode-level `intermediates/prepare/comparison.json` remains cumulative.

## Frozen clean baseline

The named profile `clean-triangulated-landmarks-v1` reproduces the validated
`pick_up_u` output independently of experimental config. It requires real
two-view triangulation, locks fill-all + SG 7/2 + landmark pose, disables
`hand_fit`, and protects the first result under `intermediates/baselines/`.
Run it with:

```bash
viki perceive <episode> --profile clean-triangulated-landmarks-v1
```

Full parameters, provenance and the reference hash are documented in
[`docs/clean_baseline_v1.md`](../../docs/clean_baseline_v1.md).

## Stubbed

- fusion weights: the caller passes detector visibility only — the range and
  incidence factors of eq. 2 are not computed.
- object-relative representation and geometry-preserving SE(3) interpolation.
## Stable fused + articulated profile

`stable-fused-hand-v1` is the default complete perception route.  It protects
the clean triangulated CLN, then creates non-destructive `40_projected.npz` and
`50_optimized.npz` variants under
`intermediates/geometry/articulated-landmarks-v1/` and installs the optimized
candidate under additive `hand_fit_*` keys.  The dense depth cloud is excluded.
Viewer routing is explicit: `smoothed_points` → **fused** and
`hand_fit_capsules` → **hand fit**.  See `docs/articulated_geometry_v1.md` for
the contract, metrics, and manual reproduction commands.

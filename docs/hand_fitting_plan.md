# Plan — articulated hand-to-point-cloud fitting

## Goal

Refine the wrist pose (`EndEffectorPose`: `position` + `R_world_palm`) by fitting
a parametric articulated capsule hand model to the per-frame hand point cloud,
instead of trusting one-shot triangulation of 21 sparse fused landmarks. Also
emit the **full per-frame joint-angle vector** as groundwork for an
anthropomorphic gripper later.

Offline only → a converging GN/LM solve per frame is fine.

## Files

| file | role |
|---|---|
| `viki/perception/hand_model.py` | parametric capsule hand (Pinocchio FK, `LM` topology), per-user calibration, landmark warm-start |
| `viki/perception/hand_fit.py` | point→capsule data term, regularisers, `scipy.least_squares` solver, `refine_cln(ep)` orchestration + hand-ROI cloud |
| `viki/prepare/run.py` | `prepare_episode` calls `refine_cln(ep)` after cln.npz is written, iff `PERCEPTION_HAND_FIT` |
| `viki/contracts.py` | `CLN_OPTIONAL_KEYS` += `hand_joint_angles`, `hand_model_nq` |
| `viki/config.py` + both JSONs | `PERCEPTION_HAND_FIT` (bool, default **false**) + tunables |
| `viki/cli.py` | `viki hand-fit <episode>` thin wrapper |
| `tests/unit_tests/perception/test_hand_fit.py` | point-to-capsule maths, residual assembly, synthetic convergence (`@slow`, needs pinocchio) |

## Open decisions (fixed here)

1. **Cloud source** → *re-deproject a dense hand-ROI cloud from raw depth per
   frame*, not the on-disk artifact. The artifact is voxel-5 mm + capped +
   stride — too sparse across a ~15 mm finger. The ROI is a ~0.12 m sphere
   around the fused wrist estimate; deproject full-res (stride 1), background
   subtracted, **no voxel**. Cheap (~one 200 px box/camera), transient, not
   persisted. Reuses `K4ACalibration.color_deproject_maps` + recorded extrinsics
   + the preset background npz — the exact maths `cloud.py::_camera_cloud` uses.
2. **Integration point** → inside `prepare_episode`, right after `cln.npz` is
   written, gated by `PERCEPTION_HAND_FIT`. `refine_cln(ep)` is also the
   standalone entry (CLI `viki hand-fit`), so it can be re-run on prepared
   episodes. Keeps `smooth_recording` (npz-only, no `raw/`) untouched.
3. **Capsule calibration** → from the recording itself: pick the N frames with
   the widest mean inter-fingertip spread (open hand), take the median of each
   segment length across them; radius = `radius_frac · seg_len`
   (`radius_frac = 0.35`, ~human finger radius/segment ratio). Fitting radius
   from cloud cross-section thickness is a documented TODO.

## Model

URDF built from a template string (`pin.buildModelFromXML`), parametrised by the
calibrated segment lengths:

- root link `palm`; **free-flyer** joint `wrist` (the 6-DOF we want).
- 4 fingers: `mcp` = 2 revolute (abduction ⟂ palm, then flexion), `pip` = 1
  revolute flexion, `dip` = 1 revolute flexion.
- thumb: `cmc` = 2 revolute, `mcp` = 1, `ip` = 1.
- ⇒ `nq ≈ 7 (free-flyer) + 4·5`, `nv ≈ 6 + 20 = 26`.

Each segment (`palm→mcp`, `mcp→pip`, …) carries a **capsule**: two Pinocchio
frames at its endpoints + a radius. `fk_capsules(q)` returns `[(a, b, r), …]` in
world.

Warm start `q_from_landmarks(fused_pts)`: wrist SE(3) from the existing palm
frame (`compute_palm_rotation`, WRIST + INDEX/MIDDLE/PINKY MCP) + wrist position;
per-finger flexion from the angle between consecutive segment vectors; abduction
from the in-palm-plane component. Clamped to joint limits.

## Functional (mirrors thesis eq. 4, separate instance — *not* robot IK)

```
E(θ) = Σ_i w_i · ρ_δ( d(x_i, M(θ)) )        data: point → nearest capsule surface, Huber
     + λ_vel · ‖θ_t ⊖ θ_{t-1}‖²             temporal velocity  (tangent space)
     + λ_acc · ‖θ_t ⊖ θ_pred‖²              optional acceleration (const-vel extrapolation)
     + λ_prior·( relu(θ−θ_max)+relu(θ_min−θ) ) + λ_posture·‖θ_fingers − θ_rest‖²
```

Solver: `scipy.optimize.least_squares(f, δθ, method='trf', loss='huber',
f_scale=δ)`, optimising the tangent increment `δθ ∈ R^nv` about a reference
config (`θ = pin.integrate(model, θ_ref, δθ)`), so the free-flyer stays on
SO(3)×R³. Jacobian: analytic via `pin.computeJointJacobians` + capsule-endpoint
chain rule; finite-difference fallback on the first pass is acceptable (**TODO**
marked). Warm start: frame 0 from landmarks, frame t from θ_{t-1}.

## Integration contract

- `positions[t]` / `rotations[t]` / `valid[t]` / `timestamps` in `cln.npz` stay
  the same shape/meaning — `refine_cln` overwrites `positions`/`rotations`
  **in place** only where `valid[t]` and the fit's median point residual is
  under a threshold; otherwise the landmark pose is kept.
- New optional `cln.npz` arrays: `hand_joint_angles` (T, nq), `hand_model_nq`
  (scalar). Nothing downstream reads them yet.
- Default **off** — existing behaviour and tests unchanged.

## Not touched

`retarget/cost.py`, object tracker, collision barriers, the visualisation cloud
artifact, no MANO/HaMeR/SMPL assets or references.

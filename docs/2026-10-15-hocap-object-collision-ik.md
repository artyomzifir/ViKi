# HO-Cap tracked-object IK collision barrier (2026-10-15)

## Scope

The explicit `--object-track-path` / `--object-id` retarget path now adds hard
signed-distance barriers to the same whole-trajectory QP as self-collision and
the tabletop. Each tracked object's canonical cloud is bounded by a local
oriented box; its per-frame pose is transformed from `hocap_tag_1` into the
robot-base frame. Every movable URDF collision body avoids all object boxes.
Only the Robotiq collision bodies may overlap the *selected* object during
closing, carry and release. From `grasp_start` through `release_start`, a box
rigidly attached at the target grasp transform also avoids the other objects.
The nominal model clearance is 5 mm. Missing, weak or misaligned tracks fail
before IK, and a final barrier violation rejects the plan.

This is discrete-frame geometry, **not** a contact-force model, continuous-time
collision proof or physical safety guarantee. Canonical clouds may omit hidden
surface; oriented boxes need not enclose the official complete mesh. The
solver uses the *recorded* poses of other objects, not their live positions in
MuJoCo after earlier contacts. It also does not solve collisions at seams when
independent object-aware clips are interpolated into one full episode.

## Checks on existing G10_3

The 235-frame `data/_retarget_diagnostics/hocap_g10_full_object_3/plan.h5`
was read, not overwritten. The aligned `object_track.npz` contains all four
objects. Against the new 5 mm box margin, the old plan's minimum robot/scene
margin is **+2.499 mm at local frame 201**. The minimum carried-G10_3 versus
other-box margin from the start of closing to release is **+21.944 mm at local
frame 63**. A 20-frame whole-trajectory QP smoke (`54:74`, two iterations,
old joint plan as warm start) completes with **+9.179 mm** total collision
margin and **8.02 mm** maximum TCP-position error. Its floor was not configured
in that isolated smoke; the production retarget path still configures the
tabletop constraint.

The prior full MuJoCo replay reports first G10_3–G10_4 mesh contact at
**16.5555 s**. The positive *nominal* box clearance above shows the new static
barrier alone does **not** explain or guarantee prevention of that simulated
contact. A plausible mechanism is divergence of the free simulated object's
pose from its recorded track after earlier contacts, plus discrepancies between
cloud boxes and mesh convex hulls; this has not been separately isolated.
The same replay reports up to 0.05085 rad arm command-tracking error, which is
another unmodelled contributor to clearance loss; it does not prove which
contact mechanism dominated.
Any claim of no unintended object collision needs a fresh full-episode replay
with explicit contact measurements, not just a passing IK margin.

## Verification

- `tests/unit_tests/retarget/test_obstacles.py`: aligned-box loader, fail-closed
  validation, contact exception and finite-difference Jacobian.
- `docker compose run --rm -T test tests/ -q -p no:cacheprovider`:
  514 passed, 10 skipped on this host.

## Complete G10 episode after the barrier

Four independent object-conditioned plans were rerun with the archived robot,
gripper and solver parameters, but with this new obstacle barrier. Inputs and
outputs are preserved separately in
`data/_retarget_diagnostics/hocap_g10_collision_20261015/`; the original plans
were not overwritten. `experiments/hocap_collision_batch.py` is the driver and
`batch_status.json` contains the full per-clip numeric results. All four IK
clips completed. The `cln.npz` files request `hand_fit` but contain no
`hand_fit_*` arrays, so the loader explicitly fell back to landmark poses.

| Clip | Frames | TCP RMSE | Robot/object margin above 5 mm | Carried/other margin above 5 mm |
|---|---:|---:|---:|---:|
| G10_2 | 190 | 35.79 mm | +0.275 mm | +54.23 mm |
| G10_3 | 235 | 12.10 mm | +0.007 mm | +12.17 mm |
| G10_4 | 180 | 29.23 mm | +0.0004 mm | +37.17 mm |
| G10_1 | 196 | 69.27 mm | +0.017 mm | +0.415 mm |

The four plans were stitched into the **complete 831-frame** trajectory and
replayed once with the same official-mesh MuJoCo scene, 2× recording time,
and opt-in first-pad-contact attachment. Raw output is
`data/experiments/hocap-rl/contact_attach_smoke/collision_20261015_full.json`;
the complete command trajectory and video are the adjacent `_trajectory.npz`
and `_full.mp4` files. The simulation reached **831/831 frames**, but only
G10_2 and G10_3 attached and lifted. G10_1 and G10_4 did not lift. Final
position errors for G10_1/2/3/4 were **11.2/21.5/126.1/116.4 mm**, where
G10_1's small error is not a success because it was not moved. There were
still G10_3–G10_4 and G10_1–G10_3 mesh contacts; their first times were
16.5745 s and 29.4970 s, with peak normal forces 1.686 N and 2.230 N.

At the first G10_3–G10_4 contact (nearest logged frame 245), G10_3 was
already attached. Its simulated displacement from the initial pose differed
from the recorded displacement by **55.7 mm**; G10_4 differed by 11.2 mm.
The maximum arm command-tracking error was 0.0647 rad. Thus the IK's positive
*recorded-pose* obstacle margin did not describe the actual simulator state.
The arbitrary first-pad weld preserves the object's contact-time offset rather
than making it follow the ideal object-to-TCP transform used by the barrier.
This is a likely mechanism for the continuing collision, not a guarantee that
it is the sole cause. Preventing it requires an actual-state-dependent
clearance/replanning rule or a better attachment/contact gate, plus a new
full-scene test. Increasing a static 5 mm margin alone cannot be called a fix.

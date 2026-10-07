# Experimental cube-aware grasp pilot (2026-10-06)

> **Later workflow update:** cube-aware grasp is now the default Retarget
> method for this UR10/Robotiq cube task, with prerequisite checks and an
> explicit hand-only opt-out. No additional trajectory or simulation success
> measurement is implied by this change. The historical pilot numbers below
> are unchanged.

The active `plan.h5` for `data/datasets/40mm-pnp/2026-10-06_08-55-49` was
rerun with UR10, Robotiq 2F-85, 11 mm adapter, robot base `[-0.7, 0, 0]` m,
and `--object-grasp --object-cube-side-mm 40`. The preceding hand-only plan was
copied to `intermediates/retarget_comparison/hand_only_plan.h5` before rerun.
`object_relative.npz` was rebuilt against the new plan. The new, unscreened
trajectory bundle is `data/exports/2026-10-06-cube-transfer-object-grasp/`;
the previous pilot export was not overwritten.

## Method and measured result

The scene track indicates approach before frame 49, closing 49–72, carrying
73–142, release 143–181 and retreat from 182. The static blue-cube model has
two well-supported perpendicular side-plane fits; this is a geometric
self-consistency check, not independently measured cube orientation. The
Robotiq contact-panel separation was checked in its orientation frame and
lies on local +Y. We orient that jaw axis toward a fitted horizontal face
normal, with the tool axis down, while the TCP follows object translation
during carry. The model's per-frame rotation is *not* used: it drifted by up
to roughly 40° during the transfer. The face correction from the original
hand-derived TCP target at pickup is 41.9°.

| Carry frames 73–142 | Hand-only plan | Cube-aware plan |
|---|---:|---:|
| URDF pad-centre separation, min–max | 34.2–74.3 mm | **38.0–38.0 mm** |
| Median absolute jaw/face-normal alignment | 0.656 | **0.985** |
| Achieved TCP/object translation-relative drift, p95 | 19.6 mm | **16.0 mm** |
| Full-plan target position RMSE | 13.0 mm | 13.3 mm |
| Full-plan target orientation RMSE | 7.3° | 10.1° |

The cube-aware IK converged. Its modelled minimum floor clearance is 22.0 mm;
the collision margin remains positive. These measurements compare two *different
target trajectories* and are only internal consistency/tracking metrics. In
particular, 16 mm TCP/object drift is still substantial for a 40 mm cube.

The first pilot used a naive linear width conversion: its nominal "38 mm"
command actually placed the Robotiq URDF pad centres about 48 mm apart.
Plan v9 now inverts the contact geometry; its normalised command is about
0.331 (28.2 mm on the old nominal scale) to achieve a **modelled 38 mm pad
gap**. The table compares both plans in that same physical URDF metric.
The 38 mm width is a positional target based on the known nominal 40 mm edge,
not a contact-force command. The reconstructed blue model spans about
55×55×59 mm, so its surfaces cannot establish the true 40 mm contact gap.
Face-fitting, pickup/release timing and contact flags are not externally
validated. No cube contact collision, friction, simulation replay or placement
success was evaluated. Do **not** treat this as proof of a stable grip or use
it as a hardware command without validation in simulation first.

At the time of this pilot the mode was opt-in. It now defaults on for the
UR10/Robotiq cube workflow, but still fails explicitly without a clear transfer,
one manipulated object, or two supported perpendicular side planes. Other tasks
can select the hand-only path explicitly.

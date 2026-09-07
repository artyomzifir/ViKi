# `viki.retarget` — calibrated poses to a robot plan

The PoC consumes the hand trajectory in `cln.npz` and writes one self-contained
`plan.h5`. The frame boundary is explicit:

- `cln.npz` has `coordinate_frame == "rig"` (reference-camera rig);
- protected V1 artifacts labelled `robot_base` are recognised as the historical
  metadata bug they are; their arrays still take this same rig-frame path;
- the episode's `raw/world_anchor.json:T_world_display` maps rig coordinates to
  the installed calibration base;
- robot axes are parallel to calibration axes;
- `RETARGET_ROBOT_BASE_POSITION = [x,y,z]` is the robot origin in that frame;
- object pose and object-relative motion are deliberately not used yet.

`T_calibration,TCP = T_calibration,rig · T_rig,hand · T_hand,TCP`, after which
positions are translated into the robot base frame for IK. The anchor is applied
exactly once. No scale, recenter, reflection, or implicit MediaPipe-to-robot
transform exists in this stage.

Orientation is part of the default objective. A provisional fixed UR10
embodiment rotation `T_hand,TCP` is stored explicitly in configuration rather
than treating the palm and URDF `wrist_3_link` axes as identical. The offset is
editable and must eventually be replaced by a physical hand-to-tool calibration.

## Interchangeable grippers

The physical gripper is a second URDF appended to the arm's mount frame (`tool0`
for UR robots). Retarget tracks the gripper's declared TCP, not the old wrist
frame. The composite arm/tool collision model is also passed to the solver and
the browser scene tags arm and gripper links separately.

There are deliberately two gripper abstractions:

- `viki.gripper.BinaryGripper` converts the observed human hand into the stable
  semantic command `closed ∈ {0,1}`;
- `viki.retarget.grippers.ToolGripperConfig` maps that command to a particular
  URDF master joint, jaw opening, mount transform, TCP and collision geometry.

This keeps the dataset/controller contract independent of the installed tool.
The current physical registry is:

| Key | Model | State |
| --- | --- | --- |
| `robotiq_2f85` | Robotiq 2F-85 | working default; BSD-2 URDF, mimic linkage and 85 mm opening |
| `robotiq_2f140` | Robotiq 2F-140 | registered; official Xacro packaging audit pending |
| `schunk_wsg50` | SCHUNK WSG-50 | registered; third-party URDF licence/source audit pending |
| `smc_mhz2_16d` | SMC MHZ2-16D | registered; manufacturer CAD found, redistributable URDF pending |
| `smc_mhz2_20d` | SMC MHZ2-20D | registered; manufacturer CAD found, redistributable URDF pending |

Unavailable entries are visible but disabled in the Retarget selector. They fail
before IK with an explicit reason instead of silently substituting approximate
geometry. Adding an audited model only requires filling its registry contract;
the append, passive-joint, solver, archive, API and viewer paths are shared.

## Solver

`solver.solve_trajectory` optimises the full `q[0:T]` trajectory. Every
Gauss–Newton iteration builds one sparse QP with:

- confidence-weighted Huber SE(3) tracking;
- velocity, acceleration and neutral-posture regularisation;
- Levenberg–Marquardt damping;
- hard joint and velocity limits;
- linearised Pink self-collision barriers from URDF collision geometry.

UR continuous joints are optimised as physical joint angles (`nv`), not as
Pinocchio's internal `[cos(q), sin(q)]` configuration embedding. OSQP solves
the sparse trajectory problem.

## Output and scene

`plan.h5` schema v3 contains arm-only `q`, the generic `(T,D)` semantic
`gripper_command`, the physical gripper master-joint position and jaw opening,
desired and achieved TCP poses, errors, joint derivatives, solver/config
metadata, and URDF-derived body origins plus parent edges for every frame. The
Retarget tab uses that geometry in the shared Three.js scene and colours the
tool separately. Rig-space cloud/input layers receive the display anchor;
calibration-space desired TCP, achieved TCP, arm and gripper do not receive it a
second time.

`Gripper` owns the command-vector contract. `BinaryGripper` currently produces
one `closed` dimension; an anthropomorphic implementation can add actuator
dimensions without changing retarget orchestration or the plan container.

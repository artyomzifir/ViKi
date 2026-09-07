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

`T_calibration,EE = T_calibration,rig · T_rig,hand · T_hand,EE`, after which
positions are translated into the robot base frame for IK. The anchor is applied
exactly once. No scale, recenter, reflection, or implicit MediaPipe-to-robot
transform exists in this stage.

Orientation is part of the default objective. A provisional fixed UR10
embodiment rotation `T_hand,EE` is stored explicitly in configuration rather
than treating the palm and URDF `wrist_3_link` axes as identical. The offset is
editable and must eventually be replaced by a physical hand-to-tool calibration.

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

`plan.h5` schema v2 contains `q`, the generic `(T,D)` `gripper_command`, desired
and achieved EE poses, errors, joint derivatives, solver/config metadata, and
URDF-derived joint origins plus parent edges for every frame. The Retarget tab
uses that geometry in the shared Three.js scene. Rig-space cloud/input layers
receive the display anchor; calibration-space desired EE, achieved EE, and robot
do not receive it a second time.

`Gripper` owns the command-vector contract. `BinaryGripper` currently produces
one `closed` dimension; an anthropomorphic implementation can add actuator
dimensions without changing retarget orchestration or the plan container.

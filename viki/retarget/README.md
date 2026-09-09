# `viki.retarget` — calibrated poses to a robot plan

The PoC consumes the hand landmarks and palm orientation in `cln.npz` and writes one self-contained
`plan.h5`. The frame boundary is explicit:

- `cln.npz` has `coordinate_frame == "rig"` (reference-camera rig);
- protected V1 artifacts labelled `robot_base` are recognised as the historical
  metadata bug they are; their arrays still take this same rig-frame path;
- the episode's `raw/world_anchor.json:T_world_display` maps rig coordinates to
  the installed calibration base;
- `RETARGET_ROBOT_BASE_POSITION = [x,y,z]` is the robot origin in that frame;
- `RETARGET_ROBOT_BASE_RPY_DEG = [roll,pitch,yaw]` rotates the robot base in
  that frame (extrinsic XYZ degrees); the Retarget tab exposes both vectors;
- object pose and object-relative motion are deliberately not used yet.

For the default two-finger semantics, target position is
`(thumb_tip + index_tip) / 2`, target orientation is the stable palm frame, and
gripper opening is the filtered thumb/index gap. Thus position, orientation and
opening remain separate signals instead of deriving an unstable orientation
from a fingertip line near closure. `wrist` remains an explicit legacy position
anchor in the UI.

`T_calibration,TCP = T_calibration,rig · T_rig,target · T_target,TCP`, after which
positions and orientations are transformed by the full robot-base SE(3) pose
for IK. The anchor is applied exactly once. No scale, recenter, reflection, or
implicit MediaPipe-to-robot transform exists in this stage.

Orientation is part of the default objective. A provisional fixed UR10
embodiment rotation `T_target,TCP` is stored explicitly in configuration rather
than treating the palm and URDF `wrist_3_link` axes as identical. The offset is
editable and must eventually be replaced by a physical hand-to-tool calibration.

## Interchangeable grippers

The physical gripper is a second URDF appended after a user-defined adapter at
the arm's mount frame (`tool0` for UR robots). Retarget tracks a declared task
point on the gripper, not the flange or wrist frame. The two transform chains are
deliberately independent:

- human target frame → desired gripper TCP is the task correspondence;
- robot flange → adapter → gripper TCP is the installed hardware geometry.

For the PoC, `GripperAdapter` is implemented by `CylinderGripperAdapter`.
The user supplies flange→gripper-base XYZ in millimetres, RPY in degrees, and
cylinder radius. Cylinder length is the norm of XYZ, so its visual and collision
geometry exactly spans the offset; zero XYZ means direct mounting. This is not
claimed to be manufacturing CAD and contributes no mass/inertia yet. A future
adapter-mesh implementation can satisfy the same abstraction.
The shipped Robotiq PoC default is `[0, 0, 11] mm`, now represented by the
cylinder instead of the old invisible transform.

For Robotiq 2F-85, the vendor `robotiq_arg2f_tcp` is fixed at 174 mm on the
tool axis, beyond the physical fingers. ViKi uses it only for orientation. The
tracked position is the dynamic midpoint of the two rubber-panel centres from
the distal-phalanx frames (130.3 mm at full opening before the adapter). This
point follows the four-bar linkage as the jaw command changes, so the target
trajectory is rendered and optimised where the object is actually clamped.

There are deliberately two gripper abstractions:

- `viki.gripper.LinearGripper` converts the observed human pinch distance into
  a filtered normalised opening (`0=closed`, `1=open`); the old binary model is
  retained only for reproducible V1 artifacts;
- `viki.retarget.grippers.ToolGripperConfig` maps that command to a particular
  URDF master joint, jaw opening, mount transform, TCP and collision geometry.

The physical gripper model rate-limits opening using the selected tool's jaw
speed before mapping it to the URDF master joint. This also turns old binary
artifacts into a finite-duration motion instead of a one-frame jump. It keeps
the dataset/controller contract independent of the installed tool.
The current physical registry is:

| Key | Model | State |
| --- | --- | --- |
| `robotiq_2f85` | Robotiq 2F-85 | working default; BSD-2 URDF, mimic linkage, 85 mm opening, 150 mm/s rate limit |
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

- confidence-weighted Huber SE(3) tracking. `RETARGET_HUBER_DELTA` is measured
  in the weighted six-dimensional residual norm, so position and orientation
  outliers are intentionally robustified as one bad pose observation;
- velocity, acceleration and neutral-posture regularisation;
- fixed Tikhonov damping of each Gauss–Newton QP (the retained configuration
  key is `RETARGET_LM_DAMPING`; this is not adaptive Levenberg–Marquardt);
- hard joint and velocity limits;
- a hard calibration-floor half-space (`z >= 0`) for the lowest support point
  of every movable URDF collision body (arm, adapter and gripper), plus both
  gripper contact centres. Mesh support vertices and primitive dimensions are
  evaluated at the current rotation, with a feasibility-preserving backtracking
  check after every non-linear step. The fixed pedestal is excluded because it
  defines the mounting plane and cannot be changed by IK. The plane is
  transformed into robot coordinates, so it remains the physical floor even
  when the base rotates;
- linearised Pink self-collision barriers from URDF collision geometry.

UR continuous joints are optimised as physical joint angles (`nv`), not as
Pinocchio's internal `[cos(q), sin(q)]` configuration embedding. OSQP solves
the sparse trajectory problem.

## Output and scene

`plan.h5` schema v8 contains arm-only `q`, the base position/RPY, target position anchor and
adapter parameters, the generic `(T,D)` continuous
`gripper_command`, normalised/metre opening, physical master-joint position,
desired and achieved TCP poses, errors, joint derivatives, solver/config
metadata, and URDF-derived body origins plus parent edges for every frame. The
metrics include a per-term objective breakdown and exact joint, velocity,
floor and collision margins over the approach plus demonstration.

For paper comparisons, `RETARGET_SEQUENTIAL_BASELINE` (or the matching UI/CLI
option) additionally runs causal frame-wise IK with warm starts, followed by a
7/2 Savitzky–Golay filter over `q`. It uses the same targets, model, gripper and
weights. Its trajectory, per-frame errors, objective breakdown and post-filter
constraint margins are archived, but it is never sent to replay; post-hoc
smoothing is allowed to expose exactly the feasibility failures being measured.
The Retarget scene shows this optional path in violet. The
Retarget tab uses that geometry in the shared Three.js scene and colours the
tool separately. Rig-space cloud/input layers receive the display anchor;
calibration-space desired TCP, achieved TCP, arm and gripper do not receive it a
second time.

`Gripper` owns the command-vector contract. `LinearGripper` produces one
`opening` dimension; an anthropomorphic implementation can add actuator
dimensions without changing retarget orchestration or the plan container.

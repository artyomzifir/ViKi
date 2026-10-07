# viki.export — datasets

**Stage 6** · retargeted episodes → a dataset · paper §3.9

Two writers, for two different needs.

| | `export_trajectories` (default) | `export_dataset` |
|---|---|---|
| output | self-contained `.npz` bundle + manifest | LeRobot dataset |
| dependencies | numpy only | optional `viki[export]` → `lerobot`, torch |
| camera video | no | yes |
| requires replay + screening | no | yes |
| intended for | replaying, inspecting, converting, shipping | policy training |

```bash
viki export <episode>... --out data/datasets/pick            # trajectory bundle
viki export <episode>... --out data/pick --format lerobot    # LeRobot
```

## Trajectory bundle

```
<out>/dataset.json                    manifest: schema, units, frames, provenance, index
<out>/episodes/<id>/trajectory.npz    the arrays
<out>/episodes/<id>/meta.json         that episode's provenance
<out>/episodes/<id>/object_models.npz optional completed scene-perception sidecar
<out>/episodes/<id>/object_relative.npz optional post-IK object-relative tracks
```

Units are metres, radians, seconds and microseconds. Poses are in the
**calibration** frame (ChArUco board origin) unless the key ends `_rig`, and the
manifest says so rather than leaving it to be assumed.

Bundle schema v4 records `segment`, `object_model`, and `object_relative`
provenance in episode metadata and copies their optional sidecars only when
their stages are complete. The trajectory arrays remain unchanged.

`object_relative.npz` has one per-frame transform for each tracked object:
`T_object_hand`, `T_object_target_tcp`, and `T_object_achieved_tcp`. It also
contains object IDs/labels, `relative_valid`, tracking confidence, rotation
information, contact flags, and gripper opening. The transforms are estimates
from the saved object poses, regardless of whether cube-aware IK was
run; a symmetric cube's yaw is not independently verified. Invalid hand rows are NaN in the hand and
target transforms and must not become policy examples.

| array | what |
|---|---|
| `q`, `joint_velocity`, `joint_acceleration` | robot joint trajectory, rad |
| `q_approach` | home → first frame, not per-frame |
| `gripper_opening`, `gripper_opening_m`, `gripper_joint_position` | normalised command, URDF pad-centre gap in plan v9 (linear nominal width in older plans), drive joint |
| `target_position_calibration`, `target_rotation_calibration` | commanded tool pose |
| `achieved_position_calibration`, `achieved_rotation_calibration` | forward kinematics |
| `position_error_m`, `orientation_error_rad` | tracking error |
| `omega` | per-frame evidence weight used by the IK |
| `hand_position_rig`, `hand_rotation_rig`, `hand_valid` | the human pose the plan came from |
| `hand_observed_mask`, `hand_pose_supported` | which frames were measured rather than filled |
| `timestamps` | host monotonic µs |

### What this version does not guarantee

**Nothing is screened.** Replay is a stub, so trajectories are exported without
being replayed, and an episode need not be labelled or rated. The manifest says
this in `screening`, and every episode's `meta.json` records what actually ran —
`replay_run`, `replay_verdict`, `labelled`, `outcome_rated`. A consumer should
not mistake a bundle for a vetted dataset.

An episode qualifies when retarget has produced a readable `plan.h5`. One
unreadable episode is skipped and listed in the manifest's `skipped`, never
fatal to the rest.

## LeRobot dataset

Eligibility is the stricter one: `retarget` and `replay` have run, the replay
verdict is not `reject`, `outcome` is not `bad`, and the task string is
non-empty. Frame schema and stub status are unchanged from before:
`annotation.object_relative` is a shape-correct placeholder until
`prepare.represent` produces a real object track — see
`docs/object_centric_decision.md` for why it does not yet.

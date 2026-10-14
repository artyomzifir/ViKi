# HO-Cap scene notes (2026-10-13)

## User feedback for the next retarget/scene iteration

- The current manipulator placement is acceptable, but the UR10 base should be moved farther away, to `y = -1.0 m` in the scene coordinate frame.
- If a two-camera configuration is used, select only the orange and red camera streams visible in the current scene; these cameras are at the same height. Verify their camera IDs against the scene before configuring the pair.

These are future configuration requirements, not changes to the running HO-Cap batch or the current scene. No camera selection or robot placement was changed on 2026-10-13 in response to this feedback.

## Table plane and collision limitation

For the forthcoming HO-Cap hand-only retarget, set the explicit tabletop plane to `z = 0 m` in `hocap_tag_1`, as in the existing zero-height floor constraint. The UR10 base remains `[0, -1.0, 0]` in that frame. This is a hard infinite-plane constraint on movable robot collision bodies and gripper contact points; it is not a finite table mesh. The tabletop height is not independently surveyed, so this does not prove physical clearance. Objects and the human are not obstacles in the current IK; their collision avoidance remains a limitation and future work.

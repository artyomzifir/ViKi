# ViKi — Pipeline Status & Next Steps

**Date:** June 2026  
**Author:** ViKi team, Innopolis University  
**Status:** Pre-thesis experimental phase

---

## Current State

The core capture and retargeting infrastructure is functional end-to-end.

**What works:**

- Multi-camera RGB-D capture (FastAPI app, RealSense + two Azure Kinects)
- Synchronized capture with hardware sync between Kinects (master/subordinate)
- Intrinsics and extrinsics extracted for both Kinects
- MediaPipe skeleton extraction (`mediapipe_extract.py`) — body + both hands, cubic spline interpolation of missing frames, confidence plots
- PINK differential IK retargeting — UR10 and iiwa14, approach phase + scene trajectory, joint smoothing, export to `.npz`
- Exploration notebook covering the full pipeline from raw landmarks to smoothed joint trajectories

**What is in progress:**

- Hand-eye calibration (`T_camera_to_robot_base`) — extrinsics between cameras are done, robot-frame transform is next
- Object-centric trajectory representation — designed, not yet implemented

---

## Three Sources of Error and Mitigation Strategy

The retargeting pipeline accumulates error across three independent stages. Each is addressed separately.

---

### Error 1 — MediaPipe skeleton noise

**Nature.** XY coordinates are reliable (direct from image pixels). Z coordinate is regressed from appearance by a neural network, not measured — this is the main source of noise. Additional degradation occurs during fast motion (blur), hand-object contact (occluded fingers), and side-view poses (orientation estimation falls apart when the palm is edge-on to camera).

**Mitigation stack:**

*Layer 1 — Better data (highest priority).* Two Azure Kinects with aligned depth give independent Time-of-Flight Z measurements per camera. Fusing them turns the Z coordinate from a neural network prediction into a geometric measurement. Implementation: unproject each MediaPipe 2D keypoint using aligned depth and camera intrinsics, transform to a common world frame using known extrinsics, fuse by weighted average of confidence scores.

```
P_world = T_cam_to_world @ unproject(u, v, depth[u,v], intrinsics)
P_fused = conf_A * P_A + conf_B * P_B  (when both valid)
```

*Layer 2 — Better model.* `model_complexity=2` in MediaPipe Hand (currently using 1) is a free upgrade. For body pose on non-standard manipulations, RTMPose (mmpose) outperforms MediaPipe Pose significantly. WiLoR is specialized for hand pose during object contact but is heavier; treat as optional.

*Layer 3 — Temporal post-processing.* Cubic spline interpolation of missing frames (already implemented). Savitzky-Golay smoothing (already implemented). Key tradeoff: larger SG window removes more noise but also smooths out contact events (peak velocity moments that carry task-relevant information). Keep window conservative — 11–15 frames for 30 fps.

**Residual error after mitigation:** ~5–10 mm at the wrist, larger at fingertips. Acceptable for the approach and scene trajectory; fingertip precision is delegated to gripper state, not IK.

---

### Error 2 — Hand-eye calibration

**Nature.** The transform `T_camera_to_robot_base` is a systematic (not random) error — every point is shifted by the same amount. A 5 mm calibration error translates directly into a 5 mm positioning error for every frame of every trajectory. Unlike MediaPipe noise, this cannot be averaged out.

**Mitigation stack:**

*ChArUco board.* Preferred over standard checkerboard — partial occlusion does not break corner detection, which allows collecting calibration poses at close range where accuracy is highest. Print on rigid backing (forex or aluminium composite), minimum 3 cm marker size for reliable detection at 1 m.

*Pose diversity.* Collect 25–40 robot configurations that cover the full workspace: near/far, tilted left/right/forward, rotated around Z. Avoid colinear sequences (all poses along one axis = ill-conditioned system).

*Algorithm.* `cv2.calibrateHandEye` with `CALIB_HAND_EYE_TSAI` or `CALIB_HAND_EYE_PARK`. Both are stable for eye-to-hand configuration (fixed external camera), which is the ViKi setup.

*Validation.* Place an ArUco marker at a physically measured position relative to the robot base. Predict its position through `T_camera_to_base`. Compare prediction to ground truth with a ruler. Target: error < 5 mm. If > 10 mm, recollect calibration data with more diverse poses.

**Residual error after mitigation:** 3–8 mm systematic offset. This cannot be reduced further without higher-precision tooling (laser tracker). Acceptable for tabletop manipulation at distances of 0.3–0.7 m.

---

### Error 3 — Absolute position dependency

**Nature.** The current retargeting maps absolute world-frame coordinates from the demonstration directly to IK targets. This means the trajectory is valid only when the robot, the object, and the camera are in the exact same spatial configuration as during recording. Moving the object 10 cm or using a robot with different arm length causes the entire trajectory to miss.

This is the most fundamental error because it prevents generalization across setups, recording sessions, and robot embodiments.

**Solution: object-centric (relative) trajectory.**

The key insight from OKAMI (Li et al., CoRL 2024): instead of recording where the hand was in world space, record how the hand moved relative to the object. The trajectory then transfers to any position of the object and any robot that can reach it.

Implementation in three steps:

*Step 1 — Localize the object in 3D.*

Use SAM2 (one click on the object in frame 0 → tracked mask for the full video) combined with aligned depth to extract the object's 3D centroid per keyframe:

```python
depth_masked = depth[sam2_mask]
pts_3d = unproject_masked(depth_masked, intrinsics)
T_object = median(pts_3d)  # robust to outliers
```

For initial experiments, a manually drawn static ROI on the first frame is sufficient — the object does not move before the hand contacts it.

*Step 2 — Express trajectory in object frame.*

```python
# Convert absolute targets to object-relative
trajectory_relative = [inv(T_object_in_robot) @ se3_target[t]
                       for t in range(T)]
```

This is what gets stored in the dataset — not absolute poses but object-relative poses.

*Step 3 — Workspace scaling for different robot embodiments.*

Object-centric representation removes position dependency but not scale dependency. A UR10 with 1.3 m reach and an iiwa14 with 0.8 m reach will execute the same relative trajectory differently. Scale the translation component of the relative trajectory to fit the target robot's reachable workspace:

```python
human_reach = max(||T_rel[t].translation|| for t in T)
robot_reach  = estimate_workspace_radius(robot)   # from FK sampling
scale        = robot_reach / human_reach
# Apply to translation only — rotation (grasp direction) is preserved
T_scaled[t].translation = T_rel[t].translation * scale
```

*Step 4 — Deployment.*

```python
T_object_new = detect_object(camera_frame, depth)       # SAM2 at deploy time
T_targets    = [T_object_new @ T_scaled[t] for t in T]  # back to absolute
q = run_retargeting(robot, EE_FRAME, T_targets, ...)     # existing PINK IK
```

**Residual error after mitigation:** object detection accuracy (5–15 mm for SAM2 + ToF centroid) plus IK tracking error (5–20 mm depending on configuration). Total position error at the object: 1–3 cm, which is at the boundary of pick-and-place reliability without closed-loop correction.

---

## Required Technical Prerequisites

| Component | Status | Notes |
|---|---|---|
| Intrinsics (both Kinects) | Done | |
| Extrinsics (Kinect A ↔ Kinect B) | Done | |
| MediaPipe extraction | Done | `mediapipe_extract.py` |
| PINK retargeting | Done | notebook + export |
| Depth fusion (stereo skeleton) | Not started | next step |
| Hand-eye calibration | In progress | extrinsics done, robot frame pending |
| SAM2 object detection | Not started | after hand-eye |
| Object-centric transform | Not started | after hand-eye |
| Workspace scaling | Not started | after object-centric |
| LeRobot export | Not started | can start in parallel |

---

## Implementation Roadmap

### Phase 1 — Better skeleton (now, ~1 week)

Implement `StereoSkeletonExtractor` using the existing Kinect extrinsics:

```python
class StereoSkeletonExtractor:
    def process_frame(self, rgb_A, depth_A, rgb_B, depth_B):
        # 1. MediaPipe on both RGB frames
        # 2. unproject each keypoint using aligned depth + intrinsics
        # 3. transform to world frame via T_cam_to_world
        # 4. fuse by confidence-weighted average
        return fused_keypoints  # (33+21+21, 3), all in world frame
```

Expected gain: Z error drops from ~20–30 mm (MediaPipe regression) to ~5–10 mm (ToF measurement).

### Phase 2 — Hand-eye calibration (~1 week)

Collect 30 ChArUco calibration poses with the UR3. Run `cv2.calibrateHandEye`. Validate with marker at known position. Target error < 5 mm.

This unlocks all subsequent steps that require coordinates in robot base frame.

### Phase 3 — Object-centric pipeline (~1–2 weeks)

Integrate SAM2 for object masking. Implement `trajectory_relative` conversion in the notebook. Add workspace scaling. Validate that the same demonstration transfers to a different object position (move the object 15 cm, check that the robot still reaches it).

### Phase 4 — Dataset and training (~ongoing)

Collect structured demonstrations: 5–10 actions × 10–20 recordings each. Export to LeRobot format. Train ACT or Diffusion Policy baseline. Evaluate on held-out object positions.

---

## Key Design Decisions to Resolve

**Object detection method.** SAM2 (one-click, good quality, requires GPU) vs ArUco marker on the table as anchor (no GPU, less flexible). Recommendation: ArUco as temporary solution for Phase 3 validation, SAM2 for final dataset.

**Which hand to retarget.** Currently right hand. For bimanual tasks, both hands need independent SE3 targets and a coordination signal. Not needed for initial single-arm experiments.

**Orientation weight in IK.** `IK_ORI_COST = 0.3` is deliberately low (position tracking prioritized). For tasks where grasp orientation matters (pouring, screwing), increase to 0.7–1.0 and verify orientation tracking error stays below 15°.

**Savitzky-Golay window.** `SG_WINDOW = 15` (0.5 s at 30 fps) is conservative. For faster motions reduce to 9–11. For very slow precision tasks can increase to 21. Always inspect the velocity profile after changing — contact events should remain as visible peaks.

**When to use a second robot.** iiwa14 has 7 DOF (kinematically redundant) which gives natural postures and avoids singularities better than UR10 (6 DOF). For experiments comparing generalization across embodiments, test both. Workspace scaling will handle the size difference.

---

## References

- Li et al., *OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation*, CoRL 2024. https://arxiv.org/abs/2410.11792
- Qin et al., *DexMV: Imitation Learning for Dexterous Manipulation from Human Videos*, ECCV 2022. https://arxiv.org/abs/2108.05877
- Caron et al., *SAM 2: Segment Anything in Images and Videos*, Meta AI 2024. https://arxiv.org/abs/2408.00714
- Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation*, arXiv 2023. https://arxiv.org/abs/2303.07399
- PINK differential IK library. https://github.com/stephane-caron/pink
# Research Proposal

**Proposed thesis title (English):** ViKi: Learning Robotic Manipulation from Multiview Human Demonstrations

**Russian translation:** ViKi: обучение роботизированным манипуляциям по многокамерным демонстрациям человека

**Project under active development:** https://github.com/artyomzifir/ViKi

## Summary

This research asks whether human manipulation demonstrations recorded with calibrated RGB-D cameras can provide useful geometric references for robot learning. ViKi reconstructs hand motion and task objects from recorded scenes, expresses the motion relative to objects, and retargets it to a robot model. The proposed next step is to test whether those geometric plans can be turned into physically successful robot demonstrations in simulation and help a small closed-loop policy learn a manipulation task. A cube-transfer task with a UR10 and a two-finger gripper will serve as the initial proof of concept, not as evidence of general performance across robots or tasks.

The work will combine recordings made with ViKi and selected public datasets with hand and object annotations. Public data can provide an independent check of perception; recordings from the actual two-camera setup are still needed to evaluate its calibration, synchronization, and end-to-end task. Simulation will be used to test contact and task success. Physical-robot deployment is outside the initial study and remains a possible later step.

## Motivation and research gap

Human demonstrations are relatively easy to record but do not directly contain robot actions. Reconstructing a hand trajectory and solving inverse kinematics produces a candidate robot motion; it does not establish that a gripper can hold an object or complete the task. A physics simulator can test these conditions, but it needs useful initial behavior and a scene model. This project studies the connection between the two: how much of a recorded human demonstration can survive reconstruction, robot retargeting, and physical execution, and whether the resulting examples help learning.

This is not a claim to invent learning from demonstrations, object-aware retargeting, or simulation-based data generation. MimicGen generates new robot demonstrations from robot examples [1], while OKAMI studies object-aware transfer from human video [2]. ViKi focuses on a measurable, camera-based route from multiview human recordings to robot-specific geometric references. The public code is available at the project link above; repository releases may lag experiments in progress.

## Research questions

1. How accurately and consistently can the pipeline recover hand and object motion from multiview RGB-D recordings? Can its predictions be evaluated against annotations in a compatible public benchmark rather than only by visual inspection?
2. What fraction of retargeted human motions can be executed by a specified robot and gripper in a contact simulation, and what are the main causes of failure?
3. In a limited cube-transfer task, do physics-validated examples derived from ViKi improve closed-loop learning compared with a matched training run without those examples?

## Method

**1. Capture and independent evaluation.** Use existing ViKi recordings of a human moving a blue cube from a red support to a green goal. Check synchronization, calibration, frame loss, reconstructed geometry, and episode-level processing yield. Adapt a small subset of a public multiview RGB-D benchmark, initially DexYCB [3], into the pipeline's input format. Keep its provided hand and object annotations out of inference and use them only for evaluation. Report benchmark errors separately from self-consistency measurements on the local camera rig. A public grasping dataset cannot by itself validate a complete cube-transfer task or the two Kinect controllers.

**2. Robot-specific geometric plans.** Reconstruct hand and object trajectories, identify candidate grasp and transfer phases, and retarget them to a UR10 with a two-finger Robotiq 2F-85 gripper. Record reachability, joint and velocity limits, trajectory error, and any manual corrections. Compare the object-aware plan with a hand-only version on the same episodes. A geometrically valid plan is a reference, not yet an executable demonstration.

**3. Physical simulation.** Build a compact MuJoCo scene with a robot, gripper, table, and cubes [4]. Execute candidate plans through a controller with simulated contacts; do not attach the cube to the gripper or teleport it to the goal. Count a rollout as successful only when the robot transfers and releases the blue cube and it remains on the green cube. Log failed attempts and their causes. Successful rollouts can then provide executed robot states and actions for training. The study will use a bounded set of scene variations rather than assume unlimited valid data generation.

**4. Learning and comparison.** Start with a compact state-based policy rather than a vision-language model. Compare reinforcement learning with and without physics-validated ViKi examples under matched rewards, evaluation scenes, and compute budgets. Report success on held-out cube placements, learning curves, training time, and variation across random seeds. A scripted or behavior-cloning controller may provide an additional baseline.

## Current basis and feasibility

ViKi already has an offline path from synchronized RGB-D recordings through hand reconstruction and robot retargeting to trajectory artifacts. A cube-transfer pilot has reached this artifact stage. This does **not** yet demonstrate a physically successful simulated grasp, a trained cube-transfer policy, or transfer to a real robot. Public-benchmark import, independent pose evaluation, a MuJoCo task, and policy experiments are planned work. The public repository demonstrates that implementation is underway, not that these planned results have been achieved.

The project can be evaluated with the existing cameras, recorded episodes, public benchmark annotations, and a modest simulation experiment. No motion-capture system or physical UR10 is required for the proposed proof of concept. Without a physical-robot experiment, the result will be described as a human-demonstration-to-simulation learning study, not as a completed Real-to-Sim-to-Real transfer.

## Work plan

First, audit the existing recordings and select representative episodes before large-scale processing. In parallel, import a small public benchmark subset and check that coordinate frames and annotations are comparable. Next, establish a physically plausible MuJoCo cube-transfer scene and measure how often the current geometric plans succeed or fail when executed. Finally, train and evaluate the small policy comparison on held-out placements. If a stage fails, the analysis will report that limitation rather than silently treating a geometric or simulated artifact as a verified robot demonstration.

## Planned outputs and decision criteria

The intended outputs are a documented pipeline and benchmark adapter, a table of errors and processing yield at each stage, a reproducible robot-and-cube simulation, a controlled learning comparison, and representative videos of successes and failures. Claims about pose accuracy will be tied to the available reference labels; claims about task success will be tied to physical simulation outcomes. The study will support a learning claim only if ViKi-derived examples improve held-out results under a matched budget. A negative result is still informative about the limits of camera-only demonstrations and retargeting.

## References

[1] A. Mandlekar et al., “MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations,” CoRL, 2023. https://arxiv.org/abs/2310.17596

[2] J. Li et al., “OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation,” 2024. https://arxiv.org/abs/2410.11792

[3] Y.-W. Chao et al., “DexYCB: A Benchmark for Capturing Hand Grasping of Objects,” CVPR, 2021. https://dex-ycb.github.io/

[4] Google DeepMind, “MuJoCo Documentation: Overview.” https://mujoco.readthedocs.io/en/stable/overview.html

# Research Questions Draft (по IU Thesis Guidelines 2022 Section 2.3)

**Дата:** 2026-11-01  
**Основа:** HO-Cap dataset (28 episodes, 8-view RGB-D captures)

## Research Problem

Generating diverse robot learning datasets from human demonstrations is expensive due to per-embodiment teleoperation requirements.

## Research Aim

To test and explain whether multiview RGB-D human demonstrations can be transformed into multi-embodiment synthetic datasets via 3D Gaussian Splatting-based reconstruction and morphology-aware retargeting.

---

## RQ1: Hand Tracking Accuracy

**What hand tracking accuracy does the ViKi perception pipeline (MediaPipe landmarks + articulated MANO fitting) achieve against HO-Cap ground truth annotations?**

- **Metrics:** MPJPE, P95 error, PCK@20mm, displacement error (mm)
- **Baseline:** HO-Cap semi-automatic multi-view MANO annotations (ground truth)
- **Method:** ViKi pipeline (MediaPipe per-camera → fusion → MANO fit)
- **Dataset:** 28 HO-Cap manipulation episodes, 8 RGB-D views

**Code:** `viki/benchmark/evaluate.py`

---

## RQ2: Object Pose Tracking Accuracy

**What object pose tracking accuracy can be achieved using SAM-based segmentation and multiview triangulation?**

- **Metrics:** Position RMSE (mm), orientation RMSE (deg), tracking success rate (%)
- **Baseline:** Manual annotations or multi-view triangulation consensus
- **Dataset:** Manipulated rigid objects (cubes, cylinders) in HO-Cap demonstrations

**Code:** `viki/perception/object_model.py`

---

## RQ3: Cross-Embodiment Retargeting Feasibility

**Can human hand trajectories be retargeted to multiple robot embodiments while satisfying kinematic and collision constraints?**

- **Metrics:** 
  - Convergence rate (% episodes with valid IK solution)
  - TCP position/orientation error (RMSE, P95 in mm/deg)
  - Constraint violations (joint limits, velocity limits, self-collision margin, floor clearance)
- **Embodiments:** UR10, UR5, Franka Panda, Kinova Gen3, xArm7 (varying DOF, reach, velocity limits)
- **Dataset:** 28 HO-Cap wrist trajectories → M robot plans

**Code:** `viki/retarget/solver.py`, `scripts/build_hocap_retarget_report.py`

---

## RQ4: Gripper State Transfer Quality

**How accurately can human pinch state be mapped to robot gripper commands?**

- **Metrics:** Gripper opening error (mm), grasp timing offset (frames), velocity satisfaction rate (%)
- **Grippers:** Robotiq 2F-85, Robotiq 3F, Allegro Hand (varying actuation, DOF)
- **Dataset:** 28 HO-Cap thumb-index trajectories → gripper commands

**Code:** `viki/gripper.py::LinearGripper`

---

## RQ5: Object-Aware Retargeting Quality

**Does incorporating object tracking improve retargeting quality over hand-only inverse kinematics?**

- **Comparison:** Hand-only IK vs. object-relative IK vs. contact-aware IK
- **Metrics:** Object pose error (mm/deg), robot-object collision rate (%), contact preservation (%)
- **Dataset:** Pick-place tasks with tracked object

**Code:** `viki/retarget/tracked_grasp.py`, `viki/retarget/object_grasp.py`

---

## RQ6: 3DGS Reconstruction Quality

**What visual reconstruction quality can 3D Gaussian Splatting achieve for manipulation scenes from RGB-D input?**

- **Metrics:** PSNR, SSIM, depth RMSE (mm)
- **Ablation:** 8-view vs. 2-view, synchronized vs. degraded timestamps
- **Dataset:** HO-Cap manipulation scenes, train on N-1 views, test on held-out view

**Code:** TBD (Nerfstudio/gsplat)

---

## RQ7: Robot Compositing Visual Quality

**Can retargeted robot models be composited into 3DGS scenes with photorealistic quality?**

- **Metrics:** LPIPS, FID, user study (Likert 1-5), edge artifact rate (%)
- **Methods:** Mesh overlay vs. joint 3DGS training vs. hybrid
- **Dataset:** M robots composited into reconstructed scenes

**Code:** TBD (3DGS + mesh rendering)

---

## RQ8: Synthetic Dataset Diversity

**What data diversity can factorial expansion (M robots × K views × P variations) achieve compared to single-embodiment teleoperation?**

- **Metrics:** Feature space coverage (%), visual entropy (bits), pairwise LPIPS distance
- **Comparison:** ViKi-generated (1 human demo → M×K×P examples) vs. teleoperation baseline
- **Dataset:** Generated synthetic datasets from HO-Cap source

**Code:** TBD (diversity metrics)

---

## Что НЕ измеряем (explicit scope exclusion)

- **Policy training:** No BC/RL policy learning experiments
- **Task success validation:** No sim/real robot execution success rate
- **Real robot deployment:** No physical robot experiments
- **Sim2real transfer:** No policy transfer from sim to real

**Justification (по IU Guidelines 2.2.2):**
This work focuses on dataset generation capability validation. Policy learning is orthogonal future work (similar to GaussianFactory's approach of validating reconstruction before training policies).

---

## Metrics Summary Table

| Pipeline Stage | Metric Group | Specific Metrics | Code Location |
|---|---|---|---|
| Hand Tracking | Reconstruction Accuracy | MPJPE, P95, PCK@20/50mm, displacement error | `viki/benchmark/evaluate.py` |
| Object Tracking | Pose Estimation | Position RMSE, orientation RMSE, tracking rate | `viki/perception/object_model.py` |
| EE Retargeting | IK Quality | Position/orientation RMSE, P95, convergence | `viki/retarget/solver.py` |
| EE Retargeting | Constraint Satisfaction | Joint/velocity limits, collision margin, floor clearance | `scripts/build_hocap_retarget_report.py` |
| Gripper Retargeting | Grasp Transfer | Opening error, timing offset, velocity satisfaction | `viki/gripper.py` |
| Object-Aware | Manipulation Quality | Object pose error, collision rate, contact preservation | `viki/retarget/tracked_grasp.py` |
| 3DGS Reconstruction | Visual Quality | PSNR, SSIM, depth RMSE | TBD (Nerfstudio) |
| Composite Rendering | Perceptual Realism | LPIPS, FID, user study, artifacts | TBD |
| Dataset Generation | Diversity | Feature coverage, entropy, LPIPS distance | TBD |

---

## Next Steps

1. **Discuss с научным руководителем:** Согласовать RQ формулировки
2. **Verify existing metrics:** Проверить что все упомянутые метрики реально работают в коде
3. **Identify missing implementations:** Какие метрики (3DGS, rendering, diversity) нужно реализовать
4. **Update Chapter 1:** Переписать Section 1.4 (Research Questions) с финальными RQ

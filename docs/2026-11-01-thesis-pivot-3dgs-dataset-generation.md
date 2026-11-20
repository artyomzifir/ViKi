# Изменение направления исследования: 3DGS-based multi-embodiment dataset generation

**Дата:** 1 ноября 2026  
**Статус:** Proposal для обсуждения с научным руководителем  
**Коммит базы:** `2470420` (dev-eval)

## Проблема текущего scope

### Что есть сейчас

**Работающий pipeline:**
- 28 HO-Cap episodes, 8-view RGB-D
- Hand tracking: MPJPE 10.7 мм
- Object tracking: 5.4 мм / 7.6°
- Retargeting: 28 UR10+Robotiq планов, все converged

**Проблемы:**
1. **MuJoCo replay физически проваливается** — Shadow/Tesollo не удерживают объект
2. **Policy learning** дорого, сложно, не хочется делать
3. **Real robot** нет доступа
4. **Sim validation** не тот домен — physics simulator != visual learning
5. **"Просто ретаргет"** не имеет новизны для диплома

### Что обещают текущие документы

**Thesis Ch1 (introduction):** обещает policy learning с sample efficiency improvement  
**Proposal:** обещает cube-transfer state policy + MuJoCo validation  
**Revision doc (2026-10-25):** предлагает сузить до morphology transfer validation

**Все три scope имеют проблемы:**
- Policy learning — дорого и не планируется
- MuJoCo validation — провалы на физике, не доказывает VLA utility
- Morphology transfer — интересно, но без downstream application зачем?

---

## Новое направление: 3DGS-based dataset generation factory

### Идея

**Цепочка:**
```
Human RGB-D demo (существующий ViKi)
  ↓
Hand + object tracking (perception)
  ↓
Multi-embodiment retargeting (retarget к M robots/grippers)
  ↓
3D Gaussian Splatting scene reconstruction
  ↓
Composite robot models в photorealistic GS scene
  ↓
Render K viewpoints × P appearance variations
  ↓
Synthetic VLA training dataset: 1 demo → M×K×P examples
```

**Research Question:**

*"Can multiview RGB-D human demonstrations, processed through hand-object reconstruction and morphology-aware retargeting, serve as a source for 3D Gaussian Splatting-based synthetic dataset generation across robot embodiments?"*

**Ключевое отличие от существующего подхода:**
- Не нужно обучать VLA
- Не проверяем task success
- Фокус на **data generation capability**: scale, diversity, visual quality
- Измеряем: reconstruction quality, retargeting feasibility, synthetic data properties

---

## Литературный обзор: что уже существует

### RoboSplat (апрель 2025, arXiv)

**Citation:** "Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation", arXiv:2504.13175v1

**Что делает:**
- Из одной **robot teleoperation demo** + multiview images
- 3DGS reconstruction сцены
- Trajectory Diffusion Model генерит новые trajectories
- Манипулирует 3D Gaussians объекта по новой траектории
- Рендерит RGB frames для policy training

**Результаты:**
- 6 типов вариаций: object pose, robot position, background, lighting, distractor, camera viewpoint
- Policy trained на RoboSplat показала 87.8% success rate в one-shot vs 57.2% для baseline с hundreds real demos + 2D augmentation
- Tested на 10 tasks

**Ограничения:**
- Source — robot demo, не human
- Один embodiment (не multi-robot)
- TDM требует большой robot trajectory dataset для pre-training

**Код:** Не опубликован (статус unknown)

**Venue:** arXiv preprint, peer-review status неизвестен

---

### GaussianFactory (сентябрь 2024, arXiv)

**Citation:** "Demonstration Synthesis from a Single Scan via Gaussian Splatting for Visuomotor Policy Learning", arXiv:2609.21112

**Что делает:**
- Video scan пустой сцены (без демонстрации)
- 3DGS reconstruction → editable replica из independent Gaussian assets
- Task sampling: автоматически комбинирует объекты в pick-and-place tasks
- Path synthesis: sampling-based geometric planning (не physics, не human motion)
- Рендерит multiview images для каждой synthetic trajectory

**Результаты:**
- Mass-production synthetic demos из single scan
- Claim: real robot deployment (конкретные цифры в abstract не приведены)

**Ограничения:**
- Motion генерится планировщиком, не из human demo
- Tasks ограничены rigid pick-and-place
- Один робот/gripper
- Path synthesis — geometric, не контролирует contacts/forces

**Код:** Не опубликован (статус unknown)

**Venue:** arXiv preprint, peer-review status неизвестен

---

### LEGS (июнь 2024, arXiv)

**Citation:** "Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World", arXiv:2606.01458v1

**Что делает:**
- Hybrid simulator: mesh robot/objects/props (rasterized в MuJoCo) + 3DGS photorealistic background
- Procedurally generated motion primitives (no teleoperation, no seed demos)
- Fine-tune pre-trained VLA (OpenVLA) на synthetic data
- Tested на Unitree G1 humanoid

**Результаты:**
- Reported real humanoid deployment с loco-manipulation tasks
- Concrete success rates не приведены в abstract/intro

**Ограничения:**
- Motion из procedural generation, не из human capture
- 3DGS только для appearance, не для motion source
- Один embodiment (G1 humanoid), нет cross-embodiment transfer
- Требует pre-trained VLA

**Код:** Не опубликован в описании (возможно на project page)

**Venue:** arXiv preprint, peer-review status неизвестен

---

### SplatSim (ICRA 2025)

**Citation:** "Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting", arXiv:2409.10161v3, ICRA 2025

**Что делает:**
- Заменяет mesh object representations в Isaac Gym на 3DGS
- Physics simulation через simplified collision proxies
- Rendering через 3DGS для photorealistic appearance
- Train RGB policies в simulator, deploy zero-shot на real robot

**Результаты:**
- Tested на 4 manipulation tasks
- Average success rate 86% (zero-shot sim2real) vs mesh-based baseline
- Improvement особенно заметен на tasks requiring fine-grained perception

**Ограничения:**
- Это про rendering в existing simulator, не про demonstration generation
- Не генерирует новые trajectories — только улучшает visual fidelity training data
- Physics остаётся mesh-based (collision proxies)

**Код:** github.com/qureshinomaan/SplatSim

**Venue:** ICRA 2025 (accepted)

---

### Real2Render2Real (R2R2R, май 2025, CoRL submission)

**Citation:** "Scaling Robot Data Without Dynamics Simulation or Robot Hardware", arXiv:2505.09601v1

**Что делает:**
- Smartphone video/RGB-D scan объектов → 3DGS reconstruction
- Извлекает objects как 3DGS assets
- Tracks 6-DoF object motion
- Automatic trajectory synthesis (не из human demo)
- Converts 3DGS → meshes для compatibility с existing VLA/IL frameworks
- Рендерит thousands robot-agnostic demonstrations с novel views/lighting

**Результаты:**
- Generates thousands of demonstrations from single video capture
- Tested на rigid и articulated objects
- Real robot validation reported (concrete success metrics не в abstract)

**Ограничения:**
- Trajectories scripted/optimized, не learned from human motion
- Robot-agnostic approach может не capture embodiment-specific constraints
- Один target robot (по описанию)
- 3DGS→mesh conversion может терять visual fidelity

**Код:** github.com/uynitsuj/real2render2real

**Venue:** CoRL 2025 submission (acceptance status неизвестен на момент публикации preprint)

---

## Ключевое отличие: где новизна ViKi

### Что делают existing works

| Работа | Source demo | Multi-embodiment | Motion source | 3DGS role |
|--------|-------------|------------------|---------------|-----------|
| RoboSplat | Robot teleop | ✗ (один робот) | TDM генерит trajectories | Scene reconstruction + object manipulation |
| GaussianFactory | Empty scene scan | ✗ (один робот) | Geometric planner | Editable scene replica |
| LEGS | Procedural | ✗ (один humanoid) | RL/scripting | Photorealistic background |
| SplatSim | N/A | ✗ | Любой (rendering tool) | Replaces mesh в simulator |
| R2R2R | Object scan | ✗ (robot-agnostic claim) | Scripted synthesis | Asset generation → mesh |

**Критический gap — никто не делает все три:**

1. **Human RGB-D demonstration** как source motion
2. **Multi-embodiment retargeting** как intermediate layer
3. **Factorial dataset expansion** через (embodiment × viewpoint × appearance)

**Ближайшие работы:**
- **RoboSplat** делает (viewpoint × appearance) но фиксирует embodiment и использует robot teleop
- **GaussianFactory** генерит много demos но motion synthetic, не из human
- **R2R2R** claims "robot-agnostic" но не показывает retargeting к разным morphologies, trajectories scripted

### ViKi pipeline

```
Human RGB-D recording
  ↓
Hand + object reconstruction (ViKi perception)
  ↓
Retarget к M embodiments (ViKi retarget)
  ↓
3DGS scene (новое)
  ↓
Composite M robots в GS scene (новое)
  ↓
Render M×K×P synthetic examples (новое)
```

**Новизна:**

1. **Human motion patterns** как demonstration source (vs scripted/RL-generated)
2. **Morphology-agnostic retargeting** перед rendering (vs fixed embodiment)
3. **Factorial data expansion** через embodiment × viewpoint × appearance
4. **Affordable RGB-D capture** (2 Kinect) → diverse multi-embodiment dataset

**Research contribution:**

*"Human RGB-D demonstration → morphology-aware retargeting → 3DGS-based multi-embodiment dataset generation. Enables factorial data scaling (1 capture → M robots × K views × P appearances) without per-robot teleoperation or procedural motion synthesis."*

**Compared to existing:**
- **vs RoboSplat:** Human source (не robot teleop), multi-embodiment (не fixed robot)
- **vs GaussianFactory:** Human motion patterns (не geometric planner), demonstrates retargeting layer
- **vs LEGS:** Human demo source (не RL/procedural), cross-morphology transfer
- **vs R2R2R:** Explicit multi-embodiment retargeting + validation (не robot-agnostic abstraction), human motion
- **vs SplatSim:** Demonstration generation (не только rendering tool)

---

## Что измерять (без обучения VLA)

### 1. Baseline: perception + retargeting (уже есть)

**Артефакты:** 28 episodes, hand/object tracking, UR10 plans

**Новые метрики:**
- Retarget к 3-5 embodiments (UR5/10, iiwa14, разные grippers)
- Cross-embodiment feasibility: % успешных retargets
- Workspace overlap между embodiments
- Joint limit violations per robot
- Grasp compatibility (parallel vs multi-finger)

**Deliverable:** Morphology transfer baseline validated.

---

### 2. 3DGS reconstruction quality

**Задача:** RGB-D → photorealistic 3D scene

**Инструменты:** Nerfstudio/gsplat (existing, не пишем с нуля)

**Метрики:**

**Reconstruction quality:**
- Train на N-1 камер, test на held-out
- PSNR/SSIM/LPIPS для RGB
- Depth error (ground truth из Kinect)
- Novel view synthesis quality

**2-view vs 8-view ablation:**
- Real setup: 2 Kinect
- HO-Cap: 8 views
- Measure degradation с меньшим числом камер

**Practical:**
- Training time per scene
- GPU memory, storage size
- Rendering speed (fps)

**Deliverable:** Доказано что 2-view RGB-D достаточно для quality 3DGS.

---

### 3. Robot compositing quality

**Задача:** Mesh robot + GS scene выглядит natural

**Технически:**
- Robot URDF → mesh
- Position через calibrated world frame
- Render: mesh rasterization + GS rendering
- Sync с retargeted trajectory

**Метрики:**

**Visual plausibility:**
- User study: humans rate "does robot belong in scene?"
- Edge artifacts detection
- Occlusion consistency

**Geometric correctness:**
- Gripper-object alignment в contact frames
- Robot base contact с floor
- Scale consistency

**Multi-embodiment:**
- Render same trajectory с 3+ robots
- Compositing time per embodiment
- Visual consistency

**Deliverable:** Believable multi-embodiment compositing demonstrated.

---

### 4. Dataset generation scale

**Задача:** 1 demo → N synthetic examples

**Процедура:**
- M embodiments (3-5)
- K viewpoints per embodiment (5-10)
- P appearance variations (3-5)
- **Total: 1 → M×K×P**

**Метрики:**

**Scale:**
- Data amplification factor
- Cost comparison bar chart:
  - Human demo: 5 мин capture + Y мин process → M×K×P outputs
  - Robot teleop: Z мин × M×K×P = огромно
- Time/compute per demo

**Diversity coverage:**
- Viewpoint: camera pose distribution, angle coverage
- Embodiment: morphology parameters (arm length, DoF, gripper type)
- Appearance: color histograms, lighting range
- Visual diversity: LPIPS distance между examples

**Trajectory fidelity:**
- Object trajectory preservation
- Gripper-object alignment в contact frames
- Temporal consistency (smoothness)

**Deliverable:** Proof что 1 capture → много diverse examples, cheaper than teleoperation.

---

### 5. Synthetic data quality validation

**Задача:** Доказать data quality без training VLA

**Метрики:**

**Feature representation:**
- Encode через pre-trained CLIP/DINOv2/R3M
- PCA/t-SNE: feature space coverage
- Если есть real robot images → domain gap (MMD, Wasserstein)

**Task-relevant content:**
- Object detection accuracy (off-the-shelf detector)
- Robot joint visibility/pose recovery
- Gripper state detection

**Sim-to-real similarity (если есть real photo):**
- PSNR/SSIM/LPIPS synthetic vs real
- FID score
- Visual Turing test

**Ablations:**
- Human demo vs scripted straight-line
- RGB-D vs RGB-only reconstruction
- 2-view vs 8-view
- With/without color augmentation

**Deliverable:** Evidence что synthetic data high-quality, diverse, task-relevant.

---

## Implementation plan

### Phase 1: Retargeting expansion (2 недели)

- Retarget existing 28 episodes к 3-5 embodiments
- Measure cross-embodiment feasibility
- Validate workspace/grasp compatibility

**Artifacts:** `plan_{ur5,iiwa14,robotiq_3finger}.h5`, feasibility report

---

### Phase 2: 3DGS reconstruction (3 недели)

- Setup Nerfstudio/gsplat для RGB-D input
- Process 10-15 episodes
- Measure reconstruction quality (PSNR/depth error)
- 2-view vs 8-view ablation

**Artifacts:** 3DGS models per scene, reconstruction metrics

---

### Phase 3: Robot compositing (3 недели)

- Implement mesh rasterization + GS rendering pipeline
- Composite 3 robot models в test scene
- Validate geometric alignment
- User study для visual plausibility

**Artifacts:** Compositing pipeline code, visual quality metrics

---

### Phase 4: Dataset generation (2 недели)

- Automate M×K×P synthetic demo generation
- Implement viewpoint/appearance variations
- Measure diversity coverage
- Cost/scale analysis

**Artifacts:** Dataset generator, scale/diversity metrics

---

### Phase 5: Quality evaluation (2-3 недели)

- Feature space analysis (CLIP/DINO)
- Task-relevant content validation
- Ablation studies
- Sim-to-real comparison если доступен real robot photo

**Artifacts:** Quality report, ablation results

---

### Phase 6: Writing (4-5 недель)

- Update Ch1 (новый RQ)
- Rewrite Ch2 (3DGS literature)
- Rewrite Ch3 (method: +3DGS, +compositing, +generation)
- Write Ch4 (experiments: все metrics выше)
- Update Ch5/6 (discussion/conclusion)

---

## Timeline

**Assuming full-time work:**

| Phase | Duration | Weeks total |
|-------|----------|-------------|
| 1. Retargeting expansion | 2 недели | 2 |
| 2. 3DGS reconstruction | 3 недели | 5 |
| 3. Robot compositing | 3 недели | 8 |
| 4. Dataset generation | 2 недели | 10 |
| 5. Quality evaluation | 2-3 недели | 12-13 |
| 6. Writing | 4-5 недель | 16-18 |

**Total: 4-4.5 месяца**

---

## Что НЕ обещаем

**Не включено в scope:**
- Policy training (ни BC, ни RL)
- VLA fine-tuning experiments
- Task success validation в sim/real
- Real robot deployment
- Proof что policy learns better from ViKi data

**Если спросят "а работает ли policy?":**

*"This work measures data generation capability — reconstruction quality, cross-embodiment transfer, and synthetic data diversity. Policy learning is orthogonal future work. Similar to GaussianFactory evaluating reconstruction before training policies, we validate the data factory itself."*

---

## Updated research question

**Main RQ:**

*"Can multiview RGB-D human demonstrations be transformed into diverse, photorealistic, multi-embodiment synthetic datasets for vision-based robot learning?"*

**Sub-questions:**

1. Can affordable 2-view RGB-D capture produce sufficient 3DGS reconstruction quality?  
   **Measured:** PSNR/SSIM/depth error

2. How many synthetic demonstrations can be generated from one human capture?  
   **Measured:** M×K×P amplification factor, cost vs teleoperation

3. Does the pipeline generalize across robot morphologies?  
   **Measured:** cross-embodiment retargeting success, diversity coverage

4. Is synthetic data visually and semantically comparable to real robot data?  
   **Measured:** feature similarity, perceptual quality, task-relevant content

---

## Contributions

**Implemented:**
1. RGB-D human demo → 3DGS scene reconstruction pipeline
2. Multi-embodiment retargeting + photorealistic rendering
3. Factorial dataset generation: 1 capture → M×K×P examples

**Experimental:**
1. Reconstruction quality validation на affordable 2-view setup
2. Cross-embodiment transfer feasibility measurement
3. Synthetic data quality characterization без policy training
4. Cost/scale analysis vs teleoperation

**Positioning:**

*"ViKi demonstrates that human RGB-D demonstrations can serve as a scalable source for multi-embodiment synthetic robot dataset generation. Unlike existing approaches that fix embodiment or generate motion procedurally, our pipeline preserves human motion patterns while enabling factorial expansion across robot morphologies through 3DGS-based rendering."*

---

## Risks and mitigation

**Risk 1: 2-view 3DGS качество недостаточно**

*Mitigation:* HO-Cap 8-view как upper bound, показать degradation gracefully. Можно предложить post-capture scan phase (двигать камеры вокруг статической сцены).

**Risk 2: Robot compositing выглядит unrealistic**

*Mitigation:* User study покажет это. Если плохо — задокументировать как limitation, предложить neural relighting future work.

**Risk 3: Reviewer спросит "но где policy results?"**

*Mitigation:* Cite GaussianFactory (тоже validated data generation before training), RoboSplat (separated data generation from policy learning). Наш scope — data factory, не end-to-end policy.

**Risk 4: Недостаточно времени**

*Mitigation:* Core phases 1-5 = 12 недель. Если затягивается, можно skip user study (phase 3), reduce ablations (phase 5). Минимальный viable = phases 1-4 (10 недель).

---

## Next steps

1. **Discuss с научным руководителем:**
   - Согласование нового scope
   - Проверка достаточности contributions для диплома
   - Timeline feasibility

2. **Verify access to compute:**
   - GPU для 3DGS training (какой доступен?)
   - Storage для synthetic dataset (сколько?)

3. **Start Phase 1:**
   - Retarget existing episodes к 2-3 новым embodiments
   - Validate cross-embodiment feasibility
   - Это можно сделать быстро, покажет technical feasibility

4. **Prototype 3DGS reconstruction:**
   - Setup Nerfstudio на 1-2 test scenes
   - Check reconstruction quality
   - Estimate compute requirements

---

## References (проверенные)

1. **RoboSplat:** "Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation", arXiv:2504.13175v1, April 2025. [https://arxiv.org/abs/2504.13175](https://arxiv.org/abs/2504.13175)

2. **GaussianFactory:** "Demonstration Synthesis from a Single Scan via Gaussian Splatting for Visuomotor Policy Learning", arXiv:2609.21112, September 2024. [https://arxiv.org/html/2609.21112](https://arxiv.org/html/2609.21112)

3. **LEGS:** "Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World", arXiv:2606.01458v1, June 2024. Project page: [https://legsvla.github.io/](https://legsvla.github.io/)

4. **SplatSim:** N. Qureshi et al., "Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting", arXiv:2409.10161v3, ICRA 2025. Code: [https://github.com/qureshinomaan/SplatSim](https://github.com/qureshinomaan/SplatSim)

5. **Real2Render2Real (R2R2R):** "Scaling Robot Data Without Dynamics Simulation or Robot Hardware", arXiv:2505.09601v1, May 2025, CoRL submission. Code: [https://github.com/uynitsuj/real2render2real](https://github.com/uynitsuj/real2render2real)

6. **HO-Cap:** Y.-W. Chao et al., "Capturing Hands Articulating Objects in the Wild", CVPR 2023.

7. **MimicGen:** A. Mandlekar et al., "A Data Generation System for Scalable Robot Learning using Human Demonstrations", CoRL 2023.

8. **RoboSimGS** (дополнительно найденная работа): "High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting", arXiv:2510.10637v1, October 2024. Hybrid 3DGS appearance + mesh physics для interactive simulation.

---

## Статус публикаций (ноябрь 2026)

**Peer-reviewed:**
- SplatSim — ICRA 2025 (accepted)
- HO-Cap — CVPR 2023
- MimicGen — CoRL 2023

**Preprints (peer-review status неизвестен):**
- RoboSplat (arXiv 2504, April 2025)
- GaussianFactory (arXiv 2609, September 2024)
- LEGS (arXiv 2606, June 2024)
- R2R2R (arXiv 2505, May 2025, CoRL submission claim)
- RoboSimGS (arXiv 2510, October 2024)

**Все 3DGS robot learning работы — 2024-2025, очень свежие.**

---

## DOE Template Compliance

Thesis structure updated to match Innopolis DOE requirements (`paper/doe-docs/thesis_template/`):

**Formatting:**
- Document class: `extreport` (14pt, Times font via `tempora`)
- Margins: 2.5cm left, 2.2cm other sides
- Line spacing: 1.5
- Bibliography: biblatex IEEE style
- Title page: separate `title.pdf` from DOE Word template

**Structure:**
- Abstract (EN + RU) — updated with 3DGS scope
- Ch1: Introduction — **needs rewrite** (new RQ)
- Ch2: Literature Review — **needs expansion** (3DGS section)
- Ch3: Methodology — **needs rewrite** (add 3DGS, compositing, generation)
- Ch4: Experiments — **new content** (reconstruction, cross-embodiment, quality metrics)
- Ch5: Discussion — **needs update**
- Ch6: Conclusion — **needs update**

**Files created:**
- `paper/thesis/thesis_doe.tex` — DOE-compliant main file
- `paper/thesis/chapters/abstract.tex` — updated abstract
- `paper/thesis/README_DOE.md` — migration guide
- `paper/thesis/DOE_CHECKLIST.md` — submission checklist

**Migration status:** Template structure ready, chapters need content adaptation.

See `paper/thesis/README_DOE.md` for detailed migration plan.

# Contact-aware human-to-robot retargeting: literature check (15 October 2026)

## Why this matters for ViKi

The 831-frame HO-Cap MuJoCo GUI run was a **hand-only** UR10/Robotiq 2F-85
plan with four freely simulated, uncalibrated box proxies; it did not test a
validated object-aware controller. The G10_4 insertion-depth and friction
trial was object-aware but used one object, an approximate contact model, and
failed its strict grasp/transfer gates. See
[`2026-10-15-hocap-full-scene-gui-smoke.md`](2026-10-15-hocap-full-scene-gui-smoke.md)
and [`2026-10-15-hocap-insertion-depth-trial.md`](2026-10-15-hocap-insertion-depth-trial.md).
Neither experiment identifies the number of fingers as the sole failure cause.

## Closest primary work

| Work | What it actually does | Transferable lesson / caveat |
|---|---|---|
| [ContactOpt, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Grady_ContactOpt_Optimizing_Contact_To_Improve_Grasps_CVPR_2021_paper.html) | Refines visually recovered human hand/object geometry using predicted surface contacts. | Contact is an inference/optimization variable, not just a fingertip-distance threshold. Geometric contact plausibility alone does **not** establish a robot grasp in dynamics. |
| [DexMV, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990562.pdf) | Multi-view human video to 3-D hand/object pose, dexterous robot retargeting in simulation, then imitation/RL. | The broad video-to-sim-to-policy concept is established prior art; ViKi cannot claim this whole pipeline as novel. |
| [ManipTrans, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_ManipTrans_Efficient_Dexterous_Bimanual_Manipulation_Transfer_via_Residual_Learning_CVPR_2025_paper.pdf) | Pretrains a hand-motion imitator, then learns residual correction under interaction constraints for bimanual dexterous transfer. | A good kinematic reference is a starting point, not a physical demonstration. This is heavier than the current single-arm, low-compute target. |
| [DexMachina, ICML 2026](https://project-dexmachina.github.io/) | Functional/object-state retargeting with a virtual object controller whose assistance decays as the policy learns. | It optimizes the *object outcome* rather than literal hand-motion matching. Assistance must be removed for evaluation; their focus is bimanual articulated-object tasks. |
| [SPIDER, 2025 preprint; v3 September 2026](https://arxiv.org/html/2511.09484v3) | Physics-based sampling of robot control sequences around a human-derived reference, with temporary contact guidance and robustness over mass/friction/contact uncertainty. | Most relevant low-training alternative: search over a short control sequence in the actual simulator and score the unassisted rollout. Their large-scale examples use substantial parallel simulation; their reported speedup over RL is not a ViKi runtime estimate. |
| [Contact-Anchored Retargeting and Residual Policy Learning, 2026 preprint](https://arxiv.org/html/2609.24093) | Reconstructs contacts with a simulated MANO hand, transfers object-surface contact anchors across robot morphologies, then uses residual RL for feasible manipulation. | Direct precedent for contact-in-object-frame plus physics repair. **Not** a like-for-like capture baseline: input uses 22 infrared mocap cameras, two RGB-D cameras, markers on hand/objects, and registered simulation geometry. |
| [Morphometric Imitation, 2026 preprint](https://arxiv.org/html/2609.28660v2) | Morphology/contact-aware IK, residual RL for object/contact tracking, then visuomotor distillation. Explicitly penalizes table collisions. | A thicker four-finger hand (Allegro) had *more* table-collision trouble than a slimmer three-finger hand on some tasks. More DoF do not automatically rescue tabletop grasping. Its reported success rates are on different hands, objects, and protocols, not directly comparable to ViKi. |
| [VideoManip, 2026 preprint](https://arxiv.org/pdf/2602.09013) | RGB video to hand/object reconstruction and contact-refined grasp data for robot policy learning. | Its own ablation reports 30.7% versus 63.75% grasp success without/with contact optimization on its 20-object test. This is evidence to test contact quality before hand complexity, **not** an expected ViKi gain. |

The 2026 entries are recent preprints except DexMachina (ICML 2026); treat
their experimental claims as author-reported. The reported numbers are from
different datasets, embodiments, and evaluators and must not be compared as a
leaderboard.

## Implication for the present two-finger setup

**Inference from these papers, not a measured ViKi result:** before swapping
Robotiq for a dexterous hand, represent the demonstrated interaction as
object-relative contact patches and an object pose trajectory, then repair the
existing robot control sequence under free-body MuJoCo dynamics. For a parallel
jaw gripper, the low-dimensional search variables could be wrist approach
pose/yaw, jaw aperture/closing timing, normal-force proxy, carry pose, and
release/retreat timing. Preserve left/right pad contact on the object during
carry; score actual free-body object motion, not just fingertip proximity or IK
error. Collision checking must include the real table and non-target objects.

The immediate experiment should be deliberately small:

1. Establish a credible simulation fixture for **one** HO-Cap object: documented
   collision geometry, correct initial pose and tabletop frame, plausible mass,
   inertia, pad geometry and friction range. Our current 5th–95th percentile
   box, 0.20 kg mass and inferred table height are not such a fixture.
2. Use the **same** object, scene, initial state, command rate and physical
   parameter sweep for hand-only reference, present object-aware IK, and an
   opt-in contact-aware physics refinement. If temporary contact assistance is
   used for search, remove it in every scored rollout.
3. Report stable lift and placement success, continuous bilateral-pad contact
   while carried, object pose error, non-target displacement, table/object
   collision and penetration, robustness across mass/friction values, and
   compute cost. Keep failed and skipped rollouts in the denominator.
4. Only then compare a dexterous-hand embodiment in the *same* scene and
   physical uncertainty sweep. If it alone succeeds repeatedly, that is
   evidence for changing morphology; if both fail, the upstream geometry,
   alignment, contact representation or physics model remains the likely
   bottleneck.

For thesis positioning, the defensible contribution is not “video to robot
demonstrations to policy” in general. It could be a **low-cost calibrated
RGB-D capture and measured error decomposition** through hand/object tracking
and retargeting, or a **well-controlled two-finger contact/physics-retarget
study**, but only if we actually validate those claims. A visually plausible
replay or unassisted object failure is not yet robot-ready training data.

## 20 October 2026 update: anthropomorphic DG5F and broader retargeting sources

The current UR10 + Tesollo DG5F test is a different embodiment from the
two-finger experiment above. The [DG5F report](2026-10-20-tesollo-dg5f.md)
records a numerically feasible 190-frame IK trajectory from **final optimized
hand-fit**, but its free-body MuJoCo replay does not lift the object. Small
TCP error against a derived target is therefore not evidence of a good grasp.
The original two-finger observations above remain historical, not DG5F results.

| Primary source | What it establishes | Relevance and limits for ViKi |
|---|---|---|
| [DexPilot (2019)](https://arxiv.org/html/1910.03135) | Uses human/robot hand geometry and a nonlinear kinematic objective to preserve useful finger relations in live Allegro hand-arm teleoperation. | A relevant classical baseline. Its human operator sees the robot and adjusts online; its demonstrated task success must not be attributed to unattended offline joint replay. Hardware also includes high-friction contact coverings and tactile fingertips. |
| [ByteDexter (2025)](https://arxiv.org/html/2507.03227) | For a 20-DoF anthropomorphic hand, optimizes 15 within-/between-finger keyvectors; short thumb-to-finger distances receive high weights for pinch poses. | A closer *kinematic* baseline than matching all 20 phalanx landmarks equally. Uses a Manus glove; the objective still does not enforce object contact forces or free-body lift. |
| [AnyDexRT (2026 preprint)](https://arxiv.org/html/2607.08341) | Learns fingertip correspondences, anchors them with a few human-guided examples, and refines pinch poses with contact classification. | Useful comparison for cross-hand calibration and pinch quality. Primarily teleoperation retargeting; a good mapping alone does not establish dynamic grasp stability. |
| [Hand2Robot (11 October 2026 preprint)](https://arxiv.org/html/2610.15769) | Jointly reasons about object geometry/pose over time and hand-object interaction while generating robot supervision from RGB-D video. | Closest reminder that an inconsistent source hand/object reconstruction contaminates any downstream contact objective. Its reported downstream benefit is policy post-training, not direct evidence that an open-loop ViKi replay would hold. |
| [ReForce (2026 preprint)](https://arxiv.org/html/2608.15560) | Adapts a kinematic reference using target and *measured* fingertip normal forces in a feedback controller. | Relevant once DG5F has calibrated finger-force feedback or a justified estimate. ViKi's current RGB-D hand-fit contains neither measured human forces nor robot fingertip forces; the published controller cannot be copied as a sensor-free offline retargeter. |
| [UMR (September 2026 preprint)](https://arxiv.org/html/2609.02134v2) | Learns reusable dense human/robot surface correspondences in a canonical pose, then optimizes pose, normals and geometric contact residuals. | **Watchlist only; not selected for the current DG5F approach.** The paper evaluates humanoid embodiments and explicitly identifies dexterous hands as future work. Its contact geometry does not by itself guarantee force closure or stable object transport. [Authors' code](https://github.com/hanyang9/UMR). |

The already listed contact/physics papers above are especially important for
interpreting the DG5F failure:

- [Contact-Anchored Retargeting](https://arxiv.org/html/2609.24093)
  transfers finger-region/object-surface contacts and then applies robot-side
  residual policy learning. Its precision capture and simulated human-hand
  physics refinement are essential inputs, not incidental implementation
  details. The [authors' repository](https://github.com/DexGEM-Lab/real2sim2real)
  currently says the code release is forthcoming.
- [SPIDER](https://arxiv.org/html/2511.09484v3) searches robot *controls*
  through physics rather than accepting the IK solution as final. The
  [released pipeline](https://github.com/facebookresearch/spider) supports
  several dexterous hands and MuJoCo Warp. Its authors omit contact guidance
  in a large generation run because noisy source demonstrations can make it
  unreliable; its GPU-scale sampling cost and reported metrics do not predict
  cost or success for one ViKi scene. Check its CC BY-NC 4.0 code licence before
  reuse outside research.
- [Morphometric Imitation](https://arxiv.org/html/2609.28660v2) separates
  contact/morphology-aware kinematic transfer from physics-based residual
  correction. Its rollout criteria include object tracking, robot-object
  contact, table collision and excessive contact force. The reported gains
  compare different hands, objects and training protocols from ours.

For an implementable kinematic baseline, also inspect
[dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting); its
examples include retargeting hand-object datasets and it makes joint-order
translation explicit. It is a software baseline, **not** evidence of a stable
physical grasp.

### Research interpretation for the DG5F case

**Inference, not a new experimental result:** the main gap is between
*kinematic resemblance* and *object transfer*. A defensible comparison would
keep the same final hand-fit, tracked object, initial scene and DG5F physics,
then separate (1) phalanx/keyvector/surface mapping, (2) object-relative
contact-region preservation, and (3) physics-based command refinement. Score
each stage by actual free-body grasp/lift, contact distribution, penetration,
object-position **mean and median**, and robustness to mass/friction and
initial-pose uncertainty. The current target-object collision omission and
convex-hull contact approximation must be disclosed. Do not treat a simulated
contact patch inferred from noisy, mutually penetrating reconstructions as a
measured human force or ground-truth contact.

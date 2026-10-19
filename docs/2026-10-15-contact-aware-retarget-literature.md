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
| [SPIDER, 2026 preprint](https://arxiv.org/html/2511.09484v3) | Physics-based sampling of robot control sequences around a human-derived reference, with temporary contact guidance and robustness over mass/friction/contact uncertainty. | Most relevant low-training alternative: search over a short control sequence in the actual simulator and score the unassisted rollout. Their large-scale examples use substantial parallel simulation; their reported speedup over RL is not a ViKi runtime estimate. |
| [Contact-Anchored Retargeting and Residual Policy Learning, 2026 preprint](https://arxiv.org/pdf/2609.24093) | Reconstructs contacts with a simulated MANO hand, transfers object-surface contact anchors across robot morphologies, then uses residual RL for feasible manipulation. | Direct precedent for contact-in-object-frame plus physics repair. **Not** a like-for-like capture baseline: input uses 22 infrared mocap cameras, two RGB-D cameras, markers on hand/objects, and registered simulation geometry. |
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

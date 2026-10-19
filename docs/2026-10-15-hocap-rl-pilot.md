# HO-Cap object-disjoint RL pilot (15 October 2026)

This is a **failed, in-sample pilot**, not an unseen-object result. No policy
has yet picked up an object reliably. Do not cite this run as evidence of
physical grasping, transfer, or sim-to-real performance.

## Data partition and physical assumptions

`experiments/hocap_rl_catalog.py` writes
`experiments/hocap_rl_split.json` from the 28 locally ingested HO-Cap scenes.
The deterministic partition has 14 training scenes (28 unique object IDs) and
14 held-out scenes (32 unique IDs), with no object identity in both partitions.
The G10_4 diagnostic scene is in training. This partition is a **design** for
later evaluation: no 14-scene policy has been trained. The manifest also
records the official mesh bounding-box dimensions. The downloaded model and
scene metadata do not give measured mass, so this pilot samples a simulator
mass of 0.1–0.3 kg and sliding friction of 0.7–1.5; these are assumptions.

## Pilot contract

The pilot uses the cropped `subject_1/20231025_170105` G10_4 object-aware
plan in `data/_retarget_diagnostics/hocap_g10_4_orient_weight_003`, frame 44
of the 180-frame crop (original frame 464), an official G10_4 mesh convex
hull aligned to the first-frame HO-Cap GT pose, UR10 and Robotiq 2F-85. The
simulation has a fitted tabletop and explicit rubber-pad contact proxies,
not calibrated surface compliance. The 5-D policy action is local TCP
translation, yaw and finger closure at 25 Hz. Observations include robot
state, pad/object relative positions and contact force, using privileged
simulator state. Success requires a 25 mm lift and bilateral contact for 13
consecutive control steps. It is **pickup only**, not carry or placement.

The default initial pose/closure is not a reliable pickup. A small manual
search of nine XY offsets and three closure targets found one candidate:
−10 mm TCP X, +10 mm TCP Y, closure target 0.45 rad, then lift. On seeds
1000–1009 it picked up G10_4 in 5/10 randomized simulations; the seed 1000
search itself selected the candidate, so 5/10 is **not** independent generalization.
The failures include sideways object ejection and short-lived contact. This
probe establishes that at least some physical pickups are reachable in this
simulator, not that the geometry/contact model is validated.

## Measured RL runs

Raw per-seed evaluation JSON and checkpoints are ignored local artifacts under
`data/experiments/hocap-rl/`. Every evaluation below is on the **same training
scene**, with randomized mass, friction and ±5 mm object XY; there is no
held-out object test.

| Local run directory | Training steps | Scripted contact seed | Pickup success | Bilateral-contact steps across 10 evaluations | Peak lift |
|---|---:|---|---:|---:|---:|
| `joint7_5000` | 5,000 | none | 0/10 | not recorded by old report | 0.23 mm |
| `tcp5_5000` | 5,000 | none | 0/10 | 0 | 3.72 mm |
| `tcp5_frame52_5000` | 5,000 | none | 0/10 | 0 | 0 mm |
| `tcp5_contact_seed_10000` | 10,000 | 30 scripted episodes, 3 successful, 2,875 transitions | 0/10 | 0 | 2.67 mm |

The visible `tcp5_5000` GUI policy moved away from G10_4 rather than grasping.
Its GUI invocation also loaded `hocap_g10_4_official_geometry/plan.h5`, while
the policy was trained on `hocap_g10_4_orient_weight_003/plan.h5`; that visual
rollout is **not** a matched evaluation. The correct-plan numerical evaluation
also fails 0/10. The two plans have different joint values even at reset
frame 44. Do not use the mismatched GUI rollout as a quantitative result.

The present reward and 5k–10k SAC training do not produce a usable contact
policy. A no-contact rollout avoids the large failed-contact terminal penalty;
the data are consistent with that local optimum, but do not isolate reward
from exploration, plan accuracy and simulator contact geometry. Merely
increasing training steps or changing friction without a controlled ablation
would not establish a fix. The scripted seed supplied successful transitions
but was not enough for the trained actor to reproduce them.

A separate, non-checkpointed diagnostic collected 100 scripted rollouts on
seeds 2000–2099; 17 succeeded. Behaviour cloning of the 947 transitions from
those successful rollouts produced 3/10 pickups on seeds 3000–3009. Starting
from that actor and training SAC for 10,000 steps on the same environment
reduced pickup success to **0/10**, with zero bilateral-contact steps in all
ten rollouts. This is evidence of policy collapse under the present SAC setup,
not proof that any one reward term is the sole cause. The exact console output
is preserved locally at `data/experiments/hocap-rl/bc_then_sac_probe.log`.

## Gate before scaling

1. Save and verify hashes of the plan, track, mesh and GT inputs for every
   checkpoint and GUI rollout. Do not compare policies across different plans.
2. Establish a repeatable single-object pickup baseline under multiple seeds,
   with bilateral-contact dwell, lift, slip and object ejection all logged.
3. Screen each HO-Cap object for a feasible two-finger grasp and label the
   actual manipulated object and pickup interval; the 28 hand-only plans do
   not encode a ready-made grasp task for every scene.
4. Only then train across the 14 training scenes and evaluate the 14 held-out
   scenes/identities using the same simulator protocol. Report assumptions
   about mass, friction and mesh/contact proxies separately from HO-Cap GT.

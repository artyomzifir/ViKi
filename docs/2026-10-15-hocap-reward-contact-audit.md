# HO-Cap G10_4: reward and contact audit (15 October 2026)

This is a **single-scene simulation diagnostic**, not an unseen-object result,
real-material calibration or a completed pickup-and-place policy. The tested
object is G10_4 in the cropped `subject_1/20231025_170105` plan at crop frame
44, with the official mesh compiled as a convex collision hull, assumed
0.1–0.3 kg mass, explicit box rubber-pad proxies and a fitted tabletop.
Nothing here validates these as accurate physical properties of G10_4.

## Sources informing the diagnosis

- [MuJoCo contact modeling and preventing slip](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip)
  distinguishes insufficient normal force, friction-cone limits, torsional
  friction, and unsuitable collision geometry. By default a contact takes the
  element-wise **maximum** of the two geom friction vectors unless one geom
  has higher `priority`; changing only object friction can change both its
  pad and table contacts.
- [Ng, Harada and Russell, ICML 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
  show why potential-difference shaping is preferable to arbitrary recurring
  progress rewards when preserving the task objective matters.
- [RLPD, ICML 2023](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)
  explicitly finds that simply adding demonstrations to SAC's replay buffer
  is insufficient; it studies balanced offline/online batches, critic LayerNorm
  and more sample-efficient value learning. Our earlier pilot was **not** RLPD.
- [Residual RL from Demonstrations](https://arxiv.org/abs/2106.08050)
  motivates a bounded learned correction around a demonstrator rather than
  letting an unconstrained actor destroy an initially useful grasp motion.
- [SPIDER, 2025/2026 preprint](https://arxiv.org/html/2511.09484v3)
  optimizes trajectories in physics, optionally using temporary contact
  guidance and robustness sweeps over friction/mass. This is a candidate
  **trajectory-generation** baseline, not evidence that ViKi already works.

## A real reward loophole

The current `experiments/hocap_grasp_env.py` gives `+0.12` **per control step**
for each pad with nonzero force even when the object remains supported by the
table. It also pays a distance term every step and leaves timeout unpenalized.
Success, by contrast, needs a 25 mm airborne lift with bilateral contact for
13 steps. The reward has no phase gate preventing indefinite contact on the
table. It additionally hides `stable_lift_steps` from the observation although
that counter determines termination.

Paired seeds 1000–1009, current randomized mass/friction/XY, same scene:

| Controller | Mean episode return | Successful pickups | Interpretation |
|---|---:|---:|---|
| No action | −4.16 | 0/10 | Safe non-contact reference. |
| Scripted offset, close, lift | +14.42 | 5/10 | Successful episodes return about +41; failures offset them. |
| Trained 10k-step SAC | −15.77 | 0/10 | Worse than no action; training failed, not just rational avoidance. |
| Scripted offset, close, **never lift** | **+25.28** | **0/10** | 109–113 bilateral-contact steps, table support on all 125 steps, peak lift at most 1.5 mm. |

Thus the no-lift contact hold is an actual accessible reward exploit. It does
**not** prove that the saved SAC actor learned this exploit: its evaluated
returns are negative and it generally did not make bilateral contact. Both
reward misalignment and off-policy learning/exploration failure remain live
issues. In the previous controlled diagnostic, 17/100 scripted seeds succeeded;
BC on successful transitions yielded 3/10 on separate seeds, then 10k SAC
steps reduced that actor to 0/10. See
[`2026-10-15-hocap-rl-pilot.md`](2026-10-15-hocap-rl-pilot.md).

## Contact-model sensitivity, with confounds removed

One 30-seed sweep changed *both* pad–object and object–table friction; it
showed 4/30 successes at friction 1 and 24/30 at friction 2, but it cannot be
used to attribute the effect to rubber alone. `condim=4` did not improve this
setup over `condim=3` at friction 2 (20/30 versus 24/30); that is not a
general verdict on torsional friction.

The second sweep fixed object geom friction to 1.2 and used geom priority to
change exactly one pair type at a time. Effective friction was read back from
`mjData.contact`, not inferred from XML. The same 30 seeds and scripted action
schedule were used in each row; mass and XY randomizations are paired by seed.

| Changed contact | Friction | Success | Lift ≥25 mm | Object ejection >120 mm |
|---|---:|---:|---:|---:|
| Pad–object, table fixed at 1.2 | 0.5 | 0/30 | 1/30 | 30/30 |
| Pad–object, table fixed at 1.2 | 1.0 | 4/30 | 11/30 | 10/30 |
| Pad–object, table fixed at 1.2 | 2.0 | 21/30 | 28/30 | 2/30 |
| Pad–object, table fixed at 1.2 | 3.0 | 9/30 | 21/30 | 1/30 |
| Object–table, pads fixed at 1.2 | 0.5 | 13/30 | 22/30 | 3/30 |
| Object–table, pads fixed at 1.2 | 1.0 | 6/30 | 16/30 | 8/30 |
| Object–table, pads fixed at 1.2 | 2.0 | 8/30 | 16/30 | 8/30 |
| Object–table, pads fixed at 1.2 | 3.0 | 9/30 | 13/30 | 17/30 |

Pad contact matters strongly, but **more friction is not monotonically
better**. Table friction also changes the outcome, consistent with closure
pushing a still-supported object sideways. This is evidence of a fragile,
uncalibrated fixture, not proof of a MuJoCo engine defect. MuJoCo 3.14.0
was used. The 0.5–3.0 values are a sensitivity study, **not measured
material coefficients**; choosing 2.0 simply because it succeeds would
invalidate a physical-accuracy claim.

Even with fixed friction 2.0 on object/pads, 17/30 scripted seeding episodes
succeeded, yet a fresh 10k-step SAC run evaluated **0/10** on seeds 1000–1009.
Better contact physics alone does not repair the current learning setup.

## Proposed fixes and gates

1. **Freeze a one-object physics fixture before learning.** Inspect pad/hull
   collision surfaces and tabletop pose in GUI, log pad normal and tangential
   forces, slip, penetration, actuator target vs actual jaw opening, object
   ejection and solver status. Compare a simple graspable box to G10_4 under
   the same controller. Vary pad friction, table friction, mass, pad compliance,
   closure speed and contact geometry *separately*. Report robustness across
   the declared range; do not select only the easiest coefficient.
2. **Replace the recurring contact reward.** Keep task success as a stable,
   unsupported lift (later, released placement). Add only one-time/gated
   progress for approach → bilateral contact → lift; while still on the table,
   contact must not earn a positive reward every step. Use a
   `gamma * Phi(next_state) - Phi(state)` potential for distance/progress rather
   than a per-step distance bonus. Include phase/dwell and controller target in
   the observation if they affect dynamics/termination. Log every reward
   component and assert that no-lift hold cannot beat a valid lift.
3. **Keep the feasible controller as a baseline.** Train bounded residual
   actions around a selected contact trajectory and compare against scratch
   SAC, scripted control and BC on identical seeds. If using prior rollouts,
   test balanced offline/online replay and critic normalization as controlled
   ablations; merely prefilling SAC is the failed baseline, not RLPD.
4. **Use a clean gate.** Require stable bilateral load-bearing pickup on at
   least two friction/table configurations and multiple unseen random seeds,
   without virtual weld/contact assistance at evaluation. Then extend to
   full carry/release and only later to the 14/14 object-disjoint split.

The raw scripts and console output for reward/no-op/scripted/SAC, contact-hold,
coupled and pair-isolated friction sweeps, and fixed-friction SAC are preserved
locally under `data/experiments/hocap-rl/audits/`. The underlying HO-Cap
recordings and meshes were not modified.

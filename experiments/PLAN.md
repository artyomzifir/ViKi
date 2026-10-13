# Cube-transfer experiment plan

`plan.h5` is a geometric UR10 trajectory, **not** evidence of a physical pick-and-place. Replay plans in MuJoCo with the old UR10, Robotiq 2F-85, a 1.2 × 0.7 m table, and three 40 mm cubes. Do not substitute a UR10e model: it has different kinematics. Red–green separation is nominally 400 mm. Use episode-relative cube positions until the remaining table-coordinate ambiguity is resolved.

## Gates

1. Complete Extract/Prepare with a named profile. Reject failed pose-quality gates; a partly written `cln.npz` is not valid.
2. Review SAM overlays at start, middle, and end in both cameras. Confirm `manipulated_object` follows the **blue** cube from red toward green. Overlay colors are mask-ID colors, **not** cube colors; compare the raw RGB video before assigning task labels. Do not reuse prompts if the raw layout changes.
3. Validate exact embodiment: old UR10 joint order/limits, FK against sampled ViKi plan states, flange–adapter–TCP transform, Robotiq pad-center gap versus `gripper_opening_m`, collision/contact geometry and actuator gains. MuJoCo Menagerie supplies UR10e and Robotiq, not the exact old UR10 combination.
4. Replay interpolated joint/jaw commands in physics with free cube bodies; never weld/teleport the grasped cube. Log grasp, lift, slip, falls, release, contact, placement and control error. Count physical success only after supported, released, stable stacking for ≥0.5 s after retreat. `success.py` implements that log-level predicate, **not** the simulator.
5. Filter unsuccessful rollouts before imitation learning. Report generated-success yield, not a claim of limitless valid data or real-robot success.

## Baselines and evaluation

One state/action contract for every method: joint positions/velocities, cube poses, target-relative pose and jaw state; bounded joint-position deltas plus jaw command. Compare physical replay yield, state behavioural cloning on successful rollouts, SAC from scratch, and BC-initialized or demonstration-mixed SAC. SAC+HER and SmolVLA are later experiments, not substitutes for physics validation.

Use identical held-out pose/dynamics seeds across methods. Primary metric: released stable stack success, Wilson 95% interval (≥50 evaluation seeds for PoC; ≥100 and three training seeds for publication). Also report pickup/lift/release rates, falls, collisions, slip, placement error, training interactions, wall-time, and successful sim rollouts per input recording. Ablate hand-only versus object-aware plans, 10/25/50 demonstrations, and held-out cube poses/physical parameters. Fifty repetitions of one layout do not demonstrate generalization.

`requirements.txt` isolates simulator/training packages from the running camera service. Installation alone does not pass the exact-model gate.

## Scope

Pouring is out of scope. The remaining study covers cube transfer only.
The original local demonstration and simulator artifacts were lost in the
October workspace deletion; the historical metrics in the thesis must be
regenerated before publication.

## Primary sources

- [MuJoCo Python API](https://mujoco.readthedocs.io/en/latest/python.html); [Menagerie model list](https://github.com/google-deepmind/mujoco_menagerie/blob/main/README.md).
- [MimicGen](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf): few source demonstrations → task-relative scene variation → physics-success filtering. Closest generation comparison.
- [robomimic](https://robomimic.github.io/docs/) for BC, [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), [HER](https://stable-baselines3.readthedocs.io/en/master/modules/her.html), [RLPD](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf) for demo-mixed RL.
- [robosuite Stack](https://robosuite.ai/docs/_sources/modules/environments.html) is a task comparator, but its [robots](https://robosuite.ai/docs/modules/robots.html) do not include the old UR10.
- [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla) is a later vision imitation baseline requiring compatible action/image data and genuine task variation.

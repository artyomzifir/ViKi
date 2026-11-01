# Shadow object-frame palm correction: carry improves, task still fails (2026-10-25)

A bounded post-retarget palm-pose experiment improved physical contact during the G10_2 Shadow Hand carry, but it did **not** yield an accepted demonstration. A **−6 mm displacement along the tracked object's local X axis** reduced carry object-track error p95 from **104.569 to 54.741 mm** and raised simultaneous positive-force opposing contact from **0.678 to 0.837** of carry physics steps. The object still loses the grasp and ends **139.421 mm** from the recorded final position, versus **25.201 mm** for the original plan. Target-task and full-scene success remain false; the exported command is a diagnostic artifact, not a production replacement.

## Question and fixed design

The preceding [hand-command polish pilot](2026-10-25-shadow-physics-polisher.md) failed to retain the object by changing finger joint targets or slowing the carry. An independent hold test showed that the original grasp slips even with the arm stopped: from frame 60, wrist-relative slip reaches 14.04 mm at 1 s and 197.64 mm at 1.5 s, when opposing force reaches zero. This motivated a local change to how the palm meets the object before acquisition.

The new family has six parameters in the tracked object's frame: XYZ translation, each bounded to ±12 mm, and XYZ rotation-vector components, each bounded to ±8°. Rotation is about the palm's own origin; it does not drag the palm position around the object center. A smooth C1 envelope rises during frames 10–23, so the correction is fully applied at first recorded contact, holds through frame 110, and returns to zero by frame 125. The 22 finger commands and all 190 saved timestamps stay bitwise unchanged. Only the six UR10 arm commands are re-solved from the desired `rh_palm` pose by arm-only Pinocchio FK/IK; no joint-angle offset is added as a stand-in for Cartesian motion.

The candidate set was frozen before simulation: zero residual; signed half- and full-bound probes on each of six axes (24); and 23 seeded Latin-hypercube points, **48 total**. The inherited physics scene, score, and task gates are exactly those of the prior pilot. Nominal ranking uses carry position-error p95, simultaneous positive-force thumb/finger contact at the same physics step, the declared vertical reference-deficit fraction, final position error, and a robot-table collision penalty. The deficit includes early lift lag and is not by itself a drop counter. Selection uses this score, not an acceptance claim.

## IK feasibility before physics

The generator solves the arm from the original arm configuration at each gated frame, quantizes to the plan's float32 command format, and checks achieved palm position ≤1.5 mm and orientation ≤1°. It rejects nonfinite values, URDF joint-position violations, and actual per-segment joint-velocity violations on the saved timestamps. It checks all URDF collision bodies against the robot-base floor and the nearest eight retained self-collision pairs per frame. Adjacent and nominal-contact pairs are excluded by the existing retarget screen; tracked scene-object collisions are checked in the later MuJoCo rollout, not claimed as an IK guarantee. Failed candidates do not enter physics simulation.

The saved zero command reproduces the archived baseline bit for bit and reproduces all original replay metrics. Pinocchio's wrist frame and its fixed wrist-to-palm transform agree with MuJoCo's imported URDF at frames 0, 42, 90, 125, and 189 within roughly 7×10⁻⁸ mm and 2.4×10⁻⁸ degrees. An independent eight-link/seven-pose check found maximum position disagreement 1.11×10⁻¹⁰ m. The original trajectory's minimum floor margin is +2.847 mm and screened self-collision margin +0.114 mm.

**30/48** candidates passed the IK/geometry/command screen; **0/30** passed the nominal target-task gate. The other **18** were rejected on negative floor margins (the recorded self-collision margins at those rejection frames were positive). The selected candidate's achieved palm error is at most **0.624 mm** and **0.000565°**, minimum floor margin +2.847 mm, minimum screened self-collision margin +0.114 mm, joint-position margin about 9.0×10⁻⁸ rad, and velocity margin +0.0392 rad/s. The tight position margin comes from the source hand configuration; commands are rejected rather than clamped after solving. Between saved physics samples, actual-joint URDF velocity margins are +0.0415 rad/s for baseline and +0.0411 rad/s for the selected plan; this sampled check does not assert a maximum between samples. The selected command changes arm values on 109 frames (13–121), by at most 0.436° at any joint. Approach frames 0–10, release frames 125–189, fingers, and timestamps are exact source copies.

## Physics outcome

The replay uses the unchanged four-object official-mesh scene, 0.08 kg per object, friction 1.2 for robot/object and 1.0 for table, gravity compensation, 0.0005 s implicitfast steps, original position-servo gains and URDF effort limits. No weld, object forcing, collision change, gain change, or online object feedback is used.

| Nominal physics | Original | Selected palm −6 mm X |
|---|---:|---:|
| Frozen score (lower is better) | 148.635 | 87.219 |
| Carry object-track error p95 (mm) | 104.569 | 54.741 |
| Simultaneous opposing contact, carry physics steps | 0.6780 | 0.8371 |
| Longest carry opposing-contact gap (s) | 0.9040 | 0.4585 |
| Carry vertical reference-deficit step fraction | 0.2223 | 0.0876 |
| Final object-track error (mm) | 25.201 | 139.421 |
| Peak robot-table / robot-other-object contact steps | 0 / 0 | 0 / 0 |
| Peak neighbor displacement (mm) | 17.262 | 17.262 |
| Target task / full scene | false / false | false / false |

The selected plan misses the target task's <30 mm carry-p95 and final-error gates. Its higher peak thumb normal force (38.69 N summed over contacts, versus 7.11 N) and roughly 10.26 mm maximum contact penetration in the convex-hull proxy also warrant caution about this simulation-only contact improvement. There is no actuator-effort saturation. Object 3 drifts about 17.19 mm even in a stationary-robot control, confounding the separate <5 mm neighbor-displacement full-scene gate; this does not explain the target's own failed carry and final placement.

An independently paused selected replay still loses the object. At frame 60 with commands held constant, wrist-relative slip is 17.74 mm after 1.5 s and 260.03 mm with zero thumb/finger force after 2 s. When paused at frame 90, contact is gone within 1 s; at frame 98, within 0.5 s. Thus the corrected palm pose delays slip but does not create a stable static grasp.

The fixed held-out 2×2 mass/friction grid also yields **0/4 target-task successes** for both original and selected commands. The selected command improves carry p95 in each setting (from 97.8–215.5 mm for the original to 52.4–64.0 mm) but never reaches the 30 mm carry gate. Its final error ranges from 9.2 to 145.0 mm and is not consistently better. This is sensitivity of one simulated scene, not a success rate or generalization evidence. The feasibility phase took 266.2 s; the full fixed experiment including rollouts and sensitivity took 358.9 s.

## Reproduce and review

From the repository root, use a fresh output directory for a new run:

```bash
docker compose -f docker-compose.experiments.yml run --rm sim experiments/hocap_shadow_palm_polisher.py --output /app/rl-output/shadow_palm_polisher_rerun

docker compose -f docker-compose.experiments.yml run --rm sim experiments/hocap_shadow_polisher.py --replay /app/rl-output/shadow_palm_polisher_rerun/final_commands.npz --output /app/rl-output/shadow_palm_independent_rerun

docker compose run --rm test
```

The durable [`paper/results/shadow_palm_polisher_2026_10_25/`](../paper/results/shadow_palm_polisher_2026_10_25/) directory contains the fixed candidate/provenance manifest, every IK feasibility entry and physics rollout, selected named and time-stamped commands, baseline and selected trajectories, sensitivity grid, source scene, and independent replay metrics. The selected export passes the existing replay's source-hash, command-shape, names, timestamp, joint-limit, and velocity checks. The parent review also ran the saved command through the original `hocap_shadow_replay.py`; object positions, rotations, actual joint tracks, contacts, and shared metrics matched bitwise, with identical compiled physics arrays. Independent frame and hold probes are saved alongside the results. The raw candidate command files are under ignored `data/experiments/hocap-rl/shadow_palm_polisher_20261025/`.

Validation: the full Docker suite passed **538 tests, 10 skipped**; the simulator experiment suite passed **80 tests**. No production retarget, replay, or export stage changed.

The independent [review verdict](../paper/results/shadow_palm_polisher_2026_10_25/parent_review.json) reproduces the implementation and concludes that this palm-only hypothesis is insufficient. The result does not rule out a coordinated palm-and-finger grasp search; that would need its own frozen protocol and static hold gate before transport optimization. `SHA256SUMS` in the result directory guards the saved command and evidence files.

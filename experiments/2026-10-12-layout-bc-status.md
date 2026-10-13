# Layout-conditioned imitation — 2026-10-12

## What was trained

The 46 successful ±20 mm and 27 successful ±50 mm MuJoCo-filtered plans from
one recovered human episode form **73** candidate demonstrations. A small
phase-conditioned network learns the six UR10 joint-command *corrections* to
the single source trajectory. Its inputs are a clock (normalized time and
eight Fourier harmonics) and the initial red/green cube XY offsets. The
offsets are read once from the simulator's privileged cube positions; they
are not estimated from camera images. The network outputs a time-dependent
6 × 4 matrix applied to the four offsets. This makes the correction exactly
zero at the original layout. Source gripper commands, 0.10 rad simulator-only
grip pressure, 3× command time and 50 Hz held-command control are unchanged.

This is **supervised imitation of object-relative trajectory corrections**,
not SAC/RL, a visual policy, or a general closed-loop manipulation skill.
The policy observes cube positions only at the beginning and does not react
to subsequent object movement; the reference time/trajectory remains known.

## Evaluation

The split is by *layout* and stratified by offset range (seed 42): 51 train,
8 validation, **14 held-out test** (nine ±20 mm, five ±50 mm). Every split
shares the **same source human motion and simulator**. No frames from a test
layout enter training or checkpoint selection. All layouts were preselected
for successful physical execution of their generated teacher plan, so these
rates are conditional on that acceptance filter.

| Layout split | Learned correction | Unchanged source commands |
|---|---:|---:|
| Train | 51/51 | 10/51 |
| Validation | 8/8 | 2/8 |
| Held-out test | **14/14** | **4/14** |
| Held-out ±20 mm only | 9/9 | 4/9 |
| Held-out ±50 mm only | 5/5 | 0/5 |

These counts use the same MuJoCo released-stable-stack/contact/tracking
predicate and the same target scenes for both controllers. A Wilson 95%
interval for 14/14 is approximately **[0.785, 1.0]**; the sample is too
small to claim reliable performance over the entire ±50 mm rectangle. The
learned policy's maximum airborne cube slip on the 14 test scenes is 3.81 mm.
**All 14 still contain robot contact with the red support.** Success does
not mean collision-free execution, robustness to different friction/pad
models, independent source motions, new cube orientations, or sim-to-real
transfer. The earlier plain 53-D state BC still has **0/17** closed-loop
successes; this new model succeeds because the fixed task phase and source
trajectory are explicitly available, not because that failure was disproven.

## Artifacts and reproduction

Checkpoint, deterministic split, training report, and per-layout physics
results are in `data/experiments/bc-v1/layout_bc_v1/`. No source recordings,
object models or canonical retarget plans were altered.

```bash
docker compose -f docker-compose.experiments.yml run --rm sim experiments/layout_bc.py train
docker compose -f docker-compose.experiments.yml run --rm sim experiments/layout_bc.py evaluate
experiments/run_viewer.sh --learned
```

The GUI playlist shows **held-out policy rollouts**, not teacher-plan replays.
Next, test on an independent layout seed, replace privileged cube positions
with perception estimates, and address red-support contact before stronger
claims. A source-disjoint split needs additional genuinely different human
demonstrations; these 73 variants cannot provide one.

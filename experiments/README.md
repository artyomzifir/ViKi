# ViKi experiments

The first experiment is physical replay of the UR10 / Robotiq 2F-85 cube-transfer trajectory in MuJoCo. Policy training starts only after replay can pick, carry, release and leave the blue cube supported by the green cube.

See [PLAN.md](PLAN.md) for gates and comparison design, and [2026-10-09-status.md](2026-10-09-status.md) for measured progress and the current physics blocker.

## Recovered code

- `scene.py`, `assemble_urdf.py`, `replay.py`, `batch_replay.py`, and
  `viewer.py` build and execute the UR10/Robotiq cube-transfer scene.
- `augment_cube.py`, `quality.py`, `quality_search.py`, and `quality_batch.py`
  generate and screen layout/placement variants.
- `observations.py`, `collect_bc.py`, `bc.py`, and `layout_bc.py` contain the
  low-dimensional imitation-learning baseline. The viewer scripts display
  replayed, augmented and learned rollouts.
- `success.py` distinguishes a released, stable stack from a mere geometric
  plan. `gripper_control.py` contains the experimental grasp-pressure bias.

The original `data/experiments/` results, local source episodes, MuJoCo model
assets and trained checkpoints were lost with the local workspace. The code and
historical reports were reconstructed from task logs; they do **not** provide
a runnable reproduction until models and recordings are restored and the
full evaluation is repeated. Use `docker compose -f docker-compose.experiments.yml`
for the isolated simulator image; see `PLAN.md` for evaluation gates.

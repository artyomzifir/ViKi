# Object-relative cube-layout sweep — 2026-10-12

## Method

The scratch plan for episode `2026-10-06_11-01-57` is the only motion source.
The red support and blue cube start together; the green support moves
independently. We sampled 48 four-dimensional layouts (red XY and green XY),
each coordinate uniformly within ±20 mm of its reconstructed source position
(seed 2026). The table, robot base, cube sizes and source grasp geometry stay
unchanged. This is a *local* layout study, not a new human dataset.

For each layout, the approach ramps from the original start to the red XY
shift. Pickup follows red; during carry, a linear blend transfers the wrist
translation from red to green; release and retreat follow green. Wrist
orientation and finger commands remain from the source plan. A full-path
MuJoCo Jacobian IK solves the six UR10 joints; candidates missing any wrist
target by more than 3 mm or 3° are rejected. The 0.10 rad simulator-only grip
pressure setting from the previous baseline is unchanged. No grasp-position
offset or cube teleportation is used. A 50 Hz held-command MuJoCo replay then
checks tracking, bilateral pad contact, airborne slip, cube support at the end
and gripper clearance after release.

## Measured result

- **46/48** candidates passed the existing released-stack physics predicate.
  One missed its IK target; one IK solution jumped to a different joint branch
  and failed physical tracking/grasp. The latter remains in the manifest as a
  rejected candidate, not a usable demonstration.
- Among accepted candidates, sampled red offsets span **−19.5 to +18.5 mm X**
  and **−17.3 to +19.6 mm Y**; green spans **−19.9 to +18.7 mm X** and
  **−19.5 to +19.2 mm Y**. Peak measured airborne blue-cube slip was **0.71 mm**
  in this simulator under its current contact model.
- The unchanged source joint commands succeeded in only **9/25** independently
  sampled ±20 mm layouts in the earlier matched-rate diagnostic. That is a
  *different random sample*, not a paired controlled estimate of improvement.
- **All 46 accepted trajectories still touch the red support with the robot**
  during the replay (784–915 logged contact-pair steps per trajectory). The
  current predicate does not reject this; therefore “physics-validated” means
  *stacked under the present MuJoCo settings*, not collision-free or robust to
  revised contact geometry. No claim of real-world success follows.

The full decision manifest and candidate plans are under
`data/experiments/bc-v1/augmented/`. These scratch artifacts do not overwrite
recordings, canonical retarget plans or calibration files.

## Inspect and reproduce

```bash
docker compose -f docker-compose.experiments.yml run --rm sim experiments/augment_cube.py --count 48 --offset-mm 20 --seed 2026
experiments/run_viewer.sh --augmented --max-scenes 12 --gap-s 1
```

The native MuJoCo playlist picks 12 spread-out successful layouts, labels the
red/green XY offsets in millimetres and uses the same source scene, shifted
cubes, generated joint plan, 3× command timing, 0.10 rad grip-pressure setting
and held 50 Hz commands as the physics check. This demonstrates local motion
transfer in one simulator. It does **not** test a learned policy, unseen task
instances outside ±20 mm, object rotation, changes in robot placement, visual
perception at execution time or sim-to-real transfer. The next learning test
must split by layout and source trajectory, and report the red-support
contacts rather than hiding them behind aggregate task success.

# viki.replay — hardware validation  · [stub stage]

**Stage 5** · `plan.h5` → `replay.h5` · paper §3.8

Execute the synthesised joint trajectory on the manipulator, log what the robot
actually attained, and screen for feasibility. Exporting the *attained* states
(not the planned ones) is what gives the dataset honest observation–action pairs.

## Files

| file | status | what |
|---|---|---|
| `driver.py` | `DryRunDriver` **working**, `UR3Driver` stub | `RobotDriver` ABC + `execute(q_traj, gripper, dt) -> ProprioLog`. Dry-run echoes the plan with a NaN residual; `UR3Driver` raises (needs `ur-rtde` + the robot). `load_driver(name)`. |
| `screen.py` | joint-limit check **working**, rest stub | `screen(q_attained, residual, robot) -> Verdict`. Joint limits from the URDF are real; singularity / collision / tracking-fault classification are not implemented → verdict falls through to `pass` / `dry-run`. |
| `run.py` | — | `replay_episode(ep, driver="dryrun", max_resolves=0)` → `replay.h5`, marks status. The re-solve loop (Fig. 3.1 feedback edge) is accepted but not implemented. |

## Contract

- **out:** `replay.h5` — keys in `contracts.REPLAY_KEYS`:
  `q_attained (T,nq)`, `gripper_attained (T,)`, `controller_residual (T,)`
  *(NaN under dry-run)*, `verdict` (`pass`｜`reject`｜`dry-run`),
  `rejection_cause`, `resolve_attempts`, `robot`, `dt`.

## To make it real

Add `ur-rtde`, implement `UR3Driver.execute` (stream `q` to the controller, read
back proprioception + tracking residual), and fill in the singularity /
collision / tracking-fault branches of `screen`.

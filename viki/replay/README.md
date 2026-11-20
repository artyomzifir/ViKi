# viki.replay — optional offline dry-run

`viki replay <episode>` copies `plan.h5` into the legacy `replay.h5` format
and checks numeric validity and available URDF joint limits. It is an explicit
compatibility tool; `viki run` ends after retarget and optional object-relative
tracks. The web interface has no Replay tab or `/api/replay` route.

Only the `dryrun` driver remains. A valid result is `dry-run`, never a task
success verdict. MuJoCo execution lives separately in `experiments/`.

| File | Role |
|---|---|
| `driver.py` | Echo planned joint and normalised gripper values |
| `screen.py` | Reject non-finite/empty inputs and joint-limit violations |
| `run.py` | Check approach + demonstration, archive demonstration frames |

The archive retains `contracts.REPLAY_KEYS` for compatibility. The historical
`q_attained` and `gripper_attained` names contain **planned copies**;
`controller_residual` is NaN and `resolve_attempts` is zero. No automatic
re-solve is supported. The optional partial LeRobot writer still consumes this
offline check; the default trajectory exporter needs only a completed plan.

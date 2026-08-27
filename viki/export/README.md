# viki.export — LeRobot dataset  · [stub stage]

**Stage 6** · labelled + screened episodes → `datasets/<name>/` · paper §3.9

Aggregate eligible episodes into one LeRobot dataset for policy training.
Delegates to the **optional** `lerobot` package (`pip install 'viki[export]'`,
which pulls in torch) — `export.run` raises a clear install error when it is
absent.

## Eligibility

An episode is exported when `retarget` and `replay` have run, its replay verdict
is not `reject`, its label `outcome` is not `bad`, and it has a non-empty
`task` string (`labeling.validate_labels(..., for_export=True)`).

## Files

| file | what |
|---|---|
| `run.py` | `export_dataset(episode_ids, out_dir, fps)` — filter, build per-frame dicts, drive the writer |
| `lerobot.py` | `LeRobotWriter` — `LeRobotDataset.create` / `add_frame(frame, task=...)` / `save_episode` / `finalize` |

## Frame schema (per timestep)

| key | source |
|---|---|
| `observation.images.<cam>` | `raw/<cam>.mp4`, decimated to `fps` |
| `observation.state` | `replay.h5:q_attained` + `gripper_attained` (falls back to `plan.h5`) |
| `action` | next-step `observation.state` |
| `task` | `EpisodeLabels.task` (per-segment when phase segments are set → frame-level task) |
| `next.success` | `outcome == "good" and verdict in {pass, dry-run}` |
| `annotation.wrist_pose` `.object_relative` `.confidence` `.replay_residual` `.phase` | `cln.npz` + `replay.h5` + labels |

## Stubbed

`annotation.object_relative` is a shape-correct placeholder until
`prepare.represent` produces a real track; `info.json` is stamped
`viki_export_status="partial"`.

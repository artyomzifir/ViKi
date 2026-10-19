# ViKi experiments

The first experiment is physical replay of the UR10 / Robotiq 2F-85 cube-transfer trajectory in MuJoCo. Policy training starts only after replay can pick, carry, release and leave the blue cube supported by the green cube.

See [PLAN.md](PLAN.md) for gates and comparison design, and [2026-10-09-status.md](2026-10-09-status.md) for measured progress and the current physics blocker.

`hocap_full_viewer.py` plays a complete saved HO-Cap hand-only plan with all
tracked objects as free box proxies in the native MuJoCo GUI. It is a visual
physics smoke, not a validated grasp benchmark; see
[`docs/2026-10-15-hocap-full-scene-gui-smoke.md`](../docs/2026-10-15-hocap-full-scene-gui-smoke.md).

For object-aware G10_4 diagnostics, `hocap_full_viewer.py` also accepts
`--grasp-dwell-s 1 --rubber-pads --min-pad-force-n 1 --fully-close`. The
simulator-only full-close mode is deliberately opt-in and may shove an object
out of the jaws; see [`docs/2026-10-15-hocap-grasp-timing-gui.md`](../docs/2026-10-15-hocap-grasp-timing-gui.md).

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

## HO-Cap RL diagnostic

`hocap_rl_catalog.py` creates the deterministic object-disjoint 14/14 scene
split in `hocap_rl_split.json`. `hocap_grasp_env.py` and
`train_hocap_grasp_sac.py` are a single-scene SAC pilot. The default
`pickup` reward retains the original lift-only baseline; optional
`--reward-mode object_track` follows a time-bounded corridor around the
perception-derived object trajectory and requires released placement. It needs
at least 9.52 s of simulated horizon for the G10_4 frame-44 crop, so the mode
defaults to 250 steps rather than 125. The current policy fails in both modes;
it is not a trained 28-scene object-generalizing model.
`view_hocap_grasp_sac.py` refuses GUI playback if the saved plan, track, mesh
or GT hash differs from the training report (older reports lack this manifest).
See [`docs/2026-10-15-hocap-rl-pilot.md`](../docs/2026-10-15-hocap-rl-pilot.md)
for the original raw-result locations, measured failures and next gates.
The object-trajectory pilot and its numerical smoke are documented in
[`docs/2026-10-15-hocap-object-track-reward.md`](../docs/2026-10-15-hocap-object-track-reward.md).

## Contact-assisted grasp diagnostic

`hocap_object_guided_replay.py --contact-attach` enables a simulation-only,
contact-gated weld for one tracked object. It requires sustained bilateral rubber-pad
force and actual finger opening to release; the default remains physical-contact-only.
Assisted placement is reported separately from physical grasp success. See
[`docs/2026-10-15-contact-assisted-grasp.md`](../docs/2026-10-15-contact-assisted-grasp.md)
for the exact gate, reproduction, raw results, video, and limitations.

The stricter four-object full-task diagnostic uses `--contact-attach --full-scene`
with `--release-opening-rad 0.03`; `--full-scene-grid` tests the 27 recorded
local grasp candidates. It skips 12 source preamble frames that penetrate the
inferred table and projects approximate neighbouring proxies onto it. The
collision audit, raw results and video are documented in
[`docs/2026-10-15-hocap-contact-assisted-full-scene.md`](../docs/2026-10-15-hocap-contact-assisted-full-scene.md).

`--full-scene-meshes` instead loads all four official HO-Cap meshes from
`data/_hocap_assets/models.tar` and lowers the held target to the table before
opening. On the same scene, the target-only placement gate passes 3/27 local
candidates, but the strict four-object gate passes 0/27 because a neighbouring
mesh drifts even in a stationary-robot control. See
[`docs/2026-10-15-hocap-four-mesh-scene.md`](../docs/2026-10-15-hocap-four-mesh-scene.md).

Use `--descriptive` when the task is to inspect one four-mesh run without an
acceptance threshold. It reports all measured values and leaves proxy success
fields unset; the reproducible run and video are documented in
[`docs/2026-10-15-hocap-four-mesh-descriptive-run.md`](../docs/2026-10-15-hocap-four-mesh-descriptive-run.md).

For new runs, the raw replay requires the current HO-Cap V4 hand-fit clean artifact
and a plan whose effective `source_pose` is `hand_fit`. The old pinch-center
plan mixed fitted orientation with landmark-derived position; that combination
is now rejected for HO-Cap, and new runs use the fitted wrist. See
[`docs/2026-10-19-hocap-replay-evaluation-audit.md`](../docs/2026-10-19-hocap-replay-evaluation-audit.md). Replaying the archived
landmark-only control requires explicit `--allow-legacy-plan`; that override is
recorded in the JSON result. The full-object batch defaults to strict bilateral
pad contact (`attach_on_any_pad=False`) and the same hand-fit provenance gates.

For the **actual full 831-frame episode**, `hocap_four_mesh_raw_replay.py` replays
all four free official meshes with the saved hand-only arm plan. It completes
all frames but grasps none of the four objects and strikes G10_1. The measured
failure, static-robot control and video are documented in
[`docs/2026-10-15-hocap-full-831-four-mesh-replay.md`](../docs/2026-10-15-hocap-full-831-four-mesh-replay.md).

`hocap_four_object_assisted_replay.py` instead stitches four object-conditioned
IK phases into one 831-frame trajectory and runs all four official meshes in
one MuJoCo scene with independent contact-gated pad welds. The measured run
completed but activated no weld and lifted no object. Its phase metrics,
frame-level diagnostics and video are documented in
[`docs/2026-10-15-hocap-four-object-conditioned-full-replay.md`](../docs/2026-10-15-hocap-four-object-conditioned-full-replay.md).

The opt-in `--attach-on-any-pad` follow-up attaches on the first positive contact
from either named pad and releases at the recorded release phase. It attached
3/4 objects but did not achieve four-object placement; the same report documents
the bystander collision that prevented the fourth pickup.

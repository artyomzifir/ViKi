# Existing cube recordings: numerical audit (2026-10-12)

## Scope and counting rules

Read-only audit of the 78 episodes listed in
`data/experiments/2026-10-09-full-audit.json`. The recording files and
canonical artifacts were not modified. Counts below refer to **episodes**
unless explicitly labelled frame groups or joint slots. Recovered object
models and retarget plans in `data/experiments/` are scratch results, not
promotions of the canonical episode files. All geometry and simulation scores
are self-consistency or model-relative measurements, not independent ground
truth or real-robot validation.

## Measurements available now

| Stage | Observation | Interpretation |
|---|---|---|
| Capture timing | 11,688 paired frame groups in 78 episodes have both Kinect device timestamps. Absolute pair-gap median 155 µs, p95 167 µs, maximum 178 µs; 0 pairs exceed 1 ms. | Hardware-timestamp pairing is resolved **for these recordings**. The host-arrival `sync_bounded` flag measures a different quantity. |
| Extract | `raw/joints3d_summary.json` reports 95.0% of joint slots solved across 78 episodes, weighted by slot count; episode fractions p10/median/p90 = 88.9% / 97.6% / 100%. | Solver coverage, not 3-D hand accuracy. Its tiny in-fit reprojection residual is not an independent accuracy measure. |
| Prepare | 73/78 pass the articulated structural gate; five fail. Among the 73 `cln.npz` artifacts, `geometry_support_mask` covers 95.2% of frames weighted by episode length; episode p10/median/p90 = 87.0% / 97.7% / 100%. Longest unsupported run per episode: median 3 frames, p95 21.2, maximum 53. | Explicitly distinguishes observed hand poses from filled poses. No independent hand-pose reference is present. |
| Object model | Original quality gate: 44/73 accepted. Independent track-recovery scratch rebuilds recover 26 of 29 rejected episodes: 70/73 accepted, three still rejected. | The 70 includes 26 scratch models; it is not the count of accepted canonical artifacts. |
| Blue object geometry | Across those 70 accepted models, the median per-axis 2nd–98th percentile extent of canonical object-frame surfels is 46.2 × 47.7 × 49.7 mm. In 35/70, at least one axis exceeds 50 mm. The physical cube edge is nominally 40 mm. | Strong model-size bias diagnostic, but a partial surfel shell is not a measured solid-cube bounding box. Cube yaw is symmetry-ambiguous. |
| Retarget | 40 canonical `plan.h5` files currently exist (38 old-mode, two later v3). Canonical position RMSE median 6.2 mm and p90 7.7 mm; one file has 122.0 mm RMSE and must not be counted as a quality-approved trajectory. Of 26 disjoint scratch recovered scenes, 22 pass the geometric plan gate, two fail retarget and two fail plan quality. | The canonical and scratch counts use different modes/gates; 40 + 22 is an artifact inventory, **not** one homogeneous success rate. |
| Physical replay | Among 24 scratch plans replayed with the provisional MuJoCo controller/contact model, 18 pass released-stack, grasp-integrity and contact checks. Of the 22 geometrically accepted scratch plans, 17 also pass physics. Robot–red-support contact occurs in 24/24, robot–green-support contact in 2/24. | A selected, tuned simulator subset; neither collision-free performance nor the full-batch physical yield is established. |

## How the figures were obtained

- Pair gaps: absolute difference of `kinect_0` and `kinect_1` entries in each
  episode's `raw/timestamps.json[frame].device_timestamps_us`.
- Extract coverage: sum of `joints_solved` divided by sum of `joints_total`
  from each `raw/joints3d_summary.json`.
- Prepare coverage and unsupported runs: `cln.npz["geometry_support_mask"]`
  over the 73 available fitted episodes.
- Object acceptance: `track-recovery-comparison.json` after-state, with
  original-model evidence in `2026-10-09-full-audit.json`. For each accepted
  scene, use the original `object_models.npz` if it was already accepted and
  `data/experiments/track-recovery-full/<episode>.npz` otherwise. Blue extent
  is the 98th minus 2nd percentile per axis of `model_xyz_object` between
  offsets for the `manipulated_object` label.
- Canonical plan count and RMSE: actual `plan.h5` files and matching
  `status.json["stages"]["retarget"]["position_rmse_mm"]`. Scratch plan statuses
  are the **last** event per episode in
  `data/experiments/recovered-retarget/status.jsonl`, not all retry events.
- Physics: `data/experiments/recovered-retarget-physics.json`, specifically
  `released_stack_success` and the red/green support-contact counts.

## Not closed by these recordings

An independent absolute accuracy claim for the hand or current large-board
calibration needs held-out, physically referenced observations at tabletop
and carry heights. A symmetric, partially visible cube does not establish
full 6-D orientation accuracy; a marked control object would. The current
fixed-layout human motions cannot establish source-motion or broad layout
generalization. The 40 canonical plans have not all been replayed under a
single current, quality-gated planner/physics protocol, and the red support
collision must be resolved before calling the trajectories safe. Pouring has
no corresponding recording set yet.

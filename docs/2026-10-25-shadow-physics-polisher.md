# Shadow physics trajectory polish: bounded negative pilot (2026-10-25)

This experiment tested whether a small open-loop change to the saved Shadow Hand command can retain object 2 during the G10_2 carry. It did not. The selected diagnostic candidate reduces the 95th-percentile carry position error from **104.569 to 96.624 mm**, but opposing contact lasts no longer, the final position error worsens from **25.201 to 36.535 mm**, and both target-task and four-object gates fail. Four separately declared carry slowdowns also fail. No polished command is accepted for the production pipeline.

## Frozen scene and scope

The input is the final 190-frame `g10_object2_190/shadow_object_plan.npz` with saved retimed timestamps and the HO-Cap oracle object track. This is one scene and one hand; no success rate or generalization claim follows. The replay uses the same UR10/Shadow URDF, four official object meshes, 0.08 kg object masses, robot/object friction 1.2, table friction 1.0, gravity compensation, implicitfast at 0.0005 s, and URDF effort-limited position actuators (arm 1800/120, hand 16/0.15). There is no weld, object pose forcing, collision deletion, gain change, friction boost, or online object feedback. The arm commands, all 190 frame indices, and release are preserved.

The source scene XML and float32 command interpolation are reproduced exactly. Zero residual gives the archived baseline's position mean 27.5206247 mm, final error 25.2007614 mm, maximum lift 243.4242156 mm, carry error p95 104.5688284 mm, carry opposing-contact frame fraction 0.690476, and zero robot-table contact steps. Preserving float32 arithmetic matters: an initial float64 promotion changed the chaotic post-drop outcome despite identical XML, and was corrected before search. `manifest.json` records hashes of the plan, track, URDF, and all four meshes.

## Search and acceptance

The initial hand-only search was frozen to four parameters: common four-finger flexion [-0.08, 0.22] rad, extra little-finger flexion [-0.10, 0.18], thumb flexion [-0.08, 0.15], and thumb opposition [-0.15, 0.15]. A C1 smooth envelope rises over frames 43–58, holds, and falls over 110–125, entirely inside the constant grasp command. The 32 deterministic Latin-hypercube candidates plus zero residual were validated against every URDF joint position and per-segment velocity limit before rollout; all 33 passed. They were scored on carry p95 tracking, same-physics-step simultaneous positive thumb and finger forces, a vertical height-deficit fraction, final position error, and robot-table contact. The height-deficit fraction includes initial lift lag; it is **not** by itself a drop counter. The source plan was never rewritten.

During review, the acceptance diagnostic was tightened **after** the first 32-candidate search: terminal placement now requires positive table support at every physics step for the last 15 frames, no hand contact in those frames, and target speed below 0.03 m/s at every step. Target-task success also requires >20 mm lift, <30 mm carry p95 and final error, no >50 mm vertical reference deficit in the carry window, and no robot-table contact. The separate four-object gate adds no robot-neighbor contact and <5 mm peak neighbor displacement. The ranking formula and candidate set were unchanged; the final ledger is a deterministic rerun under the reviewed diagnostics. This gate correction cannot convert any candidate into a pass, because none meets the carry p95 criterion.

The selected hand-search candidate is index 26, with parameter vector **[-0.038335, -0.032077, +0.054891, +0.021721] rad**. It slightly opens the four fingers and little finger while flexing/opposing the thumb. Its minimum joint-position margin is 9.0e-8 rad and velocity margin 0.03922 rad/s. Those margins show numerical proximity to a source joint limit; the file is rejected if it crosses the limit, not silently clamped. Search wall time was 82.2 s in the simulator container. The full ledger records each feasible candidate, score, and metrics.

| Nominal physics | Original | Selected hand candidate |
|---|---:|---:|
| Carry object error p95 (mm) | 104.569 | 96.624 |
| Final object error (mm) | 25.201 | 36.535 |
| Simultaneous positive-force opposing contact, carry physics steps | 0.6780 | 0.6793 |
| Opposing-contact frames, carry | 0.6905 | 0.6905 |
| Longest carry opposing-contact gap (s) | 0.9040 | 0.9015 |
| Carry vertical height-deficit step fraction | 0.2223 | 0.2306 |
| Peak neighbor displacement (mm) | 17.262 | 17.262 |
| Target-task / four-object success | false / false | false / false |

The original hand command is constant throughout frames 42–126. An independent force trace shows both thumb and finger normal force declining from about 2 N around frame 90 to near zero at frame 99, followed by loss of contact and object return to the table before intended release. The selected candidate does not prevent this. It has no robot-table or robot-neighbor collision steps and no actuator effort saturation, but its object contact penetration reaches about 12.93 mm in the convex-hull proxy. A stationary-robot control in this fixture showed object 3 drifting about 17.19 mm after settle, so the four-object <5 mm gate is confounded by the scene representation itself; target-task failure is independent of that issue. Millimetre values here are simulator-to-recorded-object-track errors, not measured physical accuracy.

## Separate timing follow-up

After the hand-only failure, four fixed follow-up rollouts slowed **only the intervals ending at carry frames 42–125**, preserving all 190 command poses, release, and per-frame object reference. Baseline and index 26 were each run at 1.5× and 2× interval duration. This changes the time correspondence to the recorded motion and is a diagnostic, not a claimed tracking improvement. All four worsen the frozen ranking score and fail the target task gate:

| Source / carry factor | Total duration (s) | Carry error p95 (mm) | Simultaneous carry contact | Final error (mm) |
|---|---:|---:|---:|---:|
| Original / 1.5× | 8.004 | 298.824 | 0.438 | 121.360 |
| Original / 2× | 9.404 | 329.936 | 0.334 | 113.082 |
| Index 26 / 1.5× | 8.004 | 250.914 | 0.418 | 150.252 |
| Index 26 / 2× | 9.404 | 329.575 | 0.299 | 98.319 |

The nominal index-26 command remains the selected **diagnostic** file. `accepted=false` in its manifest. Its saved replay reproduces all 35 evaluator metrics exactly. The exported `final_commands.npz` includes 190 timestamps, 28 named position commands, a timing factor, and source hashes; replay validates these and the URDF limits before stepping physics.

## Held-out physical sensitivity

A fixed 2×2 grid (masses 0.06/0.10 kg, robot/object friction 0.9/1.2) was withheld from nominal selection. Both the original and selected diagnostic command fail the target-task gate in all four settings. Carry p95 ranges from 97.8 to 215.5 mm for the original and 97.8 to 185.6 mm for the selected command. This is sensitivity of one simulated scene, not cross-scene validation. Full values are in `sensitivity.json`.

## Reproduce and inspect

From the repository root, run:

```bash
docker compose -f docker-compose.experiments.yml run --rm sim experiments/hocap_shadow_polisher.py --candidates 32 --output /app/rl-output/shadow_polisher_final_20261025

docker compose -f docker-compose.experiments.yml run --rm sim experiments/hocap_shadow_polisher.py --replay /app/rl-output/shadow_polisher_final_20261025/final_commands.npz --output /app/rl-output/shadow_polisher_independent_20261025

docker compose run --rm test
```

The tracked, portable result files are under [`paper/results/shadow_polisher_2026_10_25/`](../paper/results/shadow_polisher_2026_10_25/): manifest, ledger, source scene XML, all five exported command variants, separate timing reports, sensitivity grid, per-frame trajectory diagnostics, and independently replayed report. The simulator's writable raw outputs are under ignored `data/experiments/hocap-rl/`. No production retarget, replay, or export stage was changed.

Validation on this checkout: `docker compose run --rm test` passed **533 tests, 10 skipped**. The experiment tests passed **75/75** inside the simulator image with read-only `tests/` and `scripts/` mounts. Independent replay of `final_commands.npz` reproduced every reported final metric exactly. The source-hash, joint-name, timestamp, shape, and float32 checks are exercised by focused regression tests; no provenance failure is silently accepted.

The first exploratory ledger remains in ignored `data/experiments/hocap-rl/shadow_polisher_20261025/`, while the reviewed, deterministic rerun and its tracked ledger are in the result directory above. Only the latter has the corrected terminal and four-object acceptance diagnostics; both runs selected index 26 under the unchanged score.

## Independent parent review

After Sol completed implementation, the parent agent replayed `final_commands.npz` separately. All **35** final evaluator metrics matched exactly. The same commands were also converted into a diagnostic input for the unchanged original `hocap_shadow_replay.py`: object positions, object rotations, actual robot joint arrays and contact flags were **bitwise identical** between both runners. This verifies execution independently of the new evaluator.

The parent compared compiled physical model arrays (masses, inertias, gravity compensation, friction, collision filters and geometry, joint limits, actuator gains/bias/effort limits, exclusions and equality count) against the original audit scene: all checked arrays matched. Arm commands, approach through frame 43, release from frame 125 and all 190 timestamps were unchanged for the selected candidate. An intentionally altered object-track hash was rejected before simulation and produced no replay result. Details and code hashes are in `parent_review.json`, with both parent replay reports beside it.

Final validation reported by the implementation agent: **533 passed, 10 skipped** in the normal full Docker suite and **75 passed** in the simulator experiment suite. Review changes corrected placement-gate semantics, provenance checks and metric labels; no successful repair was obtained. The prototype is accepted as an opt-in experimental diagnostic, **not** as a validated fix for dropped objects.

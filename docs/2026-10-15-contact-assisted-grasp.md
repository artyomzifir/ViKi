# Contact-assisted HO-Cap grasp, 2026-10-15

## Scope and mechanism

This is an **opt-in simulator experiment**, not a change to production ViKi retargeting, a physical force-closure grasp, or an RL result. MuJoCo starts with an inactive wrist-to-object weld. The weld activates only after both named rubber-pad geoms have normal contact force at least 0.2 N for 0.08 s, the object's body origin projects between the pad centers, the closing command is at least 0.25 rad, and no non-pad robot-object contact above 0.2 N is detected. The current relative wrist/object pose is written into the weld before activation, avoiding a positional snap. It releases only once both commanded and measured finger positions have opened by at least 0.12 rad from their attachment values. A released weld cannot re-activate until reset. The attachment helper takes body and weld names and can be reused for other free objects, but this experiment instantiates it for one tracked object.

The two contact forces are an **entry condition**, not proof that the gripper could hold the object's weight during transport. During carry the weld supplies an unmeasured artificial constraint; success here must not be counted as a physical grasp or real-world transfer. Robot/object collisions still terminate the trial. The tabletop, mesh alignment, reference trajectory, and placement tolerance are simulator assumptions.

## Reproduction

The input manifest, including hashes and all source paths, is `data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json`. The scene is subject 1, G10_4, HO-Cap; the official object mesh is aligned using its GT pose. The controller follows the perception-derived object track after a contact search, with a 25 Hz control loop and 0.5 ms MuJoCo physics step. One fixed seed (1004) and the existing 3×3×3 grid of XY offsets and closing commands were used; no training or held-out evaluation was performed.

```bash
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/contact_attach_smoke/batch_report.json \
  --contact-attach
```

For the rendered successful candidate, add `--video /app/rl-output/contact_attach_smoke/success.mp4 --video-offset-x 0.005 --video-offset-y 0 --video-close-rad 0.55` and use a different `--output` path. The video is rendered from MuJoCo in two views, not camera footage. Without `--contact-attach`, the same controller retains its physical-contact-only baseline and never activates the weld.

## Observations

Raw per-trial results: `data/experiments/hocap-rl/contact_attach_smoke/batch_report.json`; the same 27 candidates without the weld are in `physical_control_report.json` beside it and passed the physical transfer proxy **0/27**. The rendered trial's report and video are `success_report.json` and `success.mp4` in that same directory (local, gitignored). The assisted placement proxy passed **3/27** candidates; physical transfer proxy is deliberately **0/27** in this mode. All three assisted passes had bilateral force at activation, explicit release, final table support, object speed below 0.001 m/s, and no detected non-pad robot-object contact. Final position errors against the track-relative goal were 12.4, 14.9, and 19.5 mm; final orientation errors were 15.4°, 14.6°, and 18.3°. In the rendered candidate, attachment occurred at 0.76 s with 0.70/0.55 N pad forces, and release at 18.40 s; the weld activation moved the object by 0.011 mm in `mj_forward` (not an end-to-end pose accuracy measurement).

The remaining candidates frequently failed because a proximal phalanx hit the object or because bilateral contact never stabilised. Consequently, even this deliberately assisted mode is **not** robust over the small grid, and it does not establish generalisation to other objects, scenes, robots, or the physical world.

## Full-task follow-up

A stricter four-object task sequence now passes 3/27 contact-assisted candidates after tabletop projection and a collision-free recorded approach. It starts at frame 12 because earlier retargeted poses penetrate the inferred table. The full protocol, paired no-weld control, raw results and rendered video are in [`2026-10-15-hocap-contact-assisted-full-scene.md`](2026-10-15-hocap-contact-assisted-full-scene.md). These assisted passes do not change the physical-grasp baseline above.

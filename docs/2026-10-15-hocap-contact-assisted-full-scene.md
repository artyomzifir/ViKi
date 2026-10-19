# HO-Cap full-task contact-assisted replay, 2026-10-15

## Question and protocol

Can one HO-Cap object-transfer task run from approach through grasp, carry, release and withdrawal with **all four tracked objects present**, without robot/table or robot/other-object collisions, if a MuJoCo weld is allowed only after measured bilateral pad contact? This is a simulator geometry check, not a physical-grasp result.

Source: `data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json` (SHA-256 paths and inputs are in that manifest), subject 1, G10_4, 180-frame crop, seed 1004. The target uses the aligned official mesh's convex collision hull; objects 1–3 are fitted box proxies. A free target and three free neighbours are simulated on an inferred tabletop. The neighbours are explicitly projected down to the tabletop before a 2 s passive settle; their vertical projection is **−23.35, −10.78 and −2.11 mm** respectively, followed by 0.20, 3.80 and 3.18 mm of passive settling. These are *model corrections*, not camera-pose accuracy measurements.

Recorded arm poses at frames 0–10 intersect the table with the right pad (up to about 8.4 mm at frame 0); frame 11 has only 0.76 mm of clearance, below the 5 mm gate. The replay therefore initializes at the first pose with at least 5 mm clearance from **both** pads: frame **12** (minimum 8.75 mm). It follows the recorded arm path from frames 12–44 at 0.08 s per frame, then uses the existing oracle-object-guided controller to complete pickup, trajectory-guided carry, release and withdrawal. Thus this is a complete **task sequence**, but **not an unmodified replay of all 180 arm frames**. The controller uses a tracked object trajectory and selected grasp offset; it does not perceive from rendered video.

The weld gate is the one in `experiments/contact_attachment.py`: both pad normal forces ≥0.2 N for 0.08 s, object centre between the pads, closing command ≥0.25 rad, and no non-pad robot/object force ≥0.2 N. In this experiment the weld releases after both commanded and measured opening reach 0.03 rad from their attached values. The 0.03 rad threshold was selected on this scene; it is not a general gripper calibration. The geometry/collision gate inspects every 0.5 ms physics substep for contact forces ≥0.2 N against the table or the other objects. A trial also requires final table support, released pads, target position error ≤50 mm, orientation error ≤30°, speed ≤0.03 m/s, and neighbouring displacement ≤5 mm after settling.

## Reproduction and raw results

```bash
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/contact_attach_smoke/full_scene_grid_final.json \
  --contact-attach --full-scene --full-scene-grid --release-opening-rad 0.03

docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/contact_attach_smoke/final_full_scene_report.json \
  --video /app/rl-output/contact_attach_smoke/final_full_scene.mp4 \
  --contact-attach --full-scene --release-opening-rad 0.03
```

The first command tests the pre-existing 3×3×3 offset/closing grid. Raw per-trial JSON and the two-view MuJoCo video are local under `data/experiments/hocap-rl/contact_attach_smoke/` (gitignored). The exact same winning candidate without `--contact-attach` is in `full_scene_physical_control.json`.

| Result | Measured value |
|---|---:|
| Assisted placement proxy | **3/27** candidates |
| Ended in align due to non-pad robot/object force | 18/27 |
| No bilateral preload after alignment | 3/27 |
| Completed scripted task without phase abort | 6/27 |
| Winning candidate (XY offset, close command) | (+5 mm, 0 mm), 0.50 rad |
| Forces at weld activation, left/right | 0.379 / 0.316 N |
| Weld activation / release | 5.32 / 22.80 s |
| Peak lift | 246.6 mm |
| Final position / orientation error | 15.26 mm / 12.92° |
| Final object speed | 0.00104 m/s |
| Robot–table or robot–neighbour force ≥0.2 N, winning run | none detected |
| Maximum neighbour displacement after settle, winning run | <0.00001 mm |
| Same candidate without weld | no stable lift; peak lift 0.43 mm |

The winning object is released, upright in the rendered video, and supported by the table after withdrawal. In the paired no-weld run it cannot lift the object. The other 24 assisted candidates do **not** constitute success; most are rejected before or during grip setup. The task is consequently still sensitive to a few-millimetre grasp offset and is not a robust pipeline.

## Interpretation and limitations

- The weld supplies artificial holding force throughout carry. Report these numbers as **contact-assisted geometric replay**, never as physical grasp success or proof of a stable two-finger grasp.
- The target mesh alignment uses HO-Cap GT, the controller uses an oracle object track, and the three box proxies require centimetre-scale height correction. No independent geometric truth is established by this replay.
- Frames 0–11 are excluded because the retargeted right pad intersects the inferred tabletop or lacks the 5 mm clearance. Fixing the upstream retargeted approach remains future work.
- There is no object-aware collision planner, no evaluated generalisation to other HO-Cap scenes or objects, and no real-world safety claim. The ≥0.2 N collision threshold can miss weaker contacts.

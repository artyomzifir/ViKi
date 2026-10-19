# HO-Cap four-mesh descriptive rerun, 2026-10-15

This is an unscored rerun of one subject-1 HO-Cap G10 scene. **No displacement, pose-error, or orientation-error threshold is applied to label the result.** The output JSON sets both success-proxy fields to `null` and reports measurements directly. This changes reporting, not MuJoCo physics, controller travel limits, contact-attachment entry rules, or the underlying collision audit. No abort was triggered in this run.

```bash
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/contact_attach_smoke/full_mesh_scene_descriptive.json \
  --video /app/rl-output/contact_attach_smoke/full_mesh_scene_descriptive.mp4 \
  --video-offset-x 0.005 --video-offset-y -0.005 --video-close-rad 0.55 \
  --contact-attach --full-scene --full-scene-meshes \
  --release-opening-rad 0.03 --descriptive
```

All four object geoms compiled as official HO-Cap meshes, not boxes. MuJoCo collides their convex hulls. The input track has 180 frames; the retargeted recorded approach starts at frame 11 because frames 0–10 intersect the inferred table, proceeds to frame 44, and then the object-guided controller finishes grasp, carry, placement, release and withdrawal. Thus this is an end-to-end **task sequence**, not a literal replay of every recorded robot frame. G10_4 is the only intentionally manipulated object.

| Direct observation | Value |
|---|---:|
| Control steps / rendered duration | 495 / 21.12 s |
| G10_4 peak lift | 263.41 mm |
| Weld attached / released | 5.40 / 23.24 s MuJoCo time |
| Two pad forces at attachment | 0.620 / 0.464 N |
| G10_4 final world position (m) | (−0.313894, 0.062093, 0.113684) |
| Final displacement from track-relative target | 14.61 mm |
| Final orientation difference from tracked end pose | 20.17° |
| G10_4 final speed | 0.00156 m/s |
| G10_4 released / supported by table | yes / yes |
| G10_1, G10_2, G10_3 displacement after settle | 1.31, 2.34, 12.85 mm |
| Detected robot–neighbour and robot–table force contacts ≥0.2 N | none |
| Non-pad robot–target contact steps ≥0.2 N | 0 |

The target's bottom mesh-to-table gap was 28.66 mm before the controlled placement descent and −1.73 mm at its end; the latter is penetration according to the convex-hull distance query, not a real-world tolerance claim. G10_3 already moves 11.49 mm in a separate 20 s static-robot control after the same initial settle/approach. It is therefore incorrect to infer from the 12.85 mm displacement that the moving robot hit G10_3; the mesh/pose/table initialization itself drifts. Conversely, the force audit cannot exclude contacts below 0.2 N.

The carry uses a contact-activated weld, so this is **not** an unaided physical-grasp demonstration. The controller uses the recorded object trajectory, and meshes are aligned using HO-Cap GT; there is no visual closed loop or independent accuracy claim. Source hashes, mesh hashes, complete trial metrics and video path are in `data/experiments/hocap-rl/contact_attach_smoke/full_mesh_scene_descriptive.json` (local, gitignored).

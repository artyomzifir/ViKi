# HO-Cap four-mesh scene replay, 2026-10-15

## Scope

One 180-frame subject-1 G10 task was replayed in MuJoCo with four **official HO-Cap `cleaned_mesh_2000.obj` meshes**, not fitted box geoms. The target is G10_4; G10_1–G10_3 are free neighbouring objects. The target is the only object intentionally grasped, transported and released. The simulator compiled all four object collision geoms as MuJoCo meshes. MuJoCo uses their convex hulls for collision, so this does **not** mean triangle-accurate contact or measured object physics. Object masses (0.2 kg nominal, with the seeded target randomisation), friction, tabletop, and rubber-pad geoms are simulator assumptions.

Meshes were mapped into each tracked-object frame from the HO-Cap GT pose at frame 420. Nearest mesh-vertex distances from the reconstructed partial clouds were 4.90/9.25 mm (median/p95) for G10_1, 3.75/8.38 mm for G10_2, and 5.09/9.53 mm for G10_3. These are alignment checks, not independent accuracy measures. The inferred table top is z=0.000337 m in the scene world frame; each neighbour is vertically projected to the table and passively settled before replay.

The recorded approach poses in frames 0–10 intersect the inferred table or lack the 5 mm clearance gate. The run starts at frame 11, follows the recorded arm through frame 44, then uses the existing oracle-object-guided controller for grasp, carry, placement and withdrawal. It is therefore a complete **task sequence**, but **not** an unchanged replay of all 180 recorded arm poses. A wrist-to-target weld is activated only after bilateral rubber-pad force/contact criteria and removed on opening; it supplies artificial holding force. The controller additionally lowers the held mesh to the table before opening. None of this demonstrates an unaided physical grasp or an RL policy.

## Reproduction

```bash
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/contact_attach_smoke/full_mesh_scene_grid.json \
  --contact-attach --full-scene --full-scene-meshes --full-scene-grid \
  --release-opening-rad 0.03

docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_object_guided_replay.py \
  --source-report data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json \
  --output /app/rl-output/contact_attach_smoke/full_mesh_scene_best.json \
  --video /app/rl-output/contact_attach_smoke/full_mesh_scene_best.mp4 \
  --video-offset-x 0.005 --video-offset-y -0.005 --video-close-rad 0.55 \
  --contact-attach --full-scene --full-scene-meshes --release-opening-rad 0.03
```

The source manifest SHA-256 guards the plan, track, target mesh and GT poses. `data/_hocap_assets/models.tar` supplies the other three meshes; their individual SHA-256 values are stored in the local output JSON. Video and full per-candidate metrics are in `data/experiments/hocap-rl/contact_attach_smoke/` and are gitignored.

## Results and failure analysis

| Metric | Result |
|---|---:|
| Tested grasp candidates on this scene | 27 |
| Completed scripted phases | 6/27 |
| Target placed within 50 mm / 30° with release and table support | 3/27 |
| **Strict full-scene gate**, including ≤5 mm neighbour displacement | **0/27** |
| Best target position / orientation error | 14.61 mm / 20.17° |
| Best candidate peak lift | 0.263 m |
| Best candidate G10_3 displacement after settling | 12.85 mm |
| Robot–neighbour or robot–table contact force ≥0.2 N in best run | none detected |

For the best candidate (+5 mm X, −5 mm Y, 0.55 rad closure), the target is released on the table and meets the target-only pose gate. **The overall strict gate fails** because G10_3 moves 12.85 mm, above the predeclared 5 mm limit. This must not be reported as a four-object scene success. No force contact ≥0.2 N with that neighbour was detected. A separate static control after the same 2 s settling and recorded approach, with the robot held still for 20 s, moved G10_3 by **11.49 mm** (G10_1 1.17 mm, G10_2 2.08 mm), also without detected robot–neighbour force contacts. Thus most of the G10_3 drift is reproducible without transport; the current mesh/table initialization is not stable enough to certify untouched-object displacement. Sub-threshold contacts and mesh/pose modelling error are not excluded.

The first tested mesh candidate (+5 mm X, 0 Y, 0.5 rad) failed more visibly: after release the target tipped and ended 181 mm / 93° from the track-relative goal. Lowering it to table support alone did not fix that offset; the selected grasp geometry matters. The best candidate should be interpreted as a scene-specific geometric demonstration, not a robust solution.

The 2 s settling moves the neighbours before the measured replay by 5.91, 2.52 and 11.08 mm respectively, and vertical table projection applies −4.04, −4.71 and +3.33 mm. Those corrections separate the simulated scene from the original GT configuration and remain an explicit limitation. The task uses an oracle object trajectory and GT-based mesh alignment, not visual feedback. There is no physical-grasp, collision-safety, generalisation, or real-world success claim.

# Full 831-frame HO-Cap four-mesh replay, 2026-10-15

## Question and source

Does the **entire** subject-1 G10 episode transfer all four HO-Cap objects with the saved UR10/Robotiq plan, without knocking another object over? This is distinct from the earlier 180-frame G10_4-only contact-assisted task sequence. The full episode has 831 frames. In the saved HO-Cap GT, all four objects move: G10_2 approximately frames 44–116, G10_3 249–334, G10_4 473–574, and G10_1 685–767. These windows are descriptive, inferred from their pose tracks, not ground-truth grasp labels.

`experiments/hocap_four_mesh_raw_replay.py` compiled all four official `G10_*.cleaned_mesh_2000.obj` assets as free MuJoCo mesh geoms, aligned each to its tracked frame using the corresponding HO-Cap GT pose at frame 0. Their SHA-256 hashes are in the raw report. MuJoCo uses mesh **convex hulls** for contact. The scene uses the existing inferred table, assumed 0.2 kg per object and rubber-pad collision geoms. It executes the original full 831-frame **hand-only** `plan.h5` twice as slowly as recorded; it never teleports objects, pins them, activates a weld, or applies an acceptance threshold. This is a physical-replay diagnostic, not an object-aware four-grasp controller.

```bash
EPISODE=$(find data/datasets/hocap_eightview -maxdepth 1 -type d \
  -name 'subject_1_20231025_170105_*' -print -quit)
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_four_mesh_raw_replay.py "$EPISODE" \
  --output /app/rl-output/contact_attach_smoke/four_object_full_raw_audit.json
docker compose -f docker-compose.experiments.yml run --rm sim \
  experiments/hocap_four_mesh_raw_replay.py "$EPISODE" \
  --output /app/rl-output/contact_attach_smoke/four_object_static_control.json \
  --static-robot
```

The same dynamic run with `--video /app/rl-output/contact_attach_smoke/four_object_full_raw.mp4` produced a 56.2 s two-view MuJoCo video and matching `four_object_full_raw_video.json`. All raw JSON/video artifacts are local and gitignored under `data/experiments/hocap-rl/contact_attach_smoke/`.

## Measured outcome

The simulator reached **831/831 frames**, 56.133 s of physics, and the maximum arm command-tracking deviation was 0.04895 rad. No object ever had simultaneous contact with both named rubber-pad geoms in this run. `Peak lift` below is height above each *initial simulated centre*, not a claim about true contact with the table. No object rose above that initial height.

| Object | Simulated final vs tracked final position | Orientation difference | Peak lift | Bilateral pad-contact steps | Robot–object contact samples | Peak robot–object normal force |
|---|---:|---:|---:|---:|---:|---:|
| G10_1 | 239.14 mm | 148.80° | 0 mm | 0 | 3,712 | 8.995 N |
| G10_2 | 122.39 mm | 92.45° | 0 mm | 0 | 0 | 0 N |
| G10_3 | 82.82 mm | 155.87° | 0 mm | 0 | 0 | 0 N |
| G10_4 | 155.43 mm | 95.21° | 0 mm | 0 | 0 | 0 N |

G10_1 first contacts a non-pad robot geom at MuJoCo time 37.00 s, before its own recorded transfer window; the measured peak normal force is 8.995 N. The scene also contains 12,266 robot–table and 3,937 object–object *contact samples* over 0.5 ms steps. These are counts of sampled contact records, not distinct collision events. The raw report has the per-frame simulated positions, contact measures and final poses for all four objects.

The static-robot control, run for the same 56.133 s with the arm held at its initial joint pose, gives final-vs-tracked errors 17.5, 31.4, 118.6 and 145.0 mm for G10_1–4. Errors against tracked *final goals* in a static control do not measure passive drift directly: G10_4, for example, is supposed to move ~148 mm in the recording but does not in the control. The control does show that the current free-mesh/table physics are not inherently stable for every untouched object (notably G10_3); the dynamic replay adds a clear G10_1 robot strike.

## Conclusion

**The requested four-object transfer is not achieved.** All four free objects change position in the simulator, but this is not four successful picks and placements: there is no bilateral grip or lift, endpoint errors are large, and G10_1 is struck. It would be misleading to report that all four were moved as intended or that no one was knocked over. The prior 180-frame G10_4 contact-assisted result demonstrates only one target with an artificial hold and cannot be extrapolated to this full episode.

To attempt the intended whole scene, the next missing capabilities are four-object phase/contact-aware retargeting and a justified stable support/contact model for these meshes. Merely replaying the hand-only joint path, or pinning/teleporting objects to their GT tracks, would not establish a physical four-object demonstration.

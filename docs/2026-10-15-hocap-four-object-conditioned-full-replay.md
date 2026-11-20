# HO-Cap G10: full object-conditioned MuJoCo replay (2026-10-15)

## Protocol

The source is the complete 831-frame right-hand HO-Cap subject-1 G10 episode. Four independent object-track-conditioned UR10/Robotiq 2F-85 IK plans were calculated in the observed order G10_2, G10_3, G10_4, G10_1. Each plan used its corresponding cloud in `object_models.npz`; no saved hand-only `plan.h5` was read for this run. The first, second and fourth plans are new diagnostic outputs in `data/_retarget_diagnostics/hocap_g10_full_object_{2,3,1}/`. G10_4 reuses the existing object-aware pilot. Input plan and official mesh SHA-256 values are in the raw JSON report.

`experiments/hocap_four_object_assisted_replay.py` copies only the object-conditioned active phase of each plan into an 831-frame joint trajectory. It interpolates joint and finger commands between those phases. **Those inter-phase transitions are not collision-planned or independently validated by IK.** The resulting trajectory is `data/experiments/hocap-rl/contact_attach_smoke/full_four_object_trajectory.npz`; it is an experimental artifact, not a production `plan.h5`.

One uninterrupted MuJoCo scene contains the four official `G10_*.cleaned_mesh_2000.obj` assets as free bodies. Collision uses their MuJoCo convex hulls, not exact non-convex triangles. The tabletop height is inferred from the initial aligned meshes (`z = -0.000933 m`), and each object has assumed mass 0.2 kg. The UR10 base is `[0, -1, 0]` in `hocap_tag_1`. Commands run at twice the recording duration with a 0.5 ms physics step. No object is teleported or pinned. Four independent wrist welds are present but each may activate only after at least 0.08 s of bilateral named rubber-pad contact at ≥0.2 N, with the object between the pads, a closing command and no non-pad robot contact. Opening both commanded and measured finger positions releases the weld. An inactive weld supplies no hold force.

Reproduction after calculating the four object plans:

```bash
EPISODE=$(find data/datasets/hocap_eightview -maxdepth 1 -type d \
  -name 'subject_1_20231025_170105_*' -print -quit)
docker compose -f docker-compose.experiments.yml run --rm -T sim \
  experiments/hocap_four_object_assisted_replay.py "$EPISODE" \
  --output /app/rl-output/contact_attach_smoke/full_four_object_assisted.json \
  --trajectory /app/rl-output/contact_attach_smoke/full_four_object_trajectory.npz \
  --video /app/rl-output/contact_attach_smoke/full_four_object_assisted.mp4
```

## Measured result

The simulator completed **831/831 frames** and wrote the two-view video. All four segment IK solvers reported convergence, but their whole-window position RMSE was 34.77, 12.03, 28.93 and 39.43 mm for G10_2, G10_3, G10_4 and G10_1 respectively. G10_1 needed a 45 mm upward shift along its cloud to find a cross-section narrower than the 85 mm jaw limit; this is an experimental grasp-site selection, not an observed hand contact.

| Object | Final position error to tracked endpoint | Orientation error | Peak lift | Bilateral-pad physics samples | Weld activated / released |
|---|---:|---:|---:|---:|---|
| G10_2 | 35.37 mm | 7.50° | 0 mm | 0 | no / no |
| G10_3 | 116.15 mm | 137.75° | 0 mm | 0 | no / no |
| G10_4 | 121.91 mm | 103.40° | 0 mm | 40 | no / no |
| G10_1 | 10.37 mm | 14.76° | 0 mm | 0 | no / no |

The 40 G10_4 bilateral samples total only 20 ms at 0.5 ms per step, below the 80 ms attachment gate. The run had 5,239 robot–table *contact samples*, not that many separate impacts; the peak measured robot force on G10_2 while it was not the active target was 4.70 N. Maximum arm command-tracking error was 0.05085 rad. G10_1's small final position error is **not** a placement success: it was never lifted or attached and started close to that endpoint. G10_3 drifts markedly even without an intended grasp, consistent with the earlier static-arm stability control.

Frame-level pad and object positions in `data/experiments/hocap-rl/contact_attach_smoke/full_four_object_assisted_diagnostic.json` show the immediate failure mechanism: by the nominal carry start, simulated objects have not lifted, while the object-conditioned arm target is already following the *recorded elevated object*. The pad midpoint is 30.4, 42.5, 60.6 and 81.5 mm above the respective free simulated centres for G10_2, G10_3, G10_4 and G10_1 at those frames. These are centre offsets, not pad-to-surface clearances. The gripper therefore misses a stable two-sided clamp. Merely enabling welds cannot remedy absent sustained contact. This is a trajectory-timing/contact discrepancy; the current evidence does not isolate one unique cause among IK error, mesh/table alignment, gripper actuation lag and the planner's pickup timing.

**Conclusion: the requested four-object transfer does not work in this model.** This is a full-scene object-conditioned test and a negative result, not another hand-only control. It does not validate physical grasp, object collision avoidance, simulated task success or real-robot use. HO-Cap masks and first-frame alignment are oracle-assisted; the scene's mass, friction, mesh convexification and tabletop height are uncalibrated. The next experiment must gate the start of the recorded lift on actual sustained contact and verify the individual object/robot geometry before repeating the full run; changing the weld threshold alone would fabricate success.

## Opt-in any-pad attachment requested after the strict run

The follow-up deliberately uses a weaker **simulation-only** rule: a positive normal force on either named rubber-pad geom of the currently active object activates its weld in one 0.5 ms physics step. It does not require both pads, 80 ms dwell, the object centre between jaws, a closing command or absence of other robot contact. An attached object is released at its object-track-derived release frame; attachment is disabled after that frame. This is a contact-triggered carry aid, not evidence of a secure two-finger grasp. The earlier strict mode remains the default.

```bash
docker compose -f docker-compose.experiments.yml run --rm -T sim \
  experiments/hocap_four_object_assisted_replay.py "$EPISODE" \
  --output /app/rl-output/contact_attach_smoke/full_four_object_any_pad_final.json \
  --trajectory /app/rl-output/contact_attach_smoke/full_four_object_trajectory.npz \
  --video /app/rl-output/contact_attach_smoke/full_four_object_any_pad_final.mp4 \
  --attach-on-any-pad
```

All **831/831 frames** ran in one continuous scene; no object was teleported or reset. Raw report and two-view video are at the paths above.

| Object | Weld activated / released (sim seconds) | Peak lift | Final position error | Orientation error |
|---|---:|---:|---:|---:|
| G10_2 | 3.065 / 8.500 | 234.6 mm | 37.3 mm | 34.25° |
| G10_3 | 16.097 / 22.900 | 286.5 mm | 126.3 mm | 128.91° |
| G10_4 | never / never | 0 mm | 116.4 mm | 104.69° |
| G10_1 | 45.111 / 51.633 | 322.1 mm | 73.2 mm | 31.08° |

Three objects were artificially attached, lifted and released, **not** accurately placed. At weld activation G10_2 had 0.222 N on the right pad and 0 on the left; G10_3 had 1.646 N on the left and 0 on the right; G10_1 had 0.791 N on the right and 0 on the left. The `3-4` object-mesh contact began at 16.5555 s during G10_3 carry, peaked at 1.157 N and displaced G10_4 from its pickup site before its own phase. It therefore never touched a pad in this run. A later `1-3` contact began at 29.8335 s and peaked at 2.246 N. These are MuJoCo force readings under assumed material properties, not measured real-world forces.

This relaxed rule answers the contact-attachment question but **does not complete the four-object task**: only 3/4 objects attach, bystander objects are struck, and final pose errors are large. It should not be scored as physical grasp success or a validated trajectory. The next blocker is object/object collision and closed-loop pickup of an object that may have moved in the simulator, not the 80 ms weld threshold.

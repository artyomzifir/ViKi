# HO-Cap G10_4 object-aware retarget pilot (2026-10-14)

## Scope and inputs

This is a single, in-sample development experiment, not a cohort result or a
successful manipulation claim. The source is the right-hand HO-Cap episode
`subject_1_20231025_170105`, object `G10_4` (body wash), frames `[420, 600)`.
The 180-frame window contains one clear lift and approximately 143 mm of
tracked horizontal transfer. The object track has 10th-percentile confidence
0.667 in this interval. In the full eight-view sequence its *relative* GT
errors were 1.23 mm translation and 1.06 degrees rotation; these are not
absolute calibration errors. The segmentation uses supplied oracle instance
masks and its first object pose is aligned to the published HO-Cap reference.

`scripts/prepare_hocap_grasp_pilot.py` copies the same prepared hand arrays
into independent hand-only and object-aware episode directories and crops the
object track to matching timestamps. Neither source episode nor its existing
plan is modified. The pilot keeps the `hocap_tag_1` frame; it does not invent a
local Kinect world anchor. The selected object ID is explicit because all four
HO-Cap objects have the non-manipulation label `other_dynamic`.

Both plans use UR10, Robotiq 2F-85, base `[0, -1, 0]` in `hocap_tag_1`,
right-hand palm-to-tool RPY `[4.325, 15.45, -47.922]` degrees, reference
policy `zero`, a nominal infinite floor `z=0`, original URDF velocity limits,
and no artificial home approach. Only the object-aware plan replaces targets
with a rigid-object-relative grasp point, geometry-derived jaw width, side
orientation, and explicit open-jaw standoff/insertion/close/carry/release
phases. The attempted fixed, center-height grip and no-standoff variants are
not the final plan. Both final IK solves converged.

## Matched geometric comparison

The common object-centre error below is the achieved gripper-pad midpoint
relative to the *same* object-fixed grasp point during frames `[54, 135)` of
the crop. This is a self-consistency metric against the tracked object, not an
independent physical-pose measurement. Ordinary IK RMSE and orientation RMSE
use different targets in the two modes and are not standalone quality gains.

| Metric | Hand-only | Object-aware |
|---|---:|---:|
| Whole-clip position RMSE to own target (mm) | 33.60 | 28.93 |
| Whole-clip orientation RMSE to own target (degrees) | 31.66 | 16.61 |
| Carry object-centre RMSE (mm) | 11.63 | 6.14 |
| Carry object-centre P95 (mm) | 16.04 | 9.31 |
| Median carry pad-centre separation (mm) | 57.75 | 55.24 |
| IK floor margin, minimum (mm) | 0.00018 | 0.04918 |
| IK reported self-collision margin, minimum (m) | 0.04125 | 0.03997 |
| Margin to unscaled URDF speed limit, minimum (rad/s) | 1.429 | 1.359 |

The object-aware plan used the Y-side cross-section (61.24 mm from its partial
model), selected a side approach closest to the observed palm/tool
orientation, commanded 55.24 mm pad separation during carry, and inserted
with jaws fully open from an 83.18 mm standoff. Its phase bounds are
`[34, 54, 135, 163]`, with insertion ending at frame 44 and closing at 52.

## Physical proxy replay

`experiments/hocap_grasp_replay.py` executes *free* object dynamics without
teleporting it. Both plans execute the same original frames `[440, 600)`;
earlier frames were excluded because the hand-only plan initially penetrated
the inferred tabletop. The tracked partial surfels become a fitted box of
approximately `81.7 × 59.3 × 157.5 mm`; its mass is an **assumed** 0.20 kg,
not a HO-Cap annotation. The tabletop top is inferred at `z=0.03987 m` from
that box, not independently measured. Therefore it is higher than the IK's
`z=0` safety half-space. The same `0.0005 s` implicit-fast physics step,
2×-slowed position-control commands, and no extra grasp-pressure command are
used for both plans. One nominally overlapping URDF pair,
`forearm_link`/`wrist_2_link`, is excluded in the MuJoCo model as in the IK
collision filter; this cannot certify self-collision safety. The UR10 wrist
kinematics and saved gripper-pad gap were cross-checked between Pinocchio and
MuJoCo (maximum sampled errors `5.3e-16 m` and `2.0e-9 m`).

| Metric | Hand-only | Object-aware |
|---|---:|---:|
| Peak object lift above start (mm) | 8.87 | 136.20 |
| Simulation steps with both distal-phalanx geoms contacting object | 0 | 2,715 |
| Both-pad contact fraction while lifted in carry | 0 | 0.863 |
| Maximum object slip relative to wrist in carry (mm) | — | 159.84 |
| Longest lost-contact interval in carry (ms) | — | 141 |
| Final translation error to observed placement (mm) | 60.44 | 110.02 |
| Final orientation error (degrees) | 92.54 | 92.09 |
| Maximum robot-table penetration (mm) | 0.60 | 0 |
| Grasp/transfer success under the strict proxy gate | no/no | no/no |

The object-aware plan **does initiate a real two-pad lift in this proxy**, unlike
the hand-only plan. It loses the object early in carry, before the planned
release, so it is **not yet a reliable grasp or pick-and-place demonstration**.
The result must not be described as success on the real HO-Cap object or as a
real-robot result. A no-standoff object-aware ablation pushed the box with one
pad and lifted only 8.8 mm. Changing the presumed mass, slowing commands,
adding 0.05–0.20 rad of simulator-only closing pressure, increasing simulated
pad friction, adding a 0.3–1.0 s post-close dwell, or moving the grip to the
proxy's middle height did not pass
the carry slip and final-placement gates in this one-scene search. These are
in-sample diagnostics, not independent hyperparameter validation.

The strict grasp gate requires over 25 mm of lift, over 80% bilateral distal-geom contact
while lifted in carry, under 10 mm carry slip, under 50 ms lost contact, under
4 mm object/table penetration, and bounded actuator tracking errors. The
transfer gate additionally requires final translation under 50 mm and
orientation under 30 degrees. Intentional release is excluded from the carry
slip/contact metrics. The control's phase window is taken from the paired
object-aware plan solely for scoring, not for commanding its gripper.

## Contact-overlap and pickup-timing follow-up

The selected rigid point cloud and its fitted box proxy were held fixed while
changing only the experimental jaw-gap target. The observed cross-section is
61.24 mm, but the 5–95% proxy box is 59.28 mm wide on the same local axis.
Thus the default 6 mm *nominal* cloud overlap commands 55.24 mm between pad
centres, about 4.04 mm smaller than the proxy. The 4 and 8 mm trials command
57.24 and 53.24 mm respectively. These are **position commands**, not measured
penetrations or calibrated squeezing forces. Every plan below converged with
positive original-URDF speed margin and nonnegative IK floor margin; all
MuJoCo runs use the same 0.20 kg free box and 2× time scale unless noted.

| Trial | Peak lift (mm) | Two-sided distal-geom contact while lifted in carry | Max carry slip (mm) | Final translation error (mm) | Max object penetration (mm) | Grasp / transfer gate |
|---|---:|---:|---:|---:|---:|---|
| Default 6 mm | 136.20 | 0.863 | 159.84 | 110.02 | 2.27 | fail / fail |
| Less overlap, 4 mm | 147.28 | 0.877 | 161.57 | 279.35 | 5.41 | fail / fail |
| More overlap, 8 mm | 134.83 | 0.863 | 159.55 | 111.40 | 2.54 | fail / fail |
| 8 mm, grasp 15 mm lower | 38.63 | 0.654 | 41.43 | 194.33 | 6.23 | fail / fail |
| 8 mm, insertion/closure 4 frames earlier | 21.41 | 0 | no lifted carry | 198.18 | 6.06 | fail / fail |
| 8 mm, gripper actuator kp 100 (sim only) | 122.28 | 0.829 | 158.85 | 261.26 | 8.88 | fail / fail |
| 8 mm, gripper actuator kp 150 (sim only) | 125.18 | 0.833 | 158.38 | 261.88 | 8.38 | fail / fail |

The last four rows are diagnostic ablations, not recommended settings. Lower
contact, earlier closure, and stiffer simulated actuation were removed from
ViKi's retarget interface after their failure; the default remains 6 mm. The
bounded `--object-contact-overlap-mm` option applies **only** when an explicit
HO-Cap object track is supplied. It does not alter the default cube pipeline.
The 4 mm trial lifted 11.08 mm higher than baseline, but lost the object and
made final placement substantially worse; no tuned setting passed the gate.

The extra baseline trace exposes a more specific failure mechanism. At 2.21 s
of replay, both *distal-phalanx collision geoms* already register contact,
although the calculated pad-centre gap is 79.7 mm—about 20 mm larger than the
box width. These geom contacts therefore do **not** establish that both
rubber-pad patches have clamped the object. In the cached Robotiq v4 URDF,
the separate rubber-pad link and its joint are commented out; only the distal
phalanx collision meshes are active. By 2.91 s the box has rotated about
49° from its initial simulated orientation; at 3.51 s its centre is about
68 mm below the tracked reference while the pad midpoint has climbed near
its upper edge. Contact is lost around 3.61 s, long before the planned release
at 7.97 s. This is an inference from the simulated pose/contact trace, not an
independent measurement of the HO-Cap object's physical motion. Better jaw
geometry/contact patches, collision-aware insertion and force-aware closure
are needed; adding nominal penetration alone is not a reliable fix.

All variant plans and replays remain under gitignored
`data/_retarget_diagnostics/hocap_g10_4_*`; full JSON metrics and plan hashes
are snapshotted in `paper/results/hocap/object_grasp_pilot/`. The baseline
`contact_trace_physics.json` adds simulated object rotation and pad-centre
poses at 0.1 s intervals; the earlier `object_physics.json` remains unchanged.

## Reproduction and next gate

On a fresh output directory, use `scripts/prepare_hocap_grasp_pilot.py` with
`--first 420 --last 600`, then run `viki retarget` on both copies with the
settings above. The control passes `--no-object-grasp`; the treatment passes
`--object-grasp --object-track-path <object-copy>/object_track.npz --object-id 4`.
The reproducible MuJoCo command is
`experiments/hocap_grasp_replay.py <plan> <object_track> --object-id 4 --frame-start 20 --frame-end 180`;
for hand-only scoring also pass `--phase-plan <object-aware-plan>`.
Run it inside the existing `viki-experiments` image with `experiments/`,
`data/`, and `models/` mounted at `/app`. Exact numerical plan metrics and
simulator traces are preserved under `paper/results/hocap/object_grasp_pilot/`.

This should remain an **opt-in pilot**. Before claiming object-driven
retarget works, we need a calibrated object shape/table frame, contact-aware
and force-aware gripper control, simulation tests on multiple held-out objects
and masses, and independent evidence of sustained carry plus placement. The
existing 28 HO-Cap hand-only plans are untouched; this mode does not add
human/object collision avoidance to their IK solver.

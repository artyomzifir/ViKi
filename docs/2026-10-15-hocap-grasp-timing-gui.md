# HO-Cap G10_4: pre-lift contact and full-close GUI trial (15 October 2026)

Scene: `subject_1/20231025_170105`, object G10_4, original frames 420–600,
replay slice `[20,180)`. The UR10/Robotiq plan was retargeted with the
official G10_4 mesh substituted for the tracked object model; geometry is
aligned using the first-frame HO-Cap GT pose. The MuJoCo collision geom is
the mesh's convex hull, not the non-convex bottle. The 0.2 kg mass, rubber-pad
boxes, and table height inferred from the mesh are simulator assumptions.

The original replay lifted as soon as the phased geometric command entered
carry. We added a simulator-only one-second hold of the pre-lift arm pose,
commanded the **final** closure target before that hold, and measured object
contact normal force on both pad geoms. The gate requires at least 1 N on
each side continuously for the final 0.1 s of the hold; otherwise playback
stops before lift. This is a diagnostic necessary condition, **not** proof
that the object will be retained. The native MuJoCo GUI displayed each trial.

| Gripper command | Peak left / right pad force | Gate | Result |
|---|---:|---|---|
| Geometric drive + 0.10 rad | 0.740 / 0.779 N | fail | Lift cancelled. |
| Geometric drive + 0.20 rad | 1.202 / 1.288 N | historical gate passed | Completed but object fell; earlier gate latched a transient pass, so this is **not** evidence of sustained pre-lift contact. |
| Full 0.8 rad closure | 3.184 / 3.089 N | fail | Object moved 59.49 mm in XY during closure; force was transient and lift was cancelled. |

Full closure is a *position command*, not a measured force command. The
gripper cannot reach 0.8 rad through a rigid object; pushing harder instead
displaced this object sideways. Higher pad friction 2–3 in earlier official-mesh
replays with +0.10 rad produced only about 2.5 mm peak lift, but those replays
did not use the new hold/gate. The separate 10/15 mm insertion-depth trials
used a fitted box rather than this mesh and also failed strict grasp/transfer;
they do not isolate insertion depth on the official mesh.

**Conclusion:** the premature lift was real, but simply increasing duration,
position-command closure, or simulated friction does not establish a stable
grasp. Contact placement/pose alignment and a load-tested closure policy are
still unresolved. Do not use this pilot as successful object-aware retarget
or as training ground truth. Raw per-run JSON is in

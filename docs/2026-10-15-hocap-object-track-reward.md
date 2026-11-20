# HO-Cap object-trajectory reward pilot (15 October 2026)

This is a **single-scene diagnostic**, not a working pick-and-place policy or a
held-out-object result. The new `object_track` mode is opt-in; the previous
`pickup` reward remains the default for historical comparison.

## Protocol

- Scene: `subject_1/20231025_170105`, object `G10_4`, cropped 180-frame
  `hocap_g10_4_orient_weight_003` plan and object track, reset frame 44.
- Reference: perception-derived 3-D object translation, indexed against the
  plan's microsecond timestamps. The track does not contain its own timestamps;
  equal frame count and common crop are currently required assumptions.
- The remaining human track lasts 4.50 s. MuJoCo replays it at 2× duration,
  matching the existing replay default: 9.00 s within a 250-step, 10.00 s
  episode at 25 Hz. The corridor allows 0.60 s of lag in simulation time and
  a 20 mm spatial radius. The object-position error is the shortest distance
  to the polyline over that limited time window, not a nearest point anywhere
  along the full path. These tolerances are pilot hyperparameters, not
  independently calibrated tracking uncertainty.
- Per-step position penalty is zero inside the corridor and
  `-0.25 * outside_m / (outside_m + 0.05)` outside. There is **no positive
  reward for remaining near a static reference**. Pad proximity contributes
  only a change in distance, not a recurring bonus. A first stable unsupported
  bilateral lift earns a one-time +15.
- Full success requires that lift, final object position within 30 mm of the
  demonstrated goal, table support, object speed no more than 30 mm/s, no pad
  force above 0.05 N, and 13 consecutive settled control steps after the
  reference ends. It pays +100. Timeout pays −25; ejection or a non-pad
  robot-object contact detected above 0.2 N at a control-step boundary pays
  −250 and terminates the episode.
- The policy observes simulated object and contact state, including the current
  reference offset. These are **privileged simulation observations**; this is
  not a camera-conditioned or real-world policy. The official mesh is anchored
  with HO-Cap GT at the first frame, while the rewarded path is ViKi's
  perception track. Mass, friction, pads and table are still assumptions.

## Measured smoke

The source inputs, hashes, checkpoint and three evaluation episodes are in
`data/experiments/hocap-rl/object_track_1024/smoke_report.json`. The exact
training call used `experiments.train_hocap_grasp_sac` with
`--reward-mode object_track --timesteps 1024 --eval-episodes 3 --seed 42`
and the paths recorded in that JSON. Training and evaluation are on the same
scene, with randomized mass/friction and ±5 mm initial XY.

| Controller | Seeds | Return | Full success | Stable lift | Final goal distance |
|---|---|---|---|---|---|
| No action | 1000–1002 | −70.40, −70.24, −70.50 | 0/3 | 0/3 | 145–152 mm |
| Close and hold | 1000–1002 | −69.63, −69.35, −69.21 | 0/3 | 0/3 | 123–133 mm |
| SAC, 1,024 steps | 1042–1044 | −72.54, −72.45, −71.44 | 0/3 | 0/3 | 150–187 mm |

The old large recurring contact bonus is gone, but close-and-hold remains
slightly better than no action because it nudges the object toward the goal
while on the table. The 1,024-step SAC policy also moved the object about
140 mm in XY without lifting it (peak lift below 0.3 mm). A further **pre-gate** 10,000-step SAC run
(`object_track_10000/smoke_report.json`) likewise achieved 0/10, no bilateral
steps and 159 mm mean XY shift. A replay of that checkpoint identified
forceful object contact from a robot geom other than either rubber pad. With
no action, the object drifts only about 2 mm, so the large shift is caused by
the controller, not passive simulator settling.

The experimental mode now terminates at a non-pad object contact of at least
0.2 N. A fresh 10,000-step SAC run under that gate is saved as
`data/experiments/hocap-rl/object_track_gate_10000/smoke_report.json`: 0/10
full successes, 0 bilateral-contact steps, peak lift 0.21 mm, and 9/10
terminated by the new contact gate. Mean return was −261.49 and mean final XY
shift 17 mm. These are **in-sample randomized seeds**, not generalization.
A MuJoCo GUI replay of the saved gated policy on seed 1042 terminated at
control step 184 with `non_pad_contact=true` and no success; the viewer emitted
host GLX/nouveau warnings but completed the episode.

The gate does not prohibit every grasp: the same scripted close-and-lift
controller reached stable unsupported bilateral lift on seeds 1004, 1005,
1008 and 1009 without a non-pad collision. It earned returns −55.81, −53.41,
−58.28 and −56.11 versus −70.19, −70.26, −70.22 and −70.17 for no action on
those seeds. None reached the final goal. Thus the reward ranks a stable lift
above inaction, but SAC still cannot discover and execute it reliably.

The trajectory reward **does not solve contact learning**. The old recurring
contact bonus is gone, but closing on the table can still slightly reduce
trajectory error without a valid pickup. Do not scale this policy to 28 scenes
or claim successful transfer from these runs. The next gate is a GUI inspection
and a controlled grasp/controller or residual-policy improvement, then a
fixed-seed comparison across contact parameters. Orientation is intentionally
not yet rewarded: this pilot tests the proposed position-in-time corridor.

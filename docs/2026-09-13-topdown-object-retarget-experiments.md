# Top-down retarget and object-conditioned grasp experiment — 2026-09-13

Follow-up object-shape, dynamic-SE(3), and gripper-width results are documented
in `2026-09-13-object-shape-se3-grasp-probe.md`.  They supersede the
translation-only object proxy below for the grasp-stabilisation question while
leaving this experiment's recorded A/B results unchanged.

## Question

Can the current `move-shipok` demonstration be retargeted with the UR10 and
Robotiq approaching from above instead of putting the wrist below the grasp
point, and can an object track suppress relative grasp jitter without changing
the intended slow motion?

No production code, default configuration, or protected baseline was changed.
All generated inputs and plans live under
`data/_retarget_lab/2026-09-13_topdown_object/`.  After review, the two selected
plans were copied to their episode roots for Viewer inspection; the prior
`plan.h5` and `status.json` files are preserved under each episode's
`intermediates/retarget/2026-09-13_before_topdown/` directory.

## Source and protocol

- Episode: `data/datasets/new-dataset/2026-09-09_11-54-10`
  (`move-shipok`, perception profile V4, `hand_fit` pose, pinch-centre target).
- Baseline: the episode's existing schema-v8 `plan.h5`.
- Tuning grid: every sixth sample (150 samples, 5 Hz) with timestamps retained,
  so duration and the SI-scaled velocity/acceleration regularisers remain
  physical.  The chosen candidate was then rerun on all 897 samples at 30 Hz.
- Existing provisional hand-to-TCP RPY:
  `[-175.675, 15.45, -47.922]` degrees.
- Top-side diagnostic RPY: `[4.325, 15.45, -47.922]` degrees, which is the
  existing local mapping post-multiplied by a 180-degree rotation about local X.
- `home-reference` means the batch solver was initialised and regularised about
  the registered UR home `[0, -pi/2, 0, -pi/2, 0, 0]` instead of zero.  The
  original experiment monkey-patched the lab process only.  The independently
  validated version of this option was later promoted to the v0.0.2 candidate
  as the explicit, reversible `reference_policy=robot_home` setting.

### Object-conditioned target

The blue object was detected independently in both RGB-D streams over manually
selected carry frames 430--700.  Detection support was 271/271 frames.  The
per-frame fused centroid was smoothed with SG(11, 2).  Rather than imposing a
constant object/hand transform, the relative vector was smoothed with SG(31, 2):

```text
p_corrected(t) = p_object,SG11(t)
               + SG31(p_pinch(t) - p_object,SG11(t))
```

A 20-frame squared-sine ramp was used at attach/release boundaries.  Both
fingertips were shifted equally, preserving gripper opening and palm
orientation.  This is deliberately a translation-only, scene-specific test,
not a general object tracker.

On the full-rate target, the correction had 1.21 mm RMS and 1.97 mm p95.  It
reduced pinch second-difference RMS from 0.412 to 0.215 mm/frame².  A strictly
constant relative vector was rejected: it required 7.67 mm RMS and 16.37 mm p95
corrections, large enough to overwrite plausible intended motion.

## 5 Hz tuning grid

All margins are exact post-solve checks.  Collision margin is clearance beyond
the configured 20 mm self-collision barrier.

| variant | position RMSE, mm | orientation RMSE | achieved axis down | floor margin, mm | collision margin, mm |
|---|---:|---:|---:|---:|---:|
| original mapping, zero reference | 21.02 | 24.12° | 0.0% | 0.00006 | 39.69 |
| flipped, `w_ori=0.01`, zero reference | 37.88 | 89.14° | 93.2% | 0.00056 | 31.77 |
| flipped, `w_ori=0.03`, zero reference | 63.38 | 74.92° | 97.0% | 0.00166 | 0.14 |
| flipped, `w_ori=0.03`, home reference | 31.32 | 24.03° | 100% | 38.29 | 0.93 |
| **flipped, `w_ori=0.01`, home reference** | **23.89** | 28.65° | **100%** | **38.29** | **18.88** |
| same + object-conditioned target | **23.89** | 28.65° | **100%** | **38.29** | **18.88** |

Increasing the scalar SO(3) weight is not the answer: it sacrifices position
and drives the solution against the collision barrier.  The home reference is
what selects the useful upper branch.  Object conditioning does not switch that
branch or consume safety margin.

On the 5 Hz carry evaluation window, object conditioning reduced target/object
second-difference RMS from 2.247 to 0.710 mm/sample² (68%) and achieved
robot/object RMS from 1.871 to 1.174 mm/sample² (37%).

## Full-rate candidate

The selected `flip + home-reference + w_ori=0.01 + object-conditioned` variant
converged in 21 iterations on all 897 frames.

| metric | existing plan | candidate |
|---|---:|---:|
| position RMSE | 18.47 mm | 20.99 mm |
| position p95 | 55.44 mm | 54.11 mm |
| orientation RMSE | 21.13° | 26.94° |
| achieved tool axis in downward hemisphere | 0% | **100%** |
| wrist height relative to grasp point, median | **−66.9 mm** | **+91.2 mm** |
| wrist above grasp point | 3.6% | **100%** |
| carry position RMSE | 5.79 mm | **3.11 mm** |
| carry position p95 | 8.94 mm | **4.83 mm** |
| floor margin | ~0 mm | **38.29 mm** |
| self-collision margin beyond 20 mm | 39.83 mm | 7.41 mm |
| velocity-limit margin | 1.577 rad/s | 0.588 rad/s |
| max joint velocity | 1.446 rad/s | 2.554 rad/s |
| max joint acceleration | 2.214 rad/s² | 3.912 rad/s² |
| target/object second-difference RMS | 0.382 | **0.036 mm/frame²** |
| achieved robot/object second-difference RMS | 0.128 | **0.104 mm/frame²** |
| achieved object-relative distance sigma | 3.03 mm | **2.74 mm** |

The side view independently confirms the branch change: the original wrist is
below the red grasp target, while the candidate wrist and gripper body are
above it.  The plot is
`data/_retarget_lab/2026-09-13_topdown_object/topdown_ab_side.png`.

## Pyramid Viewer candidate

Episode `2026-09-04_13-37-41` has only 218 originally observed target frames
out of 898.  The selected lab plan therefore uses a weak confidence of 0.02 on
the 680 interpolated frames.  This is a trajectory-shape prior for inspection,
not evidence recovered from the recording.  The target tool +Z axis is fixed to
the downward table normal, while yaw is the smoothed projection of the mapped
human hand x-axis.  The solver uses the UR home reference, `w_ori=0.01`, and the
measured table plane at calibration z=35 mm.

| metric | existing plan | semantic top-down candidate |
|---|---:|---:|
| solver | max iterations (40) | **converged (20)** |
| position RMSE, all 898 targets | **36.98 mm** | 45.89 mm |
| position RMSE, 218 originally observed frames | 17.75 mm | **13.97 mm** |
| position p95, originally observed frames | 37.35 mm | **24.44 mm** |
| orientation RMSE, all frames | 68.48° | **33.18°** |
| orientation RMSE, originally observed frames | 50.79° | **19.22°** |
| achieved tool axis within 45° of vertical down | 20.6% | **100%** |
| wrist above target | 50.2% | **100%** |
| wrist height relative to target, median | 0.31 mm | **147.04 mm** |
| floor margin | ~0 mm | **3.29 mm above the measured table** |
| self-collision margin beyond 20 mm | 39.29 mm | 39.49 mm |
| max joint velocity | 2.789 rad/s | **2.002 rad/s** |
| max joint acceleration | 4.273 rad/s² | **3.067 rad/s²** |

The all-frame position number worsens because the new plan deliberately assigns
a weak target to every long gap; the comparable observed-frame error improves.
This plan is explicitly marked `viewer_only` in the episode status.

## Runtime profile

`cProfile` was run on a duration-preserving 150-frame (stride 6), six-iteration
pyramid retarget with the same collision, floor, semantic-orientation, and weak
fill settings.  Total measured time was 54.706 s.  The profile is stored at
`data/_retarget_lab/2026-09-13_topdown_object/pyramid_runtime_profile.prof`.

| operation | cumulative time | share of total |
|---|---:|---:|
| exact whole-trajectory self-collision margin sweeps | 35.359 s | 64.6% |
| per-iteration self-collision linearisation | 16.408 s | 30.0% |
| floor exact checks + linearisation | 1.007 s | 1.8% |
| QP solve, including OSQP | 0.125 s | 0.23% |
| pose/Jacobian linearisation | 0.112 s | 0.20% |

Self-collision accounts for roughly 95% of the run.  The hot leaf is Pink's
`Configuration.update`: 49.968 s over 3,010 configuration constructions.  The
current adapter creates fresh Pinocchio model and collision data inside
`_collision_configuration()` for every frame and for every exact or linearised
collision query.  With no rejected line-search step in this profile, each
iteration still performs one exact sweep of the current trajectory and another
of the candidate, in addition to the collision-linearisation sweep.  OSQP is
not the bottleneck.

The first optimisation target should therefore be reuse of Pinocchio/Pink data
and configurations across frame queries, followed by carrying the accepted
candidate's exact collision margin into the next iteration instead of
immediately recomputing it.  Any reduction in the frequency or set of collision
checks must retain an exact full-trajectory acceptance check.

## Conclusions and limitations

1. The reported underside behaviour is real.  It is caused by the target/tool
   correspondence and reinforced by the zero solver reference; it is not a
   viewer transform error.
2. Flipping the provisional correspondence and using UR home as the solver
   reference yields a viable upper-side branch on this episode.  The full-rate
   position cost is modest (+2.52 mm RMSE), and accuracy during rigid carry
   improves substantially.
3. The result is **above-side, not yet vertically top-down**.  The achieved tool
   axis is a median 53.1° from vertical down, and essentially no frame lies
   within a 45° cone.  The flipped palm frame preserves the demonstrator's
   tilt; a semantic approach-axis term is still needed.
4. A single scalar orientation weight cannot express the task.  Tilt toward the
   table normal should be constrained or weighted separately from yaw about the
   tool axis.
5. Object conditioning is useful as a contact-phase stabiliser.  It reduces
   fast robot/object relative motion without changing the IK branch, but this
   experiment uses a manual phase and a colour-specific translation track.
6. The candidate remains an offline result, not a hardware-ready plan.  The
   floor is still the calibration-board plane rather than the measured table,
   acceleration has no hard limit, and no environment/cloud collision model is
   present.

## Recommended production work

- Make `q_reference` an explicit kinematics/solver input and use the robot's
  validated upper-workspace reference for both batch IK and approach.
- Replace the fixed full-palm SO(3) mapping with a task-semantic tool frame:
  table normal controls approach tilt; object/grasp direction controls yaw.
- Add an explicit top-down cone/hinge residual and minimum wrist/flange height
  relative to the grasp point, then use multi-start branch selection with
  clearance as a score.
- Store and use the per-episode measured table plane.
- Introduce a contact-phase object factor with confidence and attach/release
  ramps.  Do not impose constant relative pose on pushes or uncertain tracks.
- Add hard acceleration/jerk limits and an environment distance field before
  authorising replay on hardware.

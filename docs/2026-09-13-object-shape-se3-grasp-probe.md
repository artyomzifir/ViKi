# Object shape, dynamic SE(3), and grasp-width probe — 2026-09-13

## Question and scope

The object-centric representation is tested here as a **grasp stabiliser**, not
only as a cross-placement representation.  The aim is to suppress motion of the
recorded hand relative to a rigidly held object while preserving the object's
world translation and rotation.  A second aim is to stop a position-controlled
parallel gripper from closing substantially through a rigid object.

No production code, default configuration, episode plan, or protected baseline
was changed.  Lab scripts live in `data/_lab/`; artifacts live under
`data/_object_lab/2026-09-13_shape_pose/` and
`data/_retarget_lab/2026-09-13_dynamic_object/`.

## Object observations from the stored clouds

The viewer cloud artifact contains RGB after its packed XYZ array.  A colour
gate can therefore segment distinctive objects directly from the already fused
cloud, followed by voxel connectivity and temporal proximity.  This is much
cheaper than reopening both RGB-D recordings.  Colour is only an identity proxy
for these experiments; it is not proposed as the production object contract.

An object shell was accumulated while the object was resting.  A temporal
symmetric trimmed point-to-point ICP then tracked the shell.  The resulting
model is necessarily a partial visible shell, not a watertight shape.

| case | model points | robust visible dimensions | ICP residual, median / p95 | rotation step, median / p95 |
|---|---:|---:|---:|---:|
| `move-shipok`, blue bracket | 3,173 | 129.6 × 79.3 × 111.2 mm | 2.90 / 3.76 mm | 0.85 / 2.47° per 3 frames |
| `pyramid`, red block | 2,448 | 151.0 × 96.4 × 61.1 mm | 5.47 / 8.11 mm | 1.89 / 13.6° per 3 frames |

The tracked red block does capture a large deliberate rotation, but its local
orientation noise is too high for an unconditional trajectory correction.

### Static-pose repeatability

The same tracker was scored during intervals in which the object is visibly at
rest.  These are self-consistency figures, not ground-truth errors.

| case / phase | position RMS | position p95 radius | orientation RMS | orientation p95 |
|---|---:|---:|---:|---:|
| blue, before grasp | 0.53 mm | 1.59 mm | 1.23° | 2.10° |
| blue, after placement | 1.26 mm | 3.03 mm | 1.97° | 2.75° |
| red, before grasp | 2.69 mm | 8.21 mm | 2.45° | 4.28° |
| red, after placement | 1.41 mm | 4.63 mm | 4.39° | **9.76°** |

The blue track is good enough for a mild SE(3) stabilisation experiment.  The
red orientation is not: after placement, its reported static orientation alone
moves by almost 10° at p95.

### Shape repeatability

Independently reconstructed before/after shells agree in two dimensions and
fail in one view-dependent dimension:

- blue sorted span changes: 0.9%, 10.8%, **38.4%**;
- red sorted span changes: 19.7%, 2.6%, 4.4%.

Thus a partial shell can provide a robust size range and feasibility signal,
but not yet an exact collision mesh or a hard millimetre-scale width bound.
Known CAD, an object scan, a fiducial, or another useful viewpoint would be
needed for that stronger claim.

## Separating hand and object

Deleting every point inside an inflated hand capsule is unsafe at contact.  The
fraction of genuinely colour-labelled object points deleted by that rule was:

| margin around hand capsule | 0 mm | 5 mm | 10 mm | 20 mm | 30 mm |
|---|---:|---:|---:|---:|---:|
| blue, median | 0.0% | 0.5% | 2.6% | 9.2% | 15.4% |
| red, median | 3.1% | 10.1% | 20.8% | **50.2%** | 72.7% |

The enclosing grasp in `pyramid` makes geometric subtraction erase the object
it is intended to recover.  The production representation therefore needs
explicit, mutually exclusive labels for operator, manipulated object, and
environment.  At replay time the robot URDF can mask the actuator much more
accurately than a human-hand capsule, but the contact band must still be marked
ambiguous/allowed rather than deleted blindly.

## Dynamic object-relative SE(3)

The correction does not freeze world axes.  During a contact phase it computes

```text
T_OH(t) = inverse(T_WO(t)) * T_WH(t)
T'_WH(t) = T'_WO(t) * smooth(T_OH(t))
```

with attach/release ramps.  Consequently all low-frequency world motion of the
object, including rotation needed for carrying and pouring, passes through.
Only implausible fast variation of the grasp transform is suppressed.  A task
may keep slow relative motion instead of forcing `T_OH` to be constant.

On `move-shipok`, this proxy reduced world-target second-difference by 41% in
position and 36% in orientation.  The hand's net world rotation remained 12.5°
versus 12.7° recorded, so it was not world-axis locking.  Corrections were only
2.7 mm RMS and 1.24° RMS.

On the red `pyramid` block, the same unconditional orientation correction made
orientation second-difference **worse**, from 0.72° to 1.43° per sampled-frame²,
and required 8.8° RMS corrections.  That case must reject the orientation
factor and fall back to translation or a supported planar subset.  Pose
confidence must therefore be per degree of freedom, not one scalar for SE(3).

## Gripper command from object geometry

The current normalised thumb/index mapping is not a safe physical closing
command.  During the blue carry:

- human thumb/index gap: 65.4 mm median;
- visible object section along that pinch line: 71.0 mm median;
- current Robotiq opening: **37.9 mm median**;
- visible section along the current robot closing axis: **100.6 mm median**.

The old plan therefore both over-closes and presents the jaws along a poor
object axis.  A lab target aligned local gripper Y with the thumb/index line in
the moving object frame and imposed a 79 mm lower opening, taken from the
repeatable narrow model dimension.  After alignment, the visible section along
the robot closing axis was 74.3 mm median and the commanded opening had a
+3.5 mm median visible-shell clearance.  The core contact interval stayed at
79 mm; the attach ramp still dipped to 45.4 mm at its first sample, so a real
implementation must open during pre-grasp, before contact.

For the red block the measured section along the current robot jaw axis is 89.0
mm median, already beyond the 85 mm hardware maximum, with much larger values
along the demonstrated pinch line.  That should be reported as an uncertain or
infeasible grasp requiring a different grasp axis or gripper, not silently
converted into maximum closure.

## Coarse retarget A/B on `move-shipok`

Both plans use 150 duration-preserving samples, the top-side hand/TCP mapping,
UR home reference, `w_orientation=0.01`, and the same collision/floor model.
The candidate adds dynamic object-relative pose conditioning, object-relative
jaw-axis conditioning, and the 79 mm opening floor.

| metric | top-side baseline | object/grasp aware |
|---|---:|---:|
| position RMSE | 23.89 mm | 23.79 mm |
| orientation RMSE, all frames | **28.65°** | 30.36° |
| carry orientation error, median | 6.51° | **5.47°** |
| object-relative position d2 | 2.538 | **0.518 mm/sample²** |
| object-relative orientation d2 | 1.162° | **0.713°/sample²** |
| world target position d2 | 6.433 | **5.626 mm/sample²** |
| achieved position d2 | 5.711 | **5.264 mm/sample²** |
| floor margin | 38.29 mm | 38.29 mm |
| collision margin beyond barrier | 18.88 mm | 19.09 mm |

The desired effect is visible: relative translation jerk drops about 80% and
relative orientation jerk about 39%, without consuming feasibility margin or
locking the object in the world.  Global orientation RMSE rises by 1.71° because
the jaw-axis correspondence changes the requested pose over the contact phase;
inside the carry, the actual orientation error improves by about 1.0°.  This is
a correspondence trade-off, not evidence that the SE(3) smoother itself is
free of cost.

## Proposed production contract

1. Segment the scene into `operator`, `manipulated_object`, `other_object`, and
   `static_environment`.  Operator points are removed from the replay obstacle
   field because the robot replaces the operator; all other scene points remain
   obstacles except the selected object's declared gripper contact patch.
2. Associate object instances over time using appearance, geometry, proximity,
   and co-motion with the hand.  Contact phase comes from object motion plus
   hand proximity, not from the noisy gripper signal alone.
3. Export the object model, object-frame convention, per-frame pose, pose
   covariance/observable-DOF mask, dimensions, and segmentation confidence.
4. Add a robust object-relative pose factor only during supported contact
   phases.  Apply translation and each rotational degree of freedom only when
   its observability clears a measured threshold.
5. Derive the grasp frame from the contact geometry: approach axis, unsigned
   jaw line, contact centre, and feasible width.  Keep task rotation about the
   object instead of replacing it with a fixed world orientation.
6. Treat opening as a constrained command: pre-grasp clearance, contact width,
   maximum permitted compression, and hardware limits.  If geometry and
   hardware disagree, return `grasp_infeasible` rather than saturating closed.
7. Build an environment distance field from everything except the operator.
   The manipulated object is collision-enabled everywhere except explicit
   finger contact patches; other objects and the table remain obstacles.

The current evidence supports implementing this first as an optional,
confidence-gated retarget input.  It does not support making full 6-D object ICP
mandatory or applying the red-block orientation track to `pyramid`.

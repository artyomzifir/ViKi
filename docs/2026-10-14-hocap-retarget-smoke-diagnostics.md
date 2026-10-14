# HO-Cap retarget smoke: failed first attempt and repair (2026-10-14)

The first hand-only UR10 / Robotiq 2F-85 solve used the 676-frame
`subject_1-20231025_165502` episode, base `[0, -1.0, 0]`, table plane `z=0`
in `hocap_tag_1`, and disabled cube-specific object grasp. The numerical IK
ran for roughly 34 minutes, but the attempt **did not produce a plan**:
`_velocity_limited_approach` raised `could not construct an approach within
the robot velocity limits`. The gated 28-scene batch did not start.

The failure was after the main IK solve, in the synthetic home-to-first-frame
approach. The old quintic exactly matched the first three samples' estimated
velocity *and acceleration*. Under a high-curvature start, stretching its
duration multiplies the fixed endpoint acceleration contribution by duration
squared, so it can increase rather than decrease the peak velocity. A small
synthetic track reproduces the pathology, even attempting a petabyte-scale
array before the repair. This is an approach-construction defect, not evidence
that the HO-Cap hand pose or table calibration is wrong.

The repair keeps the original matched-derivative blend when feasible, then
tries a velocity-matched/zero-acceleration blend, then a zero-derivative
smoothstep. It caps approach duration at 30 seconds and rejects a source
trajectory whose own joint speed already exceeds the robot limits; no velocity
limit is relaxed. New regression tests cover high initial curvature and
overspeed rejection. The full unit suite passes (477 passed, 4 skipped).

A fresh 676-frame smoke began at approximately 01:02 MSK. Its numerical
quality and geometry must be checked before the full batch starts. The batch
remains gated on the smoke plan; this note is **not** a successful retarget
result. The tabletop is an infinite half-space of unverified real height;
objects and the person are not collision obstacles in this IK.

## Second smoke and revised benchmark scope

The second full solve also did **not** produce a plan. The repaired approach
correctly detected that the solved demonstration itself exceeded a joint
velocity limit by a value printed only as ratio 1.0000 under a stricter
threshold than the solver's `NUMERIC_TOLERANCE=1e-7 rad/s`. The approach
check now uses that same *absolute* tolerance and reports the excess in
rad/s if it is genuinely larger. This does not relax the solver constraint.

For the HO-Cap **offline hand-trajectory benchmark**, the generated path from
UR10 home to the first human pose is not part of the recorded task and is
therefore explicitly omitted with `--approach-sec 0`. The test still checks
the recorded trajectory's joint, velocity, infinite-table-plane, and
self-collision margins. No claim is made about a safe motion from robot home
to the first target or physical replay. A third smoke is required before the
28-scene batch can start.

## Third smoke: exact velocity-margin failure

The third 676-frame solve (no home approach) ended after approximately 34
minutes with `velocity=-4.92733e-05 rad/s` in the exact final trajectory
check. No plan was written and the batch did not start. The QP backend had
satisfied its own default approximate feasibility but returned a trajectory
about 0.000049 rad/s over a joint speed limit. This is a numerical QP
residual, not a tabletop or object collision result.

The solver now projects **only small** QP velocity residuals (at most
`1e-3 rad/s`) back onto the convex joint-position and inter-frame velocity
limits, using forward/backward clipping and recording the maximum joint
correction in `velocity_repair_max_rad`. A correction over `1e-3 rad` is
rejected. Position/orientation and objective are recomputed; exact floor and
self-collision checks run **after** projection. Larger violations still fail.
The smoke gate also checks the recorded repair and final velocity margin.
Regression tests cover an excess matching the observed scale and rejection of
a larger violation.

# Scene object extraction over the `40mm-cubes-play` set

Protocol and results for `viki scene` (auto-prompts → SAM 2.1 → semantic 3-D →
`object_models.npz`) run over the ten takes recorded 2026-09-21 under
calibration preset `2.3-diff`. Written 2026-09-21.

## What was run

    docker compose run --rm sam2 scene <episode> --objects 1 --force

**No per-take tuning, no flags beyond `--objects 1`.** That is the point of the
run: before the changes recorded as `bugs.md` 24–26 the stage failed on all ten,
and the question was whether it could be made to pass on defaults rather than on
a hand-picked grid and seed frame per take.

## Cross-check

There is no ground truth here. The tracked object centre — the model centroid
carried through `rotation_world_object` / `translation_world` — is compared with
the centroid of the yellow blob segmented by colour alone (`data/_lab/scene_eval.py`),
sampled every 20th frame in the rig frame.

This is a **self-consistency measure between two estimates that share the
calibration but not the tracker**, not an accuracy figure. Both are centroids of
*visible* surface, so they drift apart whenever the visible surface differs:
when one camera loses sight of the object the colour reference becomes a
single-view centroid and moves by more than the tracker does. Read the column as
an upper bound on tracking error, not as error.

## Results

| take | seed | frames | model (mm) | held | conf | >0.5 | resid | travel | vs colour: med | p90 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-51-18 | 0 | 283 | 47 × 40 × 25 | 100 % | 0.90 | 84 % | 1.9 | 292 | 21.6 | 31.2 | 15 |
| 09-51-39 | 0 | 277 | 48 × 45 × 25 | 100 % | 0.87 | 100 % | 2.2 | 246 | 4.8 | 10.0 | 14 |
| 09-52-05 | 0 | 244 | 45 × 41 × 25 | 100 % | 0.88 | 100 % | 2.1 | 320 | 9.0 | 15.3 | 13 |
| 09-53-54 | 0 | 224 | 52 × 48 × 27 | 100 % | 0.89 | 100 % | 2.0 | 287 | 11.6 | 20.4 | 12 |
| 09-54-07 | 0 | 198 | 48 × 41 × 22 | 100 % | 0.84 | 85 % | 2.5 | 292 | 9.8 | 16.6 | 10 |
| 09-54-21 | 0 | 208 | 52 × 48 × 23 | 100 % | 0.88 | 100 % | 2.1 | 300 | 8.9 | 12.1 | 11 |
| 09-56-36 | 0 | 216 | 46 × 44 × 27 | 100 % | 0.90 | 100 % | 1.9 | 283 | 6.1 | 9.0 | 11 |
| 09-57-48 | 0 | 245 | 45 × 41 × 23 | 100 % | **0.48** | **46 %** | **4.6** | 283 | 5.2 | 12.5 | 13 |
| 09-58-04 | 0 | 268 | 45 × 44 × 36 | 100 % | 0.82 | 99 % | 2.7 | 271 | 4.7 | 9.6 | 14 |
| 09-58-19 | 0 | 203 | **61 × 51 × 51** | 100 % | 0.86 | 91 % | 2.2 | 264 | 6.7 | 10.7 | 11 |

`resid` is the median per-frame model-fit residual in mm; `travel` the path
extent of the object centre in mm; `n` the number of cross-checked frames.

10/10 produced a model, all held the track on every frame. Median across takes
of the per-take median agreement: **7.8 mm**.

## Reading the outliers

- **09-51-18, 21.6 mm** is the worst agreement and is at least partly the
  reference degrading rather than the tracker: in this take the Kinect stops
  seeing the cube's yellow past frame ~170 (measured: 0 yellow pixels at frames
  180 and 240 against 300–900 earlier), so the colour centroid there comes from
  the RealSense alone. Not separated from a genuine tracking error — to do that,
  re-score restricted to frames where both cameras see the object.
- **09-57-48** carries a much weaker self-report — confidence 0.48, only 46 % of
  frames above 0.5, the highest residual at 4.6 mm — while its agreement with
  colour is among the best at 5.2 mm. The confidence signal is pessimistic here
  and does not track the cross-check; do not gate on it without understanding
  why.
- **09-58-19** is the take rescued by the colour split, and its model is visibly
  looser: 61 × 51 × 51 mm against 45–52 × 40–48 × 22–36 for the rest. The rescue
  starts from a saturated subset of a welded component, so it plausibly absorbs
  some neighbouring surface. Agreement is still 6.7 mm.

Model extents of 45–52 × 40–48 × 22–36 mm for a 40 mm cube are what a rigid
model built from two or three visible faces should look like; the short axis is
the unseen depth. An earlier version of this note left it there, which was too
generous to the algorithm — part of that depth *is* observed and refused. See
below.

## Why the short axis stays at 22–36 mm

The model is not frozen at bootstrap: a core is consensus-voxelised from the
first `bootstrap_frames` (60) frames, and `_grow_shell` then passes over the
whole take. A quarter of a typical model is shell (take 09-51-18: 134 core +
47 shell). But the shell pass can only *thicken* the surface it already has,
and two things bound it. Measured by rebuilding 09-51-18's model from the
existing masks under relaxed rules (`data/_lab/diag_model_growth.py`):

| rule | model (mm) | core+shell | conf | resid |
|---|---|---|---|---|
| default | 47 × 40 × 25 | 134 + 47 | 0.90 | 1.9 |
| `shell_distance_m` 18 → 50 mm | 47 × 40 × 25 | 134 + 47 | 0.90 | 1.9 |
| … and accept contact frames | 55 × 45 × **34** | 134 + 67 | 0.26 | 5.8 |
| … and drop `shell_max_core_ratio` | 63 × 48 × 37 | 134 + 124 | 0.35 | 4.9 |

So the 18 mm shell reach is **not** what binds — relaxing it changes nothing.
What binds is `if contact[frame_index]: continue`: the shell pass discards every
frame where the object is in contact with the operator, which is 51–69 % of
every take in this set, and is precisely the interval during which the object is
being moved and could present a face it had not shown. Removing that gate takes
the short axis from 25 to 34 mm — but naively, at a real cost: confidence 0.90 →
0.26 and residual 1.9 → 5.8 mm, and with the budget also removed the model
overshoots to 63 mm on a 40 mm cube, i.e. it starts absorbing hand and tabletop.
A blunt relaxation is a regression, not a win.

The deeper reason a 40 mm cube never becomes 40 × 40 × 40 is that **it barely
rotates**. Total orientation swept per take:

| take | 09-54-07 | 09-54-21 | 09-52-05 | 09-51-18 | 09-53-54 | 09-56-36 | 09-51-39 | 09-57-48 | 09-58-19 | 09-58-04 |
|---|---|---|---|---|---|---|---|---|---|---|
| swept | 17° | 26° | 32° | 36° | 35° | 56° | 66° | 66° | 93° | 102° |

The cube is slid across the table, not turned over. A face that never turns
toward a camera cannot enter any model, however the shell rules are set, and the
trend shows: 102° swept gives the most complete short axis at 36 mm, 17° gives
the least at 22 mm.

**What a real fix looks like** (not implemented): the contact gate is blunt, and
the information to replace it is already there. The pipeline segments the
operator as its own tracked instance, so on a contact frame the shell pass could
keep candidates and reject only those within `contact_distance_m` of the
operator's points, while raising the cross-frame support a candidate needs. That
uses what contact actually tells us — *which* points are suspect — instead of
discarding the frame.

## What did not earn its keep here

`frame_attempts` — the seed-frame search — fired on none of the ten: every take
resolved at frame 0 once the clustering grid followed the object scale. It stays
because the failure it covers (a hand resting on the object at frame zero) is
real and cheap to guard, but this set does not demonstrate it.

## The pose wobbles; it does not flip

Asked why the object's axes jump. They do not — measured over all ten takes,
**not one frame-to-frame rotation step lands near 90° or 180°**, and the maximum
step in every take sits exactly on `max_rotation_step_deg = 12.0`, the config's
per-frame clamp. Symmetry aliasing is ruled out; so is a degenerate axis, since
`rotation_information` (the inertia spectrum of the matched points, normalised)
has its smallest component at 0.32–0.47, and the per-frame rotation axis has no
preferred direction in the model frame — median |component| 0.371 / 0.436 /
0.344 along the longest / middle / thinnest principal axis of 09-51-18's model.

What is actually wrong is jitter, and it is large:

| take | step p50 | total swept | path / swept | matched coverage p50 |
|---|---|---|---|---|
| 09-51-18 | 2.40° | 36° | 22.8× | 35 % |
| 09-51-39 | 2.17° | 66° | 11.2× | 43 % |
| 09-52-05 | 2.85° | 32° | 23.8× | 49 % |
| 09-53-54 | 3.02° | 35° | 22.1× | 30 % |
| 09-54-07 | 2.29° | 17° | **27.7×** | 56 % |
| 09-54-21 | 2.52° | 26° | 23.1× | 45 % |
| 09-56-36 | 2.83° | 56° | 13.5× | 50 % |
| 09-57-48 | **0.01°** | 66° | 5.4× | 48 % |
| 09-58-04 | 2.88° | 102° | 9.4× | 43 % |
| 09-58-19 | 2.95° | 93° | 7.1× | 37 % |

`path / swept` is the summed per-frame rotation divided by the net rotation
achieved: the estimate travels **7 to 28 times further than the object turns**.
At 30 fps a 2.4° median step is ~72°/s of wobble on an object whose entire
rotation over the take is 17–102°.

The cause is visible in the last column. The model is a partial shell and only
30–56 % of it is matched in a given frame; *which* part is matched changes as
the hand occludes different faces, so ICP fits a different subset each frame and
lands on a different optimal rotation. Two rig facts feed it: the RealSense's
depth noise, and the ~10–25 ms software sync between the cameras (`bugs.md` 3),
which on a moving object makes the two halves of one observation disagree.

Nothing filters this. The only temporal machinery in `_track_model` is the
per-frame clamp and the confidence/reacquire test — a clamp caps an excursion
but does nothing to a 2–3° random walk.

09-57-48 is the opposite failure and should not be read as the best row: a
median step of 0.01° means the pose is not updating at all, consistent with its
confidence of 0.48 and 46 % of frames above 0.5.

## Smoothing the pose, and what it does not fix

`dsp.smooth_rotations` filters a rotation track without leaving SO(3): each
sample is expressed as a rotation vector relative to the track's chordal mean,
those 3-vectors are Savitzky–Golay filtered like any other trajectory, and the
result is mapped back, re-centring once on the smoothed mean. `object_model`
applies it (and plain SG on the translation) after the final track, then
**re-scores the smoothed pose against the same observations** — the quality
columns have to describe the pose that ships, not the fit it replaced. The raw
fit stays in the artifact under `*_raw`. Schema bumped to
`viki_object_models_v2`, since the meaning of `rotation_world_object` changed
under an unchanged name.

Window sweep, median across the ten takes (`data/_lab/diag_smoothing.py`):

| window | step p50 | swept | path | conf | resid (mm) | travel (mm) |
|---|---|---|---|---|---|---|
| 0 (off) | 2.68° | 45.7° | 746° | 0.88 | 2.2 | 285 |
| **15 (default)** | **0.70°** | 43.2° | **205°** | **0.87** | **2.3** | 283 |
| 31 | 0.45° | 43.0° | 142° | 0.85 | 2.4 | 282 |
| 61 | 0.38° | 41.4° | 111° | 0.78 | 2.7 | 279 |

The per-frame wobble drops by **74 %** and the cumulative rotation path by
**73 %**, at a cost of 0.01 confidence, 0.1 mm residual and 2 mm of translation
travel — the lift-carry-lower motion survives intact. Wider windows keep buying
smoothness more slowly and start costing real fit: at 61 frames take 09-54-07
falls from confidence 0.84 to 0.58 and its residual rises 2.5 → 4.3 mm. 15
frames is half a second at 30 fps, which is about as long as turning a cube over
takes, so it is also the widest window that cannot smear a real manipulation
rotation. That, rather than the sweep, is why it is the default.

**What smoothing does not fix, and it is the larger error.** `swept` — the net
orientation change over a take — barely moves: 45.7° → 43.2°. The operator
confirms the cube was lifted, carried and set down and never turned, so the
*correct* swept value is near zero and all 13–102° of it is error. It is not
jitter but slow drift, and a low-pass filter cannot touch it. The mechanism is
the same partial-model one: with 30–56 % of a two-or-three-faced model matched
per frame and the matched part shifting as the hand occludes faces, ICP walks
the model slowly around the object's surface. Candidates to attack it, none
implemented: weight the per-frame fit by coverage so thin-support frames move
the pose less; anchor the track to a reference frame instead of chaining
frame-to-frame; or exploit the flat-on-table prior between contacts.

## Reproducing

`data/_lab/run_scene_all.sh` runs the ten; `data/_lab/scene_eval.py` scores them.
Diagnostics used to find the three defects: `diag_prompt_scan.py`,
`diag_prompt_tune.py`, `diag_find_cube.py`, `diag_cube_component.py`,
`diag_candidate_scan.py`, `diag_show_components.py`, `diag_sat_split.py`.
Model growth and pose jitter: `diag_model_growth.py`, `diag_axis_jumps.py`,
`diag_axis_wobble.py`.

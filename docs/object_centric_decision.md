# Object-centric representation — where it stands, and what to do

Conclusion of the reconnaissance in `object_centric_recon.md`. This is a
recommendation, not a change: no code was touched.

## The short version

**Do not build the object-pose estimator yet. Record the data that would make it
testable first.** The dataset cannot currently validate object-centricity at
all, and the estimator we could build today is unfalsifiable, uncertain in
exactly the degree of freedom that matters, and fails outright on one scene in
five.

## Why the dataset cannot decide it

`data/datasets/new-dataset` holds **12 episodes and 12 distinct tasks**. Not one
task is recorded twice.

| | |
|---|---|
| picking, hand-move, flex, riged, pick_up, synccheck, pick_up_sync | one each |
| pick_up_u, pyramid, block-push, cup_grab, move-shipok | one each |

The object-relative form exists to generalise **across object placement**
(§3.6: "the object-relative form is what transfers"). With one demonstration per
task there is nothing to generalise across, nothing to transfer to, and no way
to tell a correct object frame from a wrong one. Any estimator built now would
be unfalsifiable on this data.

## What the one available proxy says

Two scenes both perform "reach to an object resting on a table and grasp it" —
move-shipok and pick_up_u — with different objects, operators and calibrations.
Resampling the last ~520 mm of each approach by path length and comparing them:

| form | mean separation | separation at the object | direction of travel |
|---|---|---|---|
| workspace-anchored | 81 mm | 94 mm | 61° |
| object-relative (position) | 88 mm | **54 mm** | 61° |

Object-relative helps **only near the object** — 94 → 54 mm at the contact end —
and is slightly worse far from it, where the approach depends on where the hand
started rather than on the object.

The direction of travel is **identical in both forms**, and necessarily so:
subtracting a constant offset does not change a derivative. **A translation-only
object frame cannot change the shape of a trajectory at all.** It re-datums the
position and nothing else.

Shape is what transfers between placements. Shape is governed by the object's
*orientation*, and that is precisely the quantity the reconnaissance found
unreliable: ~10° of spread between three independent estimates, on an objective
whose registration residual is flat across ±60° and therefore cannot tell a
correct rotation from a wrong one.

## What it would cost to be wrong

Three failure modes, all measured rather than supposed:

- **Segmentation fails on cup_grab.** Saturation median 18, 3.4 % of pixels
  above S = 120. Colour, which carried every other scene at 100 % detection,
  finds nothing. Geometry separates the mug but cannot keep its identity between
  frames.
- **Orientation is not identifiable from two viewpoints.** The accumulated
  before- and after-models of the *same rigid object* have principal spans of
  14.6/11.9/8.1 cm and 16.7/12.2/7.4 cm; they are partial shells, and the ICP
  objective is flat over a 60° basin.
- **The hand cannot bootstrap it.** Folding the carry into the hand frame grows
  the model without saturating (4 546 → 16 354 voxels) and inflates its spans:
  the ±8 mm grasp drift smears the shell rather than completing it.

## What to record

One afternoon of recording converts this from unfalsifiable to measurable:

**The same task, the same object, 5–6 takes, with the object deliberately moved
between takes — translated across the workspace and rotated, including at least
one placement near 90°.** Colour-distinctive object, so segmentation is not the
variable under test. Calibration unchanged across the set, so the world frame is
common.

That set makes three things measurable that cannot be measured now:

1. **Whether `T^O_H` is consistent across placements.** Its spread is a
   ground-truth-free quality metric of the whole idea, in the same spirit as the
   bone-length coefficient of variation that settled E13 — the demonstrations
   are supposed to be the *same motion relative to the object*, so their spread
   in the object frame is the error.
2. **How much variance the object pose actually explains** — the workspace-
   anchored and object-relative forms can be scored against each other on held-
   out placements instead of argued about.
3. **Whether orientation is needed at all.** If the rotated placements are
   reproduced acceptably by a position-only object frame, then `T^W_O` should be
   `(x, y, yaw)` in the table plane, or even position-only, and the
   ill-conditioned freedoms are never estimated. §3.6's full SE(3) is
   over-parameterised for tabletop pick-and-place either way.

## What to do in the meantime

Nothing in the pipeline needs to change, and two things should be recorded as
known:

- `represent.py` keeps returning `None` and `status.json` keeps saying
  `object_relative=false`. That is the honest state, and it is already honest.
- §3.6 should state that the object model and its frame convention must be
  exported alongside the trajectory. Equation 3.7 transfers by
  `T^R_O' · T^O_H · T^H_E`, which requires the consumer to express the new
  object pose in the same arbitrary frame the recording used. The chapter says
  both representations are exported; without the frame convention the
  object-relative one cannot be consumed.

Two findings from this work are worth acting on independently of the object
question, because they affect the shipped pipeline now:

- **The floor datum is wrong** — the table is 19 mm below the board plane on two
  calibrations and 33 mm above it on three. See the E14 correction.
- **The grasp anchor should be derived, not configured** — measured per
  demonstration it is the thumb–index midpoint at 1.6 mm on a pinch and the
  middle-finger MCP at 16.9 mm on a whole-hand grasp, and its residual is a free
  rigidity metric. `target_position_anchor` currently offers only
  `pinch_center` and `wrist`, and neither fits pick_up_u.

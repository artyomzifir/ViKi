# Object-centric recording — reconnaissance

Investigation of how to produce the per-frame object pose `T_world_obj` that
`viki/prepare/represent.py` has been waiting for since it was written. The
maths there is complete and correct; `object_relative()` computes
`inv(T_world_obj) @ T_world_hand` and is called with `None`.

No code was changed. Everything below is measurement on the five existing
recordings.

## 0. What the recordings actually contain

Looked at the video before reasoning about it — which should have come first,
and did not.

| scene | date | calibration | rig | task |
|---|---|---|---|---|
| move-shipok | 09-09 | `skrip` | ChArUco **flat on the table** | blue L-bracket picked up and set down elsewhere |
| cup_grab | 09-04 | `calib-v1-checkpoint` | ChArUco **vertical, off to the left** | white mug on a bare desk; a second mug in the other hand |
| block-push | 09-04 | `calib-v1-checkpoint` | same | blue block **pushed flat with the palm** — no grasp at all |
| pyramid | 09-04 | `calib-v1-checkpoint` | same | stack of red/purple/blue blocks |
| pick_up_u | 09-03 | `arteom_dont_kill_me_pls` | **no board in view** | red U-block lifted from a bare desk |

Three rig configurations, three calibrations, three different operators. The
scenes are not a homogeneous set, and several conclusions drawn earlier in
`robust_retarget_experiments.md` implicitly assumed they were.

## 1. Where the table actually is

The retarget floor is `z >= 0` in the calibration frame, i.e. the **board**
plane. The table is not that plane. Measured as the lowest dense horizontal
layer, over two frames per scene:

| scene | calibration | table z | slab spread | tilt |
|---|---|---:|---:|---:|
| move-shipok | `skrip` | **−19 mm** | 11.5 | 3.0° |
| pick_up_u | `arteom_dont_kill_me_pls` | **−19 mm** | 10.8 | 0.7° |
| cup_grab | `calib-v1-checkpoint` | +33 mm | 10.4 | 1.5° |
| block-push | `calib-v1-checkpoint` | +32 mm | 10.0 | 1.6° |
| pyramid | `calib-v1-checkpoint` | +35 mm | 9.8 | 1.1° |

Two groups, each internally repeatable to ±2 mm.

**Where the board was calibrated lying on the table, the table surface sits
19 mm *below* z = 0 — the board's own thickness.** Objects placed on the bare
table therefore sit ~19 mm lower than the calibration origin implies. Where the
board stood vertically off to the side, z = 0 has no relation to the table at
all, and it happens to fall 33 mm below it.

**Consequences for the floor constraint, which is a safety constraint:**

- `skrip` / `arteom` scenes: the floor is **19 mm too high**. The robot is
  forbidden the lowest centimetre-and-a-half in which the hand actually worked.
- `calib-v1-checkpoint` scenes: the floor is **33 mm too low**. The constraint
  that exists to stop the gripper entering the table permits it to enter by
  33 mm, on three of the five scenes.

This also corrects E14. Block-push targets sit a median 36 mm above z = 0, but
the table there is at 32 mm — so the demonstration happened **4 mm above the
table surface**, not 36 mm above it. Combined with the video, which shows a
flat-palm push rather than a grasp, the conclusion that the scene has no
parallel-jaw equivalent is strengthened, not weakened.

The floor should be the measured table plane, per episode, not the board plane.

## 2. Which signals find the object

Tried in order, on move-shipok.

**The fused point cloud is the wrong substrate.** It is capped at 40 000 points
over the whole room. In the tabletop band, 19 678 of 36 458 points are above
300 mm — the operator, not the table and not the object. Calibration-time
background subtraction removes the table but not objects placed on it
afterwards.

**Voxel occupancy over time does not work.** It cannot distinguish *absent*
from *occluded*: a static object disappears from its voxels whenever an arm
passes in front. The artifact itself is fine — consecutive frames agree to
Jaccard 0.83 at a 15 mm leaf — the scene genuinely changes because the operator
moves through it.

**Depth-image background modelling is the right level**, since occlusion is
explicit there: 81 % of the sensor gets a static model. But a persistently
present operator does not average out of the median, and a high percentile only
partly helps.

**Colour is an order of magnitude stronger than any of it.** The bracket is
saturated blue; a single HSV gate finds it in **every** sampled frame of both
cameras — 7 000–13 000 px on kinect_0, 3 000–7 000 on kinect_1, blob extent
~110×140 px. Nothing geometric came close.

That is scene-specific, and honestly so: it is a *protocol* assumption of the
same kind as the ChArUco board. The other scenes support it — the blocks are
red, purple and blue — but cup_grab's white mug on a white desk would defeat it,
and pyramid has several coloured objects with no rule yet for which one is
"the" object.

## 3. How the maths behaves

Blue blobs lifted through the k4a calibration into the calibration frame, with
a two-pass median filter (blue pixels landing on far depth otherwise inflate the
extent to two metres).

**Translation is excellent.**

| phase | object centre | stability |
|---|---|---|
| frames 20–410 | (+0.011, +0.008, +0.069) | **~1 mm over 400 frames** |
| 440–650 | moving | carried |
| 710–860 | (+0.023, +0.137, +0.064) | **<1 mm over 150 frames** |

Net displacement 129 mm, essentially in the board plane.

**The rigid-grasp assumption holds.** The object centre expressed in the hand
frame over the carry: mean (+144, +34, +133) mm, **σ = (8, 8, 8) mm**, maximum
deviation 24 mm. So the phase decomposition — static, rigid carry, static — is
supported by measurement, and during the carry the object pose can be
propagated from the hand.

**Rotation is where it breaks down, and the failure is instructive.**
Registering the before-shell to the after-shell gives a residual of 2.7 mm
median, which looks excellent. It is not evidence of a correct rotation: after
ICP refinement the yaw landscape is **flat across ±60°** (2.69–2.86 mm), and
against a carry-accumulated model it is flat across the full 360°. A small
residual here means the objective cannot discriminate, not that the answer is
right. Anyone who runs ICP and trusts the residual will get a confident wrong
number.

The cause is visible in the shapes: the accumulated "before" and "after" models
have principal spans of 14.6/11.9/8.1 cm and 16.7/12.2/7.4 cm. They are the
*same rigid object*, so identical spans are required; they differ because each
is a partial shell seen from two fixed viewpoints on one side.

**Building the model in the hand frame makes it worse, not better.** Folding
the carry into the hand frame — the operator does turn the object in front of
both cameras — produces a model that *grows without saturating* (4 546 → 16 354
unique 4 mm voxels) and whose spans inflate to 15.9/13.3/8.6 cm. A rigid body
cannot do that. The hand pose is not accurate enough to serve as a registration
prior: the ±8 mm grasp drift plus hand-orientation error exceeds the voxel size
and smears the shell.

**Constraining to the plane fixes it.** An object at rest on a known table has
**three** degrees of freedom, not six. Its in-plane principal direction is well
determined — anisotropy 1.81 before and 2.11 after — and gives a yaw change of
**+1.1°**, against 5.8° from 3-D ICP and 11.8° carried by the hand. All three
agree that the object barely rotated; their spread, about 10°, is the honest
current uncertainty. The ill-conditioned out-of-plane freedoms simply vanish if
they are not estimated.

`T^W_O` as full SE(3) is over-parameterised for tabletop pick-and-place. That is
a real gap in §3.6 of the thesis.

## 4. Side findings

**The gripper signal is not a contact detector.** At frame 230 it reads 0.00 —
fully closed — while the object sits untouched on the table. Contact must be
taken from the object's own motion; measured that way it leaves rest at frame
~420 and settles at ~700.

**The object frame convention must be exported with the trajectory.** Equation
3.7 transfers by `T^R_O' · T^O_H · T^H_E`, which requires the consumer to
supply the new object pose *in the same frame* the recording used. That frame
is defined by however the model was built and is otherwise arbitrary. §3.6 says
both representations are exported but not that the object model and its frame
convention must be too, without which the object-relative form is unusable
downstream.

## 6. The grasp anchor is not a constant, and it can be measured

Follow-up on a second scene, which corrected a conclusion from §3.

The rigidity figure reported above — σ = 8 mm — used the **centroid of the
visible object points** against `hand_fit_positions`, the wrist. Both halves of
that are wrong, and it only looked good because move-shipok happens to carry the
object almost without rotating it.

On pick_up_u the same measure gives σ = (48, 32, 40) mm and a maximum deviation
of 120 mm. Decomposing it: the *distance* |object − wrist| itself varies with
σ = 34 mm, and the direction in the hand frame scatters by 29.7° on average
and 68° at worst. Both components fail, so it is not an orientation-only
artifact.

The timeline says why. |object − wrist| runs 162 → 72 → 190 mm across a single
carry, while the gripper opening moves 0.60 → 0.30 → 0.61: **the fingers are
flexing.** The wrist cannot be the grasp anchor, because the object is held by
the fingers and the wrist-to-finger distance is a free variable.

Testing every hand landmark as the anchor, scored by the standard deviation of
its distance to the object over the carry:

| scene | grasp | best anchor | σ | runner-up |
|---|---|---|---:|---|
| move-shipok | pinch on a small bracket | **thumb–index midpoint** | **1.6 mm** | index MCP, 2.3 mm |
| pick_up_u | whole hand over a block | **middle-finger MCP (palm)** | **16.9 mm** | ring MCP, 17.9 mm |

The pinch anchor scores 82 ± 1.6 mm on move-shipok — the grasp there is rigid to
**under two millimetres**, five times better than the wrist-based figure first
reported. On pick_up_u the pinch anchor does not even reach the top five; every
palm landmark beats every fingertip, which is exactly what a whole-hand grasp
should do.

Two consequences.

**The anchor should be derived per demonstration, not configured.** Retarget
currently picks `target_position_anchor` from config, between `pinch_center` and
`wrist`. Neither is right for pick_up_u. The measurement above needs no ground
truth: choose the hand landmark whose distance to the object is most constant
while the object is being carried.

**The same minimisation yields a free quality metric.** Its residual *is* the
rigidity of the grasp — 1.6 mm where the object is pinched and held steady,
16.9 mm where it sits loosely in a palm. A demonstration whose best anchor still
scores tens of millimetres is one where the object moved in the hand, and that
is worth knowing before the trajectory is retargeted as though it had not.

Checked that the pick_up_u residual is not a segmentation artifact: on the
sparsest 30 % of frames the deviation is 9.5 mm against 9.4 mm elsewhere.

Colour detection reproduced on this scene without changes — 100 % of sampled
frames, 2 000–3 400 points — on a different object, operator and calibration.
The table sits at −19 mm there too, matching the other board-on-table
calibration to the millimetre.

## 7. Where this points

Estimate the object pose **in the table plane** — `(x, y, yaw)` — rather than in
SE(3). Segment with colour plus depth. Take yaw from the in-plane silhouette
axis, not from 3-D ICP on partial shells. Take the phase boundaries from the
object's own motion, not from the gripper.

The table plane itself must be measured per episode rather than assumed to be
the board plane, which fixes the floor constraint at the same time.

Yaw uncertainty is currently ~10° and will not improve from this data by better
fitting; it needs a new signal — **a third camera at a different azimuth**,
which attacks the partial-shell problem directly, or a fiducial on the object.

Nothing here is validated beyond move-shipok. cup_grab in particular will
defeat the colour route, and pyramid needs a rule for which object is the
subject of the demonstration.

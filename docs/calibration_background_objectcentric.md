# Calibration ↔ background subtraction ↔ object-centric — design notes

Status: **problem report + proposed workflow**, no code changed yet beyond the
SQPNP solver hardening (`fix(calibration): global SQPNP for board pose`).

## 1. Symptom

After calibrating the 2-Kinect rig with preset `popka` and recording
(`new-dataset/2026-09-03_10-10-07`, task `riged`), the fused point cloud shows
**two clouds of the same scene, rigidly offset** by roughly 5–10 cm plus a
10–20° rotation, instead of one merged scene.

## 2. What was checked (2026-09-03)

| Check | Result |
|---|---|
| Re-solve `popka` extrinsics with SQPNP + LM (global, no local basin) | **Bit-identical** to the stored `SOLVEPNP_ITERATIVE` result (Δ 0.0 mm / 0.0°) |
| Per-camera reprojection RMS | 0.8 px (kinect_1), 1.0 px (kinect_0) — nominally excellent |
| `color_deproject_maps` (fast affine map used by `cloud.py`) vs per-pixel SDK `k4a_calibration_2d_to_3d` | **Bit-exact** (0.00 mm). Depth→colour path is correct. |
| Two per-camera depth clouds, cropped to the workspace box | Centroid gap ~260–280 mm; NN median 120–190 mm |
| ICP between the two per-camera clouds | Converges with ~50–110 mm translation + rotation; residual NN 9–83 mm (≈ sensor noise) |
| Orientation spread across the 10 `popka` capture sets, per camera | **0.08° / 0.12°**; board-centre spread **< 1.1 mm**; range of board distance **1 mm** |

## 3. Root cause

**The 10 "capture sets" per camera are all the same static board pose.** The
operator pressed *Capture all* ten times without moving the ChArUco board.

A single static, near-coplanar target is the textbook cause of an **unresolvable
PnP pose ambiguity**:

- The board's out-of-plane tilt is only weakly constrained by one fronto-parallel
  view.
- Reprojection error is almost identical for the true pose and a tilt-perturbed
  ("mirror") pose — so *minimising reprojection does not pick the physically
  correct pose*. SQPNP being globally optimal doesn't help: it finds the lowest
  reprojection error, which is the wrong pose.
- Each camera independently resolves its own board→camera rotation and lands on a
  slightly different tilt interpretation. Composed through the board into the
  shared world frame, the two cameras' clouds end up rigidly offset.

The depth pipeline, the SDK calibration blob deprojection, and the cloud-building
code are all correct. **No solver can fix this from the current data** — it needs
new board observations.

Why the board was kept static: the pipeline's **background subtraction** grabs an
empty-scene depth median per camera, and the board was left in one place so that
median would be stable and the board itself would be subtracted out. That coupling
is the design mistake — see §4.

## 4. The three-way tension

Three requirements pull on the ChArUco board during setup. They only conflict
because everything is done in one shot.

| Requirement | What it needs from the board |
|---|---|
| **Accurate extrinsics** (clouds coincide) | board **moved** through 6–15 distinct poses — tilt ±20–40°, near/far, filling different parts of the frame |
| **Background subtraction** | a stable *static rig/table*, **not** the board — the board should be **out of frame** for the empty-scene grab |
| **Object-centric dataset** | board irrelevant during recording; hand/EE pose expressed in the **object** frame `T^O_H(t)`, not the world frame |

Key realisation: these are **separable in time**. "Board static" was only ever
tied to "background stable" by accident.

## 5. Proposed workflow

```
1. Calibrate       — board moved through N distinct poses            → extrinsics.json
2. Remove the board from the scene
3. Grab background — empty scene, NO board in frame                  → <dev>_bg.npz
4. Record          — human + object, no board
5. (offline) object-centric conversion                              → T^O_H(t)
```

Step 2 is the whole fix for the background coupling: if the board is gone before
step 3, it is not in the recording and never needs subtracting. Background
subtraction then only removes the genuinely static table / rig / walls.

## 6. Calibration — options

### Option 0 — quick test, zero code
Re-calibrate once, board still in one spot, but **strongly tilted (~40°) and
filling the frame** (lean it on a box). One well-tilted, frame-filling view is
far less ambiguous than a fronto-parallel one. If the two clouds then coincide,
the entire current workflow is kept — only the board angle changes. **Try this
first.**

### Option 1 — multi-pose with an anchor (recommended proper fix)
Keep a **"home" board pose** (flat on the table at a taped mark) that defines the
world frame — Z up out of the table, convenient for the viewer, the workspace
AABB, and as an interim object-frame proxy. Then require the board to be **moved**
between capture sets; the app **rejects** a set whose pose is within ~2° / 2 cm of
an existing one, forcing real diversity.

Solve as multi-view (two equivalent routes):

- **Mini bundle:** board pose free per view, camera extrinsics shared, world
  locked to the home view. `scipy.least_squares` with an analytic Jacobian — the
  machinery already exists in `viki/perception/hand_fit.py`.
- **Pure OpenCV:** `T_camᵢ_world = T_camᵢ_camref · T_camref_world`, where
  `T_camref_world` comes from the home view of the reference camera and
  `T_camᵢ_camref` from `cv2.stereoCalibrate` over all co-visible moved views.

### Option 2 — world = reference camera
Drop the board-as-world idea. World = `kinect_0` optical frame; the board is a
pure calibration target (many co-visible poses → `stereoCalibrate` →
`T_kinect1_kinect0`). Robust and standard. Downsides: world axes tied to a camera
mount (less intuitive viz / workspace box); bumping `kinect_0` invalidates
everything. **But** in a fully object-centric pipeline the world frame cancels
out downstream, so this becomes attractive once §7 lands.

### Also considered
- 3D calibration object (ChArUco cube) — non-coplanar, no single-view ambiguity,
  but needs fabrication.
- Use the visible table plane in each camera's depth to constrain the board
  normal — clever, uses existing data, fiddly.

## 7. Object-centric layer (separate track)

Need an object pose `T^W_O(t)`, then `T^O_H(t) = T^W_O(t)⁻¹ · T^W_H(t)` and the
world frame drops out of the dataset. Sources, by increasing cost:

1. **Fiducial on the object** — small ArUco marker / cube. Trivial maths; marker
   is visible in the recording.
2. **Static start pose** — object doesn't move until grasped; one annotation of
   its initial pose, propagate until contact.
3. **Mesh + ICP** — register a scanned object mesh into the fused cloud
   (needs the clouds already coinciding, i.e. §6 done).
4. **Learned 6DoF** (FoundationPose etc.) — heavy, GPU, separate integration.

Until a tracker exists, the §6 "home" board pose is a serviceable proxy: place
the object at a known offset from the board home.

The paper already specifies `T^O_H` (§3.6) with no tracker implemented — this is
the same gap.

## 8. Validation gate

After calibration, compute the two-camera cloud agreement on a live empty-scene
frame (NN median / ICP residual) and show **red/green before recording is
allowed**. This would have caught the current problem at setup instead of after
several record cycles.

## 9. Recommendation

1. **Now:** run Option 0 (tilt the board, re-calibrate) to confirm the diagnosis
   and possibly unblock immediately.
2. **Workflow:** adopt §5 — grab background with the board removed — regardless of
   which calibration option wins.
3. **Proper fix:** Option 1 (multi-pose + anchor + pose-diversity rejection in the
   capture UI) + the §8 validation gate.
4. **Parallel:** pick an object-pose source from §7 (start with fiducial-on-object
   or static-start) — this is what makes the dataset actually object-centric and
   removes the world-frame sensitivity.

## 10. Open decisions

- **World frame:** table-anchored "home" pose (recommended now — doubles as viz
  frame + interim object proxy) vs camera-anchored (simpler once object-centric).
- **Object pose source:** fiducial / static-start / mesh+ICP / learned.
- **Calibration change scope:** Option 0 only, or commit to Option 1 now.

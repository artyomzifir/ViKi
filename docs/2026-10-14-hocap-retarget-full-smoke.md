# HO-Cap full-episode retarget smoke (2026-10-14)

## Protocol

These are offline geometric IK results on independent copies of two HO-Cap
episodes, not a robot execution or a simulation grasp. The input is the
existing `cln.npz` in `hocap_tag_1`; each plan uses UR10/Robotiq 2F-85, robot
base `[0,-1,0]`, infinite table plane `z=0`, no home approach and no cube grasp.
The input requests hand-fit but does not contain `hand_fit_*` arrays, so the
retargeter explicitly falls back to landmark poses. The table height has not
been independently measured; object and human collisions are not constrained.

The left-hand experiment uses palm-to-TCP RPY
`[-175.675,15.45,-47.922]°` (diagnostic local-X half-turn of the right-hand
mapping). The right-hand experiment keeps `[4.325,15.45,-47.922]°`.
`robot_home` and `zero` below refer to the IK reference/initial joint vector,
not a commanded approach to the first pose. All rows except the stated speed
diagnostic use the unscaled URDF velocity limits.

| Episode/profile | Frames | Position RMSE / p95 (mm) | Orientation RMSE (°) | Min URDF speed margin (rad/s) | Min floor margin (mm) | Min self-collision margin (m) | Converged |
|---|---:|---:|---:|---:|---:|---:|---|
| Left, default mapping, `robot_home` | 676 | 304.50 / 843.78 | 83.98 | n/a | 0.0002 | 0.0017 | no |
| Left, axis-flipped TCP, `robot_home` | 676 | 254.84 / 772.68 | 41.15 | 0.0000 | 0.0001 | 0.0393 | yes |
| Left, axis-flipped TCP, `robot_home`, 2× velocity bound | 676 | 248.87 / 743.00 | 46.03 | −0.3460 | 0.00001 | 0.0393 | yes |
| Left, axis-flipped TCP, `robot_home`, Huber δ=0.5 | 676 | 159.12 / 464.19 | 30.68 | 0.0000 | 0.0006 | 0.0393 | yes |
| **Left, axis-flipped TCP, `zero`** | **676** | **16.21 / 31.88** | **16.86** | **1.5521** | **0.0010** | **0.0412** | **yes** |
| Right, original TCP, `robot_home` | 831 | 223.86 / 614.45 | 68.81 | 0.0000 | 0.0003 | 0.0308 | yes |
| **Right, original TCP, `zero`** | **831** | **24.12 / 50.30** | **28.24** | **1.3826** | **0.0001** | **0.0393** | **yes** |
| Left, TCP turned 90° about its local Z, `robot_home` | 676 | 287.69 / 852.53 | 45.34 | 0.0000 | 0.0001 | 0.0029 | yes |

The left input is `subject_1_20231025_165502` and the right input is
`subject_1_20231025_170105`. The `zero` left-hand plan passes the existing
676-frame numerical smoke gate: position RMSE ≤30 mm, p95 ≤50 mm,
orientation RMSE ≤45°, convergence, unscaled URDF speed, floor and
self-collision margins. The right-hand control exceeds the same 50 mm p95
threshold by **0.30 mm**; its highest errors are at the first and last frames,
where the tracked pinch point is only 7–15 mm above the nominal table and the
modelled contact centre cannot descend as far while keeping all movable
collision bodies above the plane. This is a geometric limitation, not evidence
of successful contact or of a correct table-height measurement.

## Why turning the tool was not enough

Four left-hand rotations about the downward tool axis were compared on two
60-frame windows. At frames 330–390, local-Z turns of 0°, 90°, and 180° gave
position RMSE 21.2, 20.4, and 20.6 mm respectively; 270° gave 74.4 mm. At
frames 520–580, the same turns gave 69.8, 18.2, 19.0, and 58.5 mm. Thus a
90° turn helps that local window, but the **full** 676-frame 90° run with the
same `robot_home` initialization still has 287.69 mm position RMSE. It is not
a whole-episode fix and may rotate the jaws away from a useful grasp face.

The first 500 frames of the full left-hand axis-flipped `robot_home` plan track
at roughly 8–16 mm mean position error per 100-frame block. After frame 520
the solver moves into a very different joint branch, with errors over 1 m near
frames 550 and 635 and joint speed reaching the URDF bound. The same 540–560
target window solved independently at 14.38 mm RMSE with 1.55+ rad/s speed
margin. Raising the speed bound or Huber threshold did not resolve the bad
whole-horizon basin. Initializing the entire solve from `zero` selected a
consistent branch and reduced the full error without exceeding any original
joint-speed limit. This is evidence about the measured two episodes, not a
guarantee that all 28 scenes will pass.

The HO-Cap batch script now makes both hand-side offsets and the `zero`
reference explicit, rejects stale plans from another profile, and offers a
read-only `--preflight-only`. The original first episode was rerun with the
validated `zero` profile and passed the same gate: 676 frames, 16.21 mm
position RMSE, 31.88 mm p95, 16.86° orientation RMSE, 1.552 rad/s minimum
unscaled URDF velocity margin, 0.0010 mm table margin, and 0.0412 m
self-collision margin. The previous failed plan and status were preserved under
`data/_retarget_diagnostics/original_left_smoke_robot_home_backup/`. The
read-only 28-episode preflight then passed; no other plan had been generated
at that point.

Diagnostic plans are under the ignored local `data/_retarget_diagnostics/`.
The full batch may still expose per-episode failures; report their measured
distribution rather than labeling every produced plan physically feasible.

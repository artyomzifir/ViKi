# HO-Cap full-scene MuJoCo GUI smoke (15 October 2026)

The native MuJoCo window replayed all **831 frames** of
`subject_1/20231025_170105` from the saved
`data/datasets/hocap_eightview/subject_1_20231025_170105_*/plan.h5`.
This is the original **hand-only** UR10/Robotiq 2F-85 plan in `hocap_tag_1`
with base `[0,-1,0]`; its `object_grasp` field is null. It is not the cropped
180-frame, object-aware G10_4 trial and must not be presented as a successful
object-aware grasp.

`experiments/hocap_full_viewer.py` uses the original 831 timestamps and joint/
gripper commands without pressure bias. It slows command time 2×, matching the
existing HO-Cap replay default, and uses the existing MuJoCo scene builder with
all four HO-Cap tracked objects `[1,2,3,4]` as **free** physical boxes. None
is welded to the robot or teleported along its tracked path. Initial proxy
shapes use the 5th–95th percentiles of each saved partial object cloud. Each
box is assigned the replay's assumed 0.20 kg mass, object friction 1.2, pad
friction 1.0, and the cached URDF's coarse distal collision meshes; no new
rubber-pad proxy is enabled. The tabletop top is inferred as the minimum
initial proxy bottom, `z=0.014289 m`, to avoid starting any box inside it.
Neither this tabletop, the boxes nor their masses are independently measured
HO-Cap physics parameters. In particular, several boxes start above that
inferred plane and fall during warmup.

Observed completed GUI run:

| Measure | Value |
|---|---:|
| Frames executed | 831 / 831 |
| Simulated duration | 56.133 s |
| Free object proxies | 4 |
| Maximum arm joint-position tracking error | 0.04895 rad |
| Manipulated object 4 final simulated centre | `[-0.29806, 0.00788, 0.05446]` m |
| Manipulated object 4 final saved tracked centre | `[-0.31220, 0.04719, 0.12564]` m |
| Final simulated-vs-tracked centre discrepancy | 82.53 mm (including −71.18 mm in z) |

The GUI finished its playback and remained open for inspection until its window
was closed. The 82.53 mm is a **proxy-simulation-versus-tracked endpoint**,
not an object-pose estimator error or a physical-ground-truth score. This
smoke demonstrates that the full default robot trajectory and four-object
proxy scene run end-to-end, **not** that the robot grasped any object or that
object-aware retargeting generalizes to four objects.

To repeat with the native GUI, with `DISPLAY` and `XAUTHORITY` pointing to
an accessible desktop X server:

```bash
EPISODE=$(find data/datasets/hocap_eightview -maxdepth 1 -type d \
  -name 'subject_1_20231025_170105_*' -print -quit)
docker run --rm --gpus all -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/viki-xauth -e MUJOCO_GL=glfw -e PYTHONPATH=/app \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$XAUTHORITY":/tmp/viki-xauth:ro \
  -v "$PWD/experiments":/app/experiments:ro \
  -v "$PWD/data":/app/data:ro -v "$PWD/models":/app/models:ro \
  -w /app viki-experiments python3 experiments/hocap_full_viewer.py \
  "$EPISODE/plan.h5" \
  "$EPISODE/intermediates/segmentation/hocap-oracle-v1/object_models.npz"
```

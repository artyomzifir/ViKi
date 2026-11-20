# HO-Cap PnP batch protocol (2026-10-13, corrected)

The pinned plan in `tools/hocap_pnp_plan.json` contains 28 single-hand PnP
sequences (24 right, 4 left; 29,706 source frames) from revision
`2b24836d5e51ad39e56ed4db3fc0c166e755332e`. The source tarballs are
hash-verified by `tools/download_hocap_pnp.py` and kept outside the repository.

## Frozen processing recipe

`/app/scripts/run_hocap_batch.py` stages one flat mirror archive at a time,
without expanding the entire dataset. It maps all eight RealSense views into
the importer layout; HoloLens is excluded. The importer builds the *same*
eight-view world-frame MANO reference for the two-view and eight-view runs.
The source labels are semi-automatic fits, not motion-capture measurements.

1. Import all eight views, including calibration and GT. Extract per-camera
   observations **once** with MediaPipe, then prepare with the named
   `hocap-mediapipe-triangulated-v1` profile. This uses calibrated multi-view
   ray triangulation, not the rig's `PERCEPTION_FUSE_MODE=xyz_mean` default.
2. Evaluate by exact frame index in the `hocap_tag_1` world frame. Preserve
   absolute MPJPE, PCK, matched-joint **coverage**, and failed import depth
   checks; no rigid alignment or GT-aided frame matching.
3. Import the frozen two-camera geometry and GT *without duplicating video*.
   Filter the eight-view `rec.npz` and `observations.npz` to those same two
   camera IDs, then triangulate and evaluate again. This avoids running the
   detector twice and ensures identical per-camera evidence across conditions.
4. Keep GT, per-camera tracks, prepared tracks, calibration, observations,
   acceptance checks and JSON scores. Remove generated MP4/depth media after
   each run; source tarballs remain available for reruns. A staging directory
   is removed after each sequence. Failed episodes remain for diagnosis.

The right-hand pair is `043422252387` + `105322251225` (selected during an
earlier two-sequence pilot). The left-hand pair is `105322251564` +
`037522251142` (selected on the first 120 frames of
`subject_1-20231025_165502`). These selection windows are **development**,
not held-out evaluations; later subject/sequence results must be stratified.
The left pair's 120-frame pilot yielded 16.35 mm MPJPE at 83.1% coverage,
versus 15.92 mm at 99.9% for eight views. These are pilot observations, not
whole-sequence or publication results.

## Why the first GPU run was stopped

The initial `data/datasets/hocap_batch/` artifacts used RTMPose on a verified
ONNX Runtime CUDA provider, but accidentally inherited the rig's `xyz_mean`
fusion. That is not an acceptable HO-Cap benchmark: it averaged false-positive
hands and depth-lifted points from views that failed depth consistency. On the
first left sequence the eight-view observed MPJPE was 182.9 mm, and the
right-hand-selected pair had only 1.0% coverage. A tri-only A/B on those
RTMPose detections improved the eight-view observed error to 21.9 mm but
reduced coverage to 33.6%. The GPU detector itself also had false positives
on several views. Those early rows are **diagnostics, not final results**.

MediaPipe + triangulation on a 120-frame right-hand pilot yielded 10.9 mm at
100% coverage; the same recipe on a 120-frame left-hand pilot yielded 15.9 mm
at 99.9%. We therefore prioritise the validated CPU detector over the faster
but currently unreliable GPU alternative. GPU remains available for a later
controlled detector study; this benchmark does **not** claim GPU acceleration.
Avoiding duplicated inference for the two-camera condition is the safe speedup.

## Running and interpretation

```bash
docker compose run --rm --no-deps \
  -v "$HOME/Downloads/HO-Cap-PnP-raw:/hocap-archives:ro" \
  -v "$PWD/tools:/app/tools:ro" \
  --entrypoint python3 cli /app/scripts/run_hocap_batch.py \
  --archives /hocap-archives --plan /app/tools/hocap_pnp_plan.json
```

Results go under `data/datasets/hocap_mediapipe_tri/`. Completed conditions
are resumable; an incomplete episode is retained rather than silently
overwritten. Move that generated episode before retrying it. A 10 GB disk
reserve is enforced before staging each sequence. `batch_status.json` marks
low-coverage (<80%) or high-error (>30 mm) completed rows `quality=review`;
these are operational review triggers, not a benchmark acceptance threshold.

This is the **hand-pose tract** and its calibration/depth checks, not object
pose, object-centric retargeting or physical manipulation. Eight-view GT fits
share source imagery with predictions, so they are a reference rather than
independent sensor ground truth. This RealSense dataset cannot validate the
local Azure Kinect camera calibration.

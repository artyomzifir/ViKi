# Clean triangulated landmarks baseline v1

`clean-triangulated-landmarks-v1` is the frozen perception recipe that
reproduces the visually validated `pick_up_u` trajectory. It is intentionally
not the planned final geometry-constrained pipeline: it preserves the known
good result before further experiments.

## Locked recipe

```text
raw RGB-D
  → MediaPipe, right/left side supplied per episode, all 21 landmarks
  → full-image 2-D observations + depth samples (radius 15 px)
  → strict two-view triangulation
  → coordinate gap fill, unlimited (legacy baseline behaviour)
  → Savitzky-Golay window 7, polyorder 2
  → landmark-derived wrist/palm pose
  → cln.npz
```

The profile also fixes detector confidence 0.5, confidence exponent 1.0,
binary gripper, `robot_base` coordinate-frame label, and every triangulation
parameter. Saving full 2-D/depth observations is mandatory even if the user
configuration disables it. Capsule `hand_fit` is off. Profile values override
`user_configuration.json`.

If observations or a valid two-camera triangulation are missing, the profile
fails. It never silently falls back to `xyz_mean`.

## Run after recording

The Extract/Perception page selects **clean baseline v1** by default and locks
the accuracy-affecting controls. Hand side and optional cloud generation remain
recording choices.

CLI equivalent:

```bash
docker compose run --rm cli perceive \
  data/datasets/<dataset>/<episode> \
  --profile clean-triangulated-landmarks-v1 \
  --hand right \
  --build-cloud --cloud-stride 1
```

If MediaPipe observations already exist and only the deterministic
triangulation/prepare leg must be rebuilt:

```bash
docker compose run --rm cli prepare \
  data/datasets/<dataset>/<episode> \
  --profile clean-triangulated-landmarks-v1
```

## Protection and provenance

The first output produced for an episode is copied to:

```text
intermediates/baselines/clean-triangulated-landmarks-v1/cln.npz
intermediates/baselines/clean-triangulated-landmarks-v1/manifest.json
```

The copy is never replaced by a rerun. Its manifest contains:

- the complete locked profile;
- SHA-256 of the protected `cln.npz`;
- SHA-256 of observations, triangulation, timestamps, calibration and `rec.npz`.

If the artifact hash no longer matches or somebody changes the definition of
v1 in code, the pipeline raises an error. A changed algorithm must receive a
new versioned profile instead of mutating v1.

Every active output also carries:

```text
perception_profile = clean-triangulated-landmarks-v1
active_variant = clean-triangulated-landmarks-v1
pose_source = landmarks
perception_fuse_mode = triangulate
checkpoint_stage = smoothed
```

## Reference reproduction

Reference episode: `2026-09-03_16-58-45` (`pick_up_u`).

Protected original:

```text
SHA-256 af811b702d4304809256b2b37b5534327818219f617d85ab31747dae0f7c20e8
```

Running the new profile from the saved observations reproduced all 12 canonical
arrays exactly: `positions`, `rotations`, `valid`, `omega`, `gripper`,
`timestamps`, `raw_points`, `smoothed_points`, `landmark_confidence`,
`landmark_ids`, `coordinate_frame`, and `perception_fuse_mode`. The new active
file is larger only because it now also stores stage evidence and provenance.
Episode status therefore reports both `matches_active_core` (trajectory arrays)
and `matches_active_bytes` (the complete container including metadata).

## Known baseline defects

This profile deliberately retains unlimited coordinate gap filling because that
is part of the result being frozen. On `pick_up_u` it creates 28 known
palm-collapse frames in three intervals: `451–454`, `470–479`, `482–495`.

Do not silently fix those inside v1. The next geometry-constrained pipeline
will be a separate profile and will be compared against this protected control.
See [`accuracy_audit_pick_up_u_2026-09-04.md`](accuracy_audit_pick_up_u_2026-09-04.md).

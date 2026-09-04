# Articulated landmark geometry v1

`articulated-landmarks-v1` is the first anatomical A/B stage built on the
protected clean triangulated result.  It is the hand-model stage of the locked
`stable-fused-hand-v1` perception profile.

## Why it exists

The clean `pick_up_u` baseline contains 28 visible palm-collapse frames.  All
21 triangulation confidences are zero in those frames, so their shape was not
measured by either camera: coordinate-wise gap filling and Savitzky–Golay
smoothing invented it.  Interpolating 63 independent XYZ signals does not
preserve a hand.

This stage instead calibrates one fixed hand shape from robust episode frames,
converts landmarks to the existing articulated model, and interpolates missing
joint angles on that model.  Bone lengths, palm geometry, and joint limits are
therefore fixed across the episode.

Dense depth is disabled in this experiment.  The old capsule `hand-fit` can
include pixels from the manipulated object or background in its ROI; mixing
that source back in would make it impossible to tell whether anatomical
constraints themselves help.

## Reproduce

The stable profile runs the complete route and is now the default for
`viki perceive`:

```bash
docker compose run --rm cli perceive <episode>
```

It first protects the clean fused CLN, then generates both geometry variants
and installs `50_optimized.npz` as the active `hand_fit_*` overlay.  A failure
in the declared articulated stage fails the named pipeline instead of silently
returning a fused-only result.  The quality-gate verdict is recorded even when
the candidate is retained for inspection.

For an older clean baseline, the two stages remain independently reproducible
with `--profile clean-triangulated-landmarks-v1` followed by `geometry-fit
--install-overlay optimized`.

The geometry candidates are always written separately:

```text
intermediates/geometry/articulated-landmarks-v1/
├── 40_projected.npz
├── 50_optimized.npz
└── report.json
```

- `40_projected.npz` is the minimum intervention: forward-kinematic projection
  plus joint-space interpolation where evidence is missing.
- `50_optimized.npz` adds confidence-weighted landmark and bidirectional
  temporal terms.  It never reads the dense point cloud.
- `report.json` contains calibration frames, exact recipe weights, structural
  and temporal gates, anchor residuals, and both output paths.

Both appear in the viewer's **Stage variant** picker.  Routing is fixed: the
yellow **fused** layer always reads untouched `smoothed_points`, while the blue
**hand fit** layer reads `hand_fit_capsules`.  The consumer pose preference
cannot swap these visual layers.
The protected clean file also appears as
`baseline/clean-triangulated-landmarks-v1`.

With `--install-overlay optimized`, the same blue overlay is additionally
attached to the active `<episode>/cln.npz`.  This is additive: the canonical
clean arrays are verified against the protected baseline before and after the
write.  It makes comparison available in the Extract viewer too, where there
is no Stage variant selector—enable both **fused** and **hand fit**.

## Stored audit data

Each NPZ preserves the source CLN arrays and adds:

- `geometry_source_points` — the untouched clean skeleton;
- `geometry_joint_angles` — the fitted model configuration per frame;
- `geometry_anchor_confidence` — triangulation confidence after anatomical
  outlier gating;
- `geometry_support_mask` — frames allowed to initialize palm pose;
- `geometry_edge_reference_m` — episode reference edge lengths;
- `geometry_calibration_frame_indices` and serialized hand parameters;
- exact recipe, solver diagnostics, and quality metrics as JSON scalars.

The clean wrist translation is restored exactly after both solves.  A candidate
cannot pass the quality gate if it collapses the palm, changes bone lengths,
moves the wrist, or increases joint jerk beyond the documented allowance.

## Current limitation

Hand dimensions are provisionally calibrated from robust frames in the work
episode.  The intended next refinement is a short subject/session calibration
recording with an open hand and several articulations, after which those shape
parameters should be fixed for all demonstrations from that session.

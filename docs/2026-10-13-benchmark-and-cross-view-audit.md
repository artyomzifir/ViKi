# Benchmark and cross-view audit (2026-10-13)

> Recovery note: the local 78-episode cohort, `data/experiments/` outputs and
> HO-Cap sample episodes were lost with the workspace. Values in this report
> were recovered from the original task log; commands below require the
> source artifacts to be reacquired before they can reproduce the figures.
> The local-cohort figures below are withdrawn from the thesis; see
> [the withdrawal note](2026-10-14-local-78-retraction.md).

## Newly computed on the existing 78 cube episodes

`scripts/audit_cross_view_hand.py` read the episode list from
`data/experiments/2026-10-09-full-audit.json` and compared only **directly
finite, same-capture-frame** pairs of the two world-frame `rec.npz` hand
trajectories. It excluded any frame where a camera timestamp was reused:
the original pairing sometimes repeats a camera frame, and duplicate
`rec.npz` rows with that timestamp can have different detector output.
No calibration, recording or canonical artifact was changed.

| Measure | Result |
|---|---:|
| Episodes / capture frame groups | 78 / 11,688 |
| Paired frames scored | 10,612 |
| Ambiguous repeated-timestamp frames excluded | 496 |
| Finite matched landmark pairs | 222,852 |
| Pairwise 3-D landmark separation, median / p95 | 48.34 / 80.15 mm |
| Per-episode median separation, range | 40.72–55.89 mm |

Reproduce with:

```bash
PYTHONPATH=. python3 scripts/audit_cross_view_hand.py
```

Machine-readable per-episode results are written to
`data/experiments/cross-view-hand-audit.json`. This is a **self-consistency
diagnostic**, not hand-position error against independent truth and not a
point-cloud surface-overlap score. Per-camera depth lifting can land on
different visible skin surfaces, and a 2-D detector can pick different
anatomical pixels; calibration and timing may contribute too. The large
separation must not be relabelled as either camera-extrinsic error or a
ground-truth MPJPE without further controls. It does flag that raw monocular
3-D hand tracks are not mutually interchangeable at millimetre precision.

## What the public-data connector closes

The restored `viki/benchmark/` HO-Cap importer and new evaluator now give
an independently annotated hand-pose comparison path:
`raw RGB-D -> rec.npz -> cln.npz` versus HO-Cap's world-frame 21-joint
labels. The importer, exact frame alignment, observed/fabricated distinction,
absolute MPJPE, PCK and coverage have synthetic tests. See
`viki/benchmark/README.md` for commands and limitations.

The subsequent bounded HO-Cap pilot did produce 2-view and 8-view hand-joint
MPJPE on two 120-frame sequences, recorded in
`docs/2026-10-13-hocap-real-sample-pilot.md`. Those files are now absent and
the pair was selected in-sample. It is not a held-out or local-Kinect result.

## Remaining publication gates

1. Download a bounded HO-Cap subset, freeze camera pair and held-out
   sequences, inspect reprojection overlays, resolve failed depth checks,
   then run extract/prepare and publish error **with coverage**.
2. For our Kinect rig, use a held-out large-board tabletop/carry-height
   measurement to separate calibration from hand detector/depth bias.
3. For object geometry and 6-D pose, record an asymmetric marked control
   object; a symmetric blue cube and partial shell do not fix absolute yaw.
4. Re-evaluate one frozen retarget mode and all eligible scenes under a
   single physics protocol. The current red-support collision and selected
   subset remain limitations, as documented in
   `docs/2026-10-12-existing-recordings-numerical-audit.md`.

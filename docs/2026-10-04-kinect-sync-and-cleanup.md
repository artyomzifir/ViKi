# Two-Kinect synchronization and storage cleanup — 2026-10-04

## Scope and preservation

At the first cleanup, the eight takes in `data/datasets/40mm-new-2kinects`,
calibration artifacts, depth background plates, model files, configuration, and
worktree edits were left intact. Removed old recording/test directories:

- `data/datasets/new-dataset` (about 21 GiB)
- `data/datasets/sync-test` (about 491 MiB)
- `data/_stability` (about 9.3 GiB)
- `data/k4a_raw` (about 8.0 GiB)

The Docker **build cache** was pruned, but images, volumes, and running services
were not pruned. Filesystem free space rose from about 6.5 GiB to about 69 GiB.
The two temporary synchronization takes and vendor MKVs were deleted after
their timestamps were saved under
`data/diagnostics/k4a-sync-2026-10-04/` (about 104 KiB total).

## Hardware check

Both Azure Kinects were detected at USB SuperSpeed. They now use separate xHCI
controllers: one Intel PCH and one Renesas PCIe controller. The current cable
reports `kinect_1` as master and `kinect_0` as subordinate, with a configured
160 µs delay. The cameras were stopped after each check.

The vendor recorder was asked for five seconds, but actually wrote about 22
seconds: 650 master depth frames and 666 subordinate depth frames. Among the
650 master timestamps, 647 had a subordinate depth timestamp within 1 ms, with
an exact +122 µs subordinate-minus-master offset. The other three are startup
exceptions; the subordinate depth stream has one 75,156 µs startup gap. There
were no further depth gaps over 50 ms in this capture. This establishes a
stable wired phase after startup, **not** long-run host stability. The unexpected
recording duration remains unexplained. Raw device timestamps are preserved in
`playback_timestamps.json`.

## ViKi end-to-end check

Both five-second ViKi takes used the live `/api/record/start` route at 30 fps.
The debug-only `force` flag bypassed calibration readiness for these timing
tests, not the hardware-sync gate. The first take, before the pairing fix,
contained 149 saved pairs. Four pairs (2.7%) combined captures from adjacent
33 ms cycles despite the good hardware phase. The second take, after changing
`MultiCameraSync` to choose subordinate frames by their device timestamps and
reject unmatched/stale captures, contained 150 pairs. **All 150 were within
133–167 µs** subordinate-minus-master; none differed by more than 1 ms.

Compact evidence: `viki_before_timestamps.json` and
`viki_after_timestamps.json` in `data/diagnostics/k4a-sync-2026-10-04/`.
At the time of this comparison, the eight September takes still had their
previously measured 3.6–20.3% mis-pair rates. They were subsequently deleted
on user request; the new pairing rule applies to future recordings.

`raw/sync_stats.json["bounded"]` was `false` even on the correctly paired
second take. That legacy field measures the trend of **host arrival time versus
sampling tick**, which is affected by USB/worker scheduling and is not a
measurement of inter-camera hardware phase. Do not use it to judge wired
alignment; inspect per-pair device timestamps. Revising that statistic is
separate follow-up work.

## Subsequent reset and operating status

On the user's later request, all eight September takes and all calibration
presets (including `_live`) were deleted. The active-preset pointer was cleared,
and the live extrinsics, world anchor, validation report, and depth background
plates were removed. The compact timestamp diagnostics above remain. Free
filesystem space rose to about 74 GiB after this second cleanup.

The old shared-controller host failure is closed for the current Intel-plus-
Renesas wiring; routine two-Kinect capture may proceed without the former
one-Kinect-only workaround. These short runs do not establish a general
reliability guarantee. Revisit USB stability only if a new failure appears on
this separate-controller topology. Depth geometry, cross-view extrinsics, and
object tracking still require their own calibration and data-quality checks.

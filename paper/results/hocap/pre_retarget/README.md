# HO-Cap pre-retarget raw numerical results

Immutable-by-convention snapshot of the 2026-10-13 eight-view, 28-episode
perception evaluation. These are byte-for-byte copies from
`data/datasets/hocap_eightview/`; the source files and the downloaded HO-Cap
archives remain in place. The snapshot contains **114 JSON files** (646,666
payload bytes), not camera frames, calibration files, `rec.npz`, `cln.npz`,
object models, or retarget plans.

At the root, `batch_status.json` and `object_batch_status.json` are the two
completed queue statuses. For each episode, `episodes/<id>/` mirrors four
source files: `gt/evaluation.json` (hand/Prepare versus HO-Cap annotations),
`intermediates/segmentation/hocap-oracle-v1/evaluation.json` (object tracks
using published instance masks), `acceptance.json`, and `batch_run.json`.
The relative path after `episodes/` matches the source path after
`data/datasets/hocap_eightview/`. `SHA256SUMS` records every copied JSON.
All copies were checked with `cmp` against the source at snapshot time and
with `sha256sum --check --quiet SHA256SUMS` afterward.

These metrics predate retargeting. HO-Cap reference labels are semi-automatic
annotations, not independent motion capture for the local Kinect rig. The
object evaluation uses oracle instance masks and a first-frame SE(3) alignment;
its pose errors measure relative drift, not absolute object localization.
The `G18_2` outlier is retained. See
`docs/2026-10-13-hocap-full-batch-results.md` for the full protocol and
unrounded values embedded in the daily log.

Do not replace these files when the later retarget batch runs. Put its numerical
results in a separate snapshot.

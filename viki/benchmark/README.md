# External hand-pose benchmark connector

The HO-Cap importer was restored from the older `hocap-ingest` branch. It
converts one **left- or right-hand, single-hand** sequence into ViKi's normal multi-camera
episode layout without importing the GPL-3.0 HO-Cap Toolkit or MANO code.
The [HO-Cap dataset](https://irvlutd.github.io/HOCap/) itself is CC BY 4.0;
the authors' hand labels are semi-automatic multi-view MANO estimates, not
direct motion-capture ground truth.

## Inputs and run

Download the published `calibration.zip` and one subject archive from the
official HO-Cap page, and extract them under
`data/raw_datasets/hocap/{calibration,subject_1,...}`. The example sequence
below was used in the historical pilot; it is **not** a finished accuracy
result in the current checkout.

```bash
docker compose run --rm cli hocap-import data/raw_datasets/hocap \\
  --subject subject_1 --sequence 20231025_170650 --max-frames 120
docker compose run --rm cli extract data/datasets/hocap/<imported-episode>
docker compose run --rm cli prepare data/datasets/hocap/<imported-episode>
docker compose run --rm cli hocap-evaluate \\
  data/datasets/hocap/<imported-episode> \\
  --out data/datasets/hocap/<imported-episode>/gt/evaluation.json
```

`hocap-import` selects two of the eight RealSense views by baseline and
convergence relative to a ViKi episode when `--viki-ref` is supplied.
Otherwise inspect its pair table and use `--cameras CAM_A CAM_B` to freeze
the pair, or pass any two or more unique serials for an explicit multi-view
run. `--pipeline-check` runs extract and prepare during import;
`--backend-check` measures detector drift introduced by the MP4 encoding.
The import report is written inside the episode as `hocap_import.md`;
ground truth is in `gt/joints3d_world.npz`. Re-importing over an episode
with processed artifacts is refused even with `--force`.

## Score contract

`hocap-evaluate` reports per-camera `rec.npz` error plus three prepare
views: `fused_observed` (before filling), `smoothed_observed` (only joints
with direct observation evidence), and `smoothed_all` (including finite
filled values). It maps rows to HO-Cap frames by **exact synthetic capture
timestamp** and compares all 21 joints in the stored `tag_1` world frame.
There is no time search, rigid alignment, root alignment or GT-based frame
selection. MPJPE, median, p95 and PCK are conditional on matched valid joints;
always report their `matched_joints / gt_joints` coverage beside error.
Missing `rec.npz` or `cln.npz` is explicit, never treated as zero error.

This is a ViKi-specific multi-view protocol, **not** the HO-Cap official HPE
leaderboard split or its single-view metric. Freeze subjects/sequences and
camera pairs before comparing algorithms, and keep an untouched test split.
The importer does not score object poses, retargeting or sim-to-real transfer.

The earlier branch's four-sequence import failed depth checks. A subsequent
two-sequence, 120-frame-per-sequence pilot measured smoothed-observed MPJPE
of 11.33/11.67 mm for the selected pair and 11.34/10.50 mm for all eight
views, each with 2,520/2,520 matched joints. These are historical values
recovered from the task log: the source archives and generated episodes were
lost locally. The pair was selected on these same sequences, and the depth
check still fails on one view. Reacquire source artifacts and rerun before
claiming a reproducible or held-out benchmark result. The pinned PnP archive plan includes 24 right-hand and four left-hand
sequences. The batch protocol and GPU command are in
[`docs/2026-10-13-hocap-batch-protocol.md`](../../docs/2026-10-13-hocap-batch-protocol.md).
Downloading alone does not imply that any sequence has been evaluated.

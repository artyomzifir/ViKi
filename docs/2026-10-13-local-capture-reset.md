# Local capture reset (2026-10-13)

At the user's request, all locally recorded Kinect episode datasets and their
calibration presets were removed before the next, smaller capture campaign.
Removed: `data/datasets/40mm-pnp/`, all five `data/datasets/rig-*/` datasets,
`data/share/` (the old ten-recording handoff), `data/diagnostics/`,
`data/calibrations/`, `data/depth_bases/`, and the root extrinsics, anchor, and
validation JSON files. `ACTIVE_CALIBRATION` in the local user configuration is
empty, so a new calibration must be captured before recording.

The external HO-Cap benchmark under `data/raw_datasets/hocap/` and
`data/datasets/hocap/` was retained: it cannot be recreated by re-recording this
rig. Historical aggregate experiment results, figures, and manuscript text were
also retained, but the old local cohort is no longer reproducible from raw
recordings on this machine. Do not present those numbers as a fresh measurement
on the new capture campaign.

One-camera RGB-only HO-Cap review clips are generated separately under
`data/hocap_review/right_pnp_043422252387/`, with an `index.html` gallery and
`manifest.json` identifying the exact source revision, camera, and sequences.

# viki.perception — video → 3-D hand skeleton

**Stage 2** · `episodes/<id>/raw/` → `rec.npz`

Offline. Decode the recorded colour + depth, run a pluggable hand-pose backend
per camera per frame, lift the 2-D detection to 3-D with measured depth,
transform into the workspace frame with the recorded extrinsics, and write
per-camera landmark trajectories. Cross-camera fusion is **not** done here — it
is deferred to `viki.prepare`.

## Files

| file | what |
|---|---|
| `extract.py` | offline orchestrator: `raw/` → `rec.npz`. Assumes depth aligned to colour (identity colour→depth projector). |
| `backends/` | `HandPoseBackend` ABC + implementations — see `backends/README.md` |
| `camera_prep.py` | `prepare_frame` : `Frame` → `PreparedFrame` (RGB, depth in metres, depth K) |
| `geometry.py` | `lift_to_3d` (2-D + depth → camera-frame 3-D), `camera_landmarks_to_world` (apply extrinsics) |
| `segmentation.py` | optional prompted SAM 2.1 video masks and calibrated RGB-D lift; experimental artifacts only |
| `object_model.py` | optional core/shell rigid models, robust SE(3) tracks, contact freeze and compact point decisions over a semantic cloud |
| `hand_angles.py` | `compute_end_effector_pose` — the single site that derives the wrist SE(3) pose from landmarks (also used by `prepare`) |
| `pipeline.py` | `SkeletonPipeline` — per-`SyncedFrameGroup` orchestration (kept for tooling; the offline path is `extract.py`) |
| `models.py` | compat re-export of the perception DTOs from `viki.contracts` |

## Contract

- **in:** `PreparedFrame`, `DepthProjector` (Protocol), `CalibrationExtrinsics`.
- **out:** `rec.npz` — keys in `contracts.REC_KEYS`:
  `device_ids`, `timestamps`, `points (N,21,3)`, `landmark_ids (21,)`,
  `confidence (N,21)` *(stub: detector visibility only)*.

## Palm frame

`compute_end_effector_pose` builds the wrist rotation from the MCP knuckle
spread (INDEX→PINKY), **not** the thumb: `x = norm(MIDDLE_MCP − WRIST)`,
`z = norm(x × (PINKY_MCP − INDEX_MCP))`, `y = z × x`. Missing a required
landmark falls back to the centroid of the available palm landmarks with an
identity rotation.

## Experimental scene segmentation

SAM 2.1 is intentionally isolated from the normal ViKi image because the
PyTorch CUDA wheels are several gigabytes. Build/run the dedicated profile:

For an episode recorded against a stored empty-scene depth plate, frame-zero
prompts can first be proposed from fused 3-D foreground components:

```bash
docker compose run --rm cli auto-prompts episodes/<id> \
  --objects 3 --out data/prompts/<scene>.json
```

The proposal generator treats the largest foreground component as the operator
and compact, multi-view, colour-saturated components as object candidates. It
projects the same selected 3-D components into every RGB camera to produce a
box, a positive click, and negative clicks on the other components. Candidate
objects default to `other_dynamic`: frame-zero geometry alone cannot determine
their task role. Component measurements and thresholds remain in the prompt
JSON for audit.

```bash
docker compose build sam2
docker compose run --rm sam2 segment episodes/<id> \
  --prompts data/prompts/<scene>.json --lift-3d
```

If a long 2-D run completed but the optional 3-D lift was interrupted, reuse
the masks without rerunning SAM:

```bash
docker compose run --rm cli segment-lift episodes/<id>
```

Build the non-destructive rigid object-model experiment after the 3-D lift:

```bash
docker compose run --rm cli object-model episodes/<id>
```

The official `sam2.1_hiera_small.pt` checkpoint goes in `models/sam2/`; its
expected SHA-256 is
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.
The image pins upstream commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4` and PyTorch 2.8 CUDA 12.8.
The default 100-frame chunk is chosen for the repository's 11 GiB container
ceiling; reduce it when tracking more than three simultaneous instances.

Prompts use schema `viki_sam2_prompts_v1`: declare stable positive integer
object IDs and labels (`operator`, `manipulated_object`, `other_object`, or
`other_dynamic`), then give every object a frame-0 box and/or positive/negative
points for each camera. Later prompts are corrections. Coordinates are in the
original colour-video pixels.

Outputs live under
`intermediates/segmentation/sam2.1_hiera_small/`: bit-packed mask chunks,
per-camera overlay videos, prompt/model provenance, and—when `--lift-3d` is
used—mask-supported world-frame point clouds plus a provisional centroid and
visible-dimensions track. Nothing downstream reads these artifacts implicitly.
Missing or unlabelled points remain unknown, never free space.

`object-model` adds `object_models.npz` beside those artifacts. It does not copy
or rewrite XYZ/RGB. The archive holds a temporally supported canonical
core/shell surface, `T_world_object(t)`, tracking residual/coverage/confidence,
rotation information, an operator-proximity contact flag, and compact
accepted/reassigned/rejected/ambiguous decisions referring to point indices in
each source semantic-cloud frame. Shell promotion is disabled on contact
frames. This remains an experimental perception artifact: retarget does not
load it implicitly.

Measured results and known failure modes from the first two real scenes are in
[`docs/2026-09-14-sam2-segmentation-probe.md`](../../docs/2026-09-14-sam2-segmentation-probe.md)
and the follow-on core/shell experiment is recorded in
[`docs/2026-09-14-object-model-filter-probe.md`](../../docs/2026-09-14-object-model-filter-probe.md).
The calibrated background-to-prompt experiment on the complete `pyramid` scene
is recorded in
[`docs/2026-09-14-pyramid-auto-prompts.md`](../../docs/2026-09-14-pyramid-auto-prompts.md).

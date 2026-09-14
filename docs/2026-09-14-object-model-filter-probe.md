# Rigid object core/shell filtering probe — 2026-09-14

## Scope and measurement status

This is a non-production experiment over the SAM 2.1 semantic clouds for two
real episodes: `move-shipok` and `cup_grab`.  The latter contains two separately
moving cups, so the evaluation covers three object tracks and mutual instance
assignment.  There is no ground-truth object pose or scan.  Every millimetre
and degree below is therefore a **self-consistency/tracking measure**, not an
absolute accuracy claim.

The experiment writes `object_models.npz` beside the existing segmentation
artifacts.  It does not rewrite `semantic_cloud/`, `cln.npz`, or `plan.h5`, and
retarget does not consume the result implicitly.

## Algorithm under test

1. Split the SAM-labelled object candidates into 12 mm radius components over
   the existing 4 mm voxel cloud.
2. Bootstrap a conservative component from the first 60 frames.  Prefer
   multi-camera support, then track the component by centroid and size.
3. Form the canonical `core` from voxels supported in at least 25% of the
   bootstrap observations.  Repeated detached leakage is excluded before the
   temporal vote, so a static false positive cannot become shape merely by
   persisting.
4. Track the core with robust, Cauchy-weighted trimmed point-to-point ICP.  The
   translation seed is the median same-camera visible-centroid increment; this
   avoids a fused-centroid jump when one camera surface disappears or rejoins.
5. Accumulate a model-adjacent `shell` only on confident, non-contact frames.
   A voxel needs both an absolute support count and 3% of eligible frames.  The
   shell is limited to half the core surfel count and keeps the strongest
   temporal supports, preventing sampling density from taking over ICP.
6. Refit against core + shell.  Competing object models jointly assign every
   object-candidate point.  A weak best/second-best margin becomes `ambiguous`
   rather than an arbitrary instance decision.
7. Save the canonical surfels, pose/confidence/residual/coverage/contact tracks,
   rotation information, and compact point decisions referencing the original
   per-frame point indices.

The robust residual treatment follows the same motivation as
[Open3D robust kernels](https://www.open3d.org/docs/latest/tutorial/pipelines/robust_kernels.html)
and [Trimmed ICP](https://doi.org/10.1109/ICPR.2002.1047997).  A global method
such as [TEASER++](https://arxiv.org/abs/2001.07715) remains a possible explicit
reacquisition path, not something to run on every frame.

## Rejected simpler rules

- **Minimise raw point variance:** collapses toward a small dense patch instead
  of preserving surface coverage.
- **Use a 16 mm connected component:** retained slightly more points, but on
  `move-shipok` one visible-axis spread grew from about 24 to 95 mm because the
  radius connected a contamination bridge.
- **Always keep the largest component:** on `move-shipok` frame 128, the correct
  two-camera surface split.  The largest part was single-camera and the second
  correct part was 42% of its size.
- **Freeze the first shape completely:** median residual stayed near 1.8–2.3 mm,
  but a newly visible surface reduced worst-phase retention to 45–63%.
- **Trust temporal recurrence alone:** persistent detached leakage survived.  On
  the static bootstrap it expanded one `move-shipok` dimension to about 257 mm;
  component-gated consensus gave about 138 mm.
- **Use the fused visible centroid as a pose seed:** view dropout injected a
  false 25 mm step.  Same-camera centroid increments removed it.
- **Let every repeated shell voxel into ICP:** the first cup accumulated more
  shell than core and biased tracking by sample density.  A support-ranked shell
  budget fixed this failure.

## Final real-scene results

`residual p95` below is the 95th percentile across frames of each frame's 95th
percentile inlier residual.  `retained p05` is deliberately reported beside the
median so a good typical frame cannot hide a poor phase.

| scene / object | core / shell | accepted | conf p05 / p50 | residual median / p95 | retained p05 / p50 | translation step p50 / p99 / max | rotation step p50 / p99 / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| move-shipok / 2 | 2188 / 948 | 93.70% | .813 / .918 | 1.76 / 8.51 mm | .858 / .950 | .35 / 7.66 / 9.52 mm | .21 / 1.45 / 2.63 deg |
| cup_grab / 2 | 2016 / 1008 | — | .477 / .878 | 2.17 / 10.36 mm | .713 / .963 | .76 / 13.24 / 16.35 mm | .49 / 4.38 / 6.01 deg |
| cup_grab / 3 | 2045 / 634 | — | .642 / .934 | 1.57 / 10.11 mm | .786 / .996 | .29 / 14.58 / 18.92 mm | .16 / 3.51 / 5.43 deg |

The two-cup scene-wide decisions were 93.31% accepted under the original
instance, 0.11% reassigned, 0.02% ambiguous, and 6.56% rejected.  Neither cup
had a frame below 0.1 track confidence.  An earlier fixed-template version lost
object 3 for 125 consecutive frames (492–616); the per-camera motion predictor
removed that loss without fixing object orientation.

The contact proxy marked 42.5% of `move-shipok`, 47.2% of cup 2, and 31.7% of
cup 3.  On the manually used `move-shipok` contact interval 430–700, the chosen
rule (at least 5% of object points within 12 mm of operator points) detected
99.3% of interval frames and 6.8% of frames in the pre-400 region.  This is only
a geometric freeze gate, not yet a semantic grasp label.

## Runtime and storage

| episode | frames / objects | object-model time | artifact size |
|---|---:|---:|---:|
| move-shipok | 897 / 1 | 43.54 s | 6.53 MiB |
| cup_grab | 898 / 2 | 72.73 s | 10.63 MiB |

For comparison, the existing full semantic clouds occupy approximately 361 MiB
and 180 MiB respectively.  The compact archive avoids duplicating XYZ/RGB.

## Known limitations before any retarget integration

1. The first 60 frames are assumed suitable for bootstrap.  Detecting an
   automatically stable, pre-contact window is still required for recordings
   that begin in motion.
2. The rotation-information eigenvalues are local registration information;
   they do not yet encode global rotational symmetries such as cup yaw.
3. Contact is operator proximity, not force or a confirmed grasp phase.
4. The shell budget is deliberately conservative and heuristic.  It needs
   validation on an object whose important surface becomes visible only after a
   large rotation.
5. No labelled 3-D ground truth exists, so accepted/rejected decisions need a
   small manually audited frame set before promotion.
6. Retarget must eventually consume only compact pose/confidence/model data.
   Running ICP or full-cloud collision queries inside every IK iteration would
   erase the intended stage boundary and multiply its existing collision cost.

## Reproduction

```bash
docker compose run --rm cli object-model \
  data/datasets/new-dataset/2026-09-09_11-54-10
docker compose run --rm cli object-model \
  data/datasets/new-dataset/2026-09-04_13-41-37
```

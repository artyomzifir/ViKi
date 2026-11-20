# HO-Cap object-pose benchmark (2026-10-13)

The 28 eight-view PnP sequences from `tools/hocap_pnp_plan.json` are processed
with the published HO-Cap instance masks as **oracle segmentation**. Depth and
masks are read directly from each source tarball; RGB frames are not staged or
used by this object tracker. The method lifts each visible instance into the
`hocap_tag_1` frame, builds a compact rigid model, and tracks its relative
SE(3) pose. It is an object-tracking test *conditional on given masks*, not an
end-to-end test of autonomous instance segmentation.

The object queue is `hocap-objects-batch.service`; the hand queue is
`hocap-eightview.service`. The object queue waits for each completed eight-view
hand episode and writes `data/datasets/hocap_eightview/object_batch_status.json`.
Progress is logged in `data/hocap_objects.log`. Both queues are resumable.

Per episode, keep `cln.npz`, `rec.npz`, the compact
`intermediates/segmentation/hocap-oracle-v1/object_models.npz`, its
`evaluation.json`, and `gt/object_poses_world.npz` with calibration and hand
GT/metrics. Remove the generated semantic-cloud frames immediately after the
object model and evaluation are complete. The hand queue likewise removes
transcoded MP4 and depth-frame media; original source tarballs remain outside
the repository for reruns. These four objects are marked `other_dynamic`, not a
single `manipulated_object`; this artifact is **not yet** a valid input to the
cube-specific retarget option.

The metric uses exactly one right-multiplied object-local alignment at the first
confident frame, then compares every subsequent frame to HO-Cap's published
semi-automatic object poses. It measures **relative tracking drift**, not
absolute 6-DoF localization. The calibrated `tag_1` fiducial already anchors
all cameras and points in a common world frame; it does not determine an
object model's local origin or axes. Absolute object-pose evaluation needs an
independent object-frame/CAD registration, not a second tag transform. Report
coverage, translation and rotation errors per object. Motionless periods can
have trivially low error, so assess moving segments and occlusions separately.

## Manual scene check

`subject_1-20231025_170105` (right hand, 831 frames, eight views) completed.
The hand benchmark has 100% matched-joint coverage, 11.61 mm MPJPE, and
21.46 mm p95. Inspecting RGB frames 0, 200, 400, 600, 616, 690, and 830 from
top and side cameras confirms the four object IDs, but the earlier
appearance-based labels here were imprecise. The official HO-Cap toolkit
catalog identifies `G10_1` as soup mix, `G10_2` as gel toothpaste,
`G10_3` as chocolate biscuit sticks, and `G10_4` as body wash
(https://github.com/IRVLUTD/HO-Cap/blob/main/config/hocap_info.yaml).
The positions change in the same temporal order as the images. The orange jar is hand-occluded
at frame 690, where its aligned tracking error peaks at 14.8 mm; the hand's
index-tip error rises to 33.0 mm. The red box's worst aligned translation error
is 28.3 mm around frame 616, despite no obvious RGB occlusion there; this is a
case for further diagnosis rather than a claim of universally accurate tracking.

Across all 831 frames, oracle-mask object tracking covers 100% of published GT
frames for each object. Mean aligned translation errors are 0.67, 5.57, 10.13,
and 1.23 mm respectively; mean rotation errors are 1.55°, 11.96°, 3.54°, and
1.06°. The full semantic cloud was discarded after the compact 9.5 MB model
and score were saved.

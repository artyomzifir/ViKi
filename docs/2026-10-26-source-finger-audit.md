# Source finger audit, 26 October 2026

The original human demonstration does not justify treating this as a thumb/index-only grasp. Across carry frames 42–125, both HO-Cap annotated joints and ViKi's fitted hand place the ring finger near the bottle; the little finger is also usually nearby. These are geometric proximity estimates, not measured contact forces. The robot's missing ring/little contact is therefore a retargeting hypothesis to investigate, not an established feature of the demonstration.

The audited sequence is subject_1/20231025_170105, first 190 frames, object G10_2. The convenience episode initially contained only two encoded videos. The complete local source archive `/home/zifka/Downloads/HO-Cap-PnP-raw/subject_1-20231025_170105.tar` supplies all eight synchronized RGB views; all 8×190 frames have now been extracted into the ignored pressure experiment directory. The dataset's full camera coverage is available for the next experiment. Initial two-view inspection must not be used as a reason to reject visual contact/pressure estimation.

## Measured geometric evidence

For each finger, take the minimum signed gap of its three capsule phalanges to the convex hull of the aligned official object mesh. GT-joint capsules use the fitted hand's radii (6.4–14 mm); skin thickness is approximate. Aligned OBJ vertices are in object-track coordinates, so annotated object poses are transformed by the inverse `track_from_mesh` before measuring. Directly mixing official-mesh poses with aligned vertices would be incorrect.

| Source | Thumb | Index | Middle | Ring | Little |
|---|---:|---:|---:|---:|---:|
| Annotation + annotated object, median signed gap (mm) | −0.72 | −7.05 | −3.12 | −5.58 | +1.97 |
| Annotation, fraction within 5 mm | 100% | 100% | 100% | 100% | 95.2% |
| Fitted hand + tracked object, within 5 mm | 98.8% | 100% | 100% | 100% | 83.3% |
| Fitted hand + annotated object, within 5 mm | 100% | 100% | 100% | 100% | 89.3% |

Negative values mean overlapping proxy geometry, **not pressure**. Some proxies penetrate by >10 mm. The exact anatomical skin contact and load distribution cannot be inferred from these numbers. Nonetheless, ring proximity persists across both hand representations and both object tracks, so it is not an artifact of a single fitted configuration. Human reconstruction and Shadow finger-direction fitting do not preserve contact topology automatically: different link lengths, bases, and surfaces can remove a contact even when phalanx directions look similar.

The tracked and aligned annotated object poses disagree by median 3.78 mm / 12.17°, maximum 9.02 mm / 26.87°. This is disagreement between estimates, not a calibrated physical error measurement. It also limits the precision with which a recorded target rotation can be treated as truth.

## Independent checks and retraction

The first raw run was invalid: MuJoCo compiled world-attached capsule geoms with a shared-frame fast path, so changing their model transforms did not update the actual world positions. Those numbers are retracted. The raw files remain in `data/experiments/hocap-rl/source_finger_audit_20261026/invalid_cached_world_geom_*`; they must not be used. The corrected runner disables this fast path, updates capsule bounds, and asserts actual world positions against requested endpoints.

An independent SciPy convex quadratic program uses official OBJ hull halfspaces and transforms the annotated segments directly into object coordinates, with no MuJoCo geometry transforms. It checks five distal phalanges at frames 42,60,90,120,145. All 25 optimizations converge. For the 20 cases whose centerline is outside the hull, surface distances agree within 0.000004 mm numerically. Five centerline-inside-hull cases have different signed-penetration definitions and are explicitly excluded from that numerical equality; both methods identify overlap. This validates implementation consistency, not micrometre or submillimetre anatomical accuracy.

[Results and hashes](../paper/results/source_finger_audit_2026_10_26/finger_proximity.json), [per-frame gaps](../paper/results/source_finger_audit_2026_10_26/finger_gaps.npz), [independent QP check](../paper/results/source_finger_audit_2026_10_26/independent_geometry_check.json). Reproduce with `experiments/hocap_source_finger_audit.py` and `experiments/hocap_source_finger_geometry_check.py` in the simulator container, using the source episode already on this host.

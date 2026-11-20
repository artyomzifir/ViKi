# Withdrawal of local cube-transfer results (2026-10-14)

The earlier 78-episode local cube-transfer cohort is not available for audit:
its source recordings, calibration, and downstream artifacts are missing.
Consequently, the local capture, extraction, object-reconstruction, retargeting,
MuJoCo-execution, augmentation, and policy-learning figures derived from that
cohort are **withdrawn**, not verified results. They have been removed from the
thesis results, abstracts, discussion, conclusion, and associated tables.

Historical notes and code are retained for provenance, not as evidence.
Neither recovered prose nor Git history substitutes for source data. These
figures must not be cited, combined with new cohorts, or restored to the
manuscript without preserved source episodes, calibrations, a reproducible
protocol, and regenerated metrics.

The separately archived 28-sequence HO-Cap pre-retarget numerical snapshot
remains in `paper/results/hocap/pre_retarget/` with checksums. It supports
only the external perception evaluation described in the thesis; it does not
validate the lost local cohort or downstream robot-learning claims.

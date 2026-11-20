# HO-Cap real-sample pilot (2026-10-13)

> **Recovery status:** This document is reconstructed from the original task
> log. The downloaded raw subset, imported episodes and machine-readable
> reports named below were lost with the local workspace. The numerical
> values are historical observations, not currently reproducible artifacts.
> The `043422252387`/`046122250168` pair below was an initial poor choice;
> the corrected pair and eight-view results are in the final section.

## Corrected camera choice

Two right-hand `subject_1` sequences were sampled from the
[authors' HO-Cap dataset mirror](https://huggingface.co/datasets/JWRoboticsVision/HO-Cap-Dataset/tree/main),
along with its calibration and sequence-label archives. The first 120 frames
and two fixed RealSense views (`043422252387`, `046122250168`) were imported
from each sequence; the downloaded subset and source-byte hashes are recorded
in `data/raw_datasets/hocap/sample_manifest.json`. The source labels are
semi-automatic multi-view MANO estimates, not mocap measurements. This is a
convenience pilot, **not** the official HO-Cap HPE split, a held-out result, or
an object/retarget benchmark.

| Sequence | GT frames / joints | Import depth check at GT joints | Extract | Prepare |
|---|---:|---|---|---|
| `20231025_170650` | 120 / 2,520 | failed both views | 141 observation rows | completed; 419 directly observed joints |
| `20231025_170105` | 120 / 2,520 | failed both views | 151 observation rows | failed: no resolvable palm; partial `cln.npz` ignored |

Absolute world-frame MPJPE is in millimetres. Coverage is the matched-joint
count divided by 2,520 annotated joints; there is no rigid/root alignment,
nearest-timestamp search or GT-aided frame selection.

| Sequence | Stage/view | Coverage | MPJPE | Median | P95 |
|---|---|---:|---:|---:|---:|
| `170650` | `rec`, `0434` | 2,520/2,520 | 30.6 | 27.5 | 56.8 |
| `170650` | `rec`, `0461` | 210/2,520 | 122.7 | 124.8 | 187.7 |
| `170650` | fused, observed | 419/2,520 | 13.7 | 11.7 | 28.8 |
| `170650` | smoothed, observed | 419/2,520 | 13.6 | 11.8 | 28.8 |
| `170650` | smoothed, all finite | 2,520/2,520 | 113.3 | 96.3 | 228.1 |
| `170105` | `rec`, `0434` | 2,520/2,520 | 37.9 | 34.7 | 72.4 |
| `170105` | `rec`, `0461` | 651/2,520 | 606.2 | 612.3 | 720.7 |

The 13.6 mm figure is **conditional on just 16.6% of joints and 21 frames**.
It cannot be advertised as whole-sequence accuracy. The 113.3 mm all-finite
figure includes filled/extrapolated points. The second sequence has no valid
prepare metric; its failed stage left a file but did not mark prepare complete.

## Failure localization

For each saved 2-D detector observation, project the same-frame 3-D label
through the imported camera extrinsic and colour intrinsics (including
distortion), then compare pixel coordinates. The first view is aligned in
both sequences. In the second view of `170105`, the detector locks onto the
person's face instead of the annotated hand; the frame-0 overlay makes this
unambiguous. This is a detector false positive, not evidence of a 248-pixel
calibration shift.

| Sequence / view | Detector frames | 2-D median / P95 error | Median signed depth minus GT camera-Z |
|---|---:|---:|---:|
| `170650` / `0434` | 120 | 4.0 / 10.5 px | -12.7 mm |
| `170650` / `0461` | 21 | 5.6 / 18.4 px | +142.4 mm (89 valid depth joints) |
| `170105` / `0434` | 120 | 3.1 / 9.8 px | -11.4 mm |
| `170105` / `0461` | 31 | 248.0 / 292.0 px | -192.2 mm (wrong target) |

The first sequence's `0461` pixels are close to the hand, yet sampled depth
is ~14 cm behind annotated camera-Z where valid. This separates its failure
from the face false positive, but **does not identify whether depth alignment,
occlusion/background sampling, or source labels cause that discrepancy**.
The importer's GT round-trip (~4e-16 m) and reprojection/triangulation identity
(~0.0004 mm) test conversion arithmetic, not RGB-D alignment or detector
quality. The import depth checks failed; their thresholds must not be waived.

## Artifacts and next gate

- Raw subset: `data/raw_datasets/hocap/subject_1/{20231025_170650,20231025_170105}/`
- Imported episodes and machine-readable scores: `data/datasets/hocap/subject_1_<sequence>_043422252387-046122250168/gt/evaluation.json`
- Detector-vs-GT frame-0 overlays: each episode's `gt/detection_overlay_046122250168_000000.jpg` (green GT, red detector)

Before using HO-Cap for a publication claim, investigate second-view depth
registration on visible hand pixels, add hand-region/identity checks to reject
face false positives, select and freeze an appropriate held-out split and
camera pair, then report absolute errors with coverage and failure rate.

## Corrected pair and eight-view comparison

Screening all 28 pairs on the same 120-frame windows selected
`043422252387` + `105322251225` for hand/depth overlap. The importer was
extended to accept an explicit list of any two or more unique cameras, then
Extract and Prepare were run on the selected pair and all eight cameras.
The following are **smoothed observed** scores recovered from the original
run log; every row matched 2,520/2,520 annotated joints.

| Sequence | Views | MPJPE (mm) | P95 (mm) | PCK@20 mm |
|---|---|---:|---:|---:|
| `170650` | corrected pair | 11.334 | 22.710 | 87.70% |
| `170650` | all eight | 11.338 | 22.174 | 87.50% |
| `170105` | corrected pair | 11.672 | 22.802 | 89.80% |
| `170105` | all eight | 10.501 | 20.154 | 94.88% |

Eight views improved the second sequence but barely changed mean error in
the first. Camera `105322251564` returned no hand observations in `170105`;
fusion still produced full coverage on this bounded sample. The corrected
pair was selected *on these same evaluation windows*, so none of these rows
is held-out. The `0434` view's colour–depth consistency check still failed.
Published MANO labels do not validate the local Kinect calibration, object
pose, or robot execution. Re-download and rerun before publication.

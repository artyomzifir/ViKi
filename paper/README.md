# ViKi thesis and research proposal

The editable thesis is in `paper/thesis/`; the English/Russian abstracts are
in `paper/thesis/thesis.tex`. The separate Real2Sim2Real proposal is
`paper/research_proposal_real2sim2real.md` and its submission copy is the
adjacent `.docx` file. Build the thesis with `make -C paper/thesis` (XeLaTeX).

## Evidence in the manuscript

- The local 78-episode cube-transfer cohort and MuJoCo experiments are
  **historical results**. Their original raw episodes and derived artifacts
  were lost during the October 2026 workspace deletion. The source code and
  results text have been recovered from Git and task logs, but the numbers
  require regeneration from new recordings before a reproducible submission.
- The external HO-Cap pilot used two 120-frame subject-1 sequences, a
  selected camera pair and all eight cameras. Historical smoothed-observed
  MPJPE was 10.50–11.67 mm with 2,520/2,520 matched joints per comparison.
  The pair was selected on the same sequences, and HO-Cap labels are
  semi-automatic MANO annotations, not metrological ground truth for the
  local Kinect rig. See `docs/2026-10-13-hocap-real-sample-pilot.md`.
- Simulated robot execution is not a real-robot or sim-to-real validation.
  No RL training or deployed visual policy is claimed.

Run `scripts/build_thesis_results.py` only after the input artifacts listed in
the script have been regenerated. Figures under `paper/thesis/figures/` were
not recovered from the deleted workspace. The current PDF in Git predates
the restored October source edits; rebuild it before submission.

The degree/regulation audit is in `paper/REGULATORY_AUDIT.md`; verify its
conclusions against the applicable DSAI 2023 intake rules before filing.

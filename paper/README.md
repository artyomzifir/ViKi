# ViKi thesis and research proposal

The editable thesis is in `paper/thesis/`; the English/Russian abstracts are
in `paper/thesis/thesis.tex`. The separate Real2Sim2Real proposal is
`paper/research_proposal_real2sim2real.md` and its submission copy is the
adjacent `.docx` file. Build the thesis with `make -C paper/thesis` (XeLaTeX).

## Evidence in the manuscript

- The full 28-episode HO-Cap pre-retarget numerical JSON snapshot is in
  `paper/results/hocap/pre_retarget/`, with SHA-256 hashes and source-path
  mapping. It is separate from the retarget plans and original dataset.
  Chapter 4 now reports eight-view hand and object tracking for this cohort,
  including named object outliers; Appendix B contains all 28 hand and 112
  object rows and the official object-ID catalog. Regenerate the appendix with
  `python3 scripts/build_hocap_paper_tables.py` and verify it with
  `python3 scripts/build_hocap_paper_tables.py --check`. Object names follow
  the authors' [HO-Cap toolkit catalog](https://github.com/IRVLUTD/HO-Cap/blob/main/config/hocap_info.yaml).

- Earlier local cube-transfer claims, including downstream retargeting,
  MuJoCo execution, and policy-learning figures, are withdrawn from this
  manuscript. The source recordings, calibration, and derived artifacts are
  unavailable, so these figures must not be cited as results. The withdrawal
  is documented in docs/2026-10-14-local-78-retraction.md.
- The earlier HO-Cap two-sequence selected-pair pilot is retained only as
  historical in-sample development evidence. The full eight-view cohort is
  reproducible from the archived numerical JSON; its MANO/object references
  are still semi-automatic labels, not metrological ground truth for the
  local Kinect rig. See `docs/2026-10-13-hocap-real-sample-pilot.md`.
- Simulated robot execution is not a real-robot or sim-to-real validation.
  No RL training or deployed visual policy is claimed.

The old local-result generator is not part of the current manuscript build;
do not restore its figures or tables without preserved source episodes and a
new verification protocol. The thesis PDF can be rebuilt from the retained
HO-Cap evidence and current text.

The degree/regulation audit is in `paper/REGULATORY_AUDIT.md`; verify its
conclusions against the applicable DSAI 2023 intake rules before filing.

# DOE Thesis Compliance Checklist

Use this checklist to ensure thesis meets Innopolis DOE requirements.

## ✅ Migration Status (2026-11-01)

**DOE-compliant structure created!** See `thesis_doe.tex` and supporting files.

### Completed Items
- ✅ DOE document class (`extreport`, 14pt)
- ✅ Times font (`mathptmx`)
- ✅ Double spacing (2.0)
- ✅ IEEE biblatex with biber
- ✅ DOE margins (1" all sides)
- ✅ English abstract (`abstract_doe.tex`)
- ✅ Auto TOC/LOF/LOT
- ✅ Headers and footers
- ✅ Build system (`Makefile`)

### Pending Items
- ⚠️  **Install TeX packages** (run `./install_tex_deps.sh`)
- ⚠️  **Compile and test** (`make doe`)
- [ ] Title page from DOE template
- [ ] Russian abstract
- [ ] Page numbering setup
- [ ] Content audit per checklist below

### Files Created
- `thesis_doe.tex` — Main DOE file
- `abstract_doe.tex` — DOE abstract
- `DOE_COMPLIANCE.md` — Full compliance doc
- `DOE_MIGRATION_SUMMARY.md` — Quick ref
- `DOE_DONE.md` — Migration summary
- `INSTALL.md` — Setup instructions
- `Makefile` — Build automation
- `install_tex_deps.sh` — Dep installer

---

## Pre-Submission Requirements

### Document Structure
- [ ] Title page generated from DOE Word template (`title.pdf`)
  - [ ] English title filled
  - [ ] Russian title filled
  - [ ] Supervisor name (Alexei Kornaev)
  - [ ] Author name (Artem Tuzov)
  - [ ] Group (BS23-RO-01)
  - [ ] Year (2027)
  - [ ] Converted to PDF and placed in thesis root

- [x] Table of Contents auto-generated (`thesis_doe.tex` has `\tableofcontents`)
- [x] List of Tables included (if >3 tables) (`\listoftables` present)
- [x] List of Figures included (if >3 figures) (`\listoffigures` present)
- [x] Abstract (English) — max 1 page (`abstract_doe.tex` created)
- [ ] Abstract (Russian) — max 1 page (TODO: translate abstract_doe.tex)
- [ ] Page numbering:
  - [ ] Title page = page 1 (no number shown)
  - [ ] TOC/lists = Roman numerals (ii, iii, ...)
  - [ ] Chapter 1 starts at correct page number (set `\setcounter{page}{N}`)
  - [ ] Chapters = Arabic numerals (1, 2, 3, ...)

### Formatting
- [x] Document class: `extreport` with `[oneside,final,14pt,a4paper]` ✓
- [x] Font: Times New Roman (via `mathptmx` — tempora not available) ✓
- [x] Margins: 1in all sides (DOE requirement) ✓
- [x] Line spacing: 2.0 (double) throughout body text ✓
- [x] Paragraph indent: 1.25cm first line ✓
- [x] Headers/footers: chapter/section names in header, page numbers ✓

### Chapters
- [ ] Chapter 1: Introduction
  - [ ] Problem statement
  - [ ] Research objectives
  - [ ] Scope and limitations
  - [ ] Thesis structure
- [ ] Chapter 2: Literature Review
  - [ ] Human motion retargeting
  - [ ] RGB-D perception for manipulation
  - [ ] 3D Gaussian Splatting
  - [ ] Synthetic data for robot learning
  - [ ] Gap analysis
- [ ] Chapter 3: Methodology
  - [ ] Pipeline architecture
  - [ ] Perception stage (hand + object tracking)
  - [ ] Multi-embodiment retargeting
  - [ ] 3DGS scene reconstruction
  - [ ] Robot model compositing
  - [ ] Dataset generation procedure
- [ ] Chapter 4: Implementation / Experiments
  - [ ] Hardware setup
  - [ ] Software stack
  - [ ] Evaluation metrics
  - [ ] Experimental protocol
  - [ ] Results
- [ ] Chapter 5: Discussion
  - [ ] Results interpretation
  - [ ] Comparison to related work
  - [ ] Limitations
  - [ ] Threats to validity
- [ ] Chapter 6: Conclusion
  - [ ] Summary of contributions
  - [ ] Answers to research questions
  - [ ] Future work

### Figures and Tables
- [ ] All figures have captions below: "Fig. X.Y. Caption text."
- [ ] All tables have captions above: "TABLE X.Y\nCaption text"
- [ ] Figure numbering: chapter.figure (e.g., Fig. 3.2)
- [ ] Table numbering: chapter.Roman (e.g., TABLE 2.III)
- [ ] All figures referenced in text with `\ref{label}`
- [ ] Figures source files in `figs/` or `../common/figs/`
- [ ] Image quality: min 300 DPI for photos, vector for diagrams

### Bibliography
- [x] Bibliography engine: biblatex + biber ✓
- [x] Style: IEEE (`style=ieee`) ✓
- [ ] Bibliography file: `../common/references.bib` or `ref.bib` (currently points to `references.bib`)
- [ ] All citations formatted as `\cite{key}`
- [ ] All cited works appear in references
- [ ] No uncited entries in bibliography (unless explicitly marked)
- [ ] References section titled "Bibliography cited"

### Appendices (if any)
- [ ] Appendix A, B, C... labeled correctly
- [ ] Referenced in main text where relevant

### Language
- [ ] Main text: English
- [ ] Russian abstract provided
- [ ] Grammar checked (Grammarly / LanguageTool)
- [ ] Technical terms consistent throughout
- [ ] Acronyms defined on first use

### Content Quality
- [ ] No placeholder text (`\blindtext`, "TODO", "TBD")
- [ ] All `\ph{}` macros resolved with actual values
- [ ] All figures have real content (no "[placeholder]" boxes)
- [ ] All tables filled with real data
- [ ] All experiments described with sufficient detail for reproduction
- [ ] Ethical considerations addressed (if applicable)
- [ ] Dataset licensing mentioned (HO-Cap, MediaPipe, etc.)

## Compilation Test
- [ ] Compiles with pdflatex (no errors)
- [ ] Bibliography compiles with biber (no warnings)
- [ ] All cross-references resolved (no "??" in output)
- [ ] Final PDF opens without errors
- [ ] PDF metadata set (title, author) via `\hypersetup`

## Final Checks
- [ ] Proofread by supervisor
- [ ] Proofread by peer reviewer
- [ ] DOE format validated by thesis office
- [ ] File named per DOE convention: `Tuzov_Artem_Thesis_BS_2027.pdf`
- [ ] Submitted via university portal before deadline

---

## Thesis Defense Requirements (Separate)
- [ ] Presentation slides prepared (15-20 min talk)
- [ ] Code repository accessible: https://github.com/artyomzifir/ViKi
- [ ] Demo video prepared (optional but recommended)
- [ ] Defense scheduled with thesis committee

---

**Last updated:** 2026-11-01  
**Status:** DOE migration complete, awaiting TeX package installation and first compile  
**Next step:** Run `./install_tex_deps.sh` then `make doe`  
**Expected defense:** Spring 2027

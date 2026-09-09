# Build one XeLaTeX document from a per-document subdirectory of paper/.
#
# A subdirectory Makefile sets DOC to the main .tex basename, then includes
# this file:
#
#     DOC := thesis
#     include ../common/latex.mk
#
# XeLaTeX is required (Cyrillic via polyglossia + fontspec). Two passes resolve
# the table of contents and cross-references. References are a manual
# thebibliography, so no biber/bibtex step.

# Find the shared preamble (../common) and this document's own files.
TEXINPUTS := ../common:$(CURDIR):$(TEXINPUTS)
export TEXINPUTS

LATEX    ?= xelatex
LATEXOPT ?= -interaction=nonstopmode -halt-on-error

SOURCES := $(DOC).tex $(wildcard *.tex) ../common/preamble.tex
AUX     := $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).lof $(DOC).lot \
           $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz $(DOC).bbl $(DOC).blg

.PHONY: all clean distclean
all: $(DOC).pdf

$(DOC).pdf: $(SOURCES)
	$(LATEX) $(LATEXOPT) $(DOC).tex
	$(LATEX) $(LATEXOPT) $(DOC).tex

clean:
	$(RM) $(AUX)

distclean: clean
	$(RM) $(DOC).pdf

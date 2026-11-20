# ViKi Thesis Makefile (DOE compliant)

.PHONY: all clean view

all: thesis.pdf

thesis.pdf: thesis.tex abstract.tex references.bib chapters/*.tex appendix.tex
	pdflatex thesis.tex
	biber thesis
	pdflatex thesis.tex
	pdflatex thesis.tex
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.bcf *.run.xml

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.bcf *.run.xml

view: thesis.pdf
	evince thesis.pdf &

.DEFAULT_GOAL := all

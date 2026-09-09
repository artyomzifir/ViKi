# paper/ — one self-contained XeLaTeX document per subdirectory.
#
#   make            build every document
#   make thesis     build just paper/thesis/thesis.pdf
#   make clean      remove build intermediates (keeps the PDFs)
#   make distclean  also remove the built PDFs
#
# The shared preamble lives in common/ (see common/latex.mk).

DOCS := thesis annotation-ru demo

.PHONY: all clean distclean $(DOCS)

all: $(DOCS)

$(DOCS):
	$(MAKE) -C $@

clean:
	@for d in $(DOCS); do $(MAKE) -C $$d clean; done

distclean:
	@for d in $(DOCS); do $(MAKE) -C $$d distclean; done

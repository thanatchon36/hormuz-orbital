# Build the paper.   make figures  ->  make pdf
.PHONY: all figures pdf clean

all: figures pdf

figures:
	python fonts/make_fonts.py || echo "Inter not found: figures will use a fallback font"
	python fig_maps.py
	python fig_charts.py

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex
	cp main.pdf paper.pdf

clean:
	latexmk -C

#!/bin/bash

fig_dir=data/sunflow/fig/
cor_dir=data/sunflow/correlation/
inputFiles=($fig_dir*.esp)

latexText=( 
\\documentclass{article}
\\usepackage{makeidx}
\\usepackage{multicol} 
\\usepackage[bottom]{footmisc}
\\usepackage{xcolor}
\\usepackage{graphicx}
\\usepackage{epstopdf}
\\usepackage{epsfig}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\usepackage{epigraph}
\\usepackage{color}
\\usepackage{listings}
\\usepackage{hyperref}
\\usepackage{wrapfig}
\\usepackage{lipsum}
)
eps=$(ls ${fig_dir} | grep "\.eps")
include_figs() {
	#for file in ${inputFiles[@]}; do
	for file in ${eps}; do
		echo "\begin{wrapfigure}[10]{r}{0.1\textwidth}"
		echo "\includegraphics[width= 1\textwidth]{${fig_dir}${file}}"
#		IFS='_' read -ra ADDR <<< "${file}"
#		echo "\caption{${ADDR[0]} is considered in K-means. Sorted by ${ADDR[1]}, ${ADDR[1]} and ${ADDR[2]} are showed in figure, the data was extracted from file ${ADDR[3]}}"
#		echo "Correlation coefficient data is as follows:"

		echo "\end{wrapfigure}"
	done
}

prefix() {
	for tex in ${latexText[@]}; do
		echo ${tex}
	done
}

prefix > pdf_gen.tex
echo "\begin{document}">> pdf_gen.tex
include_figs>> pdf_gen.tex
echo "\end{document}">> pdf_gen.tex



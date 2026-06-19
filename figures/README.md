# Figures

`scripts/06_make_figures.py` writes these (300 dpi, no titles, English labels,
panel letters centred below each panel). Filenames match the manuscript's
`\includegraphics{figureNN.png}` references.

| File          | Content                                                    |
|---------------|------------------------------------------------------------|
| figure01.png  | Dataset overview (tasks, rounds, age, sex, QC histograms)  |
| figure02.png  | Vergence paradigm: measured vs demanded, transitions       |
| figure03.png  | Grand-average conv/div response + first-order model fit    |
| figure04.png  | Test-retest scatter (descriptive vs model), coloured by QC |
| figure05.png  | Data-quality confound + effect-size comparison             |
| figure06.png  | Bland-Altman agreement + variance decomposition            |

Copy them into `manuscript/figures/` before compiling the LaTeX.

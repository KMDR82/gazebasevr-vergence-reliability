# Notebooks

Exploratory / reporting notebooks. The analysis was developed cell-by-cell on
Kaggle; these mirror that flow:

1. `01_data_overview` — dataset summary (Figure 1 inputs)
2. `02_pipeline` — demanded vergence, filtering, segmentation, templates, fits
3. `03_reliability` — ICC / SEM / MDC and the bootstrap ICC difference
4. `04_quality_confound` — valid-sample fraction and the quality-adjusted re-test
5. `05_figures` — regenerate the six manuscript figures

**Kaggle note:** `/kaggle/working` is wiped between sessions and manifest paths
go stale, so rebuild the recording index with a glob at the start of each
session before running anything else.

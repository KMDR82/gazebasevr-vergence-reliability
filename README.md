# gazebasevr-vergence-reliability

Analysis code and manuscript for a study on the **test–retest reliability of
binocular vergence biomarkers measured with virtual-reality eye tracking**,
using the GazeBaseVR dataset.

The study shows that (i) **descriptive** vergence biomarkers are weakly
reliable (short-term ICC ≈ 0.25–0.46) and confounded by data quality, and that
(ii) extracting the same biomarkers from a **delayed first-order exponential
model** (`v(t) = A·(1 − e^{−(t − t₀)/τ}`, mean R² ≈ 0.96) roughly doubles the
reliability of peak velocity (ICC ≈ 0.53–0.57) and removes the data-quality
confound, including a spurious sex difference. A second-order model fits
marginally better in-sample but is rejected on reproducibility grounds.

> Companion code for the article submitted to the *Journal of Eye Movement
> Research* (MDPI) Special Issue *“Digital Advances in the Evaluation of
> Binocular Vision and Eye Movements.”*

## Method

For every VRG (vergence) recording the **demanded** vergence angle is computed
purely from stimulus geometry (target position and the recorded eye-centre
coordinates), independently of the measured gaze — this is the step that avoids
circular validation. The **measured** vergence (`lx − rx`) is de-blinked with a
velocity threshold and low-pass filtered (zero-phase 10 Hz Butterworth).
Responses are segmented at the demanded-depth transitions, pooled into
convergence/divergence **ensemble-median templates**, and reduced to biomarkers
both descriptively (latency, gain, peak velocity, settling time) and by fitting
the first-order model. Reliability is quantified with ICC(2,1), bootstrap CIs,
SEM, MDC, and Spearman–Brown projections; data-quality confounds are tested with
Spearman correlations and OLS adjustment.

## Repository layout

```
gazebasevr-vergence-reliability/
├── README.md
├── LICENSE                      MIT (code)
├── CITATION.cff
├── requirements.txt / environment.yml
├── data/
│   ├── README.md                how to obtain GazeBaseVR (NOT redistributed)
│   └── raw/                     place the downloaded CSVs here (git-ignored)
├── src/
│   ├── config.py                FS, QC thresholds, filter & model constants
│   ├── io_gazebasevr.py         filename parsing, manifest, path remap, loading
│   ├── signal_processing.py     demanded vergence + clean() (blink mask + 10 Hz)
│   ├── segmentation.py          transitions, demanded step, conv/div templates
│   ├── biomarkers.py            descriptive biomarkers + per-recording extraction
│   ├── models.py                first-order (selected) + second-order (rejected)
│   ├── reliability.py           ICC(2,1), bootstrap, SEM/MDC, Spearman–Brown
│   ├── quality.py               IPD, quality confound, spurious-effect adjustment
│   └── figures.py               the six manuscript figures (300 dpi)
├── scripts/
│   ├── 01_build_manifest.py     scan raw/ -> vrg_manifest.csv
│   ├── 02_extract_biomarkers.py -> biomarker_table.csv  (descriptive + QC fields)
│   ├── 03_fit_models.py         -> model_biomarker_table.csv, model2_table.csv
│   ├── 04_reliability.py        short/long ICC, SEM/MDC, descriptive vs model
│   ├── 05_confounds.py          quality confound + spurious sex effect
│   ├── 06_make_figures.py       regenerate figure00..figure06
│   └── run_all.py               run 01 -> 06 in order
├── exploratory/
│   └── identification.py        gaze-based identification probe (NOT in paper)
├── results/tables/              CSV outputs (git-ignored)
├── figures/                     PNG outputs (git-ignored; see figures/README.md)
└── docs/pipeline.md
```

## Data

Raw data are not included. Download **GazeBaseVR**
([DOI 10.6084/m9.figshare.21308391](https://doi.org/10.6084/m9.figshare.21308391)),
extract it under `data/raw/`, and see [`data/README.md`](data/README.md) for the
expected CSV columns. Only the **VRG** task is analysed.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all.py --raw data/raw --out results --figdir figures
# or run stages individually, e.g.:
python scripts/01_build_manifest.py --raw data/raw --out results
```

All stage scripts read `--raw` (raw CSV directory) and write intermediate tables
to `--out` (default `results/`). Reliability and confound scripts print the
numbers reported in the manuscript; `06_make_figures.py` writes
`figure00`–`figure06` to `--figdir`.

## Reproducing on Kaggle

The analysis was developed on Kaggle. Because `/kaggle/working` is wiped between
sessions and manifest paths go stale, `src/io_gazebasevr.remap_paths()` rebuilds
the file index from a glob at the start of each session — see `docs/pipeline.md`.

## Citation

If you use this code, please cite the article (see `CITATION.cff`) and the
GazeBaseVR dataset (Lohr et al., 2023).

## License

Code: MIT (`LICENSE`). Manuscript text and figures follow the publishing
journal's terms (CC BY for MDPI/JEMR).

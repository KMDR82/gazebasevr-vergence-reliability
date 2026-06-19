# Pipeline

Scripts run in order; each reads `--raw` (raw CSV dir) and `--out` (tables dir).

1. **01_build_manifest** - parse `S_<rs>_S<session>_<num>_<task>.csv`, keep VRG,
   write `vrg_manifest.csv` (round = first digit of `rs`, subject = last three).
2. **02_extract_biomarkers** - per recording: demanded vergence from geometry
   (gaze-independent), `clean()` (|vel|>60 deg/s blink mask + 10 Hz Butterworth),
   segment at demanded-depth transitions, ensemble-median conv/div templates,
   descriptive biomarkers + QC fields -> `biomarker_table.csv`.
3. **03_fit_models** - on the QC subset, fit the delayed first-order model
   (`A(1-e^{-(t-t0)/tau})`, peak velocity `A/tau`) -> `model_biomarker_table.csv`;
   also fit the second-order model and report AIC/zeta -> `model2_table.csv`.
4. **04_reliability** - ICC(2,1) with bootstrap CIs (short-term S1 vs S2 and
   long-term across rounds), SEM, MDC, Spearman-Brown sessions-needed, and the
   descriptive-vs-model ICC comparison.
5. **05_confounds** - peak velocity vs valid-sample fraction (Spearman) and the
   apparent sex difference, unadjusted vs quality-adjusted (OLS).
6. **06_make_figures** - regenerate `figure01`..`figure06`.

## QC filter
`valid_frac >= 0.80`, `corr >= 0.50`, `n_conv >= 6`, `n_div >= 6` (see `src/config.py`).

## Kaggle note
`/kaggle/working` is wiped between sessions and manifest paths go stale, so
`io_gazebasevr.remap_paths()` rebuilds the basename->path index from a glob at
the start of every run.

# Data

The raw data are **not** redistributed here. Download GazeBaseVR from figshare:

- **GazeBaseVR** — Lohr, Aziz, Friedman & Komogortsev (2023).
  DOI: [10.6084/m9.figshare.21308391](https://doi.org/10.6084/m9.figshare.21308391)

Place the extracted recordings under `data/raw/`. Only the **vergence (VRG)**
task is used in this study (407 participants, up to 6 longitudinal rounds,
2 sessions per round, 250 Hz).

`data/interim/` and `data/processed/` are created by the pipeline and are
git-ignored.

## Expected CSV columns (per recording)

`n` (time, ms), `lx`, `rx` (per-eye horizontal gaze angle, deg),
`xT`, `zT` (target horizontal position and depth), and the eye-centre
coordinates `clx, cly, clz, crx, cry, crz`. `participant_details.xlsx`
(participant id, age, gender, rounds) is used for the dataset-overview figure
and the sex-difference confound check.

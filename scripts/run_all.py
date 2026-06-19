#!/usr/bin/env python3
"""Run the full pipeline (stages 01-06) in order."""
import argparse, subprocess, sys, os

STAGES = ["01_build_manifest.py", "02_extract_biomarkers.py", "03_fit_models.py",
          "04_reliability.py", "05_confounds.py", "06_make_figures.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    for s in STAGES:
        print(f"\n===== {s} =====")
        cmd = [sys.executable, os.path.join(here, s), "--out", a.out]
        if s in ("01_build_manifest.py", "02_extract_biomarkers.py",
                 "03_fit_models.py", "05_confounds.py", "06_make_figures.py"):
            cmd += ["--raw", a.raw]
        if s == "06_make_figures.py":
            cmd += ["--figdir", a.figdir]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

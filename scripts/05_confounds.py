#!/usr/bin/env python3
"""Data-quality confound and the spurious sex difference in peak velocity."""
import argparse, os
import numpy as np, pandas as pd
import _bootstrap  # noqa: F401
from src.io_gazebasevr import find_one
from src.biomarkers import apply_qc
from src.quality import quality_confound, spurious_effect_check


def col(df, *ks):
    for c in df.columns:
        if all(k in str(c).lower() for k in ks):
            return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    bt = pd.read_csv(os.path.join(a.out, "biomarker_table.csv"))
    mb = pd.read_csv(os.path.join(a.out, "model_biomarker_table.csv"))
    q = apply_qc(bt).merge(mb, on=["subject", "round", "session", "file"], how="inner")

    agg = q.groupby("subject").agg(vf=("valid_frac", "mean"),
                                   pv_desc_d=("div_peakvel", "mean"),
                                   pv_mod_d=("div_pv_m", "mean"),
                                   pv_desc_c=("conv_peakvel", "mean"),
                                   pv_mod_c=("conv_pv_m", "mean"))
    print("=== Peak-velocity vs data quality (Spearman vs valid_frac) ===")
    for nm, c in [("descriptive div", "pv_desc_d"), ("model div", "pv_mod_d"),
                  ("descriptive conv", "pv_desc_c"), ("model conv", "pv_mod_c")]:
        res = quality_confound(agg[c], agg["vf"])
        print(f"{nm:16s}: r={res['r']:+.3f} p={res['p']:.4f}")

    meta_path = find_one(a.raw, "participant_details.xlsx")
    if not meta_path:
        print("\nparticipant_details.xlsx not found - skipping sex-effect check.")
        return
    meta = pd.read_excel(meta_path)
    PID, GEN = col(meta, "participant", "id"), col(meta, "gender")
    gmap = pd.DataFrame({"subject": meta[PID],
                         "gender": meta[GEN].astype(str).str.strip()}).groupby("subject").first()
    A = agg.join(gmap, how="inner")
    A = A[A.gender.isin(["Male", "Female"])].copy()
    A["gf"] = (A.gender == "Female").astype(int)
    print("\n=== Apparent sex difference: unadjusted vs quality-adjusted (OLS) ===")
    for nm, c in [("descriptive div_pv", "pv_desc_d"), ("model div_pv", "pv_mod_d")]:
        res = spurious_effect_check(A, c, group="gf", quality="vf")
        print(f"{nm:18s}: p(unadjusted)={res['p_unadjusted']:.4f} "
              f"| p(quality-adjusted)={res['p_quality_adjusted']:.4f}")


if __name__ == "__main__":
    main()

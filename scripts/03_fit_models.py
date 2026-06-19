#!/usr/bin/env python3
"""Fit the first-order model (selected) and compare against second-order."""
import argparse, os, time
import numpy as np, pandas as pd
import _bootstrap  # noqa: F401
from src.io_gazebasevr import index_csvs
from src.signal_processing import measured_vergence, demanded_vergence, clean
from src.segmentation import templates
from src.biomarkers import apply_qc
from src.models import fit_first_order, fit_both


def _templates(path):
    d = pd.read_csv(path)
    vf, _ = clean(measured_vergence(d))
    return templates(vf, demanded_vergence(d), d["zT"].to_numpy(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    bt = pd.read_csv(os.path.join(a.out, "biomarker_table.csv"))
    man = index_csvs(a.raw)
    sub = apply_qc(bt)

    rows1, rows2, t0 = [], [], time.time()
    for _, r in sub.iterrows():
        base = dict(subject=r["subject"], round=r["round"],
                    session=r["session"], file=r["file"])
        rec1, rec2 = dict(base), dict(base)
        tpl = _templates(man[r["file"]]) if r["file"] in man else {"conv": None, "div": None}
        for dn in ("conv", "div"):
            t = tpl.get(dn)
            if t is not None:
                try:
                    fo = fit_first_order(t, dn)
                    rec1.update({f"{dn}_lat_m": fo["latency_ms"], f"{dn}_tau_m": fo["tau_ms"],
                                 f"{dn}_A_m": fo["A"], f"{dn}_pv_m": fo["peakvel"], f"{dn}_r2": fo["r2"]})
                except Exception:
                    pass
                rec2.update(fit_both(t, dn))
        rows1.append(rec1); rows2.append(rec2)
        if len(rows1) % 200 == 0:
            print(f"{len(rows1)}/{len(sub)}  ({time.time() - t0:.0f}s)")

    mb = pd.DataFrame(rows1); m2 = pd.DataFrame(rows2)
    mb.to_csv(os.path.join(a.out, "model_biomarker_table.csv"), index=False)
    m2.to_csv(os.path.join(a.out, "model2_table.csv"), index=False)
    if len(mb) == 0 or "conv_r2" not in mb:
        print("no successful model fits (check QC thresholds / data).")
        return
    print("first-order R^2: conv", round(mb["conv_r2"].mean(), 3),
          "| div", round(mb["div_r2"].mean(), 3))
    for dn in ("conv", "div"):
        if f"{dn}1_aic" not in m2 or f"{dn}2_aic" not in m2:
            continue
        pref = (m2[f"{dn}2_aic"] < m2[f"{dn}1_aic"]).mean()
        print(f"{dn}: R2 {m2[f'{dn}1_r2'].mean():.3f}->{m2[f'{dn}2_r2'].mean():.3f} "
              f"| AIC prefers 2nd-order {pref:.0%} | median zeta {m2[f'{dn}2_zeta'].median():.2f}")


if __name__ == "__main__":
    main()

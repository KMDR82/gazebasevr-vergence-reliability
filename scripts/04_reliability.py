#!/usr/bin/env python3
"""Short/long-term ICC, SEM/MDC, Spearman-Brown; descriptive vs model-based."""
import argparse, os
import numpy as np, pandas as pd
import _bootstrap  # noqa: F401
from src.config import DESC_FEATS
from src.biomarkers import apply_qc
from src.reliability import (short_term_icc, long_term_icc, pivot_pairs, sem_mdc,
                             spearman_brown, sessions_for_target, icc21,
                             bootstrap_icc_difference)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    bt = pd.read_csv(os.path.join(a.out, "biomarker_table.csv"))
    mb = pd.read_csv(os.path.join(a.out, "model_biomarker_table.csv"))
    q = apply_qc(bt)

    print("=== Descriptive biomarkers: short- and long-term ICC ===")
    rows = []
    for f in DESC_FEATS:
        s = short_term_icc(q, f); l = long_term_icc(q, f)
        x = pivot_pairs(q, f)
        sm = sem_mdc(x, s["icc"]) if len(x) >= 10 else dict(sem=np.nan, mdc=np.nan)
        kn = sessions_for_target(s["icc"])
        rows.append(dict(feature=f, n_short=s["n"], ICC_short=round(s["icc"], 2),
                         CI=f"[{s['ci'][0]:.2f},{s['ci'][1]:.2f}]",
                         SEM=round(sm["sem"], 1), MDC=round(sm["mdc"], 1),
                         ICC_long=round(l["icc"], 2), n_long=l["n"],
                         k_for_075=("inf" if not np.isfinite(kn) else int(np.ceil(kn)))))
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(a.out, "reliability_descriptive.csv"), index=False)

    print("\n=== Descriptive vs model-based ICC (short-term) ===")
    comp = [("peakvel conv", "conv_peakvel", "conv_pv_m"),
            ("peakvel div", "div_peakvel", "div_pv_m"),
            ("latency div", "div_latency_ms", "div_lat_m"),
            ("tau/settle div", "div_settle_ms", "div_tau_m")]
    out = []
    for nm, fd, fm in comp:
        a_ = q.pivot_table(index=["subject", "round"], columns="session", values=fd).dropna()
        b_ = mb.pivot_table(index=["subject", "round"], columns="session", values=fm).dropna()
        idx = a_.index.intersection(b_.index)
        Xd = a_.loc[idx].values.astype(float); Xm = b_.loc[idx].values.astype(float)
        res = bootstrap_icc_difference(Xd, Xm)
        smm = sem_mdc(Xm, res["icc_model"])
        out.append(dict(metric=nm, ICC_desc=round(res["icc_desc"], 2),
                        ICC_model=round(res["icc_model"], 2),
                        delta=round(res["delta"], 2),
                        dCI=f"[{res['ci'][0]:+.2f},{res['ci'][1]:+.2f}]",
                        SEM_model=round(smm["sem"], 1), MDC_model=round(smm["mdc"], 1)))
        print(f"{nm:16s} desc {res['icc_desc']:.2f} -> model {res['icc_model']:.2f} "
              f"| d={res['delta']:+.2f} 95%CI [{res['ci'][0]:+.2f},{res['ci'][1]:+.2f}]")
    pd.DataFrame(out).to_csv(os.path.join(a.out, "reliability_model_vs_descriptive.csv"), index=False)


if __name__ == "__main__":
    main()

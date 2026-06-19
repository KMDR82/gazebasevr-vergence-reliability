#!/usr/bin/env python3
"""Regenerate the six manuscript figures (figure01..figure06).

figure01 dataset overview | figure02 paradigm | figure03 grand average + fit
figure04 test-retest scatter | figure05 data-quality confound | figure06 agreement+variance
"""
import argparse, os, glob, re
import numpy as np, pandas as pd
import _bootstrap  # noqa: F401
from src.config import FS, DESC_FEATS
from src.io_gazebasevr import remap_paths, index_csvs, find_one
from src.signal_processing import measured_vergence, demanded_vergence, clean
from src.segmentation import transitions
from src.biomarkers import apply_qc
from src.models import fit_first_order
from src import figures as F


def col(df, *ks):
    for c in df.columns:
        if all(k in str(c).lower() for k in ks):
            return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    fp = lambda n: os.path.join(a.figdir, n)

    bt = pd.read_csv(os.path.join(a.out, "biomarker_table.csv"))
    man = remap_paths(pd.read_csv(os.path.join(a.out, "vrg_manifest.csv")), a.raw)
    idx = index_csvs(a.raw)
    allcsv = list(idx.values())

    # ---- figure01: dataset overview ----
    meta_path = find_one(a.raw, "participant_details.xlsx")
    if meta_path:
        meta = pd.read_excel(meta_path)
        PID, AGE, GEN = col(meta, "participant", "id"), col(meta, "age"), col(meta, "gender")
        RND = col(meta, "recording", "round") or col(meta, "round")
        pat = re.compile(r"_([A-Z]{3})\.csv$")
        tasks = pd.Series([pat.search(p).group(1) for p in allcsv if pat.search(p)]).value_counts()
        pp = meta.groupby(PID).agg(age=(AGE, "first"), gender=(GEN, "first"),
                                   nrounds=(RND, "nunique"))
        F.fig_dataset_overview(meta, bt, tasks, pp["age"], pp["gender"],
                               pp["nrounds"], fp("figure01.png"))
        gmap = meta.groupby(PID).agg(gender=(GEN, "first"))
    else:
        meta = gmap = None
        print("participant_details.xlsx not found - figure01/figure05 sex panel limited.")

    # ---- recompute first-order model table + grand-average templates ----
    sub = apply_qc(bt)
    L = 150
    rows, Tc, Td = [], [], []
    for _, r in sub.iterrows():
        p = idx.get(r["file"])
        if not p:
            continue
        d = pd.read_csv(p)
        vf, _ = clean(measured_vergence(d)); vd = demanded_vergence(d); zT = d["zT"].to_numpy(float)
        tr = transitions(zT)
        if len(tr) < 4:
            continue
        sl = int(np.median(np.diff(tr)))
        rec = dict(subject=r["subject"], round=r["round"], session=r["session"])
        for dn in ("conv", "div"):
            S = []
            for k in range(len(tr)):
                i0 = tr[k]; i1 = i0 + sl
                if i1 > len(vf):
                    break
                dd = (np.nanmedian(vd[i0 + int(.2 * FS):i1])
                      - np.nanmedian(vd[max(i0 - int(.3 * FS), 0):i0]))
                if (dn == "conv") == (dd > 0):
                    seg = vf[i0:i1] - np.nanmedian(vf[max(i0 - int(.2 * FS), 0):i0])
                    if np.isfinite(seg).mean() > .6:
                        S.append(seg[:sl])
            if len(S) < 3:
                rec[dn + "_pv_m"] = np.nan; rec[dn + "_lat_m"] = np.nan; continue
            tm = np.nanmedian(np.array(S), 0)
            try:
                fo = fit_first_order(tm, dn)
                rec[dn + "_lat_m"], rec[dn + "_pv_m"] = fo["latency_ms"], fo["peakvel"]
            except Exception:
                rec[dn + "_lat_m"] = rec[dn + "_pv_m"] = np.nan
            if len(Tc) < 250:
                tt = np.interp(np.linspace(0, len(tm) - 1, L), np.arange(len(tm)),
                               pd.Series(tm).interpolate(limit_direction="both").to_numpy())
                (Tc if dn == "conv" else Td).append(tt)
        rows.append(rec)
    mtab = pd.DataFrame(rows); Tc = np.array(Tc); Td = np.array(Td)
    print("model recomputed for", len(mtab), "recordings")

    # ---- figure02: paradigm (first QC recording) ----
    first = sub.merge(man[["file", "path"]], on="file", how="left")["path"].dropna().iloc[0]
    d0 = pd.read_csv(first)
    vf0, _ = clean(measured_vergence(d0))
    F.fig_paradigm(vf0, demanded_vergence(d0), d0["zT"].to_numpy(float), fp("figure02.png"))

    # ---- figure03: grand average + first-order fit ----
    F.fig_grand_average(Tc, Td, fp("figure03.png"))

    # ---- figure04: test-retest scatter (descriptive vs model) ----
    F.fig_testretest_scatter(sub, mtab, fp("figure04.png"))

    # ---- figure05: data-quality confound ----
    if gmap is not None:
        F.fig_quality_confound(sub, mtab, gmap, fp("figure05.png"))

    # ---- figure06: agreement (Bland-Altman) + variance decomposition ----
    labels = ["C lat", "D lat", "C gain", "D gain", "C pv", "D pv", "C settle", "D settle"]
    F.fig_agreement_and_variance(sub, mtab, DESC_FEATS, labels, fp("figure06.png"))

    print("figures written to", a.figdir)


if __name__ == "__main__":
    main()

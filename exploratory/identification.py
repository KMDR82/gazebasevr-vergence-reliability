#!/usr/bin/env python3
"""EXPLORATORY (not part of the manuscript): can a within-round S1->S2 vergence
template re-identify the subject? Reports Rank-1, EER and a permutation p-value
for monocular / vergence / binocular waveform features.

This is a side-probe kept for completeness; the paper's claims concern
reliability, not biometrics.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FS
from src.io_gazebasevr import remap_paths
from src.signal_processing import clean
from src.biomarkers import apply_qc
from src.segmentation import transitions


def waves(path, mode, L=50, T=int(2.0 * FS)):
    d = pd.read_csv(path)
    lx = clean(d["lx"].values)[0]; rx = clean(d["rx"].values)[0]
    sigs = {"mono": [lx], "verg": [lx - rx], "bino": [lx, rx]}[mode]
    vd = (np.degrees(np.arctan2((d["xT"] - d["clx"]).values, (d["zT"] - d["clz"]).values))
          - np.degrees(np.arctan2((d["xT"] - d["crx"]).values, (d["zT"] - d["crz"]).values)))
    zT = d["zT"].values; tr = transitions(zT)
    if len(tr) < 4:
        return None
    sl = int(np.median(np.diff(tr))); feat = []
    for sig in sigs:
        for dn in ("conv", "div"):
            S = []
            for k in range(len(tr)):
                i0 = tr[k]
                if i0 + T > len(sig):
                    break
                dd = (np.nanmedian(vd[i0 + int(.2 * FS):i0 + sl])
                      - np.nanmedian(vd[max(i0 - int(.3 * FS), 0):i0]))
                if (dn == "conv") == (dd > 0):
                    seg = sig[i0:i0 + T] - np.nanmedian(sig[max(i0 - int(.2 * FS), 0):i0])
                    if np.isfinite(seg).mean() > .6:
                        S.append(seg)
            if len(S) < 3:
                return None
            t = np.nanmedian(np.array(S), 0)
            feat.append(np.interp(np.linspace(0, len(t) - 1, L), np.arange(len(t)),
                                  pd.Series(t).interpolate(limit_direction="both").to_numpy()))
    return np.concatenate(feat)


def rank1(G, P):
    return (cdist(P, G).argmin(1) == np.arange(len(P))).mean()


def eer(G, P):
    D = cdist(P, G); gen = D[np.arange(len(P)), np.arange(len(P))]
    imp = D[~np.eye(len(P), dtype=bool)]; best, e = 1, 1
    for th in np.linspace(D.min(), D.max(), 300):
        far = (imp <= th).mean(); frr = (gen > th).mean()
        if abs(far - frr) < best:
            best = abs(far - frr); e = (far + frr) / 2
    return e


def perm_p(G, P, B=2000):
    obs = rank1(G, P); rng = np.random.default_rng(0)
    return (sum(rank1(G, P[rng.permutation(len(P))]) >= obs for _ in range(B)) + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    bt = pd.read_csv(os.path.join(a.out, "biomarker_table.csv"))
    man = remap_paths(pd.read_csv(os.path.join(a.out, "vrg_manifest.csv")), a.raw)
    q = apply_qc(bt).merge(man[["file", "path"]], on="file", how="left")
    pairs = []
    for _, g in q.groupby("subject"):
        for _, gr in g.groupby("round"):
            s = gr.set_index("session")
            if 1 in s.index and 2 in s.index:
                pairs.append((s.loc[1], s.loc[2])); break
    for mode in ("mono", "verg", "bino"):
        G, P = [], []
        for x, y in pairs:
            wa, wb = waves(x["path"], mode), waves(y["path"], mode)
            if wa is not None and wb is not None:
                G.append(wa); P.append(wb)
        G, P = np.array(G), np.array(P)
        mu, sd = G.mean(0), G.std(0) + 1e-9; G = (G - mu) / sd; P = (P - mu) / sd
        print(f"{mode:5s}: n={len(G)}  Rank-1={rank1(G, P):.1%} (chance {1/len(G):.1%})  "
              f"EER={eer(G, P):.1%}  p={perm_p(G, P):.4f}")


if __name__ == "__main__":
    main()

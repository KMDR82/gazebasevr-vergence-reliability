"""Descriptive biomarkers and per-recording extraction (-> biomarker_table)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import (FS, ONSET_FRAC, SETTLE_FRAC, MOVE_WINDOW_S, MIN_SEGMENTS,
                     ZT_EPS, SETTLE_S, PRE_S, MIN_FINITE_FRAC)
from .signal_processing import measured_vergence, demanded_vergence, clean, correlation
from .segmentation import transitions, demanded_step_abs


def descriptive_biomarkers(template, direction, ddem_abs):
    """Latency (ms), gain, peak velocity (deg/s), settling time (ms) from a template.

    Latency: first time the response exceeds 10% of amplitude for 3 samples.
    Gain: |steady level| / demanded step. Peak velocity: max signed velocity in
    the first ``MOVE_WINDOW_S`` s. Settle: first crossing of 90% amplitude.
    """
    tau = np.arange(len(template)) / FS
    sgn = 1 if direction == "conv" else -1
    steady = np.nanmedian(template[int(0.8 * len(template)):])
    amp = sgn * steady
    if not np.isfinite(amp) or amp <= 0:
        return dict(latency_ms=np.nan, gain=np.nan, peakvel=np.nan, settle_ms=np.nan)
    vel = np.gradient(template, 1 / FS)
    pv = np.nanmax(sgn * vel[:int(MOVE_WINDOW_S * FS)])
    lat = np.nan
    thr = ONSET_FRAC * amp
    for k in range(int(0.05 * FS), len(template) - 3):
        if np.all(sgn * template[k:k + 3] > thr):
            lat = tau[k]
            break
    st = np.nan
    for k in range(len(template)):
        if sgn * template[k] > SETTLE_FRAC * amp:
            st = tau[k]
            break
    return dict(latency_ms=1000 * lat if lat == lat else np.nan,
                gain=abs(steady) / ddem_abs,
                peakvel=pv,
                settle_ms=1000 * st if st == st else np.nan)


def extract_recording(path: str) -> dict:
    """Full per-recording row: QC fields + conv/div descriptive biomarkers."""
    d = pd.read_csv(path)
    vf, _ = clean(measured_vergence(d))
    vd = demanded_vergence(d)
    zT = d["zT"].to_numpy(float)
    out = dict(valid_frac=float(np.isfinite(vf).mean()),
               corr=correlation(vf, vd),
               ddem=demanded_step_abs(vd), n_conv=0, n_div=0)
    tr = transitions(zT)
    if len(tr) < 4 or not np.isfinite(out["ddem"]) or out["ddem"] < 0.2:
        return out
    seglen = int(np.median(np.diff(tr)))
    for direction in ("conv", "div"):
        segs = []
        for k in range(len(tr)):
            i0 = tr[k]
            i1 = i0 + seglen
            if i1 > len(vf):
                break
            dd = (np.nanmedian(vd[i0 + int(SETTLE_S * FS):i1])
                  - np.nanmedian(vd[max(i0 - int(PRE_S * FS), 0):i0]))
            if (direction == "conv") == (dd > 0):
                seg = vf[i0:i1] - np.nanmedian(vf[max(i0 - int(SETTLE_S * FS), 0):i0])
                if np.isfinite(seg).mean() > MIN_FINITE_FRAC:
                    segs.append(seg)
        out[f"n_{direction}"] = len(segs)
        if len(segs) >= MIN_SEGMENTS:
            bm = descriptive_biomarkers(np.nanmedian(np.array(segs), axis=0),
                                        direction, out["ddem"])
        else:
            bm = dict(latency_ms=np.nan, gain=np.nan, peakvel=np.nan, settle_ms=np.nan)
        for key, val in bm.items():
            out[f"{direction}_{key}"] = val
    return out


def apply_qc(bt: pd.DataFrame) -> pd.DataFrame:
    """Filter recordings by the QC thresholds in config.QC."""
    from .config import QC
    return bt[(bt["valid_frac"] >= QC["valid_frac"]) & (bt["corr"] >= QC["corr"])
              & (bt["n_conv"] >= QC["n_conv"]) & (bt["n_div"] >= QC["n_div"])].copy()

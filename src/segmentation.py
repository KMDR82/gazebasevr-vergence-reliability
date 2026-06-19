"""Stimulus-locked segmentation and ensemble-median templates."""
from __future__ import annotations
import numpy as np

from .config import FS, ZT_EPS, PRE_S, SETTLE_S, MIN_FINITE_FRAC, MIN_SEGMENTS


def transitions(zT: np.ndarray) -> np.ndarray:
    """Sample indices where the demanded depth steps (independent ground truth)."""
    return np.where(np.abs(np.diff(zT)) > ZT_EPS)[0] + 1


def demanded_step_abs(vd: np.ndarray) -> float:
    """Magnitude of the demanded vergence step (deg), from the two depth levels."""
    med = np.nanmedian(vd)
    return float(abs(np.nanmedian(vd[vd >= med]) - np.nanmedian(vd[vd < med])))


def templates(vf: np.ndarray, vd: np.ndarray, zT: np.ndarray, fs: float = FS):
    """Return {'conv': template|None, 'div': template|None} ensemble medians.

    Segments are cut at each demanded transition, classified conv/div by the
    sign of the demanded change, baseline-subtracted, and kept only if >60% of
    samples are finite. The per-direction template is the sample-wise median.
    """
    tr = transitions(zT)
    if len(tr) < 4:
        return {"conv": None, "div": None}
    seglen = int(np.median(np.diff(tr)))
    out = {}
    for direction in ("conv", "div"):
        segs = []
        for k in range(len(tr)):
            i0 = tr[k]
            i1 = i0 + seglen
            if i1 > len(vf):
                break
            dd = (np.nanmedian(vd[i0 + int(SETTLE_S * fs):i1])
                  - np.nanmedian(vd[max(i0 - int(PRE_S * fs), 0):i0]))
            if (direction == "conv") == (dd > 0):
                seg = vf[i0:i1] - np.nanmedian(vf[max(i0 - int(SETTLE_S * fs), 0):i0])
                if np.isfinite(seg).mean() > MIN_FINITE_FRAC:
                    segs.append(seg[:seglen])
        out[direction] = (np.nanmedian(np.array(segs), axis=0)
                          if len(segs) >= MIN_SEGMENTS else None)
        out[f"n_{direction}"] = len(segs)
    return out

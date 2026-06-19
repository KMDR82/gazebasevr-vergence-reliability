"""Demanded vergence (gaze-independent) and signal cleaning."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .config import FS, VMAX_DPS, LOWPASS_FC, BUTTER_ORDER


def measured_vergence(d: pd.DataFrame) -> np.ndarray:
    """Measured horizontal vergence from per-eye gaze angles (deg)."""
    return (d["lx"] - d["rx"]).to_numpy(float)


def demanded_vergence(d: pd.DataFrame) -> np.ndarray:
    """Geometric demanded vergence from the stimulus, independent of gaze (deg).

    Horizontal angle from each eye centre to the target, differenced L − R.
    """
    hL = np.degrees(np.arctan2((d["xT"] - d["clx"]).to_numpy(float),
                               (d["zT"] - d["clz"]).to_numpy(float)))
    hR = np.degrees(np.arctan2((d["xT"] - d["crx"]).to_numpy(float),
                               (d["zT"] - d["crz"]).to_numpy(float)))
    return hL - hR


def clean(v, fs: float = FS, vmax: float = VMAX_DPS, fc: float = LOWPASS_FC):
    """Blink/spike-mask then zero-phase low-pass; returns (filtered, mask).

    Steps: interpolate gaps -> NaN any sample whose |velocity| exceeds ``vmax``
    (blink/saccade spikes) -> re-interpolate -> Butterworth filtfilt -> restore
    the masked samples as NaN.
    """
    v = np.asarray(v, float).copy()
    interp = pd.Series(v).interpolate(limit_direction="both").to_numpy()
    if np.isnan(interp).all():
        return v, np.ones_like(v, bool)
    v[np.abs(np.gradient(interp, 1 / fs)) > vmax] = np.nan
    mask = np.isnan(v)
    s = pd.Series(v).interpolate(limit_direction="both").to_numpy()
    b, a = butter(BUTTER_ORDER, fc / (fs / 2), "low")
    vf = filtfilt(b, a, s)
    vf[mask] = np.nan
    return vf, mask


def correlation(vf: np.ndarray, vd: np.ndarray) -> float:
    """Pearson r between cleaned measured and demanded vergence (QC metric)."""
    valid = ~np.isnan(vf) & ~np.isnan(vd)
    return float(np.corrcoef(vf[valid], vd[valid])[0, 1]) if valid.sum() > 100 else np.nan

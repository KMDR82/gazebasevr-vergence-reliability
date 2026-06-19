"""Parametric vergence-transient models and fitting.

first_order  : v(t) = A·(1 − e^{−(t − t0)/τ}),  peak velocity = A/τ  (selected)
second_order : critically/under-damped step response (tested, rejected on
               reproducibility grounds despite a marginally better in-sample fit)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit

from .config import FS, M1_BOUNDS, M2_BOUNDS, MOVE_WINDOW_S


def first_order(t, t0, tau, A):
    y = np.zeros_like(t)
    m = t >= t0
    y[m] = A * (1 - np.exp(-(t[m] - t0) / tau))
    return y


def second_order(t, t0, wn, zeta, A):
    y = np.zeros_like(t)
    m = t >= t0
    td = t[m] - t0
    wd = wn * np.sqrt(max(1 - zeta ** 2, 1e-6))
    y[m] = A * (1 - np.exp(-zeta * wn * td) * (np.cos(wd * td) + (zeta * wn / wd) * np.sin(wd * td)))
    return y


def _r2(y, yh, g):
    return 1 - np.nansum((y[g] - yh[g]) ** 2) / np.nansum((y[g] - np.nanmean(y[g])) ** 2)


def aic(y, yh, k):
    n = np.isfinite(y).sum()
    return n * np.log(np.nansum((y - yh) ** 2) / n) + 2 * k


def fit_first_order(template, direction):
    """Fit first-order model to a (signed) template; return params, pv, latency, R²."""
    sgn = 1 if direction == "conv" else -1
    y = sgn * template
    t = np.arange(len(y)) / FS
    g = np.isfinite(y)
    A0 = max(np.nanmedian(y[int(0.8 * len(y)):]), 0.5)
    p, _ = curve_fit(first_order, t[g], y[g], p0=[0.15, 0.2, A0],
                     bounds=M1_BOUNDS, maxfev=6000)
    t0, tau, A = p
    yh = first_order(t, *p)
    return dict(latency_ms=1000 * t0, tau_ms=1000 * tau, A=A,
                peakvel=A / tau, r2=_r2(y, yh, g))


def fit_both(template, direction):
    """Fit first- and second-order models; return fit/PV/latency/AIC/R² for each."""
    sgn = 1 if direction == "conv" else -1
    y = sgn * template
    t = np.arange(len(y)) / FS
    g = np.isfinite(y)
    A0 = max(np.nanmedian(y[int(0.8 * len(y)):]), 0.5)
    out = {}
    try:
        p1, _ = curve_fit(first_order, t[g], y[g], p0=[0.15, 0.2, A0],
                          bounds=M1_BOUNDS, maxfev=6000)
        yh = first_order(t, *p1)
        out.update({f"{direction}1_lat": 1000 * p1[0], f"{direction}1_pv": p1[2] / p1[1],
                    f"{direction}1_r2": _r2(y, yh, g), f"{direction}1_aic": aic(y[g], yh[g], 3)})
    except Exception:
        out.update({f"{direction}1_lat": np.nan, f"{direction}1_pv": np.nan,
                    f"{direction}1_r2": np.nan, f"{direction}1_aic": np.nan})
    try:
        p2, _ = curve_fit(second_order, t[g], y[g], p0=[0.15, 8, 0.7, A0],
                          bounds=M2_BOUNDS, maxfev=10000)
        yh = second_order(t, *p2)
        pv2 = np.nanmax(np.gradient(second_order(t, *p2), 1 / FS)[:int(MOVE_WINDOW_S * FS)])
        out.update({f"{direction}2_lat": 1000 * p2[0], f"{direction}2_pv": pv2,
                    f"{direction}2_zeta": p2[2], f"{direction}2_r2": _r2(y, yh, g),
                    f"{direction}2_aic": aic(y[g], yh[g], 4)})
    except Exception:
        out.update({f"{direction}2_lat": np.nan, f"{direction}2_pv": np.nan,
                    f"{direction}2_zeta": np.nan, f"{direction}2_r2": np.nan,
                    f"{direction}2_aic": np.nan})
    return out

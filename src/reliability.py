"""Reliability statistics: ICC(2,1), bootstrap CIs, SEM/MDC, Spearman-Brown."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import N_BOOTSTRAP, TARGET_ICC


def icc21(x: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    ``x`` is an (n subjects x k sessions) array.
    """
    n, k = x.shape
    g = x.mean()
    MSR = k * ((x.mean(1) - g) ** 2).sum() / (n - 1)
    MSC = n * ((x.mean(0) - g) ** 2).sum() / (k - 1)
    MSE = (((x - g) ** 2).sum() - k * ((x.mean(1) - g) ** 2).sum()
           - n * ((x.mean(0) - g) ** 2).sum()) / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


def bootstrap_ci(x: np.ndarray, B: int = N_BOOTSTRAP, seed: int = 0):
    """Percentile bootstrap 95% CI for ICC(2,1) over resampled subjects."""
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        try:
            vals.append(icc21(x[s]))
        except Exception:
            pass
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def pivot_pairs(tbl: pd.DataFrame, feature: str) -> np.ndarray:
    """Within-round session-1 vs session-2 pairs for one feature (n x 2)."""
    piv = tbl.pivot_table(index=["subject", "round"], columns="session",
                          values=feature).dropna()
    return piv.values.astype(float)


def short_term_icc(tbl: pd.DataFrame, feature: str):
    """ICC + 95% CI for same-round S1 vs S2 (returns dict)."""
    x = pivot_pairs(tbl, feature)
    if len(x) < 10:
        return dict(icc=np.nan, ci=(np.nan, np.nan), n=len(x))
    lo, hi = bootstrap_ci(x)
    return dict(icc=icc21(x), ci=(lo, hi), n=len(x))


def long_term_icc(tbl: pd.DataFrame, feature: str):
    """ICC across rounds: per-round means, first vs second round per subject."""
    rr = tbl.groupby(["subject", "round"])[feature].mean().reset_index()
    multi = rr.groupby("subject")["round"].nunique()
    multi = multi[multi >= 2].index
    a, b = [], []
    for s in multi:
        v = rr[rr.subject == s].sort_values("round")[feature].dropna().values
        if len(v) >= 2:
            a.append(v[0])
            b.append(v[1])
    if len(a) < 10:
        return dict(icc=np.nan, n=len(a))
    return dict(icc=icc21(np.column_stack([a, b])), n=len(a))


def sem_mdc(x: np.ndarray, icc: float):
    """Standard error of measurement and 95% minimal detectable change."""
    sd = x.std(ddof=1)
    sem = sd * np.sqrt(max(1 - icc, 0))
    mdc = 1.96 * np.sqrt(2) * sem
    return dict(sem=sem, mdc=mdc)


def spearman_brown(icc1: float, k: int) -> float:
    """Predicted reliability when averaging ``k`` sessions."""
    return k * icc1 / (1 + (k - 1) * icc1)


def sessions_for_target(icc1: float, target: float = TARGET_ICC) -> float:
    """Number of averaged sessions needed to reach ``target`` reliability."""
    return np.inf if icc1 <= 0 else target * (1 - icc1) / (icc1 * (1 - target))


def variance_decomposition(tbl: pd.DataFrame, feature: str):
    """Between-subject vs within-subject (error) SD for a feature."""
    between = tbl.groupby("subject")[feature].mean().var(ddof=1)
    within = tbl.groupby("subject")[feature].var(ddof=1).mean()
    return dict(between_sd=np.sqrt(between), error_sd=np.sqrt(within),
                ratio=between / within if within > 0 else np.nan)


def bootstrap_icc_difference(Xd: np.ndarray, Xm: np.ndarray,
                             B: int = N_BOOTSTRAP, seed: int = 0):
    """Bootstrap 95% CI for ICC(model) - ICC(descriptive) on matched pairs."""
    rng = np.random.default_rng(seed)
    n = len(Xd)
    diff = [icc21(Xm[s]) - icc21(Xd[s])
            for s in (rng.integers(0, n, n) for _ in range(B))]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return dict(delta=icc21(Xm) - icc21(Xd), ci=(lo, hi),
                icc_desc=icc21(Xd), icc_model=icc21(Xm))

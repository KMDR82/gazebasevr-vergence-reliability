"""Data-quality metrics and confound analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def ipd_from_recording(path: str) -> float:
    """Median inter-eye-centre distance (a per-subject scale proxy)."""
    d = pd.read_csv(path, usecols=["clx", "cly", "clz", "crx", "cry", "crz"])
    return float(np.nanmedian(np.sqrt((d.crx - d.clx) ** 2
                                      + (d.cry - d.cly) ** 2
                                      + (d.crz - d.clz) ** 2)))


def quality_confound(values, valid_frac):
    """Spearman correlation of a biomarker with data quality (valid-sample fraction)."""
    r, p = stats.spearmanr(values, valid_frac, nan_policy="omit")
    return dict(r=r, p=p)


def spurious_effect_check(df: pd.DataFrame, feature: str, group: str = "gf",
                          quality: str = "vf"):
    """Compare a group effect with vs without adjusting for data quality (OLS)."""
    m0 = smf.ols(f"{feature} ~ {group}", data=df).fit()
    m1 = smf.ols(f"{feature} ~ {group} + {quality}", data=df).fit()
    return dict(p_unadjusted=m0.pvalues[group], p_quality_adjusted=m1.pvalues[group])


def fdr_bh(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg adjusted p-values and significance flags."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    adj[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    return adj, adj < alpha

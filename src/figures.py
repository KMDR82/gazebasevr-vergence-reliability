"""The six manuscript figures (300 dpi, no titles, English labels).

Plotting functions take already-prepared tables/arrays; the data preparation
(recomputing templates and the model table from raw recordings) lives in
scripts/06_make_figures.py, mirroring the notebook.

Manuscript mapping:
  figure01 = dataset_overview     figure02 = vergence_paradigm
  figure03 = grand_average_fit    figure04 = testretest_scatter
  figure05 = data_quality_confound figure06 = agreement_and_variance
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .config import FS, M1_BOUNDS, QC
from .models import first_order
from .reliability import icc21

plt.rcParams.update({"savefig.dpi": 300, "figure.dpi": 110, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

BLUE, RED, GREY = "#4C72B0", "#C44E52", "#B0B0B0"


def panels(axs, labels=None):
    """Centred bold panel letters below each axes."""
    axs = np.ravel(axs)
    labels = labels or [f"({c})" for c in "abcdefgh"]
    for ax, lab in zip(axs, labels):
        ax.annotate(lab, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -36),
                    textcoords="offset points", ha="center", va="top",
                    fontweight="bold", fontsize=12, annotation_clip=False)


def fig_dataset_overview(meta, bt, task_counts, age, gender, nrounds, path):
    order = ["VRG", "PUR", "VID", "TEX", "RAN"]
    vals = [int(task_counts.get(t, 0)) for t in order]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    bb = ax[0, 0].bar(order, vals, color=BLUE)
    ax[0, 0].set_ylabel("number of recordings"); ax[0, 0].set_ylim(0, max(vals) * 1.18)
    for r, v in zip(bb, vals):
        ax[0, 0].text(r.get_x() + r.get_width() / 2, v + 15, str(v), ha="center", fontsize=9)
    ax[0, 0].text(0.98, 0.96, f"total {sum(vals)}", transform=ax[0, 0].transAxes,
                  ha="right", va="top", fontsize=9)
    rc = nrounds.value_counts().sort_index()
    bb = ax[0, 1].bar(rc.index.astype(int).astype(str), rc.values, color="#55A868")
    ax[0, 1].set_xlabel("number of rounds"); ax[0, 1].set_ylabel("participants")
    for r, v in zip(bb, rc.values):
        ax[0, 1].text(r.get_x() + r.get_width() / 2, v + 3, str(int(v)), ha="center", fontsize=9)
    ax[0, 2].hist(age.dropna(), bins=20, color=RED, edgecolor="white")
    ax[0, 2].set_xlabel("age (years)"); ax[0, 2].set_ylabel("participants")
    ax[0, 2].text(0.97, 0.96, f"{age.mean():.1f}±{age.std():.1f}\n[{int(age.min())}-{int(age.max())}]",
                  transform=ax[0, 2].transAxes, ha="right", va="top", fontsize=9)
    g = gender.value_counts()
    bb = ax[1, 0].bar(g.index.astype(str), g.values, color="#8172B3")
    ax[1, 0].set_ylabel("participants")
    for r, v in zip(bb, g.values):
        ax[1, 0].text(r.get_x() + r.get_width() / 2, v + 3, str(int(v)), ha="center", fontsize=9)
    ax[1, 1].hist(bt["valid_frac"].dropna(), bins=25, color="#CCB974", edgecolor="white")
    ax[1, 1].set_xlabel("valid-sample fraction"); ax[1, 1].set_ylabel("recordings")
    ax[1, 1].axvline(bt["valid_frac"].median(), color="k", ls="--", lw=1)
    ax[1, 2].hist(bt["corr"].dropna(), bins=25, color="#64B5CD", edgecolor="white")
    ax[1, 2].set_xlabel("measured-demanded correlation"); ax[1, 2].set_ylabel("recordings")
    ax[1, 2].axvline(bt["corr"].median(), color="k", ls="--", lw=1)
    fig.subplots_adjust(left=.07, right=.98, top=.97, bottom=.15, wspace=.32, hspace=.62)
    panels(ax); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_paradigm(vf, vd, zT, path):
    t = np.arange(len(vf)) / FS
    tr = np.where(np.abs(np.diff(zT)) > 1e-6)[0] + 1
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, vf, lw=.7, color=BLUE, label="measured vergence (lx-rx)")
    ax.plot(t, vd, lw=1.3, color=RED, alpha=.8, label="demanded vergence (geometry)")
    for x in t[tr]:
        ax.axvline(x, color="grey", lw=.4, alpha=.4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("vergence (deg)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.subplots_adjust(left=.07, right=.98, top=.97, bottom=.16)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_grand_average(Tc, Td, path):
    L = Tc.shape[1]; tau = np.arange(L) / FS
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a_, A, dn, clr in [(ax[0], Tc, "conv", BLUE), (ax[1], Td, "div", RED)]:
        m = np.nanmean(A, 0); sd = np.nanstd(A, 0)
        a_.plot(tau, m, color=clr, lw=2, label="grand average")
        a_.fill_between(tau, m - sd, m + sd, color=clr, alpha=.15)
        sgn = 1 if dn == "conv" else -1
        y = sgn * m; gg = np.isfinite(y)
        pr, _ = curve_fit(first_order, tau[gg], y[gg],
                          p0=[.15, .2, max(np.nanmedian(y[-30:]), .5)],
                          bounds=M1_BOUNDS, maxfev=8000)
        r2 = 1 - np.nansum((y[gg] - first_order(tau, *pr)[gg]) ** 2) / np.nansum((y[gg] - np.nanmean(y[gg])) ** 2)
        a_.plot(tau, sgn * first_order(tau, *pr), "k--", lw=1.4,
                label=f"first-order model (R2={r2:.2f})")
        a_.axhline(0, color="grey", lw=.5)
        a_.set_xlabel("time after transition (s)"); a_.set_ylabel("Δvergence (deg)")
        a_.legend(fontsize=8, frameon=False)
    fig.subplots_adjust(left=.07, right=.98, top=.97, bottom=.2, wspace=.28)
    panels(ax); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_testretest_scatter(bt_qc, mtab, path):
    """Session-1 vs session-2 divergence peak velocity, coloured by quality."""
    vfp = bt_qc.pivot_table(index=["subject", "round"], columns="session",
                            values="valid_frac").dropna()

    def pair_xy(piv):
        piv = piv.dropna()
        j = piv.join(vfp.mean(axis=1).rename("vfm")).dropna()
        cols = list(piv.columns)
        return j[cols[0]].values.astype(float), j[cols[1]].values.astype(float), j["vfm"].values

    xd, yd, cd = pair_xy(bt_qc.pivot_table(index=["subject", "round"], columns="session", values="div_peakvel"))
    xm, ym, cm = pair_xy(mtab.pivot_table(index=["subject", "round"], columns="session", values="div_pv_m"))
    norm = Normalize(vmin=0.80, vmax=1.00); cmap = "viridis"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for a_, X, Y, C in [(ax[0], xd, yd, cd), (ax[1], xm, ym, cm)]:
        icc = icc21(np.column_stack([X, Y])); r = stats.spearmanr(X, Y)[0]
        lo = min(X.min(), Y.min()); hi = max(X.max(), Y.max()); pad = 0.05 * (hi - lo)
        lim = [lo - pad, hi + pad]
        a_.plot(lim, lim, "--", color="0.4", lw=1, zorder=1)
        a_.scatter(X, Y, c=C, cmap=cmap, norm=norm, s=22, alpha=.85,
                   edgecolor="white", lw=.3, zorder=2)
        a_.set_xlim(lim); a_.set_ylim(lim); a_.set_aspect("equal", "box"); a_.grid(alpha=.25, lw=.5)
        a_.set_xlabel("session 1 (deg/s)"); a_.set_ylabel("session 2 (deg/s)")
        a_.text(0.04, 0.96, f"ICC = {icc:.2f}\n$r_s$ = {r:.2f}\nn = {len(X)}",
                transform=a_.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=.9))
    panels(ax)
    fig.subplots_adjust(left=.08, right=.88, top=.97, bottom=.18, wspace=.28)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.90, 0.22, 0.02, 0.62])
    fig.colorbar(sm, cax=cax).set_label("valid-sample fraction")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_quality_confound(bt_qc, mtab, gmap, path):
    agg = bt_qc.groupby("subject").agg(vf=("valid_frac", "mean"), pv_desc=("div_peakvel", "mean"))
    aggm = mtab.groupby("subject").agg(pv_mod=("div_pv_m", "mean"))
    A = agg.join(aggm).join(gmap)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(A["vf"], A["pv_desc"], s=10, alpha=.4, color=GREY, label="descriptive")
    ax[0].scatter(A["vf"], A["pv_mod"], s=10, alpha=.4, color=BLUE, label="model-based")
    rd = stats.spearmanr(A["vf"], A["pv_desc"], nan_policy="omit")[0]
    rm = stats.spearmanr(A["vf"], A["pv_mod"], nan_policy="omit")[0]
    ax[0].set_xlabel("valid-sample fraction"); ax[0].set_ylabel("divergence peak velocity")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].text(0.03, 0.97, f"descriptive r={rd:.2f}\nmodel r={rm:.2f}",
               transform=ax[0].transAxes, va="top", fontsize=9)

    def cohend(a, b):
        s = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
        return (b.mean() - a.mean()) / s

    AB = A[A["gender"].isin(["Male", "Female"])]
    ds, ps, labs = [], [], ["descriptive", "model-based"]
    for c in ["pv_desc", "pv_mod"]:
        m_ = AB[AB.gender == "Male"][c].dropna(); f_ = AB[AB.gender == "Female"][c].dropna()
        ds.append(cohend(m_, f_)); ps.append(stats.mannwhitneyu(m_, f_)[1])
    bb = ax[1].bar(labs, ds, color=[GREY, BLUE]); ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_ylabel("effect size d (Female-Male)")
    for r, dv, pv in zip(bb, ds, ps):
        ax[1].text(r.get_x() + r.get_width() / 2, dv + (0.02 if dv >= 0 else -0.04),
                   f"d={dv:.2f}\np={pv:.3f}", ha="center",
                   va="bottom" if dv >= 0 else "top", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.97, bottom=.2, wspace=.3)
    panels(ax); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_agreement_and_variance(bt_qc, mtab, feats, labels, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    pvt = mtab.pivot_table(index=["subject", "round"], columns="session",
                           values="div_pv_m").dropna().values.astype(float)
    mn = pvt.mean(1); dfr = pvt[:, 0] - pvt[:, 1]; bias = dfr.mean(); loa = 1.96 * dfr.std(ddof=1)
    ax[0].scatter(mn, dfr, s=10, alpha=.4, color=BLUE)
    ax[0].axhline(bias, color="k"); ax[0].axhline(bias + loa, color=RED, ls="--")
    ax[0].axhline(bias - loa, color=RED, ls="--")
    ax[0].set_xlabel("mean divergence peak velocity (S1,S2)"); ax[0].set_ylabel("S1 - S2")
    ax[0].text(0.03, 0.97, f"bias={bias:.2f}\nLoA=±{loa:.1f}",
               transform=ax[0].transAxes, va="top", fontsize=9)
    bsd = [np.sqrt(bt_qc.groupby("subject")[f].mean().var(ddof=1)) for f in feats]
    esd = [np.sqrt(bt_qc.groupby("subject")[f].var(ddof=1).mean()) for f in feats]
    y = np.arange(len(feats)); w = .38
    ax[1].barh(y - w / 2, bsd, w, label="between-subject SD", color="#55A868")
    ax[1].barh(y + w / 2, esd, w, label="error SD", color="#DD8452")
    ax[1].set_yticks(y); ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].set_xlabel("standard deviation"); ax[1].legend(fontsize=8, frameon=False)
    fig.subplots_adjust(left=.1, right=.98, top=.97, bottom=.2, wspace=.35)
    panels(ax); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

"""Manuscript figures from the stored run outputs (nothing is refitted).

  python scripts/10_figures.py     # -> paper/figures/fig_*.{pdf,png}

Reads results/<target>/<run>/{runs.jsonl,preds,shap}, results/leakage_channels.csv and results/summary_full.csv
(written by 05_analysis.py). Schematics (battery cell, workflow) are TikZ sources in paper/figures_src/.
"""
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("an", ROOT / "scripts" / "05_analysis.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
from bmd.groups import group  # noqa: E402  (path set by 05_analysis)

RES, FIG = an.RES, an.FIG
TGT, NICE = an.TGT, dict(an.NICE)
NICE["coulombic_efficiency"] = "Coulombic efficiency"
UNIT = {"capacity": "mAh g$^{-1}$", "voltage": "V", "conductivity": "log$_{10}$(S cm$^{-1}$)",
        "coulombic_efficiency": "%", "energy": "Wh kg$^{-1}$"}
SYM = {"capacity": "$Q$", "voltage": "$V$", "conductivity": r"$\log_{10}\sigma$", "coulombic_efficiency": "CE",
       "energy": "$E$"}
PROT = [("random", "Record-level random", "#A9BED6"), ("doi", "DOI-grouped", "#2F5D8A"),
        ("temporal", "Temporal", "#C97B4A"), ("material", "Composition-grouped", "#5E9E6E")]
GC = {"composition": "#5E9E6E", "material": "#4A78A8", "material_text": "#8E7CC3", "measurement": "#D08A4E",
      "literature": "#C25B6E", "noise": "#9A9A9A"}
GN = {"composition": "Composition", "material": "Elemental properties", "material_text": "Name flags",
      "measurement": "Measurement context", "literature": "Title keywords", "noise": "Noise (control)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "pdf.fonttype": 42, "legend.frameon": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(FIG / f"{name}.pdf")


def tag(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")


R = an.load_runs()
MAIN = R[(R.tier == "t2") & (R.variant == "all")]


def stacked(t, protocol):
    d = MAIN[(MAIN.target == t) & (MAIN.protocol == protocol)]
    P = [an.preds(r.dir, r.split) for _, r in d.sort_values("split").iterrows()]
    return {k: np.concatenate([p[k] for p in P]) for k in ("y", "ens", "unc")}


# ------------------------------------------------------------------ target distributions
def fig_distributions():
    pop = pd.read_csv(RES / "population.csv").set_index("target")
    icc = {t: an.V.get(f"pop/{t}/icc1") for t in TGT}
    try:
        import json
        V = json.loads((RES / "paper_values.json").read_text()); icc = {t: V[f"pop/{t}/icc1"] for t in TGT}
    except Exception:
        pass
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.95))
    for i, (a, t) in enumerate(zip(axs, TGT)):
        D = an.data(t); s = D.Warning.fillna("").str.contains("S").values; y = D.y.values
        bins = np.histogram_bin_edges(y, 50)
        a.hist(y, bins, color="#2F5D8A", alpha=0.85, linewidth=0)
        a.hist(y[s], bins, histtype="step", color="#D08A4E", lw=1.0)
        a.set_yscale("log"); a.set_ylim(0.8, None)
        a.axvline(np.median(y), color="k", lw=0.7, ls="--")
        a.set_title(NICE[t], pad=13); a.set_xlabel(UNIT[t])
        r = pop.loc[t]
        a.text(0.5, 1.02, f"{int(r.records):,} rec. | {int(r.publications):,} pubs | ICC {icc[t]}",
               transform=a.transAxes, ha="center", va="bottom", fontsize=5.4, color="#444444")
        tag(a, "abcde"[i], x=-0.2, y=1.1)
    axs[0].set_ylabel("records (log scale)")
    h = [Patch(color="#2F5D8A", label="all records"), Line2D([], [], color="#D08A4E", lw=1.0, label="series members (S flag)"),
         Line2D([], [], color="k", lw=0.7, ls="--", label="median")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_target_distributions")


# ------------------------------------------------------------------ split protocols + leakage channels
def fig_split():
    C = pd.read_csv(RES / "leakage_channels.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.55, 1]))
    w = 0.2
    for i, t in enumerate(TGT):
        for j, (p, lab, c) in enumerate(PROT):
            d = MAIN[(MAIN.target == t) & (MAIN.protocol == p)]
            if d.empty:
                continue
            v = d.ens_r2.values; x = i + (j - 1.5) * w
            a.bar(x, v.mean(), w * 0.92, color=c, label=lab if i == 0 else None, zorder=2,
                  hatch="////" if p == "random" else None, edgecolor="white", linewidth=0)
            a.errorbar(x, v.mean(), v.std(ddof=1), color="k", lw=0.7, capsize=1.6, zorder=3)
            a.scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=3, color="k", alpha=0.55,
                      zorder=4, linewidths=0)
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(range(5)); a.set_xticklabels([NICE[t].replace(" efficiency", "\nefficiency") for t in TGT])
    a.set_ylabel("Test $R^2$ ($t_2$ ensemble)"); a.set_ylim(-0.02, 0.65)
    a.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.4, columnspacing=1.0)
    tag(a, "a", x=-0.1)
    # (b) repeated vs new values on DOI-grouped test sets
    d = C[C.protocol == "doi"]
    for i, t in enumerate(TGT):
        g = d[d.target == t]
        for j, (col, c, lab) in enumerate([("r2_rep", "#B5505A", "value reported by another paper in training"),
                                           ("r2_new", "#5A6B7B", "value without such a precedent")]):
            v = g[col].values; x = i + (j - 0.5) * 0.36
            b.bar(x, v.mean(), 0.33, color=c, label=lab if i == 0 else None, zorder=2)
            b.errorbar(x, v.mean(), v.std(ddof=1), color="k", lw=0.7, capsize=1.6, zorder=3)
            b.scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=3, color="k", alpha=0.55,
                      zorder=4, linewidths=0)
        top = max(g.r2_rep.mean() + g.r2_rep.std(), g.r2_new.mean() + g.r2_new.std(), 0)
        b.text(i, top + 0.04, f"{100 * g.rep_share.mean():.0f}%", ha="center", fontsize=6, color="#B5505A")
    b.axhline(0, color="k", lw=0.6)
    b.set_xticks(range(5)); b.set_xticklabels(["Cap.", "Volt.", "Cond.", "CE", "Energy"])
    b.set_ylabel("$R^2$ on DOI-grouped test records"); b.set_ylim(-0.12, 0.95)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), handlelength=1.2)
    tag(b, "b", x=-0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_split_by_target")


# ------------------------------------------------------------------ prediction diagnostics (DOI-grouped)
def fig_diagnostics():
    fig, axs = plt.subplots(2, 5, figsize=(7.2, 3.0), gridspec_kw=dict(height_ratios=[1.3, 0.7]))
    for i, t in enumerate(TGT):
        P = stacked(t, "doi"); y, p = P["y"], P["ens"]
        lo, hi = np.percentile(np.r_[y, p], [0.1, 99.9])
        a = axs[0, i]
        hb = a.hexbin(y, p, gridsize=38, extent=(lo, hi, lo, hi), bins="log", cmap="Blues", mincnt=1,
                      linewidths=0.1)
        a.plot([lo, hi], [lo, hi], "--", color="#C0392B", lw=0.7)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        a.text(0.04, 0.96, f"$R^2$ = {r2:.2f}\nMAE = {np.abs(y - p).mean():.3g}", transform=a.transAxes,
               va="top", fontsize=5.8)
        a.set_title(NICE[t], pad=3, fontsize=7.2); a.set_xlabel(f"reported {SYM[t]}", labelpad=1)
        if i == 0:
            a.set_ylabel("predicted")
        a.locator_params(nbins=4)
        tag(a, "abcde"[i], x=-0.3, y=1.02)
        b = axs[1, i]
        bins = np.linspace(lo, hi, 45)
        b.hist(y, bins, color="#BBBBBB", label="reported", linewidth=0)
        b.hist(p, bins, histtype="step", color="#2F5D8A", lw=0.9, label="predicted")
        b.set_yticks([]); b.set_xlabel(UNIT[t], labelpad=1); b.locator_params(axis="x", nbins=4)
        b.spines["left"].set_visible(False)
    axs[1, 0].legend(loc="upper right", fontsize=5.8, handlelength=1)
    fig.subplots_adjust(wspace=0.5, hspace=0.5, left=0.07, right=0.9, top=0.92, bottom=0.11)
    pos = axs[0, 4].get_position()
    cb = fig.colorbar(hb, cax=fig.add_axes([0.915, pos.y0, 0.01, pos.height]))
    cb.set_label("records per bin", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    save(fig, "fig_diagnostics")


# ------------------------------------------------------------------ SHAP
PROPN = {"ie1": "1st ionisation energy", "group": "group number", "en": "electronegativity", "mendeleev": "Mendeleev no.",
         "radius": "atomic radius", "Z": "atomic number", "mass": "atomic mass", "oxi_max": "highest oxidation state",
         "row": "period", "valence": "valence electrons", "oxi_min": "lowest oxidation state"}
STAT = {"mean": "mean", "min": "min.", "max": "max.", "range": "range"}


def pretty(f):
    na = f.endswith("_na"); b = f[:-3] if na else f
    m = re.match(r"mat_(.+)_(mean|min|max|range)$", b)
    if m:
        s = f"{STAT[m.group(2)]} {PROPN.get(m.group(1), m.group(1).replace('_', ' '))}"
    elif b.startswith("comp_frac_"):
        s = f"fraction of {b[10:]}"
    elif b.startswith("meas_spec_"):
        s = f"specifier: {b[10:].replace('_', ' ')}"
    elif b == "meas_log_current_Ag":
        s = "log current density"
    elif b.startswith("txt_"):
        s = f"name: {b[4:].replace('_', ' ')}"
    elif b.startswith("lit_"):
        s = f"title: {b[4:].replace('_', ' ')}"
    else:
        s = re.sub(r"^(comp|meas|mat)_", "", b).replace("_", " ")
    return s + (" (missing)" if na else "")


def shap_per_split(t, tier="t2"):
    d = R[(R.target == t) & (R.protocol == "doi") & (R.tier == tier) & (R.variant == "all")]
    per, names = [], None
    for _, r in d.sort_values("split").iterrows():
        s = np.load(Path(r.dir) / "shap" / f"{r.split}.npz", allow_pickle=True)
        names = list(s["names"]); per.append(np.abs(s["shap_ens"]).mean(0))
    return np.array(per), names


def fig_shap():
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.75)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    groups = ["composition", "material", "material_text", "measurement"]
    w = 0.19
    for i, t in enumerate(TGT):
        per, names = shap_per_split(t)
        g = np.array([group(n) for n in names])
        for j, k in enumerate(groups):
            sh = 100 * per[:, g == k].sum(1) / per.sum(1)
            x = i + (j - 1.5) * w
            a.bar(x, sh.mean(), w * 0.92, color=GC[k], label=GN[k] if i == 0 else None, zorder=2)
            a.errorbar(x, sh.mean(), sh.std(ddof=1), color="k", lw=0.6, capsize=1.2, zorder=3)
        pn, nn = shap_per_split(t, "t2_noise")
        if len(pn):
            gn = np.array([group(n) for n in nn]); ns = 100 * pn[:, gn == "noise"].sum(1) / pn.sum(1)
            a.plot([i - 2 * w, i + 2 * w], [ns.mean()] * 2, color="#555555", lw=0.9, ls=":", zorder=4,
                   label="noise share in $t_{2,\\mathrm{noise}}$" if i == 0 else None)
    a.set_xticks(range(5)); a.set_xticklabels(["Cap.", "Volt.", "Cond.", "CE", "Energy"])
    a.set_ylabel("share of mean |SHAP| (%)"); a.set_ylim(0, 80)
    a.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.2, handlelength=1.2,
             columnspacing=0.8)
    tag(a, "a", x=-0.14, y=1.2)
    per, names = shap_per_split("capacity")
    m, sd = per.mean(0), per.std(0, ddof=1)
    o = np.argsort(m)[::-1][:15][::-1]
    b.barh(range(15), m[o], xerr=sd[o], color=[GC[group(names[k])] for k in o], height=0.72,
           error_kw=dict(lw=0.6, capsize=1.2), zorder=2)
    b.set_yticks(range(15)); b.set_yticklabels([pretty(names[k]) for k in o], fontsize=6.3)
    b.set_xlabel("mean |SHAP| for capacity (mAh g$^{-1}$)")
    b.set_title("Capacity: 15 highest-ranked predictors", pad=4, fontsize=7.5)
    tag(b, "b", x=-0.62, y=1.02)
    save(fig, "fig_shap_groups_by_target")


# ------------------------------------------------------------------ uncertainty
def fig_uncertainty():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
    P = stacked("capacity", "doi"); u, e = P["unc"], np.abs(P["y"] - P["ens"])
    hb = a.hexbin(u, e, gridsize=45, bins="log", cmap="Blues", mincnt=1, extent=(0, np.percentile(u, 99.8), 0,
                                                                                  np.percentile(e, 99.8)))
    q = pd.qcut(u, 20, labels=False, duplicates="drop")
    gm = pd.DataFrame(dict(u=u, e=e, q=q)).groupby("q").agg(u=("u", "median"), e=("e", "median"),
                                                             e90=("e", lambda s: s.quantile(0.9)))
    a.plot(gm.u, gm.e, color="#C0392B", lw=1.1, label="median |error|")
    a.plot(gm.u, gm.e90, color="#C0392B", lw=0.8, ls="--", label="90th percentile")
    a.set_xlabel("ensemble member spread (mAh g$^{-1}$)"); a.set_ylabel("|error| (mAh g$^{-1}$)")
    a.set_title("Capacity, DOI-grouped test records", pad=3); a.legend(loc="upper right")
    cb = fig.colorbar(hb, ax=a, pad=0.02); cb.set_label("records per bin"); cb.ax.tick_params(labelsize=6)
    tag(a, "a", x=-0.2)
    cols = dict(zip(TGT, ["#2F5D8A", "#D08A4E", "#5E9E6E", "#8E7CC3", "#B5505A"]))
    for t in TGT:
        for p, ls in (("doi", "-"), ("random", ":")):
            P = stacked(t, p); u, e = P["unc"], np.abs(P["y"] - P["ens"])
            dec = pd.qcut(u, 10, labels=False, duplicates="drop")
            m = pd.Series(e).groupby(dec).mean()
            b.plot(m.index + 1, m / e.mean(), ls, color=cols[t], lw=1.1, marker="o" if p == "doi" else None,
                   ms=2.2)
    b.axhline(1, color="k", lw=0.5)
    b.set_xlabel("decile of member spread (1 = most confident)"); b.set_ylabel("MAE in decile / overall MAE")
    b.set_xticks(range(1, 11))
    h = [Line2D([], [], color=cols[t], lw=1.1, label=NICE[t]) for t in TGT] + \
        [Line2D([], [], color="k", lw=1.1, marker="o", ms=2.2, label="DOI-grouped"),
         Line2D([], [], color="k", lw=1.1, ls=":", label="record-level random")]
    b.legend(handles=h, loc="center left", bbox_to_anchor=(1.0, 0.5))
    tag(b, "b", x=-0.16)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_uncertainty")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_distributions, fig_split, fig_diagnostics, fig_shap, fig_uncertainty):
        f()

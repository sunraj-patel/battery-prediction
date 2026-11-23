"""Post-hoc analysis for the manuscript: every number behind a \\TBD placeholder, computed from saved run outputs.

  python scripts/05_analysis.py            # -> results/{summary_full,paired_deltas,uncertainty,overlap,series,
                                           #    correctness,error_by_warning,shap_groups_all,population_ext}.csv
                                           #    results/paper_values.json, paper/figures/fig_*.png

Nothing is refitted. Conventions: metrics are the 'ens' model, mean +/- sample s.d. over splits; paired tier
differences use identical test sets (splits depend only on the DOI groups and the seed); confidence intervals are a
publication-cluster bootstrap (DOIs resampled within each split's test set, statistic averaged over splits,
2,000 replicates, 2.5/97.5 percentiles).
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bmd.data import TARGETS, load_dataset, normalize_doi  # noqa: E402
from bmd.groups import group  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
TGT = ["capacity", "voltage", "conductivity", "coulombic_efficiency", "energy"]
NICE = {"capacity": "Capacity", "voltage": "Voltage", "conductivity": "Conductivity",
        "coulombic_efficiency": "Coulombic eff.", "energy": "Energy"}
PROPERTY = {"capacity": "Capacity", "voltage": "Voltage", "conductivity": "Conductivity",
            "coulombic_efficiency": "Coulombic Efficiency", "energy": "Energy"}
B = 2000
V = {}   # placeholder key -> formatted string


def fnum(x, kind="auto"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if kind == "r2":
        return f"{x:.3f}"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "pct":
        return f"{100 * x:.1f}\\%"
    if kind == "d":
        return f"{x:+.3f}"
    a = abs(x)
    return f"{x:.0f}" if a >= 100 else f"{x:.1f}" if a >= 10 else f"{x:.2f}" if a >= 1 else f"{x:.3f}"


def pm(m, s, kind="auto"):
    """mean +/- sd with the sd printed at the mean's precision."""
    if np.isnan(s):
        return fnum(m, kind)
    if kind == "r2":
        return f"{m:.3f} $\\pm$ {s:.3f}"
    a = abs(m); dp = 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3
    while dp < 3 and s > 0 and round(s, dp) == 0:   # never print a non-zero s.d. as 0.00
        dp += 1
    return f"{m:.{dp}f} $\\pm$ {s:.{dp}f}"


# ---------------------------------------------------------------- runs + predictions
def load_runs():
    rows = []
    for f in RES.glob("*/*/runs.jsonl"):
        R = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
        R = R[~R.get("fast", False).astype(bool)].drop_duplicates("split", keep="last")
        R["dir"] = str(f.parent)
        rows.append(R)
    R = pd.concat(rows, ignore_index=True)
    R["variant"] = np.where(R["drop_series"].astype(bool), "drop_S", "all")
    return R


def preds(run_dir, k):
    s = np.load(Path(run_dir) / "preds" / f"{k}.npz", allow_pickle=True)
    return {x: s[x] for x in s.files}


DATA = {}


def data(t, drop_series=False):
    if (t, drop_series) not in DATA:
        DATA[(t, drop_series)] = load_dataset(t, drop_series=drop_series)
    return DATA[(t, drop_series)]


def _m(y, p):
    e = y - p
    return dict(r2=r2(y, p), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------- 1. metrics + baseline
def summary(R):
    rows = []
    for (t, proto, tier, var), d in R.groupby(["target", "protocol", "tier", "variant"]):
        row = dict(target=t, protocol=proto, tier=tier, variant=var, model="ens", n_splits=len(d),
                   n_test=d.n_test.mean())
        for m in ("r2", "mae", "rmse"):
            row[m], row[f"{m}_sd"] = d[f"ens_{m}"].mean(), d[f"ens_{m}"].std(ddof=1)
        rows.append(row)
        if tier == "t2" and var == "all":   # constant-mean baseline (train+val mean) on the same test sets
            y = data(t)["y"].values
            b = []
            for _, r in d.iterrows():
                P = preds(r.dir, r.split)
                mask = np.ones(len(y), bool); mask[P["test_idx"]] = False
                e = P["y"] - y[mask].mean()
                b.append(dict(r2=r2(P["y"], np.full_like(P["y"], y[mask].mean())), mae=np.abs(e).mean(),
                              rmse=np.sqrt((e ** 2).mean())))
            b = pd.DataFrame(b)
            rows.append(dict(target=t, protocol=proto, tier="baseline", variant=var, model="mean_baseline",
                             n_splits=len(b), n_test=d.n_test.mean(),
                             **{k: b[k].mean() for k in b}, **{f"{k}_sd": b[k].std(ddof=1) for k in b}))
    S = pd.DataFrame(rows)
    S.to_csv(RES / "summary_full.csv", index=False)
    for _, r in S.iterrows():
        if r.variant != "all":
            continue
        key = f"m/{r.target}/{r.protocol}/{r.tier}"
        V[f"{key}/r2"] = pm(r.r2, r.r2_sd, "r2")
        V[f"{key}/mae"] = pm(r.mae, r.mae_sd)
        V[f"{key}/rmse"] = pm(r.rmse, r.rmse_sd)
        V[f"{key}/n"] = fnum(r.n_test, "int")
    return S


# ---------------------------------------------------------------- 2. paired tier deltas + cluster bootstrap
def _doi_sums(P, pa, pb):
    df = pd.DataFrame(dict(g=P["doi"], y=P["y"], ea=(P["y"] - pa) ** 2, eb=(P["y"] - pb) ** 2))
    df["y2"] = df.y ** 2
    a = df.groupby("g").agg(n=("y", "size"), sy=("y", "sum"), sy2=("y2", "sum"), ea=("ea", "sum"), eb=("eb", "sum"))
    return a.values  # columns n, sy, sy2, ea, eb


def _delta_from_sums(A, w):
    n, sy, sy2, ea, eb = (w @ A).T if w.ndim == 2 else w @ A
    sst = sy2 - sy ** 2 / n
    return (1 - ea / sst) - (1 - eb / sst)


def paired(R):
    rng = np.random.default_rng(0)
    rows = []
    base = R[(R.tier == "t2") & (R.variant == "all")]
    for (t, proto, tier), d in R[(R.variant == "all")].groupby(["target", "protocol", "tier"]):
        if tier == "t2":
            continue
        b = base[(base.target == t) & (base.protocol == proto)].set_index("split")
        dd, boots = [], []
        for _, r in d.iterrows():
            if r.split not in b.index:
                continue
            Pa, Pb = preds(r.dir, r.split), preds(b.loc[r.split, "dir"], r.split)
            assert np.array_equal(Pa["test_idx"], Pb["test_idx"]), "tiers must share test sets"
            dd.append(r2(Pa["y"], Pa["ens"]) - r2(Pb["y"], Pb["ens"]))
            A = _doi_sums(Pa, Pa["ens"], Pb["ens"])
            W = rng.multinomial(len(A), np.full(len(A), 1 / len(A)), size=B).astype(float)
            boots.append(_delta_from_sums(A, W))
        if not dd:
            continue
        bt = np.mean(boots, axis=0)
        rows.append(dict(target=t, protocol=proto, tier=tier, n_splits=len(dd), delta_r2=np.mean(dd),
                         delta_sd=np.std(dd, ddof=1) if len(dd) > 1 else np.nan,
                         ci_lo=np.percentile(bt, 2.5), ci_hi=np.percentile(bt, 97.5)))
    D = pd.DataFrame(rows, columns=["target", "protocol", "tier", "n_splits", "delta_r2", "delta_sd", "ci_lo", "ci_hi"])
    D.to_csv(RES / "paired_deltas.csv", index=False)
    name = {"t3": "t3-t2", "t2_id": "t2_id-t2", "t2_noise": "t2_noise-t2", "comp": "comp-t2", "t1": "t1-t2"}
    for _, r in D.iterrows():
        V[f"d/{r.target}/{r.protocol}/{name[r.tier]}/r2"] = fnum(r.delta_r2, "d")
        V[f"ci/{r.target}/{r.protocol}/{name[r.tier]}/r2"] = f"(95\\% CI {r.ci_lo:+.3f} to {r.ci_hi:+.3f})"
    # nested tier steps for capacity (t1-comp, t2-t1) from the same splits
    for t in TGT:
        m = {tier: R[(R.target == t) & (R.protocol == "doi") & (R.tier == tier) & (R.variant == "all")]
             .set_index("split").ens_r2 for tier in ("comp", "t1", "t2")}
        if all(len(v) for v in m.values()):
            V[f"d/{t}/doi/t1-comp/r2"] = fnum((m["t1"] - m["comp"]).mean(), "d")
            V[f"d/{t}/doi/t2-t1/r2"] = fnum((m["t2"] - m["t1"]).mean(), "d")
    for tier, key in (("t2_noise", "noise-t2"), ("t2_id", "t2_id-t2"), ("t3", "t3-t2")):
        x = D[(D.protocol == "doi") & (D.tier == tier)].delta_r2
        if len(x):
            V[f"d/all/doi/{key}/r2-range"] = f"{x.min():+.3f} to {x.max():+.3f}"
    return D


# ---------------------------------------------------------------- 3. uncertainty
def uncertainty(R):
    rows = []
    for (t, proto), d in R[(R.tier == "t2") & (R.variant == "all")].groupby(["target", "protocol"]):
        rho, U, E = [], [], []
        for _, r in d.iterrows():
            P = preds(r.dir, r.split)
            e = np.abs(P["y"] - P["ens"])
            rho.append(spearmanr(P["unc"], e).statistic)
            U.append(P["unc"]); E.append(e)
        U, E = np.concatenate(U), np.concatenate(E)
        q = np.quantile(U, [0.1, 0.9])
        rows.append(dict(target=t, protocol=proto, spearman=np.mean(rho), spearman_sd=np.std(rho, ddof=1),
                         mean_spread=U.mean(), mae_top_decile=E[U >= q[1]].mean(),
                         mae_bottom_decile=E[U <= q[0]].mean()))
    U = pd.DataFrame(rows)
    U.to_csv(RES / "uncertainty.csv", index=False)
    for _, r in U.iterrows():
        V[f"unc/{r.target}/{r.protocol}/spearman"] = f"{r.spearman:.2f}"
        V[f"unc/{r.target}/{r.protocol}/mae_top_decile"] = fnum(r.mae_top_decile)
        V[f"unc/{r.target}/{r.protocol}/mae_bottom_decile"] = fnum(r.mae_bottom_decile)
    for t in TGT:
        a = U[(U.target == t) & (U.protocol == "doi")].mean_spread
        b = U[(U.target == t) & (U.protocol == "random")].mean_spread
        if len(a) and len(b):
            V[f"unc/{t}/spread_ratio_doi_over_random"] = f"{a.iloc[0] / b.iloc[0]:.2f}"
    return U


# ---------------------------------------------------------------- 4. train/test overlap
def overlap(R):
    rows = []
    for (t, proto), d in R[(R.tier == "t2") & (R.variant == "all")].groupby(["target", "protocol"]):
        D = data(t)
        name = D["Name"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip().values
        g = D["doi_group"].values
        for _, r in d.iterrows():
            te = preds(r.dir, r.split)["test_idx"]
            m = np.ones(len(D), bool); m[te] = False
            rows.append(dict(target=t, protocol=proto, split=r.split, n=len(te),
                             doi_seen=np.isin(g[te], np.unique(g[m])).mean(),
                             name_seen=np.isin(name[te], np.unique(name[m])).mean()))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "overlap.csv", index=False)
    for proto, d in O.groupby("protocol"):   # record-weighted over targets and splits
        V[f"overlap/{proto}/doi_seen_share"] = fnum(np.average(d.doi_seen, weights=d.n), "pct")
        V[f"overlap/{proto}/name_seen_share"] = fnum(np.average(d.name_seen, weights=d.n), "pct")
    return O


# ---------------------------------------------------------------- 5. Warning-S sensitivity
def series(R, S):
    s = S[(S.tier == "t2") & (S.model == "ens")].set_index(["target", "protocol", "variant"])
    for t in TGT:
        for proto in ("random", "doi"):
            if (t, proto, "drop_S") in s.index:
                r = s.loc[(t, proto, "drop_S")]
                V[f"ser/{t}/{proto}/r2"] = pm(r.r2, r.r2_sd, "r2")
        try:
            gw = s.loc[(t, "random", "all")].r2 - s.loc[(t, "doi", "all")].r2
            go = s.loc[(t, "random", "drop_S")].r2 - s.loc[(t, "doi", "drop_S")].r2
            V[f"ser/{t}/gap-with-S"], V[f"ser/{t}/gap-without-S"] = f"{gw:.3f}", f"{go:.3f}"
        except KeyError:
            pass
        V[f"ser/{t}/records"] = fnum(len(data(t, True)), "int")


# ---------------------------------------------------------------- 6. Correctness sample + Warning-L errors
def label_errors(R):
    raw = pd.read_csv(ROOT / "data" / "raw" / "extracted" / "battery.csv", low_memory=False,
                      usecols=["Property", "Name", "Value", "DOI", "Correctness", "Warning"])
    lab = raw[raw.Correctness.notna()].copy()
    lab["doi_group"] = lab.DOI.map(normalize_doi)
    out, rows_w = [], []
    for t in TGT:
        D = data(t).reset_index().rename(columns={"index": "row"})
        d = R[(R.target == t) & (R.protocol == "doi") & (R.tier == "t2") & (R.variant == "all")]
        err = {}
        for _, r in d.iterrows():
            P = preds(r.dir, r.split)
            for i, e in zip(P["test_idx"], np.abs(P["y"] - P["ens"])):
                err.setdefault(i, []).append(e)
        E = pd.Series({i: np.mean(v) for i, v in err.items()})
        D["abs_err"] = D.row.map(E)
        L = D[D.abs_err.notna()]
        isL = L.Warning.fillna("").str.contains("L")
        rows_w.append(dict(target=t, n_test=len(L), L_share_records=isL.mean(),
                           L_share_abs_error=L.abs_err[isL].sum() / L.abs_err.sum(),
                           mae_L=L.abs_err[isL].mean(), mae_notL=L.abs_err[~isL].mean()))
        l = lab[lab.Property == PROPERTY[t]].merge(L[["doi_group", "Name", "Value", "abs_err"]],
                                                    on=["doi_group", "Name", "Value"], how="inner")
        l["target"] = t
        out.append(l.drop_duplicates(["doi_group", "Name", "Value"]))
    W = pd.DataFrame(rows_w); W.to_csv(RES / "error_by_warning.csv", index=False)
    C = pd.concat(out, ignore_index=True); C.to_csv(RES / "correctness.csv", index=False)
    parts = [f"{NICE[r.target].lower()}: {100 * r.L_share_records:.0f}\\% of test records, "
             f"{100 * r.L_share_abs_error:.0f}\\% of absolute error" for _, r in W.iterrows()
             if r.target in ("capacity", "voltage")]
    V["err/L-share"] = "; ".join(parts)
    # Correctness: errors are in target units, so report capacity (where most labels fall) and the count overall
    V["corr/n_test"] = fnum(len(C), "int")
    c = C[C.target == "capacity"]
    isT = c.Correctness == "T"
    V["corr/mae_T"] = f"{c.abs_err[isT].mean():.0f}~mAh~g$^{{-1}}$ ($n={isT.sum()}$, capacity)"
    V["corr/mae_F"] = f"{c.abs_err[~isT].mean():.0f}~mAh~g$^{{-1}}$ ($n={(~isT).sum()}$)"
    return W, C


# ---------------------------------------------------------------- 7. SHAP group shares
def shap_shares(R):
    rows, feats = [], []
    for (t, tier), d in R[(R.protocol == "doi") & (R.variant == "all") & R.tier.isin(["t2", "t2_noise"])] \
            .groupby(["target", "tier"]):
        per, names = [], None
        for _, r in d.iterrows():
            f = Path(r.dir) / "shap" / f"{r.split}.npz"
            if not f.exists():
                continue
            s = np.load(f, allow_pickle=True)
            names = list(s["names"])
            per.append(np.abs(s["shap_ens"]).mean(0))
        if not per:
            continue
        per = np.array(per)
        g = np.array([group(n) for n in names])
        for gn in np.unique(g):
            sh = per[:, g == gn].sum(1) / per.sum(1)
            rows.append(dict(target=t, tier=tier, group=gn, share=sh.mean(), share_sd=sh.std(ddof=1)))
        if tier == "t2":
            F = pd.DataFrame(dict(feature=names, group=g, mean_abs_shap=per.mean(0))).sort_values(
                "mean_abs_shap", ascending=False)
            F["target"] = t; F["rank"] = np.arange(1, len(F) + 1); feats.append(F)
    G = pd.DataFrame(rows); G.to_csv(RES / "shap_groups_all.csv", index=False)
    F = pd.concat(feats) if feats else pd.DataFrame(columns=["feature", "group", "mean_abs_shap", "target", "rank"])
    F.to_csv(RES / "shap_features_all.csv", index=False)
    for _, r in G.iterrows():
        if r.tier == "t2":
            V[f"shap/{r.target}/{r.group}/share"] = f"{100 * r.share:.1f}\\%"
        if r.tier == "t2_noise" and r.group == "noise":
            V[f"shap/{r.target}/noise/share"] = f"{100 * r.share:.1f}\\%"
    n = G[(G.tier == "t2_noise") & (G.group == "noise")].share
    if len(n):
        V["shap/noise/share"] = f"{100 * n.min():.1f}--{100 * n.max():.1f}\\%"
    top = F[(F.target == "capacity") & (F["rank"] == 1)].iloc[0]
    V["shapf/capacity/rank1/feature"] = "\\texttt{" + top.feature.replace("_", "\\_") + "}"
    V["shapf/capacity/rank1/mean_abs_shap"] = fnum(top.mean_abs_shap)
    return G, F


# ---------------------------------------------------------------- 8. population
def population():
    P = pd.read_csv(RES / "population.csv")
    raw = pd.read_csv(ROOT / "data" / "raw" / "extracted" / "battery.csv", low_memory=False,
                      usecols=["Property", "DOI", "Date"])
    raw["g"] = raw.DOI.map(normalize_doi)
    rows, allg = [], set()
    for t in TGT:
        D = data(t)
        y, g = D["y"].values, D["doi_group"].values
        df = pd.DataFrame(dict(y=y, g=g)); k = df.groupby("g").size(); N, a = len(df), len(k)
        gm = df.groupby("g").y.transform("mean")
        msb = ((gm - y.mean()) ** 2).sum() / (a - 1); msw = ((y - gm) ** 2).sum() / (N - a)
        n0 = (N - (k ** 2).sum() / N) / (a - 1)
        rr = raw[raw.Property == PROPERTY[t]]
        rows.append(dict(target=t, icc1=(msb - msw) / (msb + (n0 - 1) * msw), records_per_pub_median=k.median(),
                         raw_publications=rr.loc[rr.DOI.notna(), "g"].nunique()))
        allg |= set(g)
    X = P.merge(pd.DataFrame(rows), on="target", suffixes=("_old", ""))
    X = X[[c for c in X.columns if not c.endswith("_old")]]
    X.to_csv(RES / "population_ext.csv", index=False)
    for _, r in X.iterrows():
        for c in ("records", "publications", "raw_publications", "removed_bounds", "removed_duplicates",
                  "removed_calculated"):
            V[f"pop/{r.target}/{c}"] = fnum(r[c], "int")
        V[f"pop/{r.target}/records_per_pub_median"] = fnum(r.records_per_pub_median, "int")
        V[f"pop/{r.target}/icc1"] = f"{r.icc1:.2f}"
    V["pop/total/records"] = fnum(X.records.sum(), "int")
    V["pop/total/publications"] = fnum(len(allg), "int")
    V["pop/all/removed_bounds"] = fnum(X.removed_bounds.sum(), "int")
    a = X.loc[X.removed_bounds.idxmax()]
    V["pop/removed_bounds/which-target"] = (f"{NICE[a.target].lower()} records ({int(a.removed_bounds)}, mostly "
                                            "values below $10^{-12}$~S~cm$^{-1}$)" if a.target == "conductivity"
                                            else f"{NICE[a.target].lower()} records ({int(a.removed_bounds)})")
    i, j = X.icc1.idxmin(), X.icc1.idxmax()
    LONG = {"coulombic_efficiency": "Coulombic efficiency"}
    V["pop/icc1/min"], V["pop/icc1/argmin"] = f"{X.icc1[i]:.2f}", LONG.get(X.target[i], NICE[X.target[i]].lower())
    V["pop/icc1/max"], V["pop/icc1/argmax"] = f"{X.icc1[j]:.2f}", LONG.get(X.target[j], NICE[X.target[j]].lower())
    yrs = pd.to_datetime(raw.Date.str.replace("/", "-"), errors="coerce").dt.year
    V["pop/date_max_year"] = str(int(yrs.max()))
    return X


def feature_counts():
    from bmd.features import base_frame
    X = base_frame(data("capacity"), "t3")
    for k, p in (("comp", "comp_"), ("mat", "mat_"), ("txt", "txt_"), ("meas", "meas_"), ("lit", "lit_")):
        V[f"feat/{k}/n"] = str(sum(c.startswith(p) for c in X.columns))
    import importlib
    tr = importlib.import_module("02_train")
    V["cfg/grid"] = "(" + ", ".join(f"{g['depth']}/{g['l2_leaf_reg']}" for g in tr.GRID) + ")"


# ---------------------------------------------------------------- figures
def figures(S, U, G, F, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    COL = {"random": "#9DB4CF", "doi": "#2F5D8A", "temporal": "#C97B4A"}
    LAB = {"random": "Record-level random", "doi": "DOI-grouped", "temporal": "Temporal"}
    UNIT = {"capacity": "mAh g$^{-1}$", "voltage": "V", "conductivity": "log$_{10}$ S cm$^{-1}$",
            "coulombic_efficiency": "%", "energy": "Wh kg$^{-1}$"}

    # target distributions
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 1.8))
    for a, t in zip(ax, TGT):
        D = data(t); s = D.Warning.fillna("").str.contains("S")
        bins = np.histogram_bin_edges(D.y, 40)
        a.hist(D.y[~s], bins, color="#2F5D8A", alpha=.8, label="not S")
        a.hist(D.y[s], bins, color="#C97B4A", alpha=.6, label="S (series)")
        a.set_title(NICE[t]); a.set_xlabel(UNIT[t]); a.set_yticks([])
    ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fig_target_distributions.png", dpi=300); plt.close(fig)

    # split protocol by target
    E = S[(S.tier == "t2") & (S.variant == "all") & (S.model == "ens")]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, t in enumerate(TGT):
        for j, p in enumerate(("random", "doi", "temporal")):
            r = E[(E.target == t) & (E.protocol == p)]
            if r.empty:
                continue
            r = r.iloc[0]; x = i + (j - 1) * .27
            ax.bar(x, r.r2, .25, yerr=r.r2_sd, color=COL[p], capsize=1.5, label=LAB[p] if i == 0 else None)
            ax.text(x, max(r.r2, 0) + (0 if np.isnan(r.r2_sd) else r.r2_sd) + .01, f"{r.r2:.2f}", ha="center",
                    fontsize=5.5)
    ax.set_xticks(range(5)); ax.set_xticklabels([NICE[t] for t in TGT]); ax.set_ylabel("$R^2$ (t$_2$ ensemble)")
    ax.axhline(0, color="k", lw=.5); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_split_by_target.png", dpi=300); plt.close(fig)

    # SHAP groups + capacity top 15
    GC = {"composition": "#6BAF7C", "material": "#4A78A8", "material_text": "#9B8AC4", "measurement": "#C97B4A"}
    GN = {"composition": "Composition", "material": "Elemental properties", "material_text": "Name flags",
          "measurement": "Measurement context"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.2]))
    g2 = G[G.tier == "t2"]
    bottom = np.zeros(5)
    for k in GC:
        v = np.array([g2[(g2.target == t) & (g2.group == k)].share.sum() * 100 for t in TGT])
        ax[0].bar(range(5), v, .6, bottom=bottom, color=GC[k], label=GN[k]); bottom += v
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels([NICE[t] for t in TGT], rotation=30, ha="right")
    ax[0].set_ylabel("Share of mean |SHAP| (%)"); ax[0].legend(frameon=False, fontsize=6, loc="lower left")
    ax[0].set_title("(a) DOI-grouped, t$_2$")
    top = F[F.target == "capacity"].head(15).iloc[::-1]
    ax[1].barh(range(len(top)), top.mean_abs_shap, color=[GC.get(g, "#999") for g in top.group])
    ax[1].set_yticks(range(len(top))); ax[1].set_yticklabels(top.feature, fontsize=5.5)
    ax[1].set_xlabel("Mean |SHAP| (mAh g$^{-1}$)"); ax[1].set_title("(b) Capacity, top 15")
    fig.tight_layout(); fig.savefig(FIG / "fig_shap_groups_by_target.png", dpi=300); plt.close(fig)

    # uncertainty
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    d = R[(R.target == "capacity") & (R.protocol == "doi") & (R.tier == "t2") & (R.variant == "all")]
    Us, Es = [], []
    for _, r in d.iterrows():
        P = preds(r.dir, r.split); Us.append(P["unc"]); Es.append(np.abs(P["y"] - P["ens"]))
    hb = ax[0].hexbin(np.concatenate(Us), np.concatenate(Es), gridsize=45, bins="log", cmap="Blues", mincnt=1)
    ax[0].set_xlabel("Member spread (mAh g$^{-1}$)"); ax[0].set_ylabel("|error| (mAh g$^{-1}$)")
    ax[0].set_title("(a) Capacity, DOI-grouped"); fig.colorbar(hb, ax=ax[0], label="records (log)")
    for t, c in zip(TGT, ["#2F5D8A", "#C97B4A", "#6BAF7C", "#9B8AC4", "#B5505A"]):
        for p, ls in (("doi", "-"), ("random", ":")):
            d = R[(R.target == t) & (R.protocol == p) & (R.tier == "t2") & (R.variant == "all")]
            if d.empty:
                continue
            Us, Es = [], []
            for _, r in d.iterrows():
                P = preds(r.dir, r.split); Us.append(P["unc"]); Es.append(np.abs(P["y"] - P["ens"]))
            u, e = np.concatenate(Us), np.concatenate(Es)
            dec = pd.qcut(u, 10, labels=False, duplicates="drop")
            m = pd.Series(e).groupby(dec).mean()
            ax[1].plot(m.index + 1, m / e.mean(), ls, color=c, label=NICE[t] if p == "doi" else None)
    ax[1].set_xlabel("Spread decile"); ax[1].set_ylabel("MAE / overall MAE")
    ax[1].set_title("(b) solid: DOI-grouped, dotted: random"); ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fig_uncertainty.png", dpi=300); plt.close(fig)

    # high-value errors (used in text)
    rows = []
    for t in TGT:
        d = R[(R.target == t) & (R.protocol == "doi") & (R.tier == "t2") & (R.variant == "all")]
        Y, Pp = [], []
        for _, r in d.iterrows():
            P = preds(r.dir, r.split); Y.append(P["y"]); Pp.append(P["ens"])
        if not Y:
            continue
        y, p = np.concatenate(Y), np.concatenate(Pp)
        q = pd.qcut(y, 10, labels=False, duplicates="drop")
        for k in np.unique(q):
            rows.append(dict(target=t, decile=k + 1, y_mean=y[q == k].mean(), bias=(p - y)[q == k].mean(),
                             mae=np.abs(p - y)[q == k].mean()))
    pd.DataFrame(rows).to_csv(RES / "error_by_target_decile.csv", index=False)

# ---------------------------------------------------------------- 9. previous-study pipelines (07_previous_pipelines.py)
def previous():
    """Jablonka pipeline vs our ensemble on the identical test records (split 1, records their parser can featurise)."""
    f, c = RES / "previous_pipelines.csv", RES / "clever_hans.csv"
    rows = []
    for pf in sorted((RES / "previous_preds").glob("*_1.npz")):
        stem = pf.stem[:-2]
        model = "zb" if stem.startswith("zb_") else "jab"
        t, proto = stem[3:].rsplit("_", 1) if model == "zb" else stem.rsplit("_", 1)
        a = np.load(pf)
        e = RES / t / proto / "preds" / "1.npz"
        rows.append(dict(target=t, protocol=proto, model=model, n=len(a["y"]), **_m(a["y"], a["pred"])))
        if e.exists():
            b = np.load(e, allow_pickle=True)
            pos = pd.Series(np.arange(len(b["test_idx"])), index=b["test_idx"])[a["test_idx"]].values
            assert np.allclose(b["y"][pos], a["y"])
            rows.append(dict(target=t, protocol=proto, model="ens" if model == "jab" else "ens_zb", n=len(pos),
                             **_m(b["y"][pos], b["ens"][pos])))
    if rows:
        P = pd.DataFrame(rows); P.to_csv(RES / "previous_vs_ensemble.csv", index=False)
        for _, r in P.iterrows():
            base = f"prev/{r.target}/{r.protocol}/{r.model}"
            V[f"{base}/r2"], V[f"{base}/mae"], V[f"{base}/rmse"] = f"{r.r2:.3f}", fnum(r.mae), fnum(r.rmse)
    z = RES / "zhuang_pipeline.csv"
    if z.exists():
        Z = pd.read_csv(z)
        for _, r in Z[Z.evaluation == "B"].iterrows():
            V[f"zbB/{r.target}/{r.protocol}/r2"] = f"{r.r2:.3f}"
            V[f"zbB/{r.target}/{r.protocol}/mae"] = fnum(r.mae)
            V[f"zbB/{r.target}/{r.protocol}/n"] = fnum(r.n_test, "int")
        a = Z[Z.evaluation == "A"]
        if len(a):
            V["prev/zb/coverage/min"] = f"{100 * a.test_coverage.min():.0f}\\%"
    if f.exists():
        Q = pd.read_csv(f)
        if "test_coverage" in Q:
            V["prev/coverage/min"] = f"{100 * Q.test_coverage.min():.0f}\\%"
            V["prev/coverage/max"] = f"{100 * Q.test_coverage.max():.0f}\\%"
    if c.exists():
        C = pd.read_csv(c)
        for _, r in C.iterrows():
            b = f"ch/{r.target}/{r.protocol}"
            V[f"{b}/year_mae"] = f"{r.year_mae:.1f}"
            V[f"{b}/year_mae_baseline"] = f"{r.year_mae_baseline:.1f}"
            V[f"{b}/journal_f1"] = f"{r.journal_f1_micro:.2f}"
            V[f"{b}/indirect_r2"] = f"{r.indirect_r2:.3f}"

# ---------------------------------------------------------------- 10. leakage channels that a DOI split does not remove
ALIGNED = {}


def aligned(t):
    """Raw battery.csv columns row-aligned with load_dataset(t) (01_prepare_data.clean keeps file order)."""
    if not ALIGNED:
        import importlib
        p1 = importlib.import_module("01_prepare_data")
        raw = pd.read_csv(ROOT / "data" / "raw" / "extracted" / "battery.csv", low_memory=False)
        kept, _ = p1.clean(raw)
        for k in TGT:
            ALIGNED[k] = kept[kept["target"] == k].reset_index(drop=True)
    R = ALIGNED[t]
    assert len(R) == len(data(t)) and (R["Value"].values == data(t)["Value"].values).all()
    return R


SPEC_KEYS = ("theoretical", "reversib", "discharg", "charg", "specific", "electronic|electrical", "potential")


def channels(R):
    """Repeated literature values, same-paper-mean reference, theoretical capacities, per-specifier baseline."""
    rows = []
    for t in TGT:
        D, A = data(t), aligned(t)
        name = D["Name"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        key = (name + "|" + D["Value"].round(3).astype(str)).values
        spec = A["Specifier"].fillna("").str.lower()
        theo = spec.str.contains("theoretical").values
        skey = np.array(["".join("1" if __import__("re").search(k, x) else "0" for k in SPEC_KEYS) for x in spec])
        y, g = D["y"].values, D["doi_group"].values
        for proto in ("random", "doi", "temporal", "material"):
            d = R[(R.target == t) & (R.protocol == proto) & (R.tier == "t2") & (R.variant == "all")]
            for _, r in d.iterrows():
                P = preds(r.dir, r.split); te = P["test_idx"]
                m = np.ones(len(y), bool); m[te] = False
                rep = np.isin(key[te], np.unique(key[m]))
                row = dict(target=t, protocol=proto, split=r.split, n=len(te), rep_share=rep.mean(),
                           r2_rep=r2(P["y"][rep], P["ens"][rep]) if rep.sum() > 10 else np.nan,
                           r2_new=r2(P["y"][~rep], P["ens"][~rep]) if (~rep).sum() > 10 else np.nan,
                           mae_rep=np.abs(P["y"] - P["ens"])[rep].mean(), mae_new=np.abs(P["y"] - P["ens"])[~rep].mean(),
                           theo_share=theo[te].mean(),
                           r2_no_theo=r2(P["y"][~theo[te]], P["ens"][~theo[te]]))
                # per-specifier median baseline (training median of records with the same specifier keyword set)
                med = pd.Series(y[m]).groupby(skey[m]).median()
                pb = pd.Series(skey[te]).map(med).fillna(np.median(y[m])).values
                row["specbase_rmse"] = float(np.sqrt(((P["y"] - pb) ** 2).mean()))
                row["specbase_r2"] = r2(P["y"], pb)
                if proto == "random":   # same-paper mean of training records (fallback: global mean)
                    mp = pd.Series(y[m]).groupby(g[m]).mean()
                    po = pd.Series(g[te]).map(mp).fillna(y[m].mean()).values
                    row["samepaper_r2"] = r2(P["y"], po)
                rows.append(row)
    C = pd.DataFrame(rows); C.to_csv(RES / "leakage_channels.csv", index=False)
    A = C.groupby(["target", "protocol"]).mean(numeric_only=True).reset_index()
    for _, r in A.iterrows():
        b = f"rep/{r.target}/{r.protocol}"
        V[f"{b}/share"] = fnum(r.rep_share, "pct")
        V[f"{b}/r2_rep"], V[f"{b}/r2_new"] = f"{r.r2_rep:.2f}", f"{r.r2_new:.2f}"
        V[f"notheo/{r.target}/{r.protocol}/r2"] = f"{r.r2_no_theo:.3f}"
        V[f"m/{r.target}/{r.protocol}/specbase/rmse"] = fnum(r.specbase_rmse)
        V[f"m/{r.target}/{r.protocol}/specbase/r2"] = f"{r.specbase_r2:.3f}"
        if r.protocol == "random":
            V[f"oracle/{r.target}/samepaper_r2"] = f"{r.samepaper_r2:.2f}"
    V["theo/capacity/share"] = fnum(A[(A.target == "capacity") & (A.protocol == "doi")].theo_share.iloc[0], "pct")
    d = A[A.protocol == "doi"]
    V["rep/doi/share-range"] = f"{100 * d.rep_share.min():.0f}--{100 * d.rep_share.max():.0f}\\%"
    o = A[A.protocol == "random"]
    V["oracle/samepaper_r2-range"] = f"{o.samepaper_r2.min():.2f}--{o.samepaper_r2.max():.2f}"
    return C

def extras():
    """High-value error keys and the extra protocols (material, kfold) in compact form."""
    f = RES / "error_by_target_decile.csv"
    if f.exists():
        d = pd.read_csv(f)
        for t, x in d.groupby("target"):
            top, bot = x[x.decile == x.decile.max()].iloc[0], x[x.decile == 1].iloc[0]
            V[f"hv/{t}/top/ymean"], V[f"hv/{t}/top/bias"] = fnum(top.y_mean), fnum(top.bias)
            V[f"hv/{t}/bottom/ymean"], V[f"hv/{t}/bottom/bias"] = fnum(bot.y_mean), fnum(bot.bias)
    S = pd.read_csv(RES / "summary_full.csv")
    e = S[(S.tier == "t2") & (S.variant == "all") & (S.model == "ens")].set_index(["target", "protocol"]).r2
    for t in TGT:
        if (t, "temporal") in e.index and (t, "doi") in e.index:
            V[f"d/{t}/temporal-doi/r2"] = f"{e[(t, 'doi')] - e[(t, 'temporal')]:.3f}"
    for proto in ("material", "kfold"):
        x = S[(S.protocol == proto) & (S.tier == "t2") & (S.variant == "all") & (S.model == "ens")]
        for _, r in x.iterrows():
            V[f"m/{r.target}/{proto}/t2/r2"] = pm(r.r2, r.r2_sd, "r2")


def main():
    R = load_runs()
    print(R.groupby(["target", "protocol", "tier", "variant"]).size().to_string())
    S = summary(R)
    population(); feature_counts()
    D = paired(R); U = uncertainty(R); overlap(R); series(R, S); label_errors(R)
    G, F = shap_shares(R)
    figures(S, U, G, F, R)
    previous()
    channels(R)
    extras()
    (RES / "paper_values.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    print(f"{len(V)} values -> results/paper_values.json; figures -> {FIG}")


if __name__ == "__main__":
    main()

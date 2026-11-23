"""Summarise 02_train.py runs per target/protocol/tier, grouped SHAP attribution and the uncertainty proxy.

  python scripts/03_evaluate.py              # results/summary.csv, shap_{groups,features}.csv, uncertainty.csv
  python scripts/03_evaluate.py --figures    # + results/figures/{split_protocol,shap_groups}.png, shap_top15_<target>.png
  python scripts/03_evaluate.py --shap-run doi_t3   # SHAP tables/figures for another run directory (files get a _<run> suffix)

Metrics: mean +/- sample s.d. over splits (last record per split wins if a split was rerun). Runs made with --fast or
--max-train are flagged in summary.csv (they are smoke tests, not results).
SHAP: mean |SHAP| per feature over each test set; group share = group sum / total over all predictors (shares sum to 1),
missingness indicators counted with their base feature; mean and sample s.d. over splits.
Uncertainty: per split, Spearman rank correlation between the member spread (s.d. of CatBoost seeds + XGBoost) and
|error| of the ensemble; mean spread for test records whose DOI is seen vs unseen in train/val (random protocol).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bmd.data import TARGETS  # noqa: E402
from bmd.groups import group  # noqa: E402

RES = ROOT / "results"
MODELS = ("ens", "cb_bag", "xgb", "cb_tuned")
GROUP_COLOUR = {"composition": "#6BAF7C", "material": "#4A78A8", "material_text": "#9B8AC4",
                "measurement": "#C97B4A", "literature": "#D4A93C", "identity": "#888888", "noise": "#CCCCCC"}
GROUP_NAME = {"composition": "Composition", "material": "Material properties", "material_text": "Name text",
              "measurement": "Measurement", "literature": "Literature (title)", "identity": "Name identity",
              "noise": "Noise"}
PROTO_STYLE = {"random": ("#4A78A8", "//"), "doi": ("#4A78A8", None), "temporal": ("#C97B4A", None)}
PROTO_NAME = {"random": "Record-level random", "doi": "DOI-grouped", "temporal": "Temporal"}
sd1 = lambda a: a.std(0, ddof=1) if len(a) > 1 else np.full(np.shape(a)[1:], np.nan)  # noqa: E731


def runs():
    R = []
    for f in sorted(RES.glob("*/*/runs.jsonl")):
        d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l])
        R.append(d.drop_duplicates(["split"], keep="last").assign(run=f.parent.name, target=f.parent.parent.name))
    return pd.concat(R, ignore_index=True) if R else pd.DataFrame()


def summary(R):
    rows = []
    for (t, run), d in R.groupby(["target", "run"]):
        for m in MODELS:
            row = dict(target=t, run=run, protocol=d.protocol.iloc[0], tier=d.tier.iloc[0], model=m,
                       n_splits=len(d), n_test=d.n_test.mean(), n_train=d.n_train.mean(),
                       smoke=bool(d.fast.any() or d.max_train.notna().any()), doi_overlap=int(d.doi_overlap.max()))
            for k in ("r2", "mae", "rmse"):
                row[k], row[f"{k}_sd"] = d[f"{m}_{k}"].mean(), d[f"{m}_{k}"].std(ddof=1)
            rows.append(row)
    S = pd.DataFrame(rows)
    if not S.empty:
        S.to_csv(RES / "summary.csv", index=False)
        print(S[S.model == "ens"].drop(columns="model").round(3).to_string(index=False))
    return S


def shap_tables(run="doi"):
    G, F = [], []
    for t in TARGETS:
        files = sorted((RES / t / run / "shap").glob("*.npz"))
        if not files:
            continue
        per, names = [], None
        for f in files:
            s = np.load(f, allow_pickle=True)
            names = list(s["names"]) if names is None else names
            assert list(s["names"]) == names, "feature sets differ between splits"
            per.append(np.abs(s["shap_ens"]).mean(0))
        per = np.array(per)
        fg = pd.DataFrame({"feature": names, "group": [group(n) for n in names]})
        Fa = fg.assign(target=t, mean_abs_shap=per.mean(0), sd_across_splits=sd1(per)).sort_values(
            "mean_abs_shap", ascending=False)
        Fa["rank"] = np.arange(1, len(Fa) + 1)
        F.append(Fa)
        for gname in sorted(fg.group.unique()):
            shares = per[:, (fg.group == gname).values].sum(1) / per.sum(1)
            G.append(dict(target=t, run=run, group=gname, n_features=int((fg.group == gname).sum()),
                          share=shares.mean(), share_sd=sd1(shares[:, None])[0], n_splits=len(files)))
    if not G:
        print(f"No SHAP files for run {run!r}. Run 02_train.py --shap first.")
        return None, None
    G, F = pd.DataFrame(G), pd.concat(F, ignore_index=True)
    sfx = "" if run == "doi" else f"_{run}"
    G.to_csv(RES / f"shap_groups{sfx}.csv", index=False)
    F.to_csv(RES / f"shap_features{sfx}.csv", index=False)
    print(G.assign(share=lambda d: (100 * d.share).round(1), share_sd=lambda d: (100 * d.share_sd).round(1))
          .to_string(index=False))
    return G, F


def uncertainty():
    """Rank correlation of the across-member spread with |error|; spread for DOI-seen vs DOI-unseen test records."""
    from scipy.stats import spearmanr
    rows = []
    for f in sorted(RES.glob("*/*/preds/*.npz")):
        p = np.load(f, allow_pickle=True)
        if "unc" not in p:
            continue
        err, unc = np.abs(p["ens"] - p["y"]), p["unc"]
        seen = p["seen"].astype(bool)
        rows.append(dict(target=f.parents[2].name, run=f.parents[1].name, split=int(f.stem),
                         spearman=spearmanr(unc, err).statistic, mean_unc=unc.mean(),
                         unc_seen=unc[seen].mean() if seen.any() else np.nan,
                         unc_unseen=unc[~seen].mean() if (~seen).any() else np.nan,
                         abserr_seen=err[seen].mean() if seen.any() else np.nan,
                         abserr_unseen=err[~seen].mean() if (~seen).any() else np.nan, n_test=len(err)))
    if not rows:
        return None
    U = pd.DataFrame(rows)
    U.to_csv(RES / "uncertainty.csv", index=False)
    A = U.groupby(["target", "run"]).agg(n_splits=("split", "size"), spearman=("spearman", "mean"),
                                         spearman_sd=("spearman", "std"), unc_seen=("unc_seen", "mean"),
                                         unc_unseen=("unc_unseen", "mean"), abserr_seen=("abserr_seen", "mean"),
                                         abserr_unseen=("abserr_unseen", "mean")).reset_index()
    print("\nUncertainty proxy (member spread) vs |error|")
    print(A.round(3).to_string(index=False))
    return A


def figures(S, G, F, run="doi"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out = RES / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    targets = [t for t in TARGETS if S is not None and not S.empty and t in set(S.target)]
    E = S[(S.model == "ens") & (S.tier == "t2") & (S.run == S.protocol)] if targets else pd.DataFrame()
    if not E.empty:
        protos = [p for p in PROTO_STYLE if p in set(E.protocol)]
        w = .8 / len(protos)
        fig, ax = plt.subplots(figsize=(1.2 * len(targets) + 1.2, 2.8))
        for ti, t in enumerate(targets):
            for si, proto in enumerate(protos):
                r = E[(E.target == t) & (E.protocol == proto)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col, hatch = PROTO_STYLE[proto]
                x = ti + (si - (len(protos) - 1) / 2) * w
                ax.bar(x, r.r2, w * .92, yerr=r.r2_sd if r.n_splits > 1 else None, color=col, hatch=hatch,
                       edgecolor="black" if hatch else "white", lw=.5, capsize=2)
                ax.text(x, max(r.r2, 0) + (0 if np.isnan(r.r2_sd) else r.r2_sd) + .012, f"{r.r2:.2f}",
                        ha="center", fontsize=5.5)
        ax.set_xticks(range(len(targets))); ax.set_xticklabels([t.replace("_", "\n") for t in targets])
        ax.set_ylabel("$R^2$ (test)"); ax.set_ylim(min(0, E.r2.min() - .05), 1)
        ax.legend(handles=[Patch(fc=PROTO_STYLE[p][0], hatch=PROTO_STYLE[p][1], ec="black" if PROTO_STYLE[p][1]
                                 else "white", label=PROTO_NAME[p]) for p in protos],
                  frameon=False, fontsize=6.5, loc="upper right", ncol=len(protos))
        fig.tight_layout(); fig.savefig(out / "split_protocol.png", dpi=300); plt.close(fig)
    if G is not None:
        groups = [k for k in GROUP_COLOUR if k in set(G.group)]
        tg = [t for t in TARGETS if t in set(G.target)]
        w = .8 / len(groups)
        fig, ax = plt.subplots(figsize=(1.3 * len(tg) + 1.5, 2.6))
        for ai, t in enumerate(tg):
            g = G[G.target == t].set_index("group")
            for gi, k in enumerate(groups):
                if k not in g.index:
                    continue
                ax.bar(ai + (gi - (len(groups) - 1) / 2) * w, 100 * g.loc[k, "share"], w * .9,
                       yerr=100 * g.loc[k, "share_sd"] if g.loc[k, "n_splits"] > 1 else None,
                       color=GROUP_COLOUR[k], capsize=1.5, label=GROUP_NAME[k] if ai == 0 else None)
        ax.set_xticks(range(len(tg))); ax.set_xticklabels([t.replace("_", "\n") for t in tg])
        ax.set_ylabel("Share of total mean |SHAP| (%)"); ax.legend(frameon=False, fontsize=6.5, ncol=2)
        sfx = "" if run == "doi" else f"_{run}"
        fig.tight_layout(); fig.savefig(out / f"shap_groups{sfx}.png", dpi=300); plt.close(fig)
        for t in tg:
            top = F[F.target == t].head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(4.6, 3.4))
            ax.barh(range(len(top)), top.mean_abs_shap, xerr=top.sd_across_splits,
                    color=[GROUP_COLOUR[g] for g in top.group])
            ax.set_yticks(range(len(top))); ax.set_yticklabels(top.feature, fontsize=6)
            tr = TARGETS[t]
            ax.set_xlabel(f"Mean |SHAP| ({'log10 ' if tr['transform'] else ''}{tr['unit']})")
            ax.set_title(t, fontsize=8)
            ax.legend(handles=[Patch(fc=GROUP_COLOUR[k], label=GROUP_NAME[k]) for k in groups], frameon=False,
                      fontsize=6, loc="lower right")
            fig.tight_layout(); fig.savefig(out / f"shap_top15_{t}{sfx}.png", dpi=300); plt.close(fig)
    print(f"Figures -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--shap-run", default="doi")
    a = ap.parse_args()
    R = runs()
    if R.empty:
        sys.exit("No runs found under results/<target>/<run>/runs.jsonl. Run 02_train.py first.")
    S = summary(R)
    G, F = shap_tables(a.shap_run)
    uncertainty()
    if a.figures:
        figures(S, G, F, a.shap_run)

"""Adapted pipeline of Zhuang & Barnard (2024), J. Chem. Inf. Model. 64, 9205 (doi:10.1021/acs.jcim.4c01343).

Re-implemented from the paper's Methods (their regression notebook is in the ACS Supporting Information, which we did
not use): Mendeleev+ encoding = element-amount vector (global min-max scaling) + min/max/range/mean/std of tabulated
elemental properties over the elements of a formula (each feature min-max scaled); XGBoost regressor with randomised
hyperparameter search by k-fold cross-validation on the training set. Elemental properties come from their
periodic_table.csv (github.com/ZoraZhuang/battery_datasets, GPL-3.0): atomic mass, group, row, molar volume, atomic
number, block and valence; density, atomic volume and Mendeleev number are not in that table and are omitted.
Deviation, for cost: 25 random configurations, 3-fold CV on at most 20,000 training records (they searched 1,000
configurations with 10-fold stratified CV); the selected configuration is refitted on the full training set.

Two evaluations:
  A  our cleaned records and our split-1 partitions (random, doi), identical test records to the ensemble;
  B  their cleaned records (batteries_ultra_cleaned.csv), their random 85/15 split and a DOI-grouped 85/15 split.

  python scripts/08_zhuang_pipeline.py
Outputs: results/zhuang_pipeline.csv, results/previous_preds/zb_<target>_<protocol>_1.npz (evaluation A)
"""
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from bmd.data import load_dataset, normalize_doi  # noqa: E402
from bmd.evaluate import metrics  # noqa: E402
from bmd.splits import make_splits  # noqa: E402

EXT = ROOT / "data" / "external" / "zhuang2024"
OUT = ROOT / "results" / "zhuang_pipeline.csv"
PRED = ROOT / "results" / "previous_preds"
TGT = {"capacity": "Capacity", "voltage": "Voltage", "conductivity": "Conductivity",
       "coulombic_efficiency": "Coulombic Efficiency", "energy": "Energy"}
THEIR_COL = {"capacity": "Capacity/Gram^(-1.0)__Hour^(1.0)__MilliAmpere^(1.0)", "voltage": "Voltage/Volt^(1.0)",
             "conductivity": "Conductivity/CentiMeter^(-1.0)__Siemens^(1.0)",
             "coulombic_efficiency": "Coulombic Efficiency/Percent^(1.0)",
             "energy": "Energy/KiloGram^(-1.0)__WattHour^(1.0)"}
PROPS = ["atomic_mass", "group", "row", "molar_volume", "number", "block", "valance"]
SEED, N_ITER, MAX_SEARCH, THREADS = 42, 25, 20000, 4


def periodic():
    P = pd.read_csv(EXT / "periodic_table.csv").set_index("symbol")
    P["block"] = P["block"].map({"s": 0, "p": 1, "d": 2, "f": 3})
    return P[PROPS].apply(pd.to_numeric, errors="coerce")


def comp_from_extracted(s):
    try:
        out = {}
        for d in ast.literal_eval(s):
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + float(v)
        return out
    except Exception:
        return None


def comp_from_formulas(s):
    from pymatgen.core import Composition
    try:
        fs = ast.literal_eval(s)
        fs = [fs] if isinstance(fs, str) else fs
        out = {}
        for f in fs:
            for el, v in Composition(f).get_el_amt_dict().items():
                out[el] = out.get(el, 0.0) + float(v)
        return out
    except Exception:
        return None


def encode(comps, P, elements):
    """Raw Mendeleev+ block (unscaled): element amounts + unweighted stats of elemental properties."""
    rows = []
    for c in comps:
        if not c:
            rows.append([np.nan] * (len(elements) + 5 * len(PROPS)))
            continue
        amt = [c.get(e, 0.0) for e in elements]
        els = [e for e in c if e in P.index]
        V = P.loc[els].values if els else np.full((1, len(PROPS)), np.nan)
        stats = np.concatenate([np.nanmin(V, 0), np.nanmax(V, 0), np.nanmax(V, 0) - np.nanmin(V, 0),
                                np.nanmean(V, 0), np.nanstd(V, 0)])
        rows.append(amt + list(stats))
    return np.array(rows, dtype=float)


def scale(Xtr, Xte, n_el):
    """Global min-max for the amount block, per-feature min-max for the statistics (fitted on training)."""
    A, B = Xtr[:, :n_el], Xtr[:, n_el:]
    gmax = np.nanmax(A) or 1.0
    lo, hi = np.nanmin(B, 0), np.nanmax(B, 0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    f = lambda X: np.column_stack([X[:, :n_el] / gmax, (X[:, n_el:] - lo) / rng])  # noqa: E731
    return f(Xtr), f(Xte)


def fit_xgb(X, y):
    from scipy.stats import loguniform, randint, uniform
    from sklearn.model_selection import KFold, RandomizedSearchCV
    from xgboost import XGBRegressor
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), min(len(y), MAX_SEARCH), replace=False)
    space = dict(n_estimators=randint(100, 1000), max_depth=randint(3, 11), learning_rate=loguniform(0.01, 0.3),
                 subsample=uniform(0.5, 0.5), colsample_bytree=uniform(0.5, 0.5), min_child_weight=randint(1, 11),
                 reg_lambda=loguniform(0.1, 10))
    base = XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=THREADS)
    S = RandomizedSearchCV(base, space, n_iter=N_ITER, cv=KFold(3, shuffle=True, random_state=SEED),
                           scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=1).fit(X[idx], y[idx])
    m = XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=THREADS, **S.best_params_).fit(X, y)
    return m, S.best_params_


def evaluate(label, t, proto, X, y, tr, te, n_el, rows, save=None):
    ok_tr, ok_te = tr[~np.isnan(X[tr]).all(1)], te[~np.isnan(X[te]).all(1)]
    Xtr, Xte = scale(X[ok_tr], X[ok_te], n_el)
    m, bp = fit_xgb(Xtr, y[ok_tr])
    p = m.predict(Xte)
    rows.append(dict(evaluation=label, target=t, protocol=proto, n_train=len(ok_tr), n_test=len(ok_te),
                     test_coverage=len(ok_te) / len(te), best_params=str(bp), **metrics(y[ok_te], p)))
    if save:
        PRED.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PRED / save, test_idx=ok_te, y=y[ok_te], pred=p)
    print(f"{label} {t:22s} {proto:6s} R2={rows[-1]['r2']:.3f} MAE={rows[-1]['mae']:.3g} n_test={len(ok_te)}", flush=True)


def main():
    P = periodic()
    rows = []
    # ---- A: our records, our split-1 partitions
    import importlib
    p1 = importlib.import_module("01_prepare_data")
    kept, _ = p1.clean(pd.read_csv(ROOT / "data" / "raw" / "extracted" / "battery.csv", low_memory=False))
    for t in TGT:
        D = load_dataset(t)
        R = kept[kept["target"] == t].reset_index(drop=True)
        assert len(R) == len(D) and (R["Value"].values == D["Value"].values).all()
        comps = R["Extracted_name"].map(comp_from_extracted).tolist()
        elements = sorted({e for c in comps if c for e in c if e in P.index}, key=lambda e: P.loc[e, "number"])
        X = encode(comps, P, elements)
        y, g = D["y"].values, D["doi_group"].values
        for proto in ("random", "doi"):
            tr, va, te = next(make_splits(g, proto, 5, 0.2, 0.15, SEED))
            evaluate("A", t, proto, X, y, np.r_[tr, va], te, len(elements), rows, save=f"zb_{t}_{proto}_1.npz")
        pd.DataFrame(rows).to_csv(OUT, index=False)
    # ---- B: their cleaned records, their 85/15 random split vs a DOI-grouped 85/15 split
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
    import time
    while not (EXT / "SOURCE.txt").exists():   # written after the download of batteries_ultra_cleaned.csv completes
        time.sleep(30)
    Z = pd.read_csv(EXT / "batteries_ultra_cleaned.csv", low_memory=False)
    for t, prop in TGT.items():
        d = Z[(Z["Property"] == prop) & Z[THEIR_COL[t]].notna()].reset_index(drop=True)
        comps = d["Formulas"].map(comp_from_formulas).tolist()
        elements = sorted({e for c in comps if c for e in c if e in P.index}, key=lambda e: P.loc[e, "number"])
        X = encode(comps, P, elements)
        y = pd.to_numeric(d[THEIR_COL[t]], errors="coerce").values   # their units (conductivity not log-scaled)
        keep = ~np.isnan(y); X, y, d = X[keep], y[keep], d[keep].reset_index(drop=True)
        g = d["DOI"].map(normalize_doi).values
        for proto, sp in (("random", ShuffleSplit(1, test_size=0.15, random_state=SEED).split(X)),
                          ("doi", GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(X, groups=g))):
            tr, te = next(sp)
            evaluate("B", t, proto, X, y, tr, te, len(elements), rows)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows)[["evaluation", "target", "protocol", "n_test", "r2", "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

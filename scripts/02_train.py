"""Train and evaluate the ensemble per battery property under repeated holdouts.

Model: equal-weight mean of a 3-seed CatBoost bag and XGBoost. Per split, the
CatBoost configuration is selected from a predeclared 6-point grid by validation RMSE and refitted with seeds 42, 7 and
123; XGBoost is early-stopped on the same validation set. Splits: make_splits(groups, protocol, 5, test 0.2, val 0.15,
seed 42); 'doi' groups by normalised DOI, 'random' splits records, 'temporal' trains on earlier publications (records
of undated publications are excluded from this protocol only), 'kfold' is a disjoint grouped 5-fold.
Every statistic (imputation medians, missingness filter, category maps, --max-train subsample) comes from the train
indices only.

  python scripts/02_train.py --target all --protocol doi --shap
  python scripts/02_train.py --target voltage energy --protocol random --tier t3
  python scripts/02_train.py --target capacity --protocol doi --max-train 20000 --fast --splits 1   # smoke test

Outputs (<run> = <protocol> for tier t2, else <protocol>_<tier>): results/<target>/<run>/runs.jsonl,
preds/<k>.npz (test_idx, y, doi, per-model predictions, unc = s.d. across CatBoost seeds + XGBoost, seen = test DOI
present in train/val), shap/<k>.npz; models/<target>/<run>/<k>/ (CatBoost .cbm, XGBoost .json, preprocess.pkl).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import os

# threads per fit (-1 = all cores); set BMD_THREADS when several fits share a node
THREADS = int(os.environ.get("BMD_THREADS", "-1"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bmd.data import TARGETS, TIERS, load_dataset  # noqa: E402
from bmd.evaluate import metrics  # noqa: E402
from bmd.features import TrainFitPreprocessor, base_frame  # noqa: E402
from bmd.splits import make_splits  # noqa: E402

GRID = [dict(depth=6, l2_leaf_reg=3), dict(depth=8, l2_leaf_reg=3), dict(depth=10, l2_leaf_reg=3),
        dict(depth=8, l2_leaf_reg=10), dict(depth=10, l2_leaf_reg=10), dict(depth=8, l2_leaf_reg=1)]
BASE = dict(learning_rate=0.05, iterations=3000, subsample=0.8, colsample_bylevel=0.7,
            min_data_in_leaf=5, loss_function="RMSE", allow_writing_files=False)
SEEDS = [42, 7, 123]
XGB = dict(n_estimators=1500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
           tree_method="hist", early_stopping_rounds=100, random_state=42, n_jobs=THREADS)
FAST = dict(grid=GRID[1:2], iterations=1000, seeds=SEEDS[:1])   # --fast: smoke-test settings, not for results


def encode(X, cats):
    """Ordinal-encode every categorical column with the train-fitted maps (unseen -> -1) for XGBoost."""
    Z = X.copy()
    for c, mp in cats.items():
        Z[c] = Z[c].map(mp).fillna(-1)
    return Z.values.astype(np.float32)


def fit_ensemble(Xtr, ytr, Xva, yva, cat_cols, fast=False):
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    grid, seeds = (FAST["grid"], FAST["seeds"]) if fast else (GRID, SEEDS)
    base = dict(BASE, iterations=FAST["iterations"]) if fast else BASE
    ci = [i for i, c in enumerate(Xtr.columns) if c in cat_cols]
    best, grid_val = None, []
    for gp in grid:
        m = CatBoostRegressor(**base, **gp, random_seed=seeds[0], verbose=0, thread_count=THREADS)
        m.fit(Xtr, ytr, cat_features=ci, eval_set=(Xva, yva), early_stopping_rounds=150, verbose=False)
        rm = float(np.sqrt(np.mean((m.predict(Xva) - yva) ** 2)))
        grid_val.append(dict(**gp, val_rmse=rm, best_iteration=int(m.get_best_iteration())))
        if best is None or rm < best[0]:
            best = (rm, gp, m)
    _, gp, m0 = best
    cbs = [m0]
    for s in seeds[1:]:
        m = CatBoostRegressor(**base, **gp, random_seed=s, verbose=0, thread_count=THREADS)
        m.fit(Xtr, ytr, cat_features=ci, eval_set=(Xva, yva), early_stopping_rounds=150, verbose=False)
        cbs.append(m)
    cats = {c: {v: i for i, v in enumerate(sorted(Xtr[c].unique()))} for c in cat_cols if c in Xtr.columns}
    gx = XGBRegressor(**XGB)
    gx.fit(encode(Xtr, cats), ytr, eval_set=[(encode(Xva, cats), yva)], verbose=False)
    return dict(cbs=cbs, xgb=gx, cats=cats, ci=ci, selected=gp, grid=grid_val, seeds=seeds[:len(cbs)])


def predict_parts(E, X):
    p_seeds = np.array([m.predict(X) for m in E["cbs"]])
    p_x = E["xgb"].predict(encode(X, E["cats"]))
    p_bag = p_seeds.mean(0)
    # cheap epistemic proxy: spread of the individual ensemble members (CatBoost seeds + XGBoost)
    unc = np.vstack([p_seeds, p_x[None]]).std(0, ddof=1)
    return {"cb_tuned": p_seeds[0], "cb_bag": p_bag, "xgb": p_x, "ens": (p_bag + p_x) / 2}, unc


def ensemble_shap(E, X):
    """Exact TreeSHAP of the ensemble: phi = 0.5 * mean_seed(phi_CatBoost) + 0.5 * phi_XGBoost (prediction units)."""
    from catboost import Pool
    from xgboost import DMatrix
    pool = Pool(X, cat_features=E["ci"])
    sv_cb = np.mean([m.get_feature_importance(pool, type="ShapValues") for m in E["cbs"]], axis=0)
    gx = E["xgb"]
    sv_x = gx.get_booster().predict(DMatrix(encode(X, E["cats"])), pred_contribs=True,
                                    iteration_range=(0, gx.best_iteration + 1))
    sv = 0.5 * sv_cb + 0.5 * sv_x
    assert np.allclose(sv.sum(1), predict_parts(E, X)[0]["ens"], atol=1e-3 * max(1.0, float(np.abs(sv.sum(1)).max())))
    return sv, sv_cb, sv_x


def save_models(E, pre, names, d, target, tier):
    d.mkdir(parents=True, exist_ok=True)
    for s, m in zip(E["seeds"], E["cbs"]):
        m.save_model(str(d / f"catboost_seed{s}.cbm"))
    E["xgb"].save_model(str(d / "xgboost.json"))
    with open(d / "preprocess.pkl", "wb") as f:
        pickle.dump(dict(preprocessor=pre, cats=E["cats"], ci=E["ci"], columns=names, selected=E["selected"],
                         tier=tier, target=target, transform=TARGETS[target]["transform"]), f)


def material_key(df):
    """Composition identity of a record: rounded atomic fractions of the frozen element list (+ residual).
    Records whose formula yielded no element form one explicit group. Used by --protocol material."""
    cols = sorted(c for c in df.columns if c.startswith("comp_frac_"))
    F = df[cols].round(2)
    key = F.astype(str).agg("|".join, axis=1)
    return np.where(F.isna().all(1) | (F.fillna(0).sum(1) == 0), "__NOCOMP__", key)


def run_name(protocol, tier):
    return protocol if tier == "t2" else f"{protocol}_{tier}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="+", default=["all"], choices=["all", *TARGETS])
    ap.add_argument("--protocol", default="doi", choices=["doi", "random", "temporal", "kfold", "material"])
    ap.add_argument("--tier", default="t2", choices=list(TIERS))
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--max-train", type=int, default=None, help="subsample N training rows (seeded, train idx only)")
    ap.add_argument("--fast", action="store_true", help="smoke test: 1 grid point, 1000 iters, 1 CatBoost seed")
    ap.add_argument("--drop-series", action="store_true", help="sensitivity: drop Warning-S records")
    ap.add_argument("--no-save-models", action="store_true")
    a = ap.parse_args()
    targets = list(TARGETS) if "all" in a.target else a.target
    run = run_name(a.protocol, a.tier) + ("_noS" if a.drop_series else "")
    for target in targets:
        out = ROOT / "results" / target / run
        for sub in ("preds", "shap"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        df = load_dataset(target, drop_series=a.drop_series, dated_only=a.protocol == "temporal")
        y, g = df["y"].values.astype(float), df["doi_group"].values
        Xraw = base_frame(df, a.tier)
        cat_cols = [c for c in Xraw.columns if not np.issubdtype(Xraw[c].dtype, np.number)]
        n_splits = 3 if a.protocol == "temporal" else 5
        if a.protocol == "material":   # held-out materials: group by parsed composition (atomic fractions, 2 d.p.)
            groups = material_key(df)
            splits = make_splits(groups, "doi", n_splits, 0.2, 0.15, 42)
        else:
            splits = make_splits(g, a.protocol, n_splits, 0.2, 0.15, 42, dates=df["pub_date"].values)
        for k, (tr, va, te) in enumerate(splits, 1):
            if k not in a.splits:
                continue
            t0 = time.time()
            n_tr_full = len(tr)
            if a.max_train and len(tr) > a.max_train:
                tr = np.sort(np.random.default_rng(42 + k).choice(tr, a.max_train, replace=False))
            pre = TrainFitPreprocessor().fit(Xraw.iloc[tr])
            Xtr, Xva, Xte = (pre.transform(Xraw.iloc[i]) for i in (tr, va, te))
            E = fit_ensemble(Xtr, y[tr], Xva, y[va], pre.cat_, fast=a.fast)
            P, unc = predict_parts(E, Xte)
            seen_set = set(g[tr]) | set(g[va])
            seen = np.array([d in seen_set for d in g[te]])
            rec = dict(target=target, tier=a.tier, protocol=a.protocol, run=run, split=k, fast=a.fast,
                       max_train=a.max_train, drop_series=a.drop_series, n_train=len(tr), n_train_full=n_tr_full,
                       n_val=len(va), n_test=len(te), n_features=Xtr.shape[1], categoricals=pre.cat_,
                       selected=E["selected"], grid=E["grid"],
                       cb_best_iterations=[int(m.get_best_iteration()) for m in E["cbs"]],
                       xgb_best_iteration=int(E["xgb"].best_iteration),
                       doi_overlap=len(set(g[te]) & seen_set), test_seen_frac=float(seen.mean()),
                       **{f"{n}_{mk}": v for n, p in P.items() for mk, v in metrics(y[te], p).items()})
            np.savez_compressed(out / "preds" / f"{k}.npz", test_idx=te, y=y[te], doi=g[te], unc=unc, seen=seen, **P)
            if a.shap:
                sv, sv_cb, sv_x = ensemble_shap(E, Xte)
                np.savez_compressed(out / "shap" / f"{k}.npz", names=np.array(list(Xtr.columns)), test_idx=te,
                                    shap_ens=sv[:, :-1].astype(np.float32), shap_cb=sv_cb[:, :-1].astype(np.float32),
                                    shap_xgb=sv_x[:, :-1].astype(np.float32), base_ens=sv[:, -1])
            if not a.no_save_models:
                save_models(E, pre, list(Xtr.columns), ROOT / "models" / target / run / str(k), target, a.tier)
            rec["runtime_s"] = round(time.time() - t0, 1)
            with open(out / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{target} {run} split {k}: R2={rec['ens_r2']:.4f} MAE={rec['ens_mae']:.3f} "
                  f"RMSE={rec['ens_rmse']:.3f} n_train={len(tr)} n_test={len(te)} doi_overlap={rec['doi_overlap']} "
                  f"({rec['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

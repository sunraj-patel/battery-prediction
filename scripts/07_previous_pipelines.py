"""Adapted pipeline of a previous study on the same database, evaluated on our common partitions.

Jablonka (2026), "Clever Materials: When Models Identify Good Materials for the Wrong Reasons", arXiv:2602.17730,
code: github.com/lamalab-org/clever-materials-hans (MIT). Their battery task, re-implemented from their scripts
(preprocess-batteries.py, descriptor_generation.py, utils.py):
  * composition = Extracted_name components summed with int(float(stoichiometry)) (their clean_composition, kept
    verbatim, including the truncation of fractional stoichiometries);
  * features = matminer ElementProperty('magpie') + Stoichiometry + Meredig on that composition, prefixed feat_;
  * preprocessing = +-inf -> +-1e3, clip to [-1e3, 1e3], NaN left for LightGBM;
  * model = LGBMRegressor with default hyperparameters (random_state=42);
  * "Clever Hans" test = classifiers/regressors predict bibliographic metadata from the same features, and a second
    LightGBM predicts the target from the *predicted* metadata only.
Their own evaluation: battery_merged.csv, Capacity, Warning empty, random 10-fold CV.

Adaptation: their featurisation and model on OUR cleaned records and OUR split-1
partitions (identical test indices to 02_train.py: make_splits(groups, protocol, 5, 0.2, 0.15, 42)); train on
train+val (they use no validation set). Metadata available in the database itself is used (journal: top-10
one-hot, as in their add_journal_features; publication year); author lists are not in battery.csv and are not used.

  python scripts/07_previous_pipelines.py              # all targets, random + doi split 1
Outputs: results/previous_pipelines.csv, results/clever_hans.csv, data/cache/matminer_jablonka.parquet
"""
import ast
import importlib
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from bmd.data import load_dataset  # noqa: E402
from bmd.evaluate import metrics  # noqa: E402
from bmd.splits import make_splits  # noqa: E402

RAW = ROOT / "data" / "raw" / "extracted" / "battery.csv"
CACHE = ROOT / "data" / "cache" / "matminer_jablonka.parquet"
TGT = ["capacity", "voltage", "conductivity", "coulombic_efficiency", "energy"]
SEED = 42


def clean_composition(extracted_name):
    """Verbatim from clever-materials-hans preprocess-batteries.py."""
    try:
        extracted_name = ast.literal_eval(extracted_name)
        composition = defaultdict(int)
        for d in extracted_name:
            for k, v in d.items():
                composition[k] += int(float(v))
        formula = ""
        for k, v in composition.items():
            formula += f"{k}{v}"
        return formula
    except Exception:
        return None


def featurize(formulas):
    """matminer magpie + stoichiometry + meredig, cached per unique formula string."""
    from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry
    from pymatgen.core import Composition
    old = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
    todo = sorted(set(f for f in formulas if f) - set(old.index))
    if todo:
        rows, idx = [], []
        feats = [("comp", ElementProperty.from_preset("magpie")), ("stoich", Stoichiometry()), ("meredig", Meredig())]
        comps = []
        for f in todo:
            try:
                comps.append(Composition(f)); idx.append(f)
            except Exception:
                pass
        blocks = []
        for pre, fz in feats:
            fz.set_n_jobs(4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals = fz.featurize_many(comps, ignore_errors=True, pbar=False)
            names = [f"feat_{pre}_{n}" for n in fz.feature_labels()]
            vals = [v if v is not None and len(v) == len(names) else [np.nan] * len(names) for v in vals]
            blocks.append(pd.DataFrame(vals, index=idx, columns=names).apply(pd.to_numeric, errors="coerce"))
        new = pd.concat(blocks, axis=1)
        old = pd.concat([old, new]) if len(old) else new
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(CACHE)
    return old


def prep(a):
    """Their preprocess_features_for_boosting(model_type='lgb')."""
    a = np.asarray(a, dtype=np.float64)
    a = np.where(np.isposinf(a), 1e3, a); a = np.where(np.isneginf(a), -1e3, a)
    return np.where(~np.isnan(a), np.clip(a, -1e3, 1e3), a)


def lgb(task):
    import lightgbm as L
    kw = dict(random_state=SEED, n_jobs=2, verbosity=-1)
    return L.LGBMRegressor(**kw) if task == "regression" else L.LGBMClassifier(**kw)


def aligned_raw():
    """Raw columns aligned row-by-row with load_dataset(target) via the cleaning step of 01_prepare_data.py."""
    prep01 = importlib.import_module("01_prepare_data")
    raw = pd.read_csv(RAW, low_memory=False)
    kept, _ = prep01.clean(raw)
    return kept


PRED = ROOT / "results" / "previous_preds"
OUT, CH = ROOT / "results" / "previous_pipelines.csv", ROOT / "results" / "clever_hans.csv"


def _append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    kept = aligned_raw()
    done = set(map(tuple, pd.read_csv(CH)[["target", "protocol"]].values)) if CH.exists() else set()
    for t in TGT:
        if all((t, p) in done for p in ("random", "doi")):
            continue
        D = load_dataset(t)
        R = kept[kept["target"] == t].reset_index(drop=True) if "target" in kept else None
        assert R is not None and len(R) == len(D), "alignment with prepared parquet failed"
        assert (R["Value"].values == D["Value"].values).all() and (R["Name"].fillna("").values == D["Name"].fillna("").values).all()
        form = R["Extracted_name"].map(clean_composition)
        F = featurize(form.dropna().unique())
        X = prep(F.reindex(form.values).values)
        names = list(F.columns)
        y, g = D["y"].values, D["doi_group"].values
        year = pd.to_datetime(R["Date"].astype(str).str.replace("/", "-"), errors="coerce").dt.year.values
        jr = R["Journal"].fillna("").astype(str).str.strip().str.lower()
        top = [j for j in jr.value_counts().index if j][:10]
        J = np.column_stack([(jr == j).astype(int).values for j in top])
        # their run_single_analysis drops every record with ANY missing feature (dropna on feat_ columns), in train
        # and test alike; we do the same and score both models on the identical covered test subset
        valid = ~np.isnan(X).any(1)
        for proto in ("random", "doi"):
            if (t, proto) in done:
                continue
            out, ch = [], []
            tr, va, te = next(make_splits(g, proto, 5, 0.2, 0.15, SEED))
            trn = np.r_[tr, va]
            trn_v, te_v = trn[valid[trn]], te[valid[te]]
            m = lgb("regression").fit(X[trn_v], y[trn_v])
            p = m.predict(X[te_v])
            row = dict(target=t, protocol=proto, split=1, pipeline="Jablonka (2026) adapted: magpie+stoich+meredig, LightGBM",
                       n_features=len(names), n_train=len(trn_v), n_test=len(te_v), test_coverage=len(te_v) / len(te),
                       **metrics(y[te_v], p))
            out.append(row)
            PRED.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PRED / f"{t}_{proto}_1.npz", test_idx=te_v, y=y[te_v], pred=p)
            # Clever-Hans test: features -> predicted journal + year -> target
            ok = trn_v[~np.isnan(year[trn_v])]
            mj = __import__("sklearn.multioutput", fromlist=["MultiOutputClassifier"]).MultiOutputClassifier(lgb("cls"))
            mj.fit(X[ok], J[ok])
            my = lgb("regression").fit(X[ok], year[ok])
            Z_tr = np.column_stack([mj.predict(X[ok]), my.predict(X[ok])])
            Z_te = np.column_stack([mj.predict(X[te_v]), my.predict(X[te_v])])
            ind = lgb("regression").fit(prep(Z_tr), y[ok]).predict(prep(Z_te))
            yt = year[te_v]; mk = ~np.isnan(yt)
            from sklearn.metrics import f1_score
            ch.append(dict(target=t, protocol=proto, n_test=len(te_v),
                           year_mae=float(np.abs(my.predict(X[te_v])[mk] - yt[mk]).mean()),
                           year_mae_baseline=float(np.abs(year[ok].mean() - yt[mk]).mean()),
                           journal_f1_micro=float(f1_score(J[te_v], mj.predict(X[te_v]), average="micro", zero_division=0)),
                           indirect_r2=metrics(y[te_v], ind)["r2"], direct_r2=row["r2"]))
            _append(OUT, out); _append(CH, ch)   # saved per (target, protocol): the run is resumable
            print(f"{t:22s} {proto:6s} Jablonka-adapted R2={row['r2']:.3f}  indirect(meta-only) R2={ch[-1]['indirect_r2']:.3f}",
                  flush=True)
    print(pd.read_csv(OUT)[["target", "protocol", "pipeline", "r2", "mae", "rmse"]].round(3).to_string(index=False))
    print(pd.read_csv(CH).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

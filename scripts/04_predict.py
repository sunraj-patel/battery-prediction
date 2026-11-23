"""Predict a battery property for new records with ensembles saved by 02_train.py.

Input rows follow the raw battery.csv schema (Name, Extracted_name, Specifier, Info, Type, Warning, Title, ...);
features are rebuilt with the same row-level functions as 01_prepare_data.py and the frozen data/element_list.json.
Missing raw columns become missing values. With several --model directories (e.g. the five DOI-grouped split models)
the output is their mean and standard deviation, in the target's original unit (conductivity back-transformed from
log10; the log-scale prediction is also written).

  python scripts/04_predict.py --model models/voltage/doi/1 --input new.csv --out preds.csv
  python scripts/04_predict.py --model models/voltage/doi/{1,2,3,4,5} --input new.csv --out preds.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from bmd.data import ELEMENTS, inverse_y  # noqa: E402
from bmd.features import base_frame  # noqa: E402
from bmd.featurize import build_features  # noqa: E402


def load(d):
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    d = Path(d)
    if not (d / "preprocess.pkl").exists():
        sys.exit(f"Model not found: {d} (run 02_train.py first)")
    with open(d / "preprocess.pkl", "rb") as f:
        meta = pickle.load(f)
    cbs = []
    for p in sorted(d.glob("catboost_seed*.cbm")):
        m = CatBoostRegressor()
        m.load_model(str(p))
        cbs.append(m)
    gx = XGBRegressor()
    gx.load_model(str(d / "xgboost.json"))
    return meta, cbs, gx


def predict_one(meta, cbs, gx, F):
    from importlib import import_module
    train = import_module("02_train")
    X = base_frame(F, meta["tier"])
    for c in meta["preprocessor"].keep_:
        if c not in X.columns:
            X[c] = np.nan
    Z = meta["preprocessor"].transform(X)[meta["columns"]]
    p_bag = np.mean([m.predict(Z) for m in cbs], axis=0)
    p_x = gx.predict(train.encode(Z, meta["cats"]))
    return (p_bag + p_x) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.input, low_memory=False)
    F = build_features(df, json.loads(ELEMENTS.read_text())["elements"])
    F["id_name"] = F["id_name"].astype(str)
    P, targets = [], set()
    for d in a.model:
        meta, cbs, gx = load(d)
        targets.add(meta["target"])
        P.append(predict_one(meta, cbs, gx, F))
    if len(targets) > 1:
        sys.exit(f"--model directories mix targets {sorted(targets)}")
    t = targets.pop()
    P = np.stack(P)
    out = df[[c for c in ("Name", "DOI", "Property") if c in df]].copy()
    if meta["transform"]:
        out[f"{t}_pred_{meta['transform']}"] = P.mean(0)
        if len(P) > 1:
            out[f"{t}_pred_{meta['transform']}_sd"] = P.std(0, ddof=1)
    out[f"{t}_pred"] = inverse_y(P.mean(0), t)
    if len(P) > 1 and not meta["transform"]:
        out[f"{t}_pred_sd"] = P.std(0, ddof=1)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out):,} {t} predictions -> {a.out}")


if __name__ == "__main__":
    main()

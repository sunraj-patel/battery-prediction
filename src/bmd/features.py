"""Tier selection and train-fitted feature preprocessing (no statistic is learned from val/test)."""
import numpy as np
import pandas as pd

from .data import BLOCKS, CAT_ALL, CAT_ID, FORBIDDEN, NOISE_COLS, SPARSE_KEEP, TIERS

MAX_MISSING = 0.8


def base_frame(df, tier):
    """Raw (unimputed) predictor frame for a tier; row-wise, learns nothing.

    Only engineered, prefixed columns are selectable (raw text / identity / target columns can never enter by
    accident). t2_id adds the normalised material Name as a categorical (identity-admission control); t2_noise adds
    NOISE_COLS N(0,1) columns with a fixed seed (negative control)."""
    prefixes = tuple(BLOCKS[b] for b in TIERS[tier])
    cols = [c for c in df.columns if c.startswith(prefixes)]
    X = df[cols].copy()
    if tier.endswith("_id"):
        for c in CAT_ID:
            X[c] = df[c]
    if tier.endswith("_noise"):
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(rng.normal(size=(len(X), NOISE_COLS)), index=X.index,
                             columns=[f"noise_{i}" for i in range(NOISE_COLS)])
        X = pd.concat([X, noise], axis=1)
    assert not set(X.columns) & set(FORBIDDEN), "forbidden column in predictor frame"
    return X


class TrainFitPreprocessor:
    """Drop >80%-missing columns (except predeclared sparse measurement conditions observed in train), median-impute,
    add `_na` flags; categoricals (any CAT_ALL column present) are kept and missing -> 'unknown'. Fit on train only."""

    def fit(self, X):
        self.cat_ = [c for c in CAT_ALL if c in X.columns]
        self.keep_ = [c for c in X.columns if c in self.cat_ or (c in SPARSE_KEEP and X[c].notna().any())
                      or X[c].isnull().mean() <= MAX_MISSING]
        num = [c for c in self.keep_ if c not in self.cat_]
        self.na_cols_ = [c for c in num if X[c].isnull().any()]
        self.median_ = {c: float(X[c].median()) for c in self.na_cols_}
        # train medians for every numeric column: guards against NaNs that appear only in val/test
        self.median_all_ = {c: float(X[c].median()) if X[c].notna().any() else 0.0 for c in num}
        return self

    def transform(self, X):
        Z = X[self.keep_].copy()
        flags = pd.DataFrame({f"{c}_na": Z[c].isnull().astype(float)
                              for c in self.na_cols_}, index=Z.index)
        for c in self.na_cols_:
            Z[c] = Z[c].fillna(self.median_[c] if not np.isnan(self.median_[c]) else 0.0)
        Z = pd.concat([Z, flags], axis=1)
        for c, m in self.median_all_.items():  # no-op unless a column has NaN only outside train
            if Z[c].isnull().any():
                Z[c] = Z[c].fillna(m)
        for c in self.cat_:
            Z[c] = Z[c].fillna("unknown").astype(str).replace("", "unknown")
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)

"""Targets, record cleaning rules, forbidden columns, feature tiers and dataset loading.

Source: Huang & Cole (2020) Sci. Data 7, 260, figshare 10.6084/m9.figshare.11888115, file `battery.csv`
(un-merged: one row per extracted record, so the per-publication grouping is intact).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "extracted" / "battery.csv"
FEATURES = ROOT / "data" / "battery_features.parquet"
ELEMENTS = ROOT / "data" / "element_list.json"
NO_DOI_GROUP = "__NO_DOI__"
ELEMENT_MIN_FRAC = 0.005        # element enters the frozen list if present in >= 0.5% of cleaned records

# key -> raw Property, unit, transform of y, physics bounds (lo, hi, lo inclusive?) ; records outside are dropped
TARGETS = {
    "capacity": dict(prop="Capacity", unit="mAh/g", transform=None, lo=0.0, hi=4500.0, lo_incl=False),
    "voltage": dict(prop="Voltage", unit="V", transform=None, lo=0.0, hi=6.0, lo_incl=False),
    "conductivity": dict(prop="Conductivity", unit="S/cm", transform="log10", lo=1e-12, hi=1e3, lo_incl=True),
    "coulombic_efficiency": dict(prop="Coulombic Efficiency", unit="%", transform=None, lo=0.0, hi=100.0,
                                 lo_incl=False),
    "energy": dict(prop="Energy", unit="Wh/kg", transform=None, lo=0.0, hi=3000.0, lo_incl=False),
}
PROP2KEY = {v["prop"]: k for k, v in TARGETS.items()}

# Raw columns never used as predictors (reason per entry). Only engineered prefixed columns enter a tier.
FORBIDDEN = {
    "DOI": "publication identity",
    "doi_group": "publication identity (split grouping key)",
    "Value": "the target",
    "y": "the (transformed) target",
    "Raw_value": "target text",
    "Raw_unit": "target-derived (unit as written next to the value)",
    "Unit": "target-derived (normalised unit of the value)",
    "Title": "publication identity; admitted only as the t3 keyword block (lit_*)",
    "Journal": "publication identity / reporting venue",
    "Date": "publication identity / reporting time (used only for the temporal split)",
    "pub_date": "publication date (temporal split only)",
    "Correctness": "post-hoc manual evaluation label",
    "Num_records": "merged-database count (not in battery.csv; listed for completeness)",
    "Warning": "S = member of a same-paper series (identity proxy); L = value outside the expected range, i.e. a "
               "deterministic function of the target. R (Relevance: <3 records extracted from the article, Huang & Cole 2020) is paper-level constant, so it is admitted only in the t3 literature block as lit_warn_R",
    "Tag": "Calculated rows are derived (capacity x voltage) and filtered out; constant otherwise",
    "Extracted_name": "raw text; admitted only through parsed composition (comp_*) / material (mat_*) features",
    "Name": "raw material string; admitted only as id_name in the t2_id identity control tier",
    "Specifier": "raw text; admitted only as keyword flags (meas_spec_*)",
    "Info": "raw text; admitted only as parsed current density / cycle number (meas_*)",
    "Property": "selects the target (one model per property)",
    "target": "selects the target",
}

# feature blocks by column prefix (see featurize.py); tiers are unions of blocks
BLOCKS = {"composition": "comp_", "material": "mat_", "material_text": "txt_", "measurement": "meas_",
          "literature": "lit_"}
TIERS = {"comp": ("composition",),
         "t1": ("composition", "material"),
         "t2": ("composition", "material", "material_text", "measurement"),          # main
         "t3": ("composition", "material", "material_text", "measurement", "literature"),
         "t2_id": ("composition", "material", "material_text", "measurement"),       # + id_name categorical
         "t2_noise": ("composition", "material", "material_text", "measurement")}    # + NOISE_COLS N(0,1)
NOISE_COLS = 32
CAT_BASE = ["meas_type"]           # electrode role (Cathode/Anode/Electrolyte/unknown), every t2-based tier
CAT_ID = ["id_name"]               # material identity string: t2_id only (identity-admission control)
CAT_ALL = CAT_BASE + CAT_ID
# predeclared sparse measurement conditions: exempt from the >80%-missing filter (train-median + _na flag instead)
SPARSE_KEEP = ["meas_log_current_Ag", "meas_log_crate", "meas_log_cycle"]


def normalize_doi(x):
    """Canonical DOI key: strip BOM/space, lowercase, drop doi.org / doi: prefixes.

    Returns NO_DOI_GROUP for missing values. Non-DOI URLs are kept verbatim
    (lower-cased) as their own key.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DOI_GROUP
    s = str(x).replace("﻿", "").strip().lower()
    if not s or s == "nan":
        return NO_DOI_GROUP
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s


def transform_y(v, target):
    v = np.asarray(v, dtype=float)
    return np.log10(v) if TARGETS[target]["transform"] == "log10" else v


def inverse_y(y, target):
    y = np.asarray(y, dtype=float)
    return 10 ** y if TARGETS[target]["transform"] == "log10" else y


def in_bounds(v, target):
    t = TARGETS[target]
    lo = (v >= t["lo"]) if t["lo_incl"] else (v > t["lo"])
    return lo & (v <= t["hi"])


def load_dataset(target, path=FEATURES, drop_series=False, dated_only=False):
    """Records of one target from the prepared parquet (01_prepare_data.py).

    `y` is already transformed (log10 for conductivity). drop_series: sensitivity analysis without Warning-S records
    (same-paper series members). dated_only: keep records whose publication has a parsed date (temporal protocol).
    """
    if not Path(path).exists():
        raise SystemExit(f"Missing {path}: run scripts/01_prepare_data.py first")
    df = pd.read_parquet(path, filters=[("target", "==", target)])
    n0 = len(df)
    if drop_series:
        df = df[~df["Warning"].fillna("").str.contains("S")]
    if dated_only:
        df = df[df["pub_date"].notna()]
    df = df.reset_index(drop=True)
    df.attrs["n_prepared"] = n0
    return df

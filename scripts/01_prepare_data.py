"""Clean battery.csv, freeze the element list, build all row-level features once, print the modelling population.

  python scripts/01_prepare_data.py

Cleaning (per target, in this order; counts reported): drop missing Value; energy: drop Tag == "Calculated" (derived
from capacity x voltage by the database authors, not extracted); drop values outside the physics bounds in
bmd.data.TARGETS; collapse exact duplicates of (doi_group, Property, Name, Value, Specifier). Records without a DOI
form one explicit __NO_DOI__ group. Publication date = earliest parsed Date of the DOI (temporal protocol only).

Outputs: data/element_list.json (elements present in >= 0.5% of cleaned records, frozen for 02/04),
data/battery_features.parquet (one row per record: target, y [transformed], Value, doi_group, pub_date, Name,
Warning, all feature columns), results/population.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bmd.data import (ELEMENT_MIN_FRAC, ELEMENTS, FEATURES, NO_DOI_GROUP, PROP2KEY, RAW, TARGETS,  # noqa: E402
                      in_bounds, normalize_doi, transform_y)
from bmd.featurize import build_features, element_counts, parse_extracted  # noqa: E402


def parse_dates(s):
    return pd.to_datetime(s.astype(str).str.strip().str.replace("/", "-"), format="%Y-%m-%d", errors="coerce")


def clean(raw):
    """Apply the cleaning rules; return the kept records and a per-target count table."""
    raw = raw.copy()
    raw["target"] = raw["Property"].map(PROP2KEY)
    raw["doi_group"] = raw["DOI"].map(normalize_doi)
    d = parse_dates(raw["Date"])
    raw["pub_date"] = d.groupby(raw["doi_group"]).transform("min")
    nod = raw["doi_group"] == NO_DOI_GROUP                       # undated/no-DOI records are dated individually
    raw.loc[nod, "pub_date"] = d[nod]
    rows, keep = [], []
    for t, spec in TARGETS.items():
        x = raw[raw["target"] == t]
        r = dict(target=t, property=spec["prop"], unit=spec["unit"], raw_records=len(x),
                 raw_publications=x.loc[x.doi_group != NO_DOI_GROUP, "doi_group"].nunique())
        x = x[x["Value"].notna()]; r["removed_missing_value"] = r["raw_records"] - len(x)
        n = len(x)
        if t == "energy":
            x = x[x["Tag"] != "Calculated"]
        r["removed_calculated"] = n - len(x); n = len(x)
        x = x[in_bounds(x["Value"], t)]; r["removed_bounds"] = n - len(x); n = len(x)
        x = x[~x.assign(_s=x["Specifier"].fillna("")).duplicated(["doi_group", "Property", "Name", "Value", "_s"])]
        r["removed_duplicates"] = n - len(x)
        pub = x.loc[x.doi_group != NO_DOI_GROUP, "doi_group"]
        r.update(records=len(x), publications=pub.nunique(), no_doi=int((x.doi_group == NO_DOI_GROUP).sum()),
                 dated=int(x["pub_date"].notna().sum()), type_known=int(x["Type"].notna().sum()),
                 warning_S=int(x["Warning"].fillna("").str.contains("S").sum()),
                 records_per_pub_median=float(pub.value_counts().median()) if len(pub) else np.nan)
        rows.append(r); keep.append(x)
    return pd.concat(keep).sort_index(), pd.DataFrame(rows)


def main():
    if not RAW.exists():
        sys.exit(f"Missing: {RAW}\nSee data/README.md")
    t0 = time.time()
    raw = pd.read_csv(RAW, low_memory=False)
    print(f"battery.csv: {len(raw):,} records; {raw['DOI'].map(normalize_doi).nunique():,} DOI groups")
    df, P = clean(raw)

    # composition parse rate (per unique string, broadcast)
    parsed = df["Extracted_name"].map({s: parse_extracted(s) is not None for s in df["Extracted_name"].unique()})
    print(f"Extracted_name parsed to >=1 known element: {parsed.mean():.2%} of cleaned records")
    P["comp_parsed"] = [int(parsed[df.target == t].sum()) for t in P.target]

    cnt = element_counts(df["Extracted_name"].values)
    frac = {e: n / len(df) for e, n in sorted(cnt.items(), key=lambda kv: -kv[1])}
    elements = [e for e, f in frac.items() if f >= ELEMENT_MIN_FRAC]
    ELEMENTS.write_text(json.dumps(dict(min_record_frac=ELEMENT_MIN_FRAC, n_records=len(df), elements=elements,
                                        record_frac={e: round(f, 5) for e, f in frac.items()}), indent=1))
    print(f"Frozen element list ({len(elements)} elements, >= {ELEMENT_MIN_FRAC:.1%} of records) -> {ELEMENTS}")

    X = build_features(df, elements)
    y = np.empty(len(df))
    for t in TARGETS:
        m = (df["target"] == t).values
        y[m] = transform_y(df.loc[m, "Value"].values, t)
    meta = df[["target", "Value", "doi_group", "pub_date", "Name", "Warning"]].assign(y=y)
    out = pd.concat([meta, X], axis=1).reset_index(drop=True)
    out["Name"] = out["Name"].astype(str)
    out.to_parquet(FEATURES, index=False)
    nfeat = {b: sum(c.startswith(b) for c in X.columns) for b in ("comp_", "mat_", "txt_", "meas_", "lit_")}
    print(f"Features {X.shape[1]} {nfeat} -> {FEATURES} ({len(out):,} rows)")

    (ROOT / "results").mkdir(exist_ok=True)
    P.to_csv(ROOT / "results" / "population.csv", index=False)
    cols = ["target", "raw_records", "removed_calculated", "removed_bounds", "removed_duplicates", "records",
            "publications", "no_doi", "dated", "type_known", "warning_S", "comp_parsed"]
    print("\nModelling population (Table 1)")
    print(P[cols].to_string(index=False))
    print(f"Total records {P.records.sum():,}; publications (any target) "
          f"{out.loc[out.doi_group != NO_DOI_GROUP, 'doi_group'].nunique():,}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()

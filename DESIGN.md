# Battery port: design spec (shared by code + paper work)

Source: Huang & Cole (2020), *Sci. Data* 7, 260, "A database of battery materials auto-generated
using ChemDataExtractor", figshare 10.6084/m9.figshare.11888115 ("Battery Data.zip").
Raw files: `data/raw/extracted/{battery,battery_merged}.{csv,json,db}`. We use **`battery.csv`**
(un-merged, one row per extracted record, 292,313 rows, 42,873 DOIs). `battery_merged.csv` collapses
identical (name, property, value) records across papers (`Num_records`), so it destroys the
per-publication grouping and is not used for modelling.

## Raw schema (long format: one property value per row)
`Property` (Capacity 144,359 / Voltage 114,240 / Energy 15,543 / Coulombic Efficiency 11,003 /
Conductivity 7,168), `Name` (material string as written, e.g. "Fe2O3 / C", "TiO2 electrodes"),
`Value` (float, normalised unit), `Raw_value` (text, may hold a series "521.4 and 448.0"),
`Raw_unit`, `Unit` (normalised; Capacity has a small Ah/kg minority = mAh/g numerically),
`Extracted_name` (list of dicts of element->stoichiometry, e.g. `[{'Zn':'2.0','Sn':'1.0','O':'4.0'}]`),
`DOI`, `Specifier` (e.g. "discharge capacities", "theoretical capacity"), `Tag` (CDE / Calculated:
6,718 Energy rows computed by the authors from capacity x voltage), `Warning` (S = one of a series
from the same paper, L = value outside expected range, R = ratio/relative, combos LS/RL),
`Type` (Cathode/Anode/Electrolyte; 97% missing), `Info` (dict text, e.g. current density
`{'current_value': '0.1 , 0.2', 'current_units': 'Ag−1'}`), `Title`, `Journal`, `Date`
(mixed `YYYY/MM/DD`, `YYYY-MM-DD`), `Correctness` (manual evaluation sample, ~500 rows).

## Study framing
Publication-aware prediction of battery material properties from literature-extracted records.
Headline comparison: record-level random split vs DOI-grouped split (plus temporal by publication
date), same model and features, 5 repeated holdouts (test 0.2, val 0.15, seed 42).

## Targets (one model per property; `TARGETS` dict in `src/bmd/data.py`)
| key | Property | transform | physics bounds (records outside are dropped, counted) |
|---|---|---|---|
| capacity | Capacity (mAh/g) | none | 0 < v <= 4500 (Si ~4200, Li metal ~3860) |
| voltage | Voltage (V) | none | 0 < v <= 6.0 (electrochemical window) |
| conductivity | Conductivity (S/cm) | log10 | 1e-12 <= v <= 1e3 |
| coulombic_efficiency | Coulombic Efficiency (%) | none | 0 < v <= 100 |
| energy | Energy (Wh/kg) | none | 0 < v <= 3000 practical-scale; Tag=="Calculated" rows excluded (derived from capacity x voltage) |

Bounds are documented filters; report counts removed.

## Record cleaning
- Drop rows with missing Value/DOI-less rows go into one explicit `__NO_DOI__` group (as before).
- Normalise DOI with the existing `normalize_doi`.
- Exact duplicates of (doi_group, Property, Name, Value, Specifier) collapsed to one row.
- Rows whose `Extracted_name` cannot be parsed to >=1 element keep NaN composition features (no drop).

## Feature tiers (`base_frame(df, tier)`)
Groups (for SHAP, `src/bmd/groups.py`):
- **composition**: element fractions for a fixed predeclared element list (all elements occurring
  in >= 0.5% of records, computed from the full corpus at prep time and frozen in a JSON), number of
  elements, n_components (len of Extracted_name list), carrier-ion flags (Li/Na/K/Mg/Zn/Al present).
- **material**: Magpie-style stoichiometry-weighted mean/min/max/range of elemental properties via
  pymatgen `Element` (Z, atomic mass, Pauling electronegativity, atomic radius, row, group,
  Mendeleev number, ionization energy, valence/oxidation-state proxy).
- **material_text**: regex flags from `Name` that a materials scientist would read as morphology /
  architecture: carbon/graphene/CNT/rGO composite, "@" core-shell, "/" composite, doped, nano*,
  hollow/porous, film, "anode"/"cathode"/"electrode" words.
- **measurement**: `Specifier` keyword flags (theoretical, initial/first, reversible, discharge,
  charge, specific, areal, volumetric, average, open-circuit, plateau, cut-off, ionic, electronic),
  current density parsed from `Info` (normalised to A/g where possible, log10) + `_na`,
  `Type` (categorical Cathode/Anode/Electrolyte/unknown), Warning-L/R flags.
  `Warning` S is NOT a predictor (it is literally "part of a same-paper series" = identity proxy);
  it is retained for analysis (sensitivity: drop S rows).
- **literature (paper-level)**: keyword flags from `Title` only (battery chemistry:
  lithium-ion, sodium-ion, potassium, Li-S, Li-O2/air, zinc, magnesium, supercapacitor, solid-state,
  electrolyte, anode, cathode). Constant within a publication, so it is tested under both random and DOI splits.
- **noise**: 32 N(0,1) columns (negative control).

Tiers: `comp` (composition only), `t1` (composition + material), `t2` = **main**
(t1 + material_text + measurement), `t3` (t2 + literature title flags), `t2_id` (t2 + `Name`
lower-cased/whitespace-normalised string as a CatBoost categorical: identity-admission control),
`t2_noise` (t2 + noise).

FORBIDDEN (never predictors, with reason): DOI/doi_group (identity), Value/Raw_value (target),
Raw_unit/Unit (target-derived), Title/Journal/Date (identity / reporting; Title only via the
`t3` keyword block), Correctness (post-hoc label), Num_records, Warning S, Tag (Calculated is a
derived record; excluded by filter), Extracted_name raw text, Name raw text (except `t2_id`).

## Splits
Reuse `make_splits` unchanged: `random`, `doi`, `kfold` (grouped), `temporal` (publication date =
earliest parsed Date per DOI; records with no date excluded from temporal only).

## Model
CatBoost 6-point grid x 3 seeds bagged + XGBoost, 50/50 average.
`encode()` and `ci` must use the full categorical list present in the frame (fixes the known bug).
Add `--max-train N` subsample option (for speed on capacity, ~140k rows) and `--fast` (CatBoost grid
reduced to 1 point, iterations 1000) for smoke tests; full settings are default.

## Outputs
`results/<target>/<protocol>/{runs.jsonl,preds,shap}`; `models/<target>/<protocol>/<k>/`;
`results/summary.csv` with a `target` column; figures per target. Same schema as
`reference_results/` plus `target`.

## Uncertainty
Report mean +/- s.d. across 5 holdouts (as before) and the across-model spread (3 CatBoost seeds +
XGBoost) per record as a cheap epistemic proxy; evaluate its rank correlation with |error| and
whether it rises for DOI-unseen test records.

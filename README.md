# Publication-aware prediction of battery material properties

Leakage-aware ML pipeline and manuscript (`paper/`) for the Huang & Cole (2020) *Battery Materials Property
Database* (*Sci. Data* 7, 260).
One model per property (capacity, voltage, conductivity, Coulombic efficiency, energy) is evaluated under a
record-level random split, a DOI-grouped split and a temporal split, with the same features and estimator, to measure
how much record-level evaluation overstates generalisation to unseen publications. The binding spec is `DESIGN.md`.

## Setup

```bash
pip install -r requirements.txt
```
Download the data as described in `data/README.md` (figshare "Battery Data.zip" -> `data/raw/extracted/battery.csv`).

## Pipeline

```bash
python scripts/01_prepare_data.py                         # clean, freeze element list, build all features (~10 s)
python scripts/02_train.py --target all --protocol doi --shap
python scripts/02_train.py --target all --protocol random --shap
python scripts/02_train.py --target all --protocol temporal --shap
python scripts/03_evaluate.py --figures                   # summary.csv, SHAP groups, uncertainty, figures
python scripts/04_predict.py --model models/voltage/doi/{1,2,3,4,5} --input new.csv --out preds.csv
```

Useful options of `02_train.py`: `--tier {comp,t1,t2,t3,t2_id,t2_noise}` (default `t2`), `--protocol kfold`,
`--splits 1` (one split), `--max-train N` (seeded subsample of the training indices, e.g. for capacity),
`--fast` (smoke test: 1 grid point, 1000 iterations, 1 CatBoost seed; not for results), `--drop-series`
(sensitivity analysis without Warning-S records), `--no-save-models`.

| Path | Content |
|---|---|
| `src/bmd/data.py` | `TARGETS` (property, unit, transform, physics bounds), `FORBIDDEN` (with reasons), tiers, `load_dataset` |
| `src/bmd/featurize.py` | row-level features from the raw schema (shared by 01 and 04) |
| `src/bmd/features.py` | `base_frame(df, tier)`, `TrainFitPreprocessor` (train-only imputation / missingness filter) |
| `src/bmd/groups.py` | SHAP feature groups (by column prefix) |
| `src/bmd/splits.py` | `make_splits` (random / doi / kfold / temporal; `material` via `02_train.py`) |
| `results/<target>/<protocol>[_<tier>]/` | `runs.jsonl`, `preds/<k>.npz` (incl. per-record uncertainty), `shap/<k>.npz` |
| `models/<target>/<protocol>[_<tier>]/<k>/` | CatBoost `.cbm` x3, XGBoost `.json`, `preprocess.pkl` |
| `results/population.csv` | Table-1 style population counts per target |

## Feature blocks (column prefix = SHAP group)

`comp_` composition (element fractions over a frozen list, counts, carrier-ion flags) · `mat_` stoichiometry-weighted
elemental statistics (pymatgen) · `txt_` morphology words in `Name` · `meas_` measurement conditions (Specifier
keywords, current density / C-rate / cycle from `Info`, electrode `Type`, Warning-R) · `lit_` Title keywords (`t3`
only; constant within a publication) · `id_name` normalised material name (`t2_id` only) · `noise_` (`t2_noise`).
Main tier `t2` = comp + mat + txt + meas.

## Analysis, comparison and manuscript

```
python scripts/05_analysis.py          # every manuscript number -> results/paper_values.json, CSV tables, paper/figures/
python scripts/07_previous_pipelines.py # Jablonka (2026) pipeline adapted to our split-1 partitions + bibliographic-fingerprint test
python scripts/06_fill_paper.py        # paper/{ml,supplementary}_template.tex -> paper/{ml,supplementary}.tex
```

Edit prose only in the `*_template.tex` files; `ml.tex` and `supplementary.tex` are generated. Remaining red `\TBD`
markers are listed by `06_fill_paper.py`.

Extra protocols: `--protocol material` (composition-grouped: no parsed composition in two partitions) and
`--protocol kfold` (disjoint grouped 5-fold check). `cluster/check_splits.py` prints an md5 of every split; run it on
each machine and diff the output before pairing results from different machines.

## Running on Param Sanganak

`cluster/node.sbatch` runs one shard file (`cluster/shard{1..4}.txt`) on a whole node, 4 tasks at a time with
`BMD_THREADS=12`. The account QoS allows 4 running / 8 submitted jobs; `--time` above ~6 h is rejected.

```
sbatch -p hm --export=ALL,SHARD=cluster/shard1.txt cluster/node.sbatch   # from /scratch/sunrajp23/battery
bash cluster/run_shard_local.sh cluster/shard4.txt                      # same shard format, locally
```

Use the `sanganak` host alias (key auth + ControlMaster in `~/.ssh/config`) and do not poll the cluster in tight
loops: bursts of new SSH connections get the client IP blocked by the institute firewall.

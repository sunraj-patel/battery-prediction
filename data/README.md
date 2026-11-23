# Data

Source: Huang, S. & Cole, J. M. (2020) "A database of battery materials auto-generated using ChemDataExtractor",
*Sci. Data* 7, 260; figshare 10.6084/m9.figshare.11888115 (CC BY 4.0).

## Download

```bash
# metadata (file list, checksums): https://api.figshare.com/v2/articles/11888115
mkdir -p data/raw/extracted
curl -L -o data/raw/battery_data.zip https://ndownloader.figshare.com/files/22789760   # "Battery Data.zip"
unzip -j data/raw/battery_data.zip -d data/raw/extracted
```

| File | Used | Description |
|------|------|-------------|
| `raw/extracted/battery.csv` | yes (01) | 292,313 extracted records, one property value per row (long format), 42,873 DOIs |
| `raw/extracted/battery_merged.csv` | no | identical (name, property, value) records merged across papers; destroys the per-publication grouping |
| `raw/extracted/*.json`, `*.db` | no | same content in other formats |
| `element_list.json` | 02, 04 | frozen element list (elements in >= 0.5% of cleaned records), written by 01 |
| `battery_features.parquet` | 02 | one row per cleaned record: target, y (transformed), doi_group, pub_date, Name, Warning, features |

Modelling population (01_prepare_data.py, also `results/population.csv`):

| target | records | publications | removed by bounds | removed (Calculated) |
|---|---:|---:|---:|---:|
| capacity (mAh/g) | 144,322 | 22,833 | 37 | 0 |
| voltage (V) | 114,176 | 32,747 | 64 | 0 |
| conductivity (S/cm, log10) | 6,957 | 3,351 | 211 | 0 |
| coulombic_efficiency (%) | 11,003 | 5,861 | 0 | 0 |
| energy (Wh/kg) | 8,785 | 3,113 | 40 | 6,718 |

No exact duplicates of (DOI, Property, Name, Value, Specifier) exist in `battery.csv`; the only DOI-less records are the
6,718 `Tag == "Calculated"` energy rows, which are excluded, so every modelled record has a DOI and a publication date.

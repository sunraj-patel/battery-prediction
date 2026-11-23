#!/bin/bash
# Build the Digital Discovery submission package in submission/ :
#   manuscript.pdf, supplementary.pdf, toc.png, cover_letter.md, latex_source.zip, code_archive.zip (for Zenodo)
# Requires the run outputs in results/. Usage: bash make_submission.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -W ignore scripts/05_analysis.py > /dev/null
python3 scripts/fig_toc.py > /dev/null
python3 scripts/06_fill_paper.py
OUT=submission; BUILD=$(mktemp -d); mkdir -p $OUT
cp -r paper/ml.tex paper/supplementary.tex paper/references_battery.bib paper/figures "$BUILD"/
( cd "$BUILD" && for f in ml supplementary; do pdflatex -interaction=nonstopmode $f.tex >/dev/null; bibtex $f >/dev/null;
    pdflatex -interaction=nonstopmode $f.tex >/dev/null; pdflatex -interaction=nonstopmode $f.tex >/dev/null;
    echo "$f: $(grep -c '^!' $f.log) LaTeX errors; $(grep -c 'undefined' $f.log) undefined refs"; done )
cp "$BUILD/ml.pdf" $OUT/manuscript.pdf; cp "$BUILD/supplementary.pdf" $OUT/supplementary.pdf
cp paper/figures/toc.png paper/cover_letter.md $OUT/
( cd "$BUILD" && zip -qr latex_source.zip ml.tex supplementary.tex references_battery.bib ml.bbl supplementary.bbl figures )
cp "$BUILD/latex_source.zip" $OUT/
# code + processed data + result tables (no raw database: it is on figshare; no per-record predictions or models)
zip -qr $OUT/code_archive.zip README.md DESIGN.md requirements.txt src scripts cluster/*.py cluster/*.sh cluster/*.sbatch \
    cluster/shard*.txt cluster/splits_local.txt data/README.md data/element_list.json data/battery_features.parquet \
    data/external/zhuang2024/SOURCE.txt results/*.csv results/paper_values.json \
    -x '*/__pycache__/*' '*.pyc'
ls -la $OUT
grep -o '\\TBD[a-z]*{[^}]*}' paper/ml.tex paper/supplementary.tex | sort -u | sed 's/^/  remaining: /' || true

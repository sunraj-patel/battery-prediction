#!/bin/bash
# Build the J. Chem. Inf. Model. (ACS) submission package in submission/:
#   Patel_CoverLetter.pdf, Patel_Manuscript.pdf (achemso, figures/tables embedded, TOC graphic on last page),
#   Patel_SupportingInformation.pdf, Patel_TOC_graphic.{tif,png}, Patel_Fig1..7.{pdf,png},
#   Patel_LaTeX_source.zip, Patel_Code_and_data.zip
# All PDFs are built from .tex with tectonic. Requires the run outputs in results/. Usage: bash make_submission.sh  (SKIP_ANALYSIS=1 to reuse figures/values)
set -euo pipefail
cd "$(dirname "$0")"
[ "${SKIP_ANALYSIS:-0}" = 1 ] || { python3 -W ignore scripts/05_analysis.py > /dev/null; python3 -W ignore scripts/12_repeated_diagnostics.py > /dev/null; }
python3 -W ignore scripts/10_figures.py > /dev/null 2>&1   # data figures
python3 scripts/11_schematics.py > /dev/null               # TikZ schematics (tectonic)
python3 scripts/fig_toc.py > /dev/null
python3 scripts/06_fill_paper.py
OUT=submission; BUILD=$(mktemp -d); rm -rf $OUT; mkdir -p $OUT
cp -r paper/ml.tex paper/supplementary.tex paper/cover_letter.tex paper/references_battery.bib paper/figures paper/figures_src "$BUILD"/
# every PDF is compiled from its .tex with tectonic (handles bibtex and reruns itself)
( cd "$BUILD"
  for f in ml supplementary cover_letter; do
    tectonic --keep-intermediates --keep-logs $f.tex > $f.tectonic.out 2>&1 || { echo "$f: tectonic FAILED"; tail -20 $f.tectonic.out; exit 1; }
    echo "$f: $(grep -c '^!' $f.log) LaTeX errors; $(grep -ci 'undefined' $f.log) undefined refs/cites"; done )
cp "$BUILD/ml.pdf" $OUT/Patel_Manuscript.pdf
cp "$BUILD/supplementary.pdf" $OUT/Patel_SupportingInformation.pdf
# figures, numbered as in the manuscript (numbers read from ml.aux)
for pair in cell:battery_cell workflow:workflow dist:fig_target_distributions split:fig_split_by_target \
            diag:fig_diagnostics unc:fig_uncertainty shap:fig_shap_groups_by_target; do
  lab=${pair%%:*}; file=${pair#*:}
  n=$(sed -n "s/^\\\\newlabel{fig:$lab}{{\([0-9]*\)}.*/\1/p" "$BUILD/ml.aux")
  cp paper/figures/$file.pdf $OUT/Patel_Fig$n.pdf; cp paper/figures/$file.png $OUT/Patel_Fig$n.png
done
cp paper/figures/toc.png $OUT/Patel_TOC_graphic.png; cp paper/figures/toc.tif $OUT/Patel_TOC_graphic.tif
cp "$BUILD/cover_letter.pdf" $OUT/Patel_CoverLetter.pdf
( cd "$BUILD" && zip -qr latex_source.zip ml.tex supplementary.tex references_battery.bib ml.bbl supplementary.bbl figures figures_src )
cp "$BUILD/latex_source.zip" $OUT/Patel_LaTeX_source.zip
# code + processed data + result tables (no raw database: it is on figshare; no per-record predictions or models)
zip -qr $OUT/Patel_Code_and_data.zip README.md DESIGN.md requirements.txt src scripts cluster/*.py cluster/*.sh cluster/*.sbatch \
    cluster/shard*.txt cluster/splits_local.txt data/README.md data/element_list.json data/battery_features.parquet \
    data/external/zhuang2024/SOURCE.txt results/*.csv results/paper_values.json \
    -x '*/__pycache__/*' '*.pyc'
ls -la $OUT
grep -o '\\TBD[a-z]*{[^}]*}' paper/ml.tex paper/supplementary.tex | sort -u | sed 's/^/  remaining: /' || true

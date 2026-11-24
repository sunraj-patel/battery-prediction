"""Make the ACS (JCIM) LaTeX source from the generated manuscript, following the ACS LaTeX guidelines:
minimal packages, no custom \\newcommand/\\def, all sections and the reference list in one .tex file,
and only the graphics the document references.

  python scripts/13_acs_source.py <build_dir_with_ml.tex_and_ml.bbl> <out_dir>

Writes <out_dir>/Patel_Manuscript.tex, references_battery.bib and figures/<referenced files>.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
build, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
s = (build / "ml.tex").read_text()

# 1. packages: keep only those the content needs
DROP = ["float", "enumitem", "stfloats", "tabularx", "xcolor", "lmodern", "microtype", "inconsolata"]
for p in DROP:
    s = re.sub(r"\\usepackage(\[[^\]]*\])?\{" + p + r"\}\n", "", s)

# 2. remove the placeholder macros (all placeholders are filled) and the preamble comment
s = re.sub(r"% TBD placeholders.*?\n", "", s)
s = re.sub(r"\\newif\\ifshowTBDkeys\n\\showTBDkeysfalse\n", "", s)
s = re.sub(r"\\newcommand\{\\(TBD|TBDtext|TBDcell|FIGTBD)\}.*\n", "", s)
assert not re.search(r"\\(TBD|TBDcell|TBDtext|FIGTBD)\b", s), "unfilled placeholder left"

# 3. enumerate without enumitem options
s = re.sub(r"\\begin\{enumerate\}\[[^\]]*\]", r"\\begin{enumerate}", s)

assert "\\newcommand" not in s and "\\def\\" not in s   # (the achemso .bbl inserted below has its own standard header)

# 4. reference list inside the main file
bbl = (build / "ml.bbl").read_text()
s = re.sub(r"\\bibliography\{[^}]*\}", lambda m: bbl.strip(), s)

# 5. referenced graphics only
figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)))
for f in figs:
    (out / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(build / f, out / f)
(out / "Patel_Manuscript.tex").write_text(s)
shutil.copy(build / "references_battery.bib", out / "references_battery.bib")
print(f"Patel_Manuscript.tex ({len(figs)} graphics) -> {out}")

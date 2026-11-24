"""Compile the TikZ schematics in paper/figures_src/ with tectonic -> paper/figures/<name>.{pdf,png} (600 dpi)."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
SRC, FIG = ROOT / "paper" / "figures_src", ROOT / "paper" / "figures"
for tex in sorted(SRC.glob("*.tex")):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["tectonic", "--outdir", d, str(tex)], check=True, capture_output=True)
        pdf = FIG / f"{tex.stem}.pdf"
        shutil.copy(Path(d) / f"{tex.stem}.pdf", pdf)
    fitz.open(pdf)[0].get_pixmap(dpi=600).save(FIG / f"{tex.stem}.png")
    print(pdf)

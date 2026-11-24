"""Collect the manuscript tables into a stand-alone paper/tables.tex (one table per page, numbered as in
the manuscript). Compile in the directory holding ml.aux and ml.bbl: labels come from ml.aux via xr and
citation numbers from ml.bbl, so they match the manuscript."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "paper" / "ml.tex").read_text()
pre = src[:src.index("\\begin{document}")]
pre = pre.replace("\\usepackage{hyperref}", "\\usepackage{xr}\n\\externaldocument{ml}\n\\usepackage{hyperref}")
tabs = re.findall(r"\\begin\{(table\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}", src, flags=re.S)
body = ["\\begin{center}{\\Large\\bfseries Tables}\\\\[4pt]{\\large Publication-aware prediction of battery material "
        "properties from literature-extracted data}\\\\[4pt]Sunraj Patel\\end{center}",
        "\\setbox0\\vbox{\\input{ml.bbl}}"]
for i, (_, t) in enumerate(tabs):
    body.append(("\\clearpage\n" if i else "") + "\\begin{table}[H]" + t + "\\end{table}")
(ROOT / "paper" / "tables.tex").write_text(pre + "\\begin{document}\n" + "\n\n".join(body) + "\n\\end{document}\n")
print(f"{len(tabs)} tables -> paper/tables.tex")

"""Fill paper/ml.tex from paper/ml_template.tex and results/paper_values.json (written by 05_analysis.py).

Edit prose in ml_template.tex only; ml.tex is generated. Unfilled keys stay as red \\TBD markers, and the script
lists them. \\FIGTBD{path}{caption} becomes \\includegraphics when paper/<path> exists.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
V = json.loads((ROOT / "results" / "paper_values.json").read_text())
missing = set()


def sub(m):
    k = m.group(2)
    if k in V:
        return V[k]
    missing.add(k)
    return m.group(0)


for src, dst in (("ml_template.tex", "ml.tex"), ("supplementary_template.tex", "supplementary.tex")):
    if not (P / src).exists():
        continue
    s = (P / src).read_text()
    s = re.sub(r"\\(TBD|TBDcell)\{([^{}]+)\}", sub, s)
    s = re.sub(r"\\FIGTBD\{([^{}]+)\}\{(?:[^{}]|\{[^{}]*\})*\}",
               lambda m: (f"\\includegraphics[width=\\linewidth]{{{m.group(1)}}}" if (P / m.group(1)).exists()
                          else m.group(0)), s)
    (P / dst).write_text(s)
    missing |= {f"TBDtext:{k}" for k in re.findall(r"\\TBDtext\{([^{}]+)\}", s)}
left = sorted(missing)
print(f"filled {len(V)} available values; {len(left)} placeholders left")
for k in left:
    print("  ", k)

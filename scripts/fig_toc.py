"""Table-of-contents graphic (8 cm x 4 cm) -> paper/figures/toc.png, from results/summary_full.csv and leakage_channels.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_csv(ROOT / "results" / "summary_full.csv")
C = pd.read_csv(ROOT / "results" / "leakage_channels.csv").groupby(["target", "protocol"]).mean(numeric_only=True)
T = ["capacity", "voltage", "energy"]
LAB = {"capacity": "Capacity", "voltage": "Voltage", "energy": "Energy"}
E = S[(S.tier == "t2") & (S.variant == "all") & (S.model == "ens")].set_index(["target", "protocol"]).r2

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(8 / 2.54, 4 / 2.54))
prot = [("random", "Random", "#9DB4CF"), ("doi", "DOI", "#2F5D8A"), ("temporal", "Time", "#C97B4A"),
        ("material", "Compos.", "#6BAF7C")]
for i, t in enumerate(T):
    for j, (p, lab, c) in enumerate(prot):
        if (t, p) in E.index:
            ax[0].bar(i + (j - 1.5) * 0.2, E[(t, p)], 0.19, color=c, label=lab if i == 0 else None)
ax[0].set_xticks(range(len(T))); ax[0].set_xticklabels([LAB[t] for t in T]); ax[0].set_ylabel("$R^2$")
ax[0].set_title("Split protocol", fontsize=6.5); ax[0].legend(frameon=False, fontsize=5, ncol=2, loc="upper right")
ax[0].set_ylim(0, 0.75)
for i, t in enumerate(T):
    r = C.loc[(t, "doi")]
    ax[1].bar(i - 0.15, r.r2_rep, 0.3, color="#B5505A", label="value seen in\nanother paper" if i == 0 else None)
    ax[1].bar(i + 0.15, r.r2_new, 0.3, color="#5A6B7B", label="new value" if i == 0 else None)
ax[1].set_xticks(range(len(T))); ax[1].set_xticklabels([LAB[t] for t in T]); ax[1].set_ylim(0, 0.9)
ax[1].set_title("DOI-grouped test records", fontsize=6.5); ax[1].legend(frameon=False, fontsize=5, loc="upper right")
fig.tight_layout(pad=0.3)
out = ROOT / "paper" / "figures" / "toc.png"
fig.savefig(out, dpi=600)
print(out)

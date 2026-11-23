"""Draw paper/figures/battery_workflow.png (pipeline schematic)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(__file__).resolve().parents[1] / "paper" / "figures" / "battery_workflow.png"
steps = [("Huang–Cole\ndatabase", "battery.csv\n292,313 records\n42,873 DOIs"),
         ("Cleaning", "DOI normalisation\nrange filters\ndrop derived energy"),
         ("Features\n(per target)", "composition, elemental\nproperties, name flags,\nmeasurement context"),
         ("Splits", "record-level random\nDOI-grouped\ntemporal"),
         ("Ensemble", "CatBoost (3 seeds)\n+ XGBoost"),
         ("Evaluation", "R², MAE, RMSE\ngrouped SHAP\nmember spread")]
fig, ax = plt.subplots(figsize=(7.2, 1.55)); ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis("off")
for i, (h, b) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i + .06, .08), .80, .84, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc="#EAF0F6" if i != 3 else "#FBE9DD", ec="#2F5D8A" if i != 3 else "#C97B4A", lw=.8))
    ax.text(i + .46, .76, h, ha="center", va="center", fontsize=6.6, weight="bold")
    ax.text(i + .46, .36, b, ha="center", va="center", fontsize=5.6, linespacing=1.3)
    if i < len(steps) - 1:
        ax.annotate("", (i + 1.06, .5), (i + .87, .5), arrowprops=dict(arrowstyle="->", lw=.8))
fig.tight_layout(pad=.1); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print(out)

"""Feature provenance / interpretive group assignment (used for SHAP aggregation); defined by column prefix."""
GROUPS = ("composition", "material", "material_text", "measurement", "literature", "identity", "noise")
PREFIX = {"comp_": "composition", "mat_": "material", "txt_": "material_text", "meas_": "measurement",
          "lit_": "literature", "id_": "identity", "noise_": "noise"}


def base(f):
    return f[:-3] if f.endswith("_na") else f


def group(f):
    b = base(f)
    for p, g in PREFIX.items():
        if b.startswith(p):
            return g
    raise ValueError(f"feature {f!r} has no group prefix")

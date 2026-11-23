"""Row-level feature construction from the raw battery.csv schema (shared by 01_prepare_data.py and 04_predict.py).

Every feature is a function of one record's own fields (Extracted_name, Name, Specifier, Info, Type, Warning-R, Title)
plus fixed external tables (pymatgen elemental data, the frozen element list), so nothing is learned from other
records and featurisation can be done once, before splitting. Column prefixes define the SHAP groups / tiers:
comp_ (composition), mat_ (Magpie-style elemental statistics), txt_ (Name morphology words), meas_ (measurement
conditions), lit_ (Title keywords, paper-level constant), id_name (identity control).
"""
import ast
import re
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

CARRIERS = ("Li", "Na", "K", "Mg", "Zn", "Al")
ELEM_PROPS = ("Z", "mass", "en", "radius", "row", "group", "mendeleev", "ie1", "oxi_max")
STATS = ("mean", "min", "max", "range")
SYMBOLIC_VALUE = 0.1    # stoichiometry variables (x, y, 1-x, ...) evaluated at x = 0.1 (small dopant level)

# Specifier is a closed 54-term CDE vocabulary in battery.csv; the DESIGN.md keywords initial/first, areal,
# volumetric, average, open-circuit, plateau, cut-off and ionic never occur in it and are omitted (constant columns).
# "potential" (vs "voltage") is added: it distinguishes electrode potentials from cell voltages.
SPEC_FLAGS = {"theoretical": r"theoretical", "reversible": r"reversib", "discharge": r"discharg",
              "charge": r"(?<!dis)charg", "specific": r"specific", "electronic": r"electronic|electrical",
              "potential": r"potential"}
NAME_FLAGS = {"carbon": r"carbon|graphen|graphite|\bcnts?\b|nanotube|\br?go\b|\bcnf", "graphene": r"graphen|\br?go\b",
              "cnt": r"\bcnts?\b|nanotube", "core_shell": r"@", "composite": r"/|composite", "doped": r"dop",
              "nano": r"nano", "porous": r"hollow|porous|mesopor|macropor|micropor", "film": r"film",
              "anode": r"anode", "cathode": r"cathode", "electrode": r"electrode"}
C_TOKEN = r"(?:^|[\s/@\-–(])C(?:$|[\s/@\-–),])"   # stand-alone carbon component, e.g. "Fe2O3 / C", "Si@C"
TITLE_FLAGS = {"lithium_ion": r"LITHIUMION|LIION", "sodium_ion": r"SODIUMION|NAION|SODIUMSTORAGE",
               "potassium": r"POTASSIUM", "li_s": r"LITHIUMSULFUR|LITHIUMSULPHUR|LISULFUR|LISBATTER",
               "li_o2": r"LITHIUMOXYGEN|LITHIUMAIR|LIO2|LIAIR", "zinc": r"ZINC", "magnesium": r"MAGNESIUM",
               "supercapacitor": r"SUPERCAPACIT|ULTRACAPACIT|PSEUDOCAPACIT",
               "solid_state": r"SOLIDSTATE|ALLSOLID|SOLIDELECTROLYTE", "electrolyte": r"ELECTROLYTE",
               "anode": r"ANODE", "cathode": r"CATHODE"}
ORDINALS = {"first": 1, "initial": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "tenth": 10}


@lru_cache(maxsize=None)
def element_props(sym):
    """pymatgen elemental properties (NaN where pymatgen has no value); None for unknown symbols."""
    from pymatgen.core import Element
    try:
        e = Element(sym)
    except Exception:
        return None

    def get(f):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = f(e)
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan
    return np.array([get(lambda e: e.Z), get(lambda e: e.atomic_mass), get(lambda e: e.X),
                     get(lambda e: e.atomic_radius), get(lambda e: e.row), get(lambda e: e.group),
                     get(lambda e: e.mendeleev_no), get(lambda e: e.ionization_energy),
                     get(lambda e: max(e.common_oxidation_states) if e.common_oxidation_states else None)])


def _amount(v):
    """Stoichiometry string -> (float > 0 or None, symbolic?). Variables are set to SYMBOLIC_VALUE."""
    try:
        a = float(v)
        return (a if a > 0 else None), False
    except (TypeError, ValueError):
        pass
    expr = re.sub(r"[A-Za-z]+", str(SYMBOLIC_VALUE), str(v))
    if not re.fullmatch(r"[0-9.+\-*/() ]+", expr):
        return SYMBOLIC_VALUE, True
    try:
        a = float(eval(expr, {"__builtins__": {}}, {}))  # arithmetic only (checked above)
    except Exception:
        return SYMBOLIC_VALUE, True
    return (a if a > 0 else SYMBOLIC_VALUE), True


def parse_extracted(s):
    """Extracted_name text -> (n_components, {element: summed amount}, symbolic) or None if unparsable.

    Components of a composite ("Fe2O3 / C" -> [{Fe:2,O:3},{C:1}]) are pooled by summing element amounts (as a
    pymatgen Composition sum). Unknown element symbols are ignored."""
    if not isinstance(s, str):
        return None
    try:
        L = ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(L, dict):
        L = [L]
    if not isinstance(L, (list, tuple)):
        return None
    amt, sym = {}, False
    for comp in L:
        if not isinstance(comp, dict):
            continue
        for el, v in comp.items():
            if element_props(str(el)) is None:
                continue
            a, s_ = _amount(v)
            sym |= s_
            if a is not None:
                amt[str(el)] = amt.get(str(el), 0.0) + a
    return (len(L), amt, sym) if amt else None


def element_counts(extracted):
    """Number of records containing each element (for the frozen element list)."""
    vc = pd.Series(extracted).value_counts(dropna=False)
    cnt = {}
    for s, n in vc.items():
        p = parse_extracted(s)
        for el in (p[1] if p else ()):
            cnt[el] = cnt.get(el, 0) + int(n)
    return cnt


def composition_block(extracted, elements):
    """comp_* and mat_* features; computed once per unique Extracted_name string and broadcast."""
    codes, uniq = pd.factorize(pd.Series(extracted).astype(object).where(pd.notna(extracted), None))
    el_idx = {e: i for i, e in enumerate(elements)}
    nE, nP = len(elements), len(ELEM_PROPS)
    comp_names = ([f"comp_frac_{e}" for e in elements] + ["comp_frac_other", "comp_n_elements", "comp_n_components",
                  "comp_symbolic_stoich"] + [f"comp_has_{c}" for c in CARRIERS])
    mat_names = [f"mat_{p}_{s}" for p in ELEM_PROPS for s in STATS]
    U = np.full((len(uniq), len(comp_names) + len(mat_names)), np.nan)
    for u, s in enumerate(uniq):
        p = parse_extracted(s)
        if p is None:
            continue
        ncomp, amt, sym = p
        els = list(amt)
        w = np.array([amt[e] for e in els]); w = w / w.sum()
        row = np.zeros(nE + 1)
        for e, f in zip(els, w):
            row[el_idx[e] if e in el_idx else nE] += f
        flags = [float(c in amt) for c in CARRIERS]
        V = np.array([element_props(e) for e in els])          # (n_el, nP)
        ok = ~np.isnan(V)
        with np.errstate(invalid="ignore", divide="ignore"):
            ww = np.where(ok, w[:, None], 0.0)
            mean = (np.where(ok, V, 0.0) * ww).sum(0) / ww.sum(0)   # fraction-weighted over elements with data
            mn = np.where(ok.any(0), np.nanmin(np.where(ok, V, np.inf), 0), np.nan)
            mx = np.where(ok.any(0), np.nanmax(np.where(ok, V, -np.inf), 0), np.nan)
        mean = np.where(ok.any(0), mean, np.nan)
        mat = np.stack([mean, mn, mx, mx - mn], 1).ravel()     # property-major, stat-minor (matches mat_names)
        U[u] = np.concatenate([row, [len(els), ncomp, float(sym)], flags, mat])
    out = np.full((len(codes), U.shape[1]), np.nan)
    ok = codes >= 0
    out[ok] = U[codes[ok]]
    return pd.DataFrame(out, columns=comp_names + mat_names)


def _flags(s, patterns, prefix, case=False):
    s = pd.Series(s).astype(object)
    na = s.isna()
    t = s.fillna("").astype(str)
    out = {f"{prefix}{k}": t.str.contains(p, regex=True, case=case).astype(float) for k, p in patterns.items()}
    return pd.DataFrame(out), na


def _parse_info(s):
    """Info dict text -> (log10 current density in A/g, log10 C-rate, log10 cycle number); NaN when absent."""
    out = [np.nan, np.nan, np.nan]
    try:
        D = ast.literal_eval(s) if isinstance(s, str) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        D = None
    if not isinstance(D, dict):
        return out
    m = re.search(r"\d+(?:\.\d+)?", str(D.get("current_value", "")))
    u = str(D.get("current_units", "")).replace("−", "-").replace(" ", "").lower()
    if m and float(m.group()) > 0:
        v = float(m.group())
        if u == "c":
            out[1] = np.log10(v)
        elif u.startswith("ma") and "g" in u:
            out[0] = np.log10(v / 1000)
        elif u.startswith("a") and "kg" in u:
            out[0] = np.log10(v / 1000)
        elif u.startswith("a") and "g" in u:
            out[0] = np.log10(v)
    c = str(D.get("cycle_value", "")).lower()
    m = re.search(r"\d+", c)
    n = float(m.group()) if m else next((float(v) for k, v in ORDINALS.items() if k in c), np.nan)
    if n > 0:
        out[2] = np.log10(n)
    return out


def build_features(df, elements):
    """All row-level features for raw-schema records `df` (index-aligned output). Missing raw columns -> NaN."""
    col = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index, dtype=object)  # noqa: E731
    parts = [composition_block(col("Extracted_name").values, elements)]
    name = col("Name")
    txt, _ = _flags(name, NAME_FLAGS, "txt_")
    txt["txt_carbon"] = np.maximum(txt["txt_carbon"], name.fillna("").astype(str).str.contains(C_TOKEN).astype(float))
    txt.loc[name.isna().values] = np.nan
    parts.append(txt)
    spec, na = _flags(col("Specifier"), SPEC_FLAGS, "meas_spec_")
    spec["meas_spec_missing"] = na.astype(float).values
    parts.append(spec)
    info = col("Info").fillna("").astype(str)       # "" -> all NaN (NaN keys are not hashable-equal in a dict)
    vc = pd.Series(info.unique())
    lut = dict(zip(vc, (_parse_info(s) for s in vc)))
    parts.append(pd.DataFrame([lut[s] for s in info], columns=["meas_log_current_Ag", "meas_log_crate",
                                                                "meas_log_cycle"]))
    parts.append(pd.DataFrame({"meas_type": col("Type").where(col("Type").notna(), None).values,
                               "lit_warn_R": col("Warning").fillna("").astype(str).str.contains("R")
                               .astype(float).values}))
    title = col("Title").fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    lit, _ = _flags(title, TITLE_FLAGS, "lit_", case=True)
    lit.loc[(title == "").values] = np.nan
    parts.append(lit)
    parts.append(pd.DataFrame({"id_name": name.fillna("").astype(str).str.lower()
                               .str.replace(r"\s+", " ", regex=True).str.strip().values}))
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    X.index = df.index
    return X

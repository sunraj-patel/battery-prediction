"""Diagnostics of repeated literature values on the DOI-grouped test sets (nothing is refitted).

  python scripts/12_repeated_diagnostics.py
    -> results/repeated_diagnostics.csv          (tidy: target, split, metric, value; split="all" = whole set)
       results/repeated_diagnostics_summary.csv  (per target, mean over the 5 DOI splits)
       results/repeated_pairs_sample.csv         (50 repeated pairs for manual classification; manual_class empty)
       results/paper_values.json                 (new keys under rpd/, existing keys untouched)

Runs: tier t2, variant "all", protocol "doi" (5 splits per target), ensemble predictions ('ens').

Definitions
-----------
key            lower-cased, whitespace-normalised Name + "|" + Value at 6 significant figures (as in 05_analysis.channels).
repeated       a test record whose key occurs among the training records (all records not in that split's test
               set). Under DOI grouping every such match is from a different publication.
publication    doi_group; size = number of records of that target in the whole modelling set with that doi_group.
year           year of pub_date (earliest parsed Date of the publication; computed in 01_prepare_data.py and the
               same date the temporal split uses).

1. R flag / size  R_share_rep, R_share_new: share of repeated / non-repeated test records whose Warning contains "R".
               pubsize_rep_median, pubsize_new_median: median size of the test record's own publication.
               pubsize_match_median: median over repeated test records of the largest matching training publication
               (a max over several publications, so biased upward); pubsize_match_med_median: same with the median.
               topdec_rep, topdec_new: share of records whose own publication is in the top decile of publication
               size (size >= 90th percentile over the publications of that target).
2. Dates       for each repeated test record, with the set M of years of its matching training publications:
               later   = test year > max(M);  earlier = test year < min(M);  sameyear = otherwise (test year equals
               some match year, or lies between the match years). Records without a year are excluded.
               NOTE: under a random DOI-grouped split the test publication is an exchangeable draw among the
               publications sharing a key, so later and earlier are expected to be equal *whatever* the mechanism
               (quotation or re-measurement); a test-later excess cannot arise from the split itself.
               Whole-set (split="all"): over keys held by >=2 publications, sym_pairs_diffyear = fraction of unordered
               publication pairs sharing a key whose years differ; sym_keys_later = fraction of such keys that are
               re-reported in a year later than the first year they appear; sym_gap_median = median (last - first)
               year over those keys; sym_later_larger = among keys with a unique earliest publication, fraction where
               the largest later publication is larger than the earliest one (reviews/compilations tend to be large).
3. Rule classes (each repeated test record, mutually exclusive, in this order)
               theo  : test Specifier contains "theoretical", or (capacity only) the test Name matches one of the
                       materials in THEO and the Value is within +-1% of one of its theoretical capacities;
               quote : not theo, and the test year is later than at least one matching publication's year
                       (necessary, not sufficient, for restatement/quotation);
               other : the rest (same year or earlier than every match: possible independent re-measurement,
                       concurrent reporting, or undated).
4. Tolerance   tol1: repeated if a training record has the same normalised Name and |v_test - v_train| <= 0.01|v_test|
               (raw Value, so conductivity is matched in S/cm, not log10). sig2: key with Value rounded to two
               significant figures. exact6: key with Value at 6 significant figures (%.6g) instead of round(3);
               differs from the main key only where round(3) is coarse, i.e. conductivity in S/cm, where every
               value < 5e-4 rounds to 0.0 so the main key reduces to the name. For all three: share of test records repeated, R2 on repeated vs non-repeated.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("an", ROOT / "scripts" / "05_analysis.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
from bmd.data import TARGETS  # noqa: E402  (path set by 05_analysis)

RES, TGT, fnum, r2 = an.RES, an.TGT, an.fnum, an.r2
SEED, NSAMPLE, MINPER = 12, 50, 5

# material -> (name regex on the raw Name, theoretical capacities in mAh/g)
THEO = {
    "LiFePO4": (r"LiFePO4|\bLFP\b", [170]), "LiCoO2": (r"LiCoO2|\bLCO\b", [274, 140]),
    "graphite": (r"(?i)graphit", [372]), "Si": (r"(?<![A-Za-z])Si(?![a-z0-9])|(?i:silicon)", [4200, 3579]),
    "Li": (r"(?<![A-Za-z])Li(?![a-zA-Z0-9])|(?i:lithium metal)", [3860]),
    "S": (r"(?<![A-Za-z])S(?![a-z0-9])|(?i:sulfur|sulphur)", [1675]), "LiMn2O4": (r"LiMn2O4|\bLMO\b", [148]),
    "Li4Ti5O12": (r"Li4Ti5O12|\bLTO\b", [175]), "SnO2": (r"SnO2", [782, 1494]), "Fe2O3": (r"Fe2O3", [1007]),
    "Co3O4": (r"Co3O4", [890]), "NiO": (r"(?<![A-Za-z])NiO(?![0-9a-z])", [718]), "MoS2": (r"MoS2", [670]),
    "TiO2": (r"TiO2", [335, 168]), "V2O5": (r"V2O5", [294]), "Sn": (r"(?<![A-Za-z])Sn(?![a-zA-Z0-9])", [994]),
}


def norm_name(s):
    return s.fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def sig2(v):
    v = np.asarray(v, float); out = np.zeros_like(v)
    nz = v != 0
    e = np.floor(np.log10(np.abs(v[nz])))
    out[nz] = np.round(v[nz] / 10 ** (e - 1)) * 10 ** (e - 1)
    return np.array([f"{x:.6g}" for x in out])


def theo_value(name, value):
    hit = np.zeros(len(name), bool)
    for pat, caps in THEO.values():
        m = name.str.contains(pat, regex=True).values
        close = np.zeros(len(name), bool)
        for c in caps:
            close |= np.abs(value - c) <= 0.01 * c
        hit |= m & close
    return hit


def tol_match(nm_tr, v_tr, nm_te, v_te, rel=0.01):
    """True where some training record with the same name has |v_te - v_tr| <= rel*|v_te|."""
    o = np.lexsort((v_tr, nm_tr)); n_s, v_s = nm_tr[o], v_tr[o]
    un, start = np.unique(n_s, return_index=True); end = np.r_[start[1:], len(n_s)]
    pos = dict(zip(un, zip(start, end)))
    out = np.zeros(len(v_te), bool)
    tol = rel * np.abs(v_te)
    for nm, idx in pd.Series(np.arange(len(v_te))).groupby(nm_te).groups.items():
        if nm not in pos:
            continue
        a, b = pos[nm]; vs = v_s[a:b]; idx = np.asarray(idx)
        lo = np.searchsorted(vs, v_te[idx] - tol[idx], "left")
        hi = np.searchsorted(vs, v_te[idx] + tol[idx], "right")
        out[idx] = hi > lo
    return out


def r2s(y, p, m):
    return r2(y[m], p[m]) if m.sum() > 10 else np.nan


def whole_set(t, key, g, year, size):
    """Symmetric cross-publication statistics over the full modelling set."""
    P = pd.DataFrame(dict(key=key, g=g, year=year, size=size)).drop_duplicates(["key", "g"])
    P = P[P.groupby("key").g.transform("size") >= 2].dropna(subset=["year"])
    n_keys = P.key.nunique()
    # unordered publication pairs per key, split by same/different year
    pairs = diff = 0
    for _, yr in P.groupby("key").year:
        c = yr.value_counts().values; k = c.sum()
        tot = k * (k - 1) // 2; same = int((c * (c - 1) // 2).sum())
        pairs += tot; diff += tot - same
    a = P.groupby("key").year.agg(["min", "max"])
    first = P[P.year == P.key.map(a["min"])]
    uniq_first = first.groupby("key").size() == 1
    f1 = first[first.key.map(uniq_first)].set_index("key")["size"]
    later = P[P.year > P.key.map(a["min"])].groupby("key")["size"].max()
    common = f1.index.intersection(later.index)
    return dict(sym_n_keys=n_keys, sym_pairs=pairs, sym_pairs_diffyear=diff / pairs if pairs else np.nan,
                sym_keys_later=float((a["max"] > a["min"]).mean()),
                sym_gap_median=float((a["max"] - a["min"])[a["max"] > a["min"]].median()),
                sym_later_larger=float((later[common] > f1[common]).mean()) if len(common) else np.nan)


def main():
    R = an.load_runs()
    R = R[(R.tier == "t2") & (R.variant == "all") & (R.protocol == "doi")]
    LC = pd.read_csv(RES / "leakage_channels.csv"); LC = LC[LC.protocol == "doi"].set_index(["target", "split"])
    rows, cands = [], []
    for t in TGT:
        D, A = an.data(t), an.aligned(t)
        nm = norm_name(D["Name"]).values
        val = D["Value"].values.astype(float)
        key = np.char.add(np.char.add(nm.astype(str), "|"), np.array([f"{x:.6g}" for x in val]))  # same as 05_analysis
        key6 = np.char.add(np.char.add(nm.astype(str), "|"), np.array([f"{x:.6g}" for x in val]))
        key2 = np.char.add(np.char.add(nm.astype(str), "|"), sig2(val))
        g = D["doi_group"].values
        year = pd.to_datetime(D["pub_date"]).dt.year.values.astype(float)
        size = D.groupby("doi_group")["doi_group"].transform("size").values
        q90 = np.quantile(D.groupby("doi_group").size().values, 0.9)
        rflag = D["Warning"].fillna("").str.contains("R").values
        spec_s = A["Specifier"].fillna("")
        theo = spec_s.str.lower().str.contains("theoretical").values
        if t == "capacity":
            theo |= theo_value(D["Name"].fillna(""), val)
        for k, v in whole_set(t, key, g, year, size).items():
            rows.append(dict(target=t, split="all", metric=k, value=v))
        rows.append(dict(target=t, split="all", metric="pubsize_q90", value=q90))
        for _, r in R[R.target == t].sort_values("split").iterrows():
            P = an.preds(r.dir, r.split); te = P["test_idx"]; y, p = P["y"], P["ens"]
            tr = np.ones(len(D), bool); tr[te] = False; tri = np.where(tr)[0]
            rep = np.isin(key[te], np.unique(key[tr]))
            out = dict(n=len(te), rep_share=rep.mean(), r2_rep=r2s(y, p, rep), r2_new=r2s(y, p, ~rep),
                       rep_share_check=rep.mean() - LC.loc[(t, r.split), "rep_share"])
            # 1. R flag and publication size
            out.update(R_share_rep=rflag[te][rep].mean(), R_share_new=rflag[te][~rep].mean(),
                       R_share_all=rflag[te].mean(),
                       pubsize_rep_median=np.median(size[te][rep]), pubsize_new_median=np.median(size[te][~rep]),
                       topdec_rep=(size[te][rep] >= q90).mean(), topdec_new=(size[te][~rep] >= q90).mean())
            # matching training publications of each repeated test record
            T = pd.DataFrame(dict(key=key[tri], g=g[tri], year=year[tri], size=size[tri], i=tri))
            Tp = T.drop_duplicates(["key", "g"])
            ri = te[rep]
            Q = pd.DataFrame(dict(i=ri, key=key[ri], year=year[ri]))
            agg = Tp.groupby("key").agg(mmin=("year", "min"), mmax=("year", "max"), msize=("size", "max"),
                                        msize_med=("size", "median"),
                                        npub=("g", "size"))
            Q = Q.join(agg, on="key")
            nrec = T.groupby("key").size(); Q["nrec"] = Q.key.map(nrec).values
            sk = set(zip(Tp.key, Tp.year)); Q["same"] = [(a, b) in sk for a, b in zip(Q.key, Q.year)]
            out["pubsize_match_median"] = Q.msize.median()
            out["pubsize_match_med_median"] = Q.msize_med.median()
            out["match_npub_median"] = Q.npub.median()
            # 2. date asymmetry
            dated = Q.year.notna() & Q.mmin.notna()
            later = dated & (Q.year > Q.mmax); earlier = dated & (Q.year < Q.mmin)
            nd = dated.sum()
            out.update(date_coverage=dated.mean(), later_share=later.sum() / nd, earlier_share=earlier.sum() / nd,
                       sameyear_share=(dated & ~later & ~earlier).sum() / nd,
                       sameyear_exact_share=(dated & ~later & ~earlier & Q.same).sum() / nd)
            # 3. rule classes
            th = theo[ri]; qt = ~th & dated.values & (Q.year > Q.mmin).values
            out.update(class_theo=th.mean(), class_quote=qt.mean(), class_other=(~th & ~qt).mean())
            cls = np.where(th, "theo", np.where(qt, "quote", "other"))
            # sample candidates: earliest matching training record (the most plausible source)
            Te = T.sort_values(["year", "i"]).drop_duplicates("key")
            m1 = Q[["i", "key"]].merge(Te[["key", "i"]].rename(columns={"i": "j"}), on="key")
            m1 = m1.assign(target=t, split=r.split, rule_class=pd.Series(cls, index=Q.i).loc[m1.i].values,
                           n_matches=Q.set_index("i").loc[m1.i, "nrec"].values,
                           n_match_pubs=Q.set_index("i").loc[m1.i, "npub"].values)
            cands.append(m1)
            # 4. tolerance matching
            tol = tol_match(nm[tri], val[tri], nm[te], val[te])
            s2 = np.isin(key2[te], np.unique(key2[tr]))
            e6 = np.isin(key6[te], np.unique(key6[tr]))
            out.update(exact6_share=e6.mean(), exact6_r2_rep=r2s(y, p, e6), exact6_r2_new=r2s(y, p, ~e6))
            out.update(tol1_share=tol.mean(), tol1_r2_rep=r2s(y, p, tol), tol1_r2_new=r2s(y, p, ~tol),
                       sig2_share=s2.mean(), sig2_r2_rep=r2s(y, p, s2), sig2_r2_new=r2s(y, p, ~s2))
            rows += [dict(target=t, split=int(r.split), metric=k, value=float(v)) for k, v in out.items()]
            print(t, r.split, f"rep={out['rep_share']:.4f} (check {out['rep_share_check']:+.1e})", flush=True)
    L = pd.DataFrame(rows); L.to_csv(RES / "repeated_diagnostics.csv", index=False)
    S = L.groupby(["target", "metric"]).value.mean().unstack().reindex(TGT)
    S.to_csv(RES / "repeated_diagnostics_summary.csv")

    # 50-pair sample, stratified by target (>= MINPER each, rest proportional to repeated counts)
    C = pd.concat(cands, ignore_index=True).drop_duplicates(["target", "i"])
    cnt = C.target.value_counts().reindex(TGT)
    extra = NSAMPLE - MINPER * len(TGT)
    w = cnt / cnt.sum() * extra; nper = MINPER + np.floor(w).astype(int)
    for k in (w - np.floor(w)).sort_values(ascending=False).index[:NSAMPLE - nper.sum()]:
        nper[k] += 1
    rng = np.random.default_rng(SEED); out = []
    for t in TGT:
        c = C[C.target == t]; c = c.iloc[np.sort(rng.choice(len(c), nper[t], replace=False))]
        D, A = an.data(t), an.aligned(t)
        yr = pd.to_datetime(D["pub_date"]).dt.year
        for _, x in c.iterrows():
            i, j = int(x.i), int(x.j)
            out.append(dict(target=t, name=D.Name[i], value=D.Value[i], value_match=D.Value[j], unit=TARGETS[t]["unit"],
                            specifier_test=A.Specifier[i], specifier_match=A.Specifier[j],
                            test_doi=A.DOI[i], test_title=A.Title[i], test_year=yr[i],
                            match_doi=A.DOI[j], match_title=A.Title[j], match_year=yr[j],
                            n_matches=int(x.n_matches), n_match_pubs=int(x.n_match_pubs), split=int(x.split),
                            rule_class=x.rule_class, manual_class=""))
    pd.DataFrame(out).to_csv(RES / "repeated_pairs_sample.csv", index=False)

    # paper values
    f = RES / "paper_values.json"; V = json.loads(f.read_text())
    pct = ["R_share_rep", "R_share_new", "later_share", "earlier_share", "sameyear_share", "class_theo",
           "class_quote", "class_other", "tol1_share", "sig2_share", "exact6_share", "topdec_rep", "topdec_new",
           "sym_pairs_diffyear", "sym_keys_later", "sym_later_larger"]
    r2k = ["tol1_r2_rep", "tol1_r2_new", "sig2_r2_rep", "sig2_r2_new", "exact6_r2_rep", "exact6_r2_new"]
    num = ["pubsize_rep_median", "pubsize_new_median", "pubsize_match_median", "pubsize_match_med_median"]
    for t in TGT:
        for k in pct:
            V[f"rpd/{t}/{k}"] = fnum(S.loc[t, k], "pct")
        for k in r2k:
            V[f"rpd/{t}/{k}"] = f"{S.loc[t, k]:.2f}"
        for k in num:
            V[f"rpd/{t}/{k}"] = f"{S.loc[t, k]:.0f}"
    for k in pct:
        V[f"rpd/{k}-range"] = f"{100 * S[k].min():.0f}--{100 * S[k].max():.0f}\\%"
    for k in r2k:
        V[f"rpd/{k}-range"] = f"{S[k].min():.2f}--{S[k].max():.2f}"
    f.write_text(json.dumps(V, indent=1, sort_keys=True))
    pd.set_option("display.width", 250)
    print(S.T.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

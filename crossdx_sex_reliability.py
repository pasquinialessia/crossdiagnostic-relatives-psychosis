#!/usr/bin/env python3
"""
Cross-diagnostic map correlations — sex-permutation reliability (reviewer analysis)
==================================================================================
The cross-diagnostic analysis (05) correlates each disorder's dx-vs-CN t-map with
another disorder's, split by sex, and tests each correlation only against a spatial
(spin/shuffle) null. This script adds the missing SEX-permutation reference:

  * observed cross-disorder correlation for the whole sample, males, and females;
  * a 1,000x sex-permutation null: shuffle sex labels, split into pseudo-male /
    pseudo-female groups of the SAME sizes, recompute each disorder's dx-vs-CN
    t-map in each pseudo group, correlate the disorder pair -> null distribution.

From it we get, per measure x disorder-pair:
  - null_mean_pseudo_r  : the concordance a random sex split gives (a ceiling for
    the by-sex correlations — if observed r_male/r_female sit here, they carry no
    sex-specific signal);
  - the female - male difference vs its sex-permuted null (p_diff): tests whether
    the "female-stronger" cross-disorder sharing (§4) exceeds chance.

dx-vs-CN t-maps are computed by vectorized OLS
(region ~ age_z + C(study_site) + C(dx, ref=CN)); the df constant cancels in the
across-region correlation. Sample: dataset.csv (loaded via load_crossdx_matched(), CN matched to clinical).

Outputs -> ../results/qc_sensitivity/crossdx_reliability/
"""
import os, warnings, numpy as np, pandas as pd
from scipy import stats
from data_utils import load_crossdx_matched
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "06_qc_sensitivity", "crossdx_reliability")
os.makedirs(OUT, exist_ok=True)

SEED, N_PERM = 42, 1000
rng = np.random.default_rng(SEED)
DX = ["BD", "SCZ", "Schizoaffective"]              # vs CN reference
PAIRS = [("BD", "SCZ"), ("BD", "Schizoaffective"), ("SCZ", "Schizoaffective")]

def cortical(df, pref):
    return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]
def subc(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def build_design(df):
    """Intercept + age_z + site dummies + dx dummies (CN ref). Returns X and the
    column indices of each dx dummy."""
    site = pd.get_dummies(df["study_site"], drop_first=True).astype(float)
    dxd  = pd.get_dummies(df["dx"], drop_first=False).astype(float)
    dxd  = dxd[[d for d in DX if d in dxd.columns]]     # CN dropped (reference)
    X = np.column_stack([np.ones(len(df)), df["age_z"].values, site.values, dxd.values])
    base = 2 + site.shape[1]
    idx = {d: base + i for i, d in enumerate(dxd.columns)}
    return X, idx

def dx_tmaps(X, Y, idx):
    """Vectorized OLS -> t-map (region vector) for each dx dummy."""
    XtXi = np.linalg.pinv(X.T @ X)
    B = XtXi @ (X.T @ Y)
    resid = Y - X @ B
    ssr = (resid ** 2).sum(0)
    out = {}
    for d, j in idx.items():
        se = np.sqrt(XtXi[j, j] * ssr)
        out[d] = B[j] / se
    return out

def corr(a, b):
    return float(stats.spearmanr(a, b)[0])

df = load_crossdx_matched(DATA)
df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)

rows = []
print("=== Cross-diagnostic sex-permutation reliability (1,000 perms) ===")
for meas, cols in [("CT", cortical(df, "CT")), ("GMV", cortical(df, "GM")),
                   ("SA", cortical(df, "SA")), ("Subcortical", subc(df))]:
    keep = ["age_z", "sex", "study_site", "dx"] + cols
    d = df[keep].dropna().reset_index(drop=True)
    sex = d["sex"].astype(str).values
    Y = d[cols].values.astype(float)
    X, idx = build_design(d)
    nM = int((sex == "Male").sum()); nF = int((sex == "Female").sum())
    # observed dx maps for whole / male / female
    maps_w = dx_tmaps(X, Y, idx)
    maps_m = dx_tmaps(X[sex == "Male"], Y[sex == "Male"], idx)
    maps_f = dx_tmaps(X[sex == "Female"], Y[sex == "Female"], idx)
    # sex-permutation null (per pair: pseudo r_male, r_female, and their diff)
    n = len(d); null_pseudo = {p: [] for p in PAIRS}; null_diff = {p: [] for p in PAIRS}
    for _ in range(N_PERM):
        perm = rng.permutation(n); a, b = perm[:nM], perm[nM:]
        mm = dx_tmaps(X[a], Y[a], idx); ff = dx_tmaps(X[b], Y[b], idx)
        for (d1, d2) in PAIRS:
            rm = corr(mm[d1], mm[d2]); rf = corr(ff[d1], ff[d2])
            null_pseudo[(d1, d2)] += [rm, rf]
            null_diff[(d1, d2)].append(rf - rm)
    for (d1, d2) in PAIRS:
        r_w = corr(maps_w[d1], maps_w[d2]); r_m = corr(maps_m[d1], maps_m[d2]); r_f = corr(maps_f[d1], maps_f[d2])
        diff = r_f - r_m
        npd = np.asarray(null_diff[(d1, d2)]); nps = np.asarray(null_pseudo[(d1, d2)])
        p_diff = float((1 + np.sum(np.abs(npd) >= abs(diff))) / (1 + len(npd)))
        rows.append({"measure": meas, "dx1": d1, "dx2": d2,
                     "r_whole": round(r_w, 3), "r_male": round(r_m, 3), "r_female": round(r_f, 3),
                     "female_minus_male": round(diff, 3),
                     "sexperm_ceiling_mean": round(float(nps.mean()), 3),
                     "sexperm_ceiling_2.5": round(float(np.percentile(nps, 2.5)), 3),
                     "sexperm_ceiling_97.5": round(float(np.percentile(nps, 97.5)), 3),
                     "p_female_vs_male_diff": round(p_diff, 4), "n_perm": N_PERM})
        print(f"{meas:11} {d1:4}-{d2:15} r F {r_f:+.2f} M {r_m:+.2f} | F-M {diff:+.2f} "
              f"(p={p_diff:.3f}) | sex-split ceiling {nps.mean():+.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "crossdx_sex_reliability_summary.csv"), index=False)
print("\nSaved ->", OUT)

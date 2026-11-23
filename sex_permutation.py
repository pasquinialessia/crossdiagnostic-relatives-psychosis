#!/usr/bin/env python3
"""
Sex-map divergence: observed r vs SSE null (beta difference)
=============================================================
The OBSERVED statistic is the Pearson correlation between the male and female
region-wise case-control effect maps (t-statistics of dx coefficient).
A lower r means the two sex-stratified maps point in different directions —
genuine male/female topographic divergence.

The NULL DISTRIBUTION is built by permuting sex labels (n = 1,000), keeping
the same group sizes as the real male/female split. For each permutation the
test statistic is the SUM OF SQUARED DIFFERENCES in the dx beta between the
two sub-group OLS models, summed across all brain regions:
  SSE = Σ_k (β_A[k] − β_B[k])²
This directly measures how much the two group-specific effect maps diverge.
The p-value is one-sided upper-tail:
  p = P(null_SSE > obs_SSE)
A small p means the real sex split produces MORE map divergence than random
splits — consistent with genuine sex-specific brain effects.

Outputs -> ../results/qc_sensitivity/map_reliability/
"""
import os, warnings, numpy as np, pandas as pd
from scipy import stats
from data_utils import load_bd_matched, load_scz_dataset
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "06_qc_sensitivity", "map_reliability")
os.makedirs(OUT, exist_ok=True)

SEED, N_PERM = 42, 1000
rng = np.random.default_rng(SEED)

def cortical(df, pref):
    return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]

def subcortical(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def build_design(df):
    """Return X for y ~ dx + age_z + site + dx:age_z, and index of dx coefficient.
    Columns: [intercept, dx, age_z, site_dummies..., dx*age_z]
    """
    site  = pd.get_dummies(df["study_site"], drop_first=True).astype(float)
    cond  = (df["Condition_type"].astype(str) == "Clinical").astype(float).values
    age_z = df["age_z"].values
    dxage = cond * age_z
    X = np.column_stack([np.ones(len(df)), cond, age_z, site.values, dxage])
    return X, 1     # dx is column 1

def ols_beta_and_tmap(X, Y, j):
    """Vectorized OLS for a sub-group.
    Returns:
      beta  : shape (R,) — raw coefficient j per region
      tmap  : shape (R,) — t-statistic of coefficient j per region
    """
    XtX  = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    B    = XtXi @ (X.T @ Y)           # (p, R)
    resid = Y - X @ B
    ssr   = (resid ** 2).sum(0)        # (R,) per-region residual SS
    se    = np.sqrt(XtXi[j, j] * ssr) # (R,)
    return B[j], B[j] / se            # beta, t-map

_LOADERS_06C = None
def _get_loaders():
    global _LOADERS_06C
    if _LOADERS_06C is None:
        _LOADERS_06C = {"scz": load_scz_dataset(DATA), "bd": load_bd_matched(DATA)}
    return _LOADERS_06C

def run(loader_key, label, dx_keep=None):
    df = _get_loaders()[loader_key].copy()
    if dx_keep is not None:
        df = df[df["dx"].isin(dx_keep) | (df["Condition_type"].astype(str) == "Control")].copy()
    df["Condition_type"] = df["Condition_type"].astype(str)
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)
    out = {}
    measures = [("CT",          cortical(df, "CT")),
                ("GMV",         cortical(df, "GM")),
                ("SA",          cortical(df, "SA")),
                ("Subcortical", subcortical(df))]
    for meas, cols in measures:
        keep = ["age_z", "sex", "study_site", "Condition_type"] + cols
        d   = df[keep].dropna().reset_index(drop=True)
        sex = d["sex"].astype(str).values
        nM  = int((sex == "Male").sum())
        nF  = int((sex == "Female").sum())
        X, jc = build_design(d)
        Y     = d[cols].values.astype(float)

        # --- Observed statistics ---
        betaM, tM = ols_beta_and_tmap(X[sex == "Male"],   Y[sex == "Male"],   jc)
        betaF, tF = ols_beta_and_tmap(X[sex == "Female"], Y[sex == "Female"], jc)
        r_obs   = float(stats.spearmanr(tM, tF)[0])          # map correlation (Spearman)
        sse_obs = float(np.sum((betaM - betaF) ** 2))        # sum of squared beta diffs

        # --- Permutation null: null[i] = Σ(β_A − β_B)² under random sex-label split ---
        n    = len(d)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            perm = rng.permutation(n)
            a, b = perm[:nM], perm[nM:]
            bA, _ = ols_beta_and_tmap(X[a], Y[a], jc)
            bB, _ = ols_beta_and_tmap(X[b], Y[b], jc)
            null[i] = float(np.sum((bA - bB) ** 2))

        # --- p-value: one-sided upper-tail ---
        # small p = real sex split produces MORE map divergence than random splits
        p_above = float((1 + np.sum(null > sse_obs)) / (1 + N_PERM))
        p95_thr = float(np.percentile(null, 95))

        rec = {
            "dataset":           label,
            "measure":           meas,
            "n":                 n,
            "n_male":            nM,
            "n_female":          nF,
            "r_observed":        round(r_obs,    4),
            "sse_observed":      round(sse_obs,  6),
            "null_mean_sse":     round(float(null.mean()),                6),
            "null_sd_sse":       round(float(null.std(ddof=1)),           6),
            "null_ci2.5_sse":    round(float(np.percentile(null,  2.5)), 6),
            "null_ci97.5_sse":   round(float(np.percentile(null, 97.5)), 6),
            "sse_p95_threshold": round(p95_thr,  6),
            "p_above":           p_above,
            "n_perm":            N_PERM,
        }
        out[meas] = rec
        np.savez_compressed(
            os.path.join(OUT, f"null_{label}_{meas}.npz"),
            null=null,
            r_observed=r_obs,
            sse_observed=sse_obs,
            null_mean_sse=null.mean(),
            p_above=p_above,
        )
        print(
            f"[{label}] {meas}: r_obs={r_obs:+.3f} | "
            f"SSE_obs={sse_obs:.4f} null_mean={null.mean():.4f} "
            f"[{np.percentile(null,2.5):.4f},{np.percentile(null,97.5):.4f}] "
            f"| p_above={p_above:.4f}",
            flush=True,
        )
    return list(out.values())

rows  = []
rows += run("scz", "SCZ_matched", dx_keep=["SCZ"])
rows += run("scz", "SAD_matched", dx_keep=["Schizoaffective"])
rows += run("bd",  "BD_matched")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "map_reliability_summary.csv"), index=False)
print("\nSaved ->", OUT)

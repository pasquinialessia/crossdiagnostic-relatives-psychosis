#!/usr/bin/env python3
"""
Case-vs-Relative (proband vs first-degree relative) analysis — BSNIP
======================================================================
Question: how does the brain of a proband differ from their unaffected first-
degree relative?  This contrast removes shared familial factors (genetic
background, upbringing) and isolates effects of diagnosis, medication and
illness course.

Design: NDAR_BSNIP, Proband + Relative only (controls excluded).
Model (per region, vectorized):
    region ~ 1 + age_z + sex_bin + site_dummies + relative_flag

where relative_flag = 1 for Relative, 0 for Clinical (reference).
A positive beta means the relative shows MORE than the clinical group.
Separate model per disorder × measure.

Outputs (../results/relatives/case_vs_relative/):
  _supp_case_vs_relative.csv   — cohens_d, t, p (uncorrected & BH-FDR) per region
  {disorder}_{measure}_cohen_d.npy  — Cohen's d arrays for brain-map plotting
"""

import os, warnings
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from data_utils import load_bsnip_relatives
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "08_relatives", "case_vs_relative")
os.makedirs(OUT, exist_ok=True)

df = load_bsnip_relatives(DATA)
df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)

# ── helpers ──────────────────────────────────────────────────────────────────
def cortical(df, pref):
    return [c for c in df.columns
            if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]

def subc(df):
    return [c for c in df.columns
            if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def build_X(sub):
    site    = pd.get_dummies(sub["study_site"].astype(str), drop_first=True).astype(float)
    sex_bin = (sub["sex"].astype(str).str.startswith(("M","m"))).astype(float).values
    relative_flag = (sub["role"].astype(str) == "Relative").astype(float).values
    X = np.column_stack([np.ones(len(sub)), sub["age_z"].values, sex_bin,
                         site.values, relative_flag])
    j_rel = 3 + site.shape[1]          # index of relative_flag (Clinical = ref)
    return X, j_rel

def ols(X, Y, j):
    """Return beta[j], t[j], p[j] (two-sided) for every column of Y."""
    XtXi = np.linalg.pinv(X.T @ X)
    B    = XtXi @ (X.T @ Y)             # (p, R)
    resid = Y - X @ B
    ssr   = (resid ** 2).sum(0)
    dof   = X.shape[0] - X.shape[1]
    se    = np.sqrt(XtXi[j, j] * ssr / max(dof, 1))
    beta  = B[j]
    t     = beta / se
    from scipy.stats import t as tdist
    p     = 2 * tdist.sf(np.abs(t), df=dof)
    return beta, t, p

DISORDERS = ["SCZ", "SAD", "BD"]
MEASURES  = [("CT",          lambda d: cortical(d,"CT")),
             ("GMV",         lambda d: cortical(d,"GM")),
             ("SA",          lambda d: cortical(d,"SA")),
             ("Subcortical", subc)]

rows = []
for dx in DISORDERS:
    sub = df[df["dx_group"].isin([dx, "Control"])].copy()
    sub = sub[sub["role"].isin(["Proband","Relative"])].copy()
    sub = sub.reset_index(drop=True)

    n_prob = int((sub["role"]=="Proband").sum())
    n_rel  = int((sub["role"]=="Relative").sum())
    print(f"\n{dx}: {n_prob} clinical, {n_rel} relatives  [Clinical = reference]")

    for meas, col_fn in MEASURES:
        cols = col_fn(sub)
        keep = ["age_z","sex","study_site","role"] + cols
        d    = sub[keep].dropna().reset_index(drop=True)

        Y  = d[cols].values.astype(float)
        X, j_rel = build_X(d)

        beta, t, p = ols(X, Y, j_rel)

        # Cohen's d: standardise by pooled SD of raw Y
        sd_pool = Y.std(axis=0, ddof=1)
        sd_pool[sd_pool == 0] = np.nan
        cohen_d = beta / sd_pool

        # BH FDR within measure
        reject, q, _, _ = multipletests(p, method="fdr_bh")

        np.save(os.path.join(OUT, f"{dx}_{meas}_cohen_d.npy"), cohen_d)
        np.save(os.path.join(OUT, f"{dx}_{meas}_t.npy"), t)

        for i, c in enumerate(cols):
            rows.append(dict(
                disorder=dx, measure=meas, region=c,
                n_clinical=n_prob, n_relative=n_rel,
                beta=float(beta[i]), t=float(t[i]),
                p=float(p[i]), q_fdr=float(q[i]),
                cohen_d=float(cohen_d[i])
            ))
        n_sig = int(reject.sum())
        print(f"  {meas}: {len(cols)} regions, {n_sig} FDR-sig (q<0.05)")

res = pd.DataFrame(rows)
out_csv = os.path.join(OUT, "_supp_case_vs_relative.csv")
res.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}  ({len(res)} rows)")

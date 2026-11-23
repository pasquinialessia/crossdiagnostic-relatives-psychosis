#!/usr/bin/env python3
"""Generate separate SCZ and SAD permutation null distributions.
Run from the repo root:  python code/generate_separate_nulls.py
"""
import os, warnings, numpy as np, pandas as pd
from scipy import stats
from data_utils import load_scz_dataset
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "results", "06_qc_sensitivity", "map_reliability")
os.makedirs(OUT, exist_ok=True)

SEED, N_PERM = 42, 1000
rng = np.random.default_rng(SEED)

def cortical(df, pref):
    return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]

def subcortical(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def build_design(df):
    site = pd.get_dummies(df["study_site"], drop_first=True).astype(float)
    cond = (df["Condition_type"].astype(str) == "Clinical").astype(float).values
    X = np.column_stack([np.ones(len(df)), df["age_z"].values, site.values, cond])
    return X, X.shape[1] - 1

def condition_tmap(X, Y, jcond):
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    B = XtXi @ (X.T @ Y)
    resid = Y - X @ B
    ssr = (resid ** 2).sum(0)
    se = np.sqrt(XtXi[jcond, jcond] * ssr)
    return B[jcond] / se

_LOADERS = None

def _get_loaders():
    global _LOADERS
    if _LOADERS is None:
        _LOADERS = {
            "scz": load_scz_dataset(DATA),
        }
    return _LOADERS

def run(dsfile, label, dx_keep=None):
    df = _get_loaders()["scz"].copy()
    if dx_keep is not None:
        df = df[df["dx"].isin(dx_keep) | (df["Condition_type"].astype(str) == "Control")].copy()
    df["Condition_type"] = df["Condition_type"].astype(str)
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)
    out = {}
    measures = [("CT", cortical(df, "CT")), ("GMV", cortical(df, "GM")),
                ("SA", cortical(df, "SA")), ("Subcortical", subcortical(df))]
    for meas, cols in measures:
        keep = ["age_z", "sex", "study_site", "Condition_type"] + cols
        d = df[keep].dropna().reset_index(drop=True)
        sex = d["sex"].astype(str).values
        nM = int((sex == "Male").sum()); nF = int((sex == "Female").sum())
        X, jc = build_design(d)
        Y = d[cols].values.astype(float)
        mM = condition_tmap(X[sex == "Male"], Y[sex == "Male"], jc)
        mF = condition_tmap(X[sex == "Female"], Y[sex == "Female"], jc)
        r_obs = float(stats.spearmanr(mM, mF)[0])
        n = len(d)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            perm = rng.permutation(n)
            a, b = perm[:nM], perm[nM:]
            ma = condition_tmap(X[a], Y[a], jc)
            mb = condition_tmap(X[b], Y[b], jc)
            null[i] = stats.spearmanr(ma, mb)[0]
        null = np.asarray(null)
        p_below = float((1 + np.sum(null <= r_obs)) / (1 + N_PERM))
        rec = {"dataset": label, "measure": meas, "n": n, "n_male": nM, "n_female": nF,
               "r_observed": r_obs, "null_mean_ceiling": float(null.mean()),
               "p_below_ceiling_1sided": p_below}
        out[meas] = rec
        out_path = os.path.join(OUT, f"null_{label}_{meas}.npz")
        np.savez_compressed(out_path, null=null,
                            r_observed=r_obs, null_mean_ceiling=float(null.mean()))
        print(f"[{label}] {meas}: n={n} (M={nM}/F={nF}) | r_obs={r_obs:+.3f} | "
              f"ceiling={null.mean():+.3f} | p={p_below:.4f}", flush=True)
    return list(out.values())

rows = []
print("=== SCZ-only null ===")
rows += run("dataset.csv", "SCZ_matched", dx_keep=["SCZ"])
print("\n=== SAD-only null ===")
rows += run("dataset.csv", "SAD_matched", dx_keep=["Schizoaffective"])

pd.DataFrame(rows).to_csv(
    os.path.join(OUT, "map_reliability_SCZ_SAD_separate.csv"), index=False)
print("\nAll done.")

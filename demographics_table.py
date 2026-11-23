#!/usr/bin/env python3
"""
Summary demographics (Table 1) for the analysis samples.
Outputs -> ../results/demographics/  (CSV tables)
  demographics_clinical_matched.csv   per-group N / sex / age / sites (SCZ+SAD sample)
  demographics_BD_matched.csv    per-group N / sex / age / sites (BD sample)
  group_comparisons.csv          case-vs-control age (Welch t) and sex (chi2) tests
"""
import os, numpy as np, pandas as pd
from scipy import stats
from data_utils import load_bd_matched, load_scz_dataset

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "01_demographics")
os.makedirs(OUT, exist_ok=True)

def describe(d):
    n = len(d); f = int((d.sex == "Female").sum()); m = int((d.sex == "Male").sum())
    return {"N": n, "Female_n": f, "Female_pct": round(100*f/n, 1),
            "Male_n": m, "Male_pct": round(100*m/n, 1),
            "Age_mean": round(d.age.mean(), 1), "Age_sd": round(d.age.std(ddof=1), 1),
            "Age_min": round(d.age.min(), 1), "Age_max": round(d.age.max(), 1),
            "N_sites": int(d.study_site.nunique())}

def table(df, groups, label):
    rows = [{"group": name, **describe(df[mask])} for name, mask in groups]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, f"demographics_{label}.csv"), index=False)
    print(f"\n===== {label} =====\n" + t.to_string(index=False))
    return t

def compare(df, case_mask, ctrl_mask, sample, contrast):
    """Case-vs-control age (Welch t) and sex (chi-square) tests -> dict row."""
    tage = stats.ttest_ind(df.loc[case_mask, "age"], df.loc[ctrl_mask, "age"], equal_var=False)
    sub = df[case_mask | ctrl_mask].copy()
    sub["grp"] = np.where(case_mask[case_mask | ctrl_mask], "case", "control")
    chi = stats.chi2_contingency(pd.crosstab(sub["grp"], sub["sex"]))
    return {"sample": sample, "contrast": contrast,
            "n_case": int(case_mask.sum()), "n_control": int(ctrl_mask.sum()),
            "age_welch_t": round(float(tage.statistic), 3), "age_p": round(float(tage.pvalue), 4),
            "sex_chi2": round(float(chi[0]), 3), "sex_p": round(float(chi[1]), 4)}

# ---- 1. Matched SCZ+SAD (SCZ + schizoaffective) with site-matched controls ----
clin_df = load_scz_dataset(DATA)
ctrl = clin_df.Condition_type == "Control"; clin = clin_df.Condition_type == "Clinical"
scz  = clin_df.dx == "SCZ"; scza = clin_df.dx.astype(str).str.startswith("Schizo")
table(clin_df, [("Controls", ctrl), ("Clinical (SCZ+SAD)", clin),
            ("  SCZ", scz), ("  Schizoaffective", scza),
            ("Total", clin_df.index == clin_df.index)], "clinical_matched")

# ---- 2. Site/sex-matched BD sample ----
bd = load_bd_matched(DATA)
table(bd, [("Controls (CN)", bd.dx == "CN"), ("BD", bd.dx == "BD"),
           ("Total", bd.index == bd.index)], "BD_matched")

# ---- 3. Group comparison tests -> CSV ----
comparisons = pd.DataFrame([
    compare(clin_df, clin, ctrl, "clinical_matched", "Clinical vs Control"),
    compare(bd, bd.dx == "BD", bd.dx == "CN", "BD_matched", "BD vs Control"),
])
comparisons.to_csv(os.path.join(OUT, "group_comparisons.csv"), index=False)
print("\n===== group_comparisons =====\n" + comparisons.to_string(index=False))
print("\nSaved ->", OUT)

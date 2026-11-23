#!/usr/bin/env python3
"""
Hemispheric correction — QC / sensitivity analysis (reviewer analysis)
=====================================================================
ICV is unavailable, so the pre-computed hemispheric aggregate for the matching
measure/side (HEMI.<side>.<measure>) is used as a global-size proxy. Here the
correction is applied as a NEW OUTCOME (region minus its hemispheric value),
rather than as a covariate:

    uncorrected outcome:  y = region
    corrected   outcome:  y = region - HEMI.<side>.<measure>

Both are fit with the whole-group model (both sexes) including the sex×condition
and age×condition interactions:

    y ~ age_z + C(sex) + C(study_site)
        + C(Condition_type, Treatment(reference="Control"))
        + C(Condition_type, Treatment(reference="Control")):C(sex)     # sex × dx
        + C(Condition_type, Treatment(reference="Control")):age_z      # age × dx

Per region we record the condition (dx) main effect, the sex×dx and age×dx
interactions, for both the uncorrected and corrected outcome; FDR across regions.
Run on BOTH matched samples: clinical SCZ+SAD (dataset.csv via load_scz_dataset) and
BD (BD vs matched CN, loaded via load_bd_matched()). Per-sample CSVs are written
with the sample label, plus a combined summary with a `dataset` column.

NOTE on the aggregate: HEMI.<side>.cortical.thickness is a hemispheric mean, while
HEMI.<side>.{surface.area, greymatter.volume, subcortex} are hemispheric totals.
So `region - hemi` is a within-hemisphere deviation for CT, but for SA/GMV/subcortex
it subtracts the hemispheric total. Adjust the aggregate if a different global
reference is intended.

Outputs -> ../results/qc_sensitivity/hemispheric_correction/
"""
import os, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
from data_utils import load_bd_matched, load_scz_dataset
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "06_qc_sensitivity", "hemispheric_correction")
os.makedirs(OUT, exist_ok=True)

FORM = ('y ~ age_z + C(sex) + C(study_site)'
        ' + C(Condition_type, Treatment(reference="Control"))'
        ' + C(Condition_type, Treatment(reference="Control")):C(sex)'
        ' + C(Condition_type, Treatment(reference="Control")):age_z')

HEMI = {"CT": "cortical.thickness", "GM": "greymatter.volume", "SA": "surface.area"}

def hemi_col(region):
    side = "left" if region.startswith("lh.") or ".Left." in region else "right"
    if region.startswith("SubcGM."):
        return f"HEMI.{side}.subcortex"
    return f"HEMI.{side}.{HEMI[region.split('.')[1]]}"

def cortical(df, pref):
    return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]
def subc(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def fit(d):
    m = smf.ols(FORM, data=d).fit()
    kmain = [k for k in m.params.index if "Clinical" in k and ":" not in k][0]
    ksex  = [k for k in m.params.index if "Clinical" in k and ":" in k and "sex" in k and "age_z" not in k][0]
    kage  = [k for k in m.params.index if "Clinical" in k and ":age_z" in k][0]
    return {"t_dx": m.tvalues[kmain], "p_dx": m.pvalues[kmain],
            "t_dxXsex": m.tvalues[ksex], "p_dxXsex": m.pvalues[ksex],
            "t_dxXage": m.tvalues[kage], "p_dxXage": m.pvalues[kage]}

_LOADERS_06B = None
def _get_loaders():
    global _LOADERS_06B
    if _LOADERS_06B is None:
        _LOADERS_06B = {"scz": load_scz_dataset(DATA), "bd": load_bd_matched(DATA)}
    return _LOADERS_06B

def run(loader_key, label):
    df = _get_loaders()[loader_key].copy()
    df["Condition_type"] = pd.Categorical(df["Condition_type"],
                                          categories=["Control", "Clinical"], ordered=False)
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)

    blocks = {"CT": cortical(df, "CT"), "GMV": cortical(df, "GM"),
              "SA": cortical(df, "SA"), "Subcortical": subc(df)}

    summary = []
    print(f"\n=== [{label}] Hemispheric correction (new outcome = region - hemi); whole-group model ===")
    for meas, cols in blocks.items():
        rows = []
        for col in cols:
            hc = hemi_col(col)
            d = df[[col, hc, "age_z", "sex", "study_site", "Condition_type"]].dropna().rename(
                columns={col: "region", hc: "hemi"})
            if d["Condition_type"].nunique() < 2 or len(d) < 40:
                continue
            r = {"region": col, "n": len(d)}
            for tag, y in [("uncorr", d["region"]), ("corr", d["region"] - d["hemi"])]:
                for k, v in fit(d.assign(y=y)).items():
                    r[f"{k}_{tag}"] = v
            rows.append(r)
        r = pd.DataFrame(rows)
        for tag in ["uncorr", "corr"]:
            r[f"pfdr_dx_{tag}"]     = multipletests(r[f"p_dx_{tag}"], method="fdr_bh")[1]
            r[f"pfdr_dxXsex_{tag}"] = multipletests(r[f"p_dxXsex_{tag}"], method="fdr_bh")[1]
            r[f"pfdr_dxXage_{tag}"] = multipletests(r[f"p_dxXage_{tag}"], method="fdr_bh")[1]
        r.to_csv(os.path.join(OUT, f"hemicorr_{label}_{meas}.csv"), index=False)
        s = {"dataset": label, "measure": meas, "n_regions": len(r),
             "dx_FDR_uncorr": int((r.pfdr_dx_uncorr < 0.05).sum()),
             "dx_FDR_corr":   int((r.pfdr_dx_corr < 0.05).sum()),
             "dxXsex_FDR_uncorr": int((r.pfdr_dxXsex_uncorr < 0.05).sum()),
             "dxXsex_FDR_corr":   int((r.pfdr_dxXsex_corr < 0.05).sum()),
             "dxXage_FDR_uncorr": int((r.pfdr_dxXage_uncorr < 0.05).sum()),
             "dxXage_FDR_corr":   int((r.pfdr_dxXage_corr < 0.05).sum()),
             "mean_abs_t_dx_uncorr": round(r.t_dx_uncorr.abs().mean(), 2),
             "mean_abs_t_dx_corr":   round(r.t_dx_corr.abs().mean(), 2),
             "map_r_uncorr_vs_corr": round(float(stats.spearmanr(r.t_dx_uncorr, r.t_dx_corr)[0]), 3)}
        summary.append(s)
        print(f"{meas:12} dx FDR {s['dx_FDR_uncorr']:2}->{s['dx_FDR_corr']:2} | "
              f"dx*sex {s['dxXsex_FDR_uncorr']}->{s['dxXsex_FDR_corr']} | "
              f"dx*age {s['dxXage_FDR_uncorr']}->{s['dxXage_FDR_corr']} | "
              f"mean|t_dx| {s['mean_abs_t_dx_uncorr']}->{s['mean_abs_t_dx_corr']} | map r={s['map_r_uncorr_vs_corr']}")
    return summary

summary = []
summary += run("scz", "clinical_matched")
summary += run("bd",  "BD_matched")
pd.DataFrame(summary).to_csv(os.path.join(OUT, "hemispheric_correction_summary.csv"), index=False)
print("\nSaved ->", OUT)

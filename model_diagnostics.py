#!/usr/bin/env python3
"""
Model diagnostics, condition x age_z interaction, and laterality (reviewer point 14 + 23)
=========================================================================================
Sample: SCZ and Schizoaffective analysed SEPARATELY from dataset.csv
         (each restricted to its own sites + matched CN).

Conventions fixed per request:
  * age_z is standardized ONCE on the whole sample at load (not per region/sex).
  * The canonical per-region model now includes a condition x age_z interaction:

        region ~ age_z + C(sex) + C(study_site)
                 + C(Condition_type, Treatment(reference="Control"))
                 + C(Condition_type, Treatment(reference="Control")):age_z

Per region we record: the condition main effect and the condition:age_z
interaction (t, p, FDR), plus model diagnostics:
  * residual normality        — Jarque-Bera p
  * heteroscedasticity        — Breusch-Pagan p
  * influential observations  — Cook's distance (# with D > 4/n, max D)
  * collinearity              — max VIF over predictors, and VIF of condition/age_z

Laterality: for each L/R homologous pair an asymmetry index
AI = (L - R) / ((L + R)/2) is modelled with the same covariates; the condition
effect on AI tests whether diagnosis alters structural asymmetry (FDR across pairs).

Outputs -> ../results/qc_sensitivity/diagnostics/
NEW analysis; existing scripts untouched.
"""
import os, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from data_utils import load_scz_dataset
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "06_qc_sensitivity", "diagnostics")
os.makedirs(OUT, exist_ok=True)

FORM = ('y ~ age_z + C(sex, Sum) + C(study_site)'
        ' + C(Condition_type, Treatment(reference="Control"))'
        ' + C(Condition_type, Treatment(reference="Control")):C(sex, Sum)'   # condition x sex
        ' + C(Condition_type, Treatment(reference="Control")):age_z')        # condition x age

_full = load_scz_dataset(DATA)

DIAGNOSES = {
    "scz": "SCZ",
    "sad": "Schizoaffective",
}

def build_dx_sample(raw, dx_value, seed=42):
    """Site-restrict then site×sex-match CN to cases (same approach as BD)."""
    sites  = raw.loc[raw["dx"] == dx_value, "study_site"].unique()
    sub    = raw[raw["study_site"].isin(sites) & raw["dx"].isin(["CN", dx_value])].copy()
    cases  = sub[sub["dx"] == dx_value]
    cns    = sub[sub["dx"] == "CN"]
    sampled = []
    for (site, sex), cell in cases.groupby(["study_site", "sex"]):
        cn_cell = cns[(cns["study_site"] == site) & (cns["sex"] == sex)]
        if len(cn_cell) == 0:
            continue
        sampled.append(cn_cell.sample(n=min(len(cell), len(cn_cell)), random_state=seed))
    matched_cn = pd.concat(sampled) if sampled else cns.iloc[:0]
    out = pd.concat([cases, matched_cn]).reset_index(drop=True)
    out["Condition_type"] = np.where(out["dx"] == dx_value, "Clinical", "Control")
    out["Condition_type"] = pd.Categorical(out["Condition_type"],
                                           categories=["Control", "Clinical"], ordered=False)
    out["age_z"] = (out["age"] - out["age"].mean()) / out["age"].std(ddof=1)
    return out

def cortical(df, pref):
    return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]
def subc(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def diagnose(col, df):
    d = df[[col, "age_z", "sex", "study_site", "Condition_type"]].dropna().rename(columns={col: "y"})
    m = smf.ols(FORM, data=d).fit()
    resid = m.resid
    exog = m.model.exog
    names = m.model.exog_names
    # key coefficients
    kmain = [k for k in m.params.index if "Clinical" in k and ":" not in k][0]
    kint  = [k for k in m.params.index if "Clinical" in k and ":age_z" in k][0]
    ksex  = [k for k in m.params.index if "Clinical" in k and ":" in k and "sex" in k and "age_z" not in k][0]
    # diagnostics
    jb_p = jarque_bera(resid)[1]
    bp_p = het_breuschpagan(resid, exog)[1]
    infl = m.get_influence().cooks_distance[0]
    n = len(d); n_infl = int((infl > 4 / n).sum()); max_cook = float(np.nanmax(infl))
    vifs = {}
    for i, nm in enumerate(names):
        if nm == "Intercept":
            continue
        try:
            vifs[nm] = variance_inflation_factor(exog, i)
        except Exception:
            vifs[nm] = np.nan
    max_vif = np.nanmax(list(vifs.values()))
    vif_age = vifs.get("age_z", np.nan)
    vif_cond = next((v for k, v in vifs.items() if "Clinical" in k and ":" not in k), np.nan)
    return {
        "region": col, "n": n,
        "t_condition": m.tvalues[kmain], "p_condition": m.pvalues[kmain],
        "t_cond_x_sex": m.tvalues[ksex], "p_cond_x_sex": m.pvalues[ksex],
        "t_cond_x_age": m.tvalues[kint], "p_cond_x_age": m.pvalues[kint],
        "resid_JB_p": jb_p, "residual_nonnormal": int(jb_p < 0.05),
        "BreuschPagan_p": bp_p, "heteroscedastic": int(bp_p < 0.05),
        "cook_n_influential": n_infl, "cook_max": max_cook, "cook_flag": int(max_cook > 1.0),
        "max_VIF": max_vif, "VIF_age_z": vif_age, "VIF_condition": vif_cond,
        "high_collinearity": int(max_vif > 10),
    }

all_diag_summary = []
for dxkey, dx_value in DIAGNOSES.items():
    df = build_dx_sample(_full, dx_value)
    print(f"\n=== MODEL DIAGNOSTICS — {dx_value} (n={len(df)}) ===")
    diag_summary = []
    for meas, cols in [("CT", cortical(df, "CT")), ("GMV", cortical(df, "GM")),
                       ("SA", cortical(df, "SA")), ("Subcortical", subc(df))]:
        rows = [diagnose(c, df) for c in cols]
        r = pd.DataFrame(rows)
        r["p_fdr_condition"]  = multipletests(r["p_condition"],  method="fdr_bh")[1]
        r["p_fdr_cond_x_sex"] = multipletests(r["p_cond_x_sex"], method="fdr_bh")[1]
        r["p_fdr_cond_x_age"] = multipletests(r["p_cond_x_age"], method="fdr_bh")[1]
        r.to_csv(os.path.join(OUT, f"diagnostics_{dxkey}_{meas}.csv"), index=False)
        s = {
            "dx": dxkey, "measure": meas, "n_regions": len(r),
            "pct_resid_nonnormal": round(100 * r.residual_nonnormal.mean(), 1),
            "pct_heteroscedastic": round(100 * r.heteroscedastic.mean(), 1),
            "mean_n_influential": round(r.cook_n_influential.mean(), 1),
            "max_cook_over_regions": round(r.cook_max.max(), 3),
            "regions_cook_gt1": int(r.cook_flag.sum()),
            "max_VIF_over_regions": round(r.max_VIF.max(), 2),
            "regions_highVIF": int(r.high_collinearity.sum()),
            "n_sig_condition_FDR": int((r.p_fdr_condition < 0.05).sum()),
            "n_sig_cond_x_sex_FDR": int((r.p_fdr_cond_x_sex < 0.05).sum()),
            "n_sig_cond_x_age_FDR": int((r.p_fdr_cond_x_age < 0.05).sum()),
        }
        diag_summary.append(s)
        all_diag_summary.append(s)
        print(f"{meas:12} nonnormal {s['pct_resid_nonnormal']:5}% | heterosced {s['pct_heteroscedastic']:5}% "
              f"| infl/region {s['mean_n_influential']:4} maxCook {s['max_cook_over_regions']:.2f} "
              f"| maxVIF {s['max_VIF_over_regions']:.1f} | cond FDR {s['n_sig_condition_FDR']:2} "
              f"| cond×sex FDR {s['n_sig_cond_x_sex_FDR']:2} | cond×age FDR {s['n_sig_cond_x_age_FDR']:2}")
    pd.DataFrame(diag_summary).to_csv(os.path.join(OUT, f"diagnostics_summary_{dxkey}.csv"), index=False)
pd.DataFrame(all_diag_summary).to_csv(os.path.join(OUT, "diagnostics_summary.csv"), index=False)

# ---------------- LATERALITY ----------------
def pairs(df, pref, sub=False):
    if sub:
        L = [c for c in df.columns if c.startswith("SubcGM.Left.") and "Vent" not in c and "Cereb" not in c]
        return [(l, l.replace("SubcGM.Left.", "SubcGM.Right.")) for l in L
                if l.replace("SubcGM.Left.", "SubcGM.Right.") in df.columns]
    L = [c for c in df.columns if c.startswith(f"lh.{pref}.")]
    return [(l, l.replace("lh.", "rh.")) for l in L if l.replace("lh.", "rh.") in df.columns]

all_lat_summary = []
for dxkey, dx_value in DIAGNOSES.items():
    df = build_dx_sample(_full, dx_value)
    print(f"\n=== LATERALITY — {dx_value} (n={len(df)}) ===")
    lat_summary = []
    for meas, pref, sub in [("CT", "CT", False), ("GMV", "GM", False),
                            ("SA", "SA", False), ("Subcortical", None, True)]:
        rows = []
        for lcol, rcol in pairs(df, pref, sub):
            d = df[[lcol, rcol, "age_z", "sex", "study_site", "Condition_type"]].dropna().copy()
            denom = (d[lcol] + d[rcol]) / 2.0
            d = d[denom != 0]
            d["y"] = (d[lcol] - d[rcol]) / ((d[lcol] + d[rcol]) / 2.0)
            m = smf.ols(FORM, data=d).fit()
            k = [k for k in m.params.index if "Clinical" in k and ":" not in k][0]
            rows.append({"pair": lcol.replace("lh.", "").replace("SubcGM.Left.", ""),
                         "beta_AI_condition": m.params[k], "t": m.tvalues[k], "p": m.pvalues[k]})
        r = pd.DataFrame(rows)
        r["p_fdr"] = multipletests(r["p"], method="fdr_bh")[1]
        r["sig"] = (r["p_fdr"] < 0.05).astype(int)
        r.to_csv(os.path.join(OUT, f"laterality_{dxkey}_{meas}.csv"), index=False)
        nsig = int(r.sig.sum())
        entry = {"dx": dxkey, "measure": meas, "n_pairs": len(r),
                 "n_sig_asym_condition_FDR": nsig,
                 "min_p_fdr": round(float(r.p_fdr.min()), 4)}
        lat_summary.append(entry); all_lat_summary.append(entry)
        top = r.sort_values("p").head(3)[["pair", "t", "p_fdr"]].to_dict("records")
        print(f"{meas:12} pairs={len(r):2} | asym altered by condition (FDR): {nsig:2} | top: "
              + ", ".join(f"{t['pair']}(t={t['t']:.2f},pFDR={t['p_fdr']:.3f})" for t in top))
    pd.DataFrame(lat_summary).to_csv(os.path.join(OUT, f"laterality_summary_{dxkey}.csv"), index=False)
pd.DataFrame(all_lat_summary).to_csv(os.path.join(OUT, "laterality_summary.csv"), index=False)
print("\nSaved ->", OUT)

#!/usr/bin/env python3
"""
Sex x diagnosis interaction — SCZ, SAD and BD analysed separately (reviewer analysis)
===================================================================================
Directly tests whether the case-control effect DIFFERS by sex, per region, by
adding a diagnosis x sex interaction to the SAME linear model each analysis uses:

  * SCZ (dataset.csv, restricted to SCZ sites + CN), mirroring sex_maps.py:
  * SAD (dataset.csv, restricted to Schizoaffective sites + CN):
        region ~ age_z + C(sex) + C(study_site)
                 + C(Condition_type, Treatment(reference="Control"))
                 + C(Condition_type, Treatment(reference="Control")):C(sex)   # sex x dx
                 + C(Condition_type, Treatment(reference="Control")):age_z     # age x dx

  * BD (loaded via load_bd_matched()), mirroring bd_casecontrol.py:
        region ~ age_z + C(sex) + C(study_site)
                 + C(dx, Treatment(reference="CN"))
                 + C(dx, Treatment(reference="CN")):C(sex)                     # sex x dx
                 + C(dx, Treatment(reference="CN")):age_z                      # age x dx

The coefficient of interest is diagnosis[case] : sex[Male] — the sex difference in
the case-vs-control effect. FDR (BH) across regions, per measure. Matched samples
only (the pooled dataset.csv is not analysed).

Outputs -> ../results/qc_sensitivity/interaction/
"""
import os, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from data_utils import load_bd_matched, load_scz_dataset
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "06_qc_sensitivity", "interaction")
os.makedirs(OUT, exist_ok=True)

MEAS = {"CT": "CT", "GMV": "GM", "SA": "SA"}   # cortical prefixes

def regions(df, prefix):
    return [c for c in df.columns if c.startswith(f"lh.{prefix}.") or c.startswith(f"rh.{prefix}.")]
def subc_regions(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def fit_interaction(df, cols, cond_expr, case_token):
    """Fit the (SCZ- or BD-style) model per region; return diagnosis x sex interaction."""
    form = (f'y ~ age_z + C(sex) + C(study_site) + {cond_expr}'
            f' + {cond_expr}:C(sex) + {cond_expr}:age_z')
    rows = []
    for col in cols:
        d = df[[col, "age_z", "sex", "study_site", "cond"]].dropna().rename(columns={col: "y"})
        try:
            m = smf.ols(form, data=d).fit()
            key = [k for k in m.params.index
                   if case_token in k and ":" in k and "sex" in k and "age_z" not in k]
            if not key:
                continue
            k = key[0]
            rows.append({"region": col, "beta_interaction": m.params[k],
                         "t_interaction": m.tvalues[k], "p_interaction": m.pvalues[k],
                         "n": int(m.nobs)})
        except Exception:
            rows.append({"region": col, "beta_interaction": np.nan,
                         "t_interaction": np.nan, "p_interaction": np.nan, "n": np.nan})
    r = pd.DataFrame(rows)
    ok = r["p_interaction"].notna()
    r["p_fdr"] = np.nan
    if ok.sum():
        r.loc[ok, "p_fdr"] = multipletests(r.loc[ok, "p_interaction"], method="fdr_bh")[1]
    r["sig_fdr"] = (r["p_fdr"] < 0.05).astype(int)
    return r

# dataset -> (file, condition column, reference level, case-coefficient token)
# (name, file, dx_filter, condcol, ref, case_token)
# dx_filter: str -> filter dataset.csv to that dx + CN sites and derive Condition_type
#            None -> use the file as-is (BD matched sample)
CONFIGS = [
    ("scz_matched", "_scz",  "SCZ",                     "Condition_type", "Control", "Clinical"),
    ("sad_matched", "_scz",  "Schizoaffective","Condition_type", "Control", "Clinical"),
    ("BD_matched",  "_bd",   None,                      "dx",             "CN",      "BD"),
]

summary = []
for dsname, fname, dx_filter, condcol, ref, case_token in CONFIGS:
    _loaders06a = {"_scz": load_scz_dataset(DATA), "_bd": load_bd_matched(DATA)}
    df = _loaders06a[fname]
    if dx_filter is not None:
        # site-restrict then site×sex-match CN to cases (same approach as BD)
        sites  = df.loc[df["dx"] == dx_filter, "study_site"].unique()
        _sub   = df[df["study_site"].isin(sites) & df["dx"].isin(["CN", dx_filter])].copy()
        _cases = _sub[_sub["dx"] == dx_filter]
        _cns   = _sub[_sub["dx"] == "CN"]
        _samp  = []
        for (_st, _sx), _cell in _cases.groupby(["study_site", "sex"]):
            _cn = _cns[(_cns["study_site"] == _st) & (_cns["sex"] == _sx)]
            if len(_cn) == 0:
                continue
            _samp.append(_cn.sample(n=min(len(_cell), len(_cn)), random_state=42))
        _mcn = pd.concat(_samp) if _samp else _cns.iloc[:0]
        df = pd.concat([_cases, _mcn]).reset_index(drop=True)
        df["Condition_type"] = np.where(df["dx"] == dx_filter, "Clinical", "Control")
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)
    df["cond"] = df[condcol]                                   # generic condition column
    cond_expr = f'C(cond, Treatment(reference="{ref}"))'
    blocks = {m: regions(df, p) for m, p in MEAS.items()}
    blocks["Subcortical"] = subc_regions(df)
    for meas, cols in blocks.items():
        res = fit_interaction(df, cols, cond_expr, case_token)
        res.to_csv(os.path.join(OUT, f"interaction_{dsname}_{meas}.csv"), index=False)
        nsig = int(res["sig_fdr"].sum())
        nunc = int((res["p_interaction"] < 0.05).sum())
        summary.append({"dataset": dsname, "model": f"{case_token} vs {ref}",
                        "measure": meas, "n_regions": len(cols),
                        "n_sig_uncorrected_p05": nunc, "n_sig_FDR": nsig,
                        "min_p_fdr": np.nanmin(res["p_fdr"]) if res["p_fdr"].notna().any() else np.nan})
        print(f"{dsname:12} {meas:12} ({case_token} vs {ref})  regions={len(cols):3} "
              f"sig(sex*dx) uncorrected p<.05: {nunc:3} | FDR<.05: {nsig:3}")

pd.DataFrame(summary).to_csv(os.path.join(OUT, "interaction_summary.csv"), index=False)
print("\nSaved ->", OUT)

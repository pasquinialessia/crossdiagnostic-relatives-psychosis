# %%
"""
Sliding-window peak-age permutation — SCZ and Schizoaffective separately.
Each diagnosis is run against its own matched controls (sites that recruited that dx).
Symmetric with peak_age_bd.py.

Inputs
------
  raw     : ../data/dataset.csv   (contains SCZ + Schizoaffective + CN)
  summary : per-diagnosis bootstrap sliding-window Cohen's d files, named
              {DX_PREFIX}-<Group>-regional-swa.csv
            If a file is absent the script falls back to point-estimate only.

Outputs -> ../results/swa_peak_age/{dxkey}/
  results_<measure>_<group>.csv  and  peak_age_panels.png
"""
import os, importlib, tempfile, swa_lib
importlib.reload(swa_lib)
import swa_lib as swp
import pandas as pd
import numpy as np
from data_utils import load_scz_dataset

RAW     = None  # loaded via data_utils.load_scz_dataset
DATA    = "../data"

DIAGNOSES = {
    "scz": {
        "dx":           "SCZ",
        "label":        "Schizophrenia",
        "summary_pfx":  "SCZ",          # bootstrap files: SCZ-regional-swa.csv, etc.
    },
    "sad": {
        "dx":           "Schizoaffective",
        "label":        "Schizoaffective disorder",
        "summary_pfx":  "SAD",
    },
}

MEASURES = ("CT", "SA", "GM", "subcortical")
SEXES    = (None, "Male", "Female")          # None -> whole sample ("All")
N_PERM   = 1000


def build_diagnosis_sample(raw_df, dx_value, seed=42):
    """Return dx_value + site×sex-matched CN (same approach as BD)."""
    sites = raw_df.loc[raw_df["dx"] == dx_value, "study_site"].unique()
    sub   = raw_df[raw_df["study_site"].isin(sites) & raw_df["dx"].isin(["CN", dx_value])].copy()
    cases = sub[sub["dx"] == dx_value]
    cns   = sub[sub["dx"] == "CN"]
    sampled = []
    for (site, sex), cell in cases.groupby(["study_site", "sex"]):
        cn_cell = cns[(cns["study_site"] == site) & (cns["sex"] == sex)]
        if len(cn_cell) == 0:
            continue
        sampled.append(cn_cell.sample(n=min(len(cell), len(cn_cell)), random_state=seed))
    matched_cn = pd.concat(sampled) if sampled else cns.iloc[:0]
    out = pd.concat([cases, matched_cn]).reset_index(drop=True)
    out["Condition_type"] = np.where(out["dx"] == dx_value, "Clinical", "Control")
    return out


def _summary_for(group, pfx):
    """Resolve bootstrap summary path for (group, prefix); None if missing."""
    fname = f"{pfx}-regional-swa.csv" if group == "All" else f"{pfx}-{group}-regional-swa.csv"
    path  = os.path.join(DATA, fname)
    if os.path.exists(path):
        return path
    print(f"  [summary] no bootstrap file '{path}' -> point-estimate only")
    return None


# %%
_full = load_scz_dataset()

for dxkey, cfg in DIAGNOSES.items():
    dx_value = cfg["dx"]
    pfx      = cfg["summary_pfx"]
    label    = cfg["label"]
    out_dir  = os.path.join("..", "results", "03_swa_peak_age", dxkey)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Diagnosis : {label}  ({dxkey})")
    print(f"  Output    : {out_dir}")
    print(f"{'='*60}")

    # Filter to this dx + matched controls
    sub_df = build_diagnosis_sample(_full, dx_value)

    # Write temp CSV so run_pipeline (which takes a file path) can read it
    with tempfile.NamedTemporaryFile(
            suffix=f"_{dxkey}.csv", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
        sub_df.to_csv(tmp, index=False)

    try:
        raw_cols = sub_df.columns
        results = {}
        for meas in MEASURES:
            rc = swp._region_cols_for(pd.DataFrame(columns=raw_cols), meas)
            if not rc:
                print(f"[skip] no columns for measure '{meas}'"); continue
            for sx in SEXES:
                gname = sx or "All"
                print(f"\n  --- {dxkey} | {meas} | {gname} ---")
                res, _ = swp.run_pipeline(
                    tmp_path, _summary_for(gname, pfx),
                    measure_prefix=meas,
                    region_cols=rc, sex=sx, n_perm=N_PERM, n_jobs=-1,
                    do_cortical_plot=False,
                    summary_d_col="auto",
                    results_csv=os.path.join(
                        out_dir, f"results_{meas}_{gname}.csv"))
                results[(meas, gname)] = res

        # Build the 4×3 peak-age panel figure
        swp.make_peak_age_panels(
            results,
            out_png=os.path.join(out_dir, "peak_age_panels.png"),
            outline_width=2, tick_step=2)

    finally:
        os.unlink(tmp_path)

print("\nDone — per-diagnosis peak-age analyses complete.")

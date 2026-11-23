# %%
"""
Sliding-window peak-age permutation — BIPOLAR (BD vs matched CN).
Mirror of peak_age_ssd.py (SCZ + schizoaffective), pointed at the BD matched sample.

Inputs
------
  raw     : ../data/dataset.csv (BD + matched CN, subset built by load_bd_matched())
  summary : per-sex bootstrap sliding-window Cohen's d files, same format as the
            clinical ones, named  BD-<Group>-regional-swa.csv  (user-provided).
            The d column is auto-detected ('Mean_Cohen_d_*'), so BD-named columns
            (e.g. Mean_Cohen_d_BD) work without editing the module config.
            A group whose summary file is absent still runs on the point-estimate
            peak (the panels use the point-estimate anyway); only p_boot is skipped.

Outputs -> ../results/swa_peak_age/bd/
  results_<measure>_<group>.csv  and  bd_peak_age_panels.png
"""
import os, importlib, swa_lib
importlib.reload(swa_lib)
import swa_lib as swp

RAW     = None  # written to a temp CSV below (run_pipeline takes a file path)
OUT_DIR = "../results/swa_peak_age/bd"
os.makedirs(OUT_DIR, exist_ok=True)

# Per-group bootstrap summary files (edit names here if yours differ).
# Set a value to None to force point-estimate-only for that group.
SUMMARY_FILES = {
    "All":    "../data/BD-regional-swa.csv",
    "Male":   "../data/BD-Male-regional-swa.csv",
    "Female": "../data/BD-Female-regional-swa.csv",
}

MEASURES = ("CT", "SA", "GM", "subcortical")
SEXES    = (None, "Male", "Female")          # None -> whole sample ("All")
N_PERM   = 1000


def _summary_for(group):
    """Resolve the summary path for a group name; None if the file is missing."""
    path = SUMMARY_FILES.get(group)
    if path and os.path.exists(path):
        return path
    print(f"  [summary] no bootstrap file for '{group}' "
          f"({path or 'unset'}) -> point-estimate only")
    return None


# %%
# 1. run every measure x group combination
import tempfile, pandas as pd
from data_utils import load_bd_matched
bd_df    = load_bd_matched()
raw_cols = bd_df.columns

# Write temp CSV so run_pipeline (which takes a file path) can read it
with tempfile.NamedTemporaryFile(suffix="_bd.csv", delete=False, mode="w") as tmp:
    RAW = tmp.name
    bd_df.to_csv(tmp, index=False)

try:
    results = {}
    for meas in MEASURES:
        rc = swp._region_cols_for(pd.DataFrame(columns=raw_cols), meas)
        if not rc:
            print(f"[skip] no columns for measure '{meas}'"); continue
        for sx in SEXES:
            gname = sx or "All"
            print(f"\n===== BD | {meas} | {gname} =====")
            res, _ = swp.run_pipeline(
                RAW, _summary_for(gname), measure_prefix=meas,
                region_cols=rc, sex=sx, n_perm=N_PERM, n_jobs=-1,
                do_cortical_plot=False,
                summary_d_col="auto",                    # BD-named d column
                results_csv=os.path.join(OUT_DIR, f"results_{meas}_{gname}.csv"))
            results[(meas, gname)] = res
finally:
    os.unlink(RAW)

# %%
# 2. build the 4x3 peak-age panel figure (point-estimate significance)
swp.make_peak_age_panels(
    results, out_png=os.path.join(OUT_DIR, "bd_peak_age_panels.png"),
    outline_width=2, tick_step=2)

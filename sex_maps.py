#!/usr/bin/env python3
"""
ENIGMA — Male vs Female map comparison, per diagnosis (cortical spin + subcortical shuffle)
==========================================================================================
Each diagnosis is analysed on its own against controls: schizophrenia (SCZ) vs CN
and schizoaffective disorder (SAD) vs CN. There is no pooled group — the two
diagnoses are never merged. For each diagnosis the sample is drawn from
dataset.csv by keeping controls plus that diagnosis (dx == 'SCZ' or
dx == 'Schizoaffective'), restricted to the sites that recruited that
diagnosis so cases and controls stay site-matched. Results are written to a
per-diagnosis sub-folder (results/casecontrol/{scz,sad}_lm_by_sex/).

Analysis
--------
For each measure and each sex group, an OLS model is fit per region:

    region ~ age_z
             + C(study_site)
             + C(Condition_type, Treatment(reference="Control"))
             + C(Condition_type, Treatment(reference="Control")):age_z

`age_z` is z-scored within each sex group. The t-statistic of the main
Condition_type[T.Clinical] term gives one brain map per sex. The male and
female maps are then compared:
    * Cortical (CT, GMV, SA): spin_test  (spatial-autocorrelation-preserving rotation)
    * Subcortical volumes:    shuf_test  (label shuffling; spheres N/A subcortically)

Reproducibility (this script maximises what is written to disk)
--------------------------------------------------------------
Everything needed to reproduce/re-plot every number in the paper is saved under
RESULTS_DIR:
    run_manifest.json        — timestamp, seed, n_rot, input file + SHA-256,
                               sample sizes, resolved package versions
    environment.txt          — key package versions (pip-freeze style)
    run_log.txt              — full console log of the run
    tables/                  — per-region stats (beta, SE, t, p, 95% CI, n, df) for
                               every measure x sex, plus the master combined table
    tables/summary_all.csv   — one row per measure: r, p, null mean/sd, n_rot, test type
    t_maps/                  — the exact t-statistic maps fed into each test
    null_distributions/      — every permutation null distribution (.csv + .npz)
    figures/                 — null-distribution + scatter plots (PNG 300 dpi + PDF)

Seeding policy
--------------
A single BASE_SEED (default 42) is set globally at start, and re-applied
immediately before every permutation test so each test is independently
reproducible regardless of execution order.

Usage
-----
    python scz_lm_by_sex.py                 # uses defaults below
    python scz_lm_by_sex.py --n-rot 10000   # more rotations for the final run
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import argparse
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from enigmatoolbox.datasets import load_fc
from enigmatoolbox.permutation_testing import spin_test, shuf_test
from data_utils import load_scz_dataset


# =============================================================================
# CONFIG
# =============================================================================
DEFAULT_DATA_PATH = None  # loaded via data_utils.load_scz_dataset
DEFAULT_RESULTS_ROOT = os.path.join("..", "results", "02_casecontrol")
BASE_SEED = 42
DEFAULT_N_ROT = 1000

# Diagnoses analysed separately (never pooled). key -> (dx value in dataset.csv, folder)
DIAGNOSES = {
    "scz": {"dx": "SCZ",                     "label": "Schizophrenia",         "dir": "scz_lm_by_sex"},
    "sad": {"dx": "Schizoaffective", "label": "Schizoaffective disorder", "dir": "sza_lm_by_sex"},
}


def build_diagnosis_sample(raw: pd.DataFrame, dx_value: str, seed: int = 42) -> pd.DataFrame:
    """Controls + one diagnosis, site-matched then site×sex-matched.
    CN are sampled to match case count per site×sex cell (same approach as BD)."""
    sites = raw.loc[raw["dx"] == dx_value, "study_site"].unique()
    sub   = raw[raw["study_site"].isin(sites) & raw["dx"].isin(["CN", dx_value])].copy()
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

# Cortical measures: ENIGMA fc label -> dataframe column prefix
MEASURE_PREFIX = {"CT": "CT", "GMV": "GM", "SA": "SA"}
COLORS = {"CT": "#6B3FA0", "GMV": "#A8221C", "SA": "#324F7D", "Subcortical": "#2A7F62"}

# Model spec recorded verbatim in the manifest
MODEL_FORMULA = (
    '{region} ~ age_z + C(study_site)'
    ' + C(Condition_type, Treatment(reference="Control"))'
    ' + C(Condition_type, Treatment(reference="Control")):age_z'
)


# =============================================================================
# LOGGING
# =============================================================================
log = logging.getLogger("scz_lm_by_sex")


def setup_logging(results_dir: str) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(os.path.join(results_dir, "run_log.txt"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    log.addHandler(fh)


# =============================================================================
# REPRODUCIBILITY HELPERS
# =============================================================================
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict:
    import importlib
    pkgs = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib",
            "sklearn", "nibabel", "vtk", "enigmatoolbox"]
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in pkgs:
        try:
            out[p] = importlib.import_module(p).__version__
        except Exception:
            out[p] = "not found"
    return out


def write_environment(results_dir: str, versions: dict) -> None:
    with open(os.path.join(results_dir, "environment.txt"), "w") as f:
        for k, v in versions.items():
            f.write(f"{k}=={v}\n")


# =============================================================================
# CORE MODEL
# =============================================================================
def fit_lm_per_region(data: pd.DataFrame, regions: list) -> pd.DataFrame:
    """OLS per region; returns per-region beta, SE, t, p, 95% CI, n, df for the
    main Condition_type[T.Clinical] effect. Identical model to the source notebooks,
    with extra reproducibility columns (SE, CI, df_resid)."""
    records = []
    for col in regions:
        safe = col.replace(".", "_")
        tmp = data[[col, "age", "study_site", "Condition_type"]].copy().dropna()
        tmp["age_z"] = (tmp["age"] - tmp["age"].mean()) / tmp["age"].std(ddof=1)
        tmp = tmp.rename(columns={col: safe})
        model = smf.ols(MODEL_FORMULA.format(region=safe), data=tmp).fit()
        coef_key = [k for k in model.params.index if "Clinical" in k and ":" not in k][0]
        ci = model.conf_int().loc[coef_key]
        records.append({
            "col":       col,
            "beta":      model.params[coef_key],
            "se":        model.bse[coef_key],
            "t_val":     model.tvalues[coef_key],
            "p_unc":     model.pvalues[coef_key],
            "ci_low":    ci[0],
            "ci_high":   ci[1],
            "n":         int(model.nobs),
            "df_resid":  int(model.df_resid),
        })
    return pd.DataFrame(records).set_index("col")


def region_table(res_male, res_female, cols, measure, labels=None) -> pd.DataFrame:
    """Combine male & female per-region stats into one tidy table with FDR."""
    out = pd.DataFrame({
        "measure":      measure,
        "region":       cols,
        "beta_male":    res_male.loc[cols, "beta"].values,
        "se_male":      res_male.loc[cols, "se"].values,
        "t_male":       res_male.loc[cols, "t_val"].values,
        "p_unc_male":   res_male.loc[cols, "p_unc"].values,
        "ci_low_male":  res_male.loc[cols, "ci_low"].values,
        "ci_high_male": res_male.loc[cols, "ci_high"].values,
        "n_male":       res_male.loc[cols, "n"].values,
        "beta_female":    res_female.loc[cols, "beta"].values,
        "se_female":      res_female.loc[cols, "se"].values,
        "t_female":       res_female.loc[cols, "t_val"].values,
        "p_unc_female":   res_female.loc[cols, "p_unc"].values,
        "ci_low_female":  res_female.loc[cols, "ci_low"].values,
        "ci_high_female": res_female.loc[cols, "ci_high"].values,
        "n_female":       res_female.loc[cols, "n"].values,
    })
    if labels is not None:
        out.insert(2, "enigma_label", list(labels))
    _, out["p_fdr_male"], _, _ = multipletests(out["p_unc_male"], method="fdr_bh")
    _, out["p_fdr_female"], _, _ = multipletests(out["p_unc_female"], method="fdr_bh")
    return out


# =============================================================================
# PLOTS
# =============================================================================
def save_fig(fig, results_dir, name):
    fdir = os.path.join(results_dir, "figures")
    fig.savefig(os.path.join(fdir, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(fdir, name + ".pdf"), bbox_inches="tight")  # vector for paper
    plt.close(fig)


def plot_null(results_dir, name, null_dist, r_emp, p_val, measure, test_label, color):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_dist, bins=50, density=True, color=color,
            edgecolor="white", linewidth=0.4, alpha=0.88)
    ax.axvline(r_emp, lw=2, ls="--", color="k", dashes=(2, 3),
               label=f"$\\rho$ = {r_emp:+.3f}\n$p_{{{test_label}}}$ = {p_val:.3f}")
    ax.set_xlabel(f"Null correlations ({test_label})", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{measure} | Condition effect: Male vs Female map", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    save_fig(fig, results_dir, name)


def plot_scatter(results_dir, name, t_male, t_female, cols, r_emp, p_val,
                 measure, test_label, color, annotate="all"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t_male, t_female, c=color, alpha=0.78,
               edgecolors="white", linewidths=0.4, s=70)
    for col, tm, tf in zip(cols, t_male, t_female):
        if annotate == "all" or (
            abs(tm) > np.percentile(np.abs(t_male), 90) or
            abs(tf) > np.percentile(np.abs(t_female), 90)
        ):
            short = col.replace("SubcGM.", "").split(".", 2)[-1].replace(".", " ")
            ax.annotate(short, (tm, tf), fontsize=6.5, alpha=0.85,
                        xytext=(4, 2), textcoords="offset points")
    lims = [min(t_male.min(), t_female.min()) - 0.4,
            max(t_male.max(), t_female.max()) + 0.4]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="identity")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("t-statistic – Males (Condition effect)", fontsize=10)
    ax.set_ylabel("t-statistic – Females (Condition effect)", fontsize=10)
    ax.set_title(f"{measure} – Males vs Females\n"
                 f"$\\rho$ = {r_emp:+.3f},  $p_{{{test_label}}}$ = {p_val:.3f}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, results_dir, name)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA_PATH)
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--diagnosis", choices=["scz", "sad", "both"], default="both",
                    help="which diagnosis to run against controls (default: both, separately)")
    ap.add_argument("--n-rot", type=int, default=DEFAULT_N_ROT)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    data_sha = sha256_of(args.data)
    raw = load_scz_dataset(os.path.join("..", "data")) if args.data is None else pd.read_csv(args.data)
    versions = package_versions()
    selected = ["scz", "sad"] if args.diagnosis == "both" else [args.diagnosis]
    for dxkey in selected:
        run_one(raw, dxkey, DIAGNOSES[dxkey], args, versions, data_sha)


def run_one(raw, dxkey, meta, args, versions, data_sha):
    # ---- output scaffolding --------------------------------------------------
    rd = os.path.join(args.results_root, meta["dir"])
    for sub in ("", "tables", "figures", "t_maps", "null_distributions"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)
    setup_logging(rd)

    np.random.seed(args.seed)
    write_environment(rd, versions)

    log.info("=" * 70)
    log.info(f"ENIGMA {meta['label']} vs controls — Male vs Female map comparison")
    log.info("=" * 70)
    log.info(f"data            : {args.data}")
    log.info(f"diagnosis       : {meta['label']} (dx == '{meta['dx']}')")
    log.info(f"results_dir     : {rd}")
    log.info(f"base seed       : {args.seed}")
    log.info(f"n_rot           : {args.n_rot}")

    # ---- load (controls + this diagnosis only, site-matched) -----------------
    df = build_diagnosis_sample(raw, meta["dx"])
    df["Condition_type"] = pd.Categorical(
        df["Condition_type"], categories=["Control", "Clinical"], ordered=False)
    males = df[df["sex"] == "Male"].reset_index(drop=True)
    females = df[df["sex"] == "Female"].reset_index(drop=True)

    n_info = {
        "n_total": int(df.shape[0]),
        "n_male": int(len(males)),
        "n_female": int(len(females)),
        "n_male_clinical": int((males["Condition_type"] == "Clinical").sum()),
        "n_male_control": int((males["Condition_type"] == "Control").sum()),
        "n_female_clinical": int((females["Condition_type"] == "Clinical").sum()),
        "n_female_control": int((females["Condition_type"] == "Control").sum()),
    }
    log.info(f"\nLoaded {n_info['n_total']} subjects")
    log.info(f"  Males  : n={n_info['n_male']} "
             f"(Clinical={n_info['n_male_clinical']}, Control={n_info['n_male_control']})")
    log.info(f"  Females: n={n_info['n_female']} "
             f"(Clinical={n_info['n_female_clinical']}, Control={n_info['n_female_control']})")

    # ---- region sets ---------------------------------------------------------
    _, fc_ctx_labels, _, _ = load_fc()

    def label_to_col(label, prefix):
        hemi, region = label.split("_", 1)
        h = "lh" if hemi == "L" else "rh"
        return f"{h}.{prefix}.{region}"

    cortical = {m: [label_to_col(l, MEASURE_PREFIX[m]) for l in fc_ctx_labels]
                for m in MEASURE_PREFIX}
    for m, cols in cortical.items():
        missing = [c for c in cols if c not in df.columns]
        assert not missing, f"[{m}] missing columns: {missing}"

    all_sctx = [c for c in df.columns if c.startswith("SubcGM.")]
    subc_cols = [c for c in all_sctx if "Vent" not in c and "Cereb" not in c]
    log.info(f"\nCortical regions per measure: {len(fc_ctx_labels)}")
    log.info(f"Subcortical regions (ventricles excluded): {len(subc_cols)}")

    master_rows = []       # per-region across every measure
    summary_rows = []      # one row per test
    tmaps = {}             # measure -> DataFrame(region, t_male, t_female)

    def run_test(measure, cols, kind, labels=None):
        log.info("\n" + "=" * 60)
        log.info(f"  {measure}  ({kind})")
        log.info("=" * 60)
        res_m = fit_lm_per_region(males, cols)
        res_f = fit_lm_per_region(females, cols)
        t_m = res_m.loc[cols, "t_val"].values
        t_f = res_f.loc[cols, "t_val"].values
        log.info(f"  t range Males  : [{t_m.min():.3f}, {t_m.max():.3f}]")
        log.info(f"  t range Females: [{t_f.min():.3f}, {t_f.max():.3f}]")

        # per-region table
        tbl = region_table(res_m, res_f, cols, measure, labels)
        tbl.to_csv(os.path.join(rd, "tables", f"regions_{measure}.csv"), index=False)
        master_rows.append(tbl)

        # t-maps
        tmap = pd.DataFrame({"region": cols, "t_male": t_m, "t_female": t_f})
        if labels is not None:
            tmap.insert(1, "enigma_label", list(labels))
        tmap.to_csv(os.path.join(rd, "t_maps", f"tmap_{measure}.csv"), index=False)
        tmaps[measure] = tmap

        # permutation test (seed reset -> independent reproducibility)
        np.random.seed(args.seed)
        if kind == "spin":
            p_val, null = spin_test(t_m, t_f, surface_name="fsa5",
                                    parcellation_name="aparc", type="spearman",
                                    n_rot=args.n_rot, null_dist=True)
            test_label = "spin"
        else:
            p_val, null = shuf_test(t_m, t_f, n_rot=args.n_rot,
                                    type="spearman", null_dist=True)
            test_label = "shuf"
        null = np.asarray(null, dtype=float)
        r_emp = float(stats.spearmanr(t_m, t_f)[0])
        log.info(f"  r = {r_emp:+.3f}  |  p_{test_label} = {p_val:.4f}")

        # save null distribution
        np.savez_compressed(
            os.path.join(rd, "null_distributions", f"null_{measure}.npz"),
            null=null, r_emp=r_emp, p_val=p_val)
        pd.DataFrame({f"null_r_{measure}": null}).to_csv(
            os.path.join(rd, "null_distributions", f"null_{measure}.csv"), index=False)

        # figures
        color = COLORS.get(measure, "#444444")
        plot_null(rd, f"null_{measure}", null, r_emp, p_val, measure, test_label, color)
        plot_scatter(rd, f"scatter_{measure}", t_m, t_f, cols, r_emp, p_val,
                     measure, test_label, color,
                     annotate="all" if kind == "shuf" else "extremes")

        summary_rows.append({
            "measure": measure, "test": test_label, "n_regions": len(cols),
            "r_pearson": r_emp, "p_value": float(p_val), "n_rot": args.n_rot,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p2.5": float(np.percentile(null, 2.5)),
            "null_p97.5": float(np.percentile(null, 97.5)),
        })

    # cortical spin tests
    for m in ("CT", "GMV", "SA"):
        run_test(m, cortical[m], "spin", labels=fc_ctx_labels)
    # subcortical shuffle test
    run_test("Subcortical", subc_cols, "shuf")

    # ---- combined tables -----------------------------------------------------
    master = pd.concat(master_rows, ignore_index=True)
    master.to_csv(os.path.join(rd, "tables", "regions_all.csv"), index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(rd, "tables", "summary_all.csv"), index=False)
    log.info("\n" + "=" * 60)
    log.info("  SUMMARY (all tests)")
    log.info("=" * 60)
    log.info("\n" + summary.to_string(index=False))

    # ---- manifest ------------------------------------------------------------
    manifest = {
        "script": os.path.basename(__file__),
        "diagnosis": meta["label"],
        "diagnosis_dx_value": meta["dx"],
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": args.seed,
        "n_rot": args.n_rot,
        "model_formula": MODEL_FORMULA,
        "fdr_method": "fdr_bh",
        "cortical_test": "spin_test (surface=fsa5, parcellation=aparc, pearson)",
        "subcortical_test": "shuf_test (pearson; ventricles excluded)",
        "seeding_policy": "np.random.seed(base_seed) reset before every permutation test",
        "input_data": os.path.abspath(args.data),
        "input_sha256": data_sha,
        "sample_sizes": n_info,
        "n_cortical_regions": len(fc_ctx_labels),
        "n_subcortical_regions": len(subc_cols),
        "package_versions": versions,
        "outputs": {
            "tables": ["regions_<measure>.csv", "regions_all.csv", "summary_all.csv"],
            "t_maps": ["tmap_<measure>.csv"],
            "null_distributions": ["null_<measure>.npz", "null_<measure>.csv"],
            "figures": ["null_<measure>.{png,pdf}", "scatter_<measure>.{png,pdf}"],
        },
    }
    with open(os.path.join(rd, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"\nAll outputs written to: {rd}/")
    log.info("Manifest: run_manifest.json  |  Log: run_log.txt  |  Env: environment.txt")


if __name__ == "__main__":
    main()

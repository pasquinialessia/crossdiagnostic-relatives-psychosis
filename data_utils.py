"""
data_utils.py — Dataset loading helpers for cross-sectional-psychosis analyses.

Each function loads the relevant raw file(s) and applies the subsample or
matching logic that used to live in the standalone prep scripts (01a2, 01b,
01c, 01d), eliminating the need to pre-generate and store intermediate CSVs.

Raw input files expected in data/ (relative to code/: ../data/):
  dataset.csv                 — main analysis file (provided with the release)
  bsnip_relatives_dataset.csv — BSNIP probands / relatives / controls

Usage
-----
    from data_utils import load_scz_dataset, load_bd_matched
    from data_utils import load_crossdx_matched, load_bsnip_relatives

All functions accept an optional data_dir argument (default "../data").
All matching uses SEED=42 for reproducibility.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data")


def load_scz_dataset(data_dir=None):
    """
    Load the SCZ/SAD case-control sample (age 18-50 already applied).

    Reads dataset.csv (the main QC'd sample provided with the release)
    and returns the SSD arm: SCZ + Schizoaffective + CN rows.
    The 18-50 age filter is already applied in dataset.csv.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing dataset.csv. Defaults to ../data relative
        to this file.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = data_dir or _DEFAULT_DATA
    df = pd.read_csv(os.path.join(data_dir, "dataset.csv"))
    out = df[df["dx"].isin(["SCZ", "Schizoaffective", "CN"])].reset_index(drop=True)
    return out



def build_diagnosis_sample(raw_df, dx_value, seed=SEED):
    """
    Filter a loaded DataFrame to one diagnosis + site×sex-matched CN.

    Restricts to sites that recruited *dx_value*, then samples CN within each
    study_site × sex cell to match the clinical case count (same approach used
    for BD in load_bd_matched). Adds a Condition_type column (Clinical/Control).

    Parameters
    ----------
    raw_df : pd.DataFrame
        Full dataset loaded by one of the load_* helpers (e.g. load_scz_dataset()).
    dx_value : str
        Value in the dx column, e.g. "SCZ" or "Schizoaffective".
    seed : int
        Random state for reproducibility (default SEED=42).

    Returns
    -------
    pd.DataFrame
    """
    sites  = raw_df.loc[raw_df["dx"] == dx_value, "study_site"].unique()
    sub    = raw_df[raw_df["study_site"].isin(sites) & raw_df["dx"].isin(["CN", dx_value])].copy()
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
    return out

def load_bd_matched(data_dir=None):
    """
    Build the BD site × sex-matched case-control sample on the fly.

    Controls (CN) are matched to BD cases within each study_site × sex cell;
    cells with fewer CN than BD cases keep all available CN. Matching is
    deterministic (random_state=SEED). Replaces the pre-generated
    site_sex_matched_dataset.csv (not distributed; matching is recomputed here)

    Parameters
    ----------
    data_dir : str, optional
        Directory containing dataset.csv. Defaults to ../data.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = data_dir or _DEFAULT_DATA
    df = pd.read_csv(os.path.join(data_dir, "dataset.csv")).copy()
    df = df[df["dx"].isin(["CN", "BD"])].copy()
    bd = df[df["dx"] == "BD"]
    cn = df[df["dx"] == "CN"]
    sampled = []
    for (site, sex), bd_cell in bd.groupby(["study_site", "sex"]):
        cn_cell = cn[(cn["study_site"] == site) & (cn["sex"] == sex)]
        if len(cn_cell) == 0:
            continue
        sampled.append(
            cn_cell.sample(n=min(len(bd_cell), len(cn_cell)), random_state=SEED)
        )
    matched_cn = pd.concat(sampled) if sampled else cn.iloc[:0]
    return pd.concat([bd, matched_cn]).reset_index(drop=True)


def load_crossdx_matched(data_dir=None):
    """
    Build the cross-diagnostic matched dataset (CN matched to full clinical group).

    Controls (CN) are matched to the combined clinical group (BD + SCZ +
    schizoaffective) by study_site × sex: within each cell, keep as many CN
    as there are clinical cases (or all available CN if fewer), sampled
    deterministically. Control-only sites (e.g. NACC) are removed automatically
    because their cells request zero CN. Replaces the pre-generated
    dataset_cn_matched.csv (not distributed; matching is recomputed here).

    Parameters
    ----------
    data_dir : str, optional
        Directory containing dataset.csv. Defaults to ../data.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = data_dir or _DEFAULT_DATA
    df       = pd.read_csv(os.path.join(data_dir, "dataset.csv")).copy()
    clinical = df[df["Condition_type"] == "Clinical"].copy()
    cn       = df[df["Condition_type"] == "Control"].copy()
    need     = clinical.groupby(["study_site", "sex"]).size()
    picked   = []
    for (site, sex), k in need.items():
        pool = cn[(cn["study_site"] == site) & (cn["sex"] == sex)]
        picked.append(pool.sample(n=min(k, len(pool)), random_state=SEED))
    cn_matched = pd.concat(picked) if picked else cn.iloc[:0]
    return pd.concat([clinical, cn_matched]).reset_index(drop=True)


def load_bsnip_relatives(data_dir=None):
    """
    Load the BSNIP proband / relative / control sample.

    Reads bsnip_relatives_dataset.csv from data_dir.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing bsnip_relatives_dataset.csv. Defaults to ../data.

    Returns
    -------
    pd.DataFrame
        Columns: INDEX.ID, age, sex, study_site, dx, dx_group, role,
                 lh.* / rh.* cortical columns, SubcGM.* subcortical columns.
    """
    data_dir = data_dir or _DEFAULT_DATA
    return pd.read_csv(os.path.join(data_dir, "bsnip_relatives_dataset.csv"), low_memory=False)

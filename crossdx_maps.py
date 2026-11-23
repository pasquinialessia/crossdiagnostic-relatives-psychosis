# %%
"""
Transdiagnostic Surface Maps – CT, GMV, SA  |  Condition_type (Clinical vs Control)
=====================================================================================
Uses Condition_type (Clinical vs Control) as the predictor instead of dx.
All clinical groups (BD, SCZ, SA) are collapsed into "Clinical".

Model per region:
    centile ~ age_z + [C(sex)] + C(study_site)
           + C(Condition_type, Treatment(reference="Control"))
           + C(Condition_type):age_z
           + [C(Condition_type):C(sex)]

Groups: Whole / Males / Females  (3 rows per figure)
Cohen's d = t / sqrt(n), NaN where p-FDR >= 0.05

Outputs
-------
  transdx_cond_cortical_{measure}.png      3 rows × 4 views
  transdx_cond_subcortical_{measure}.png   3 rows stacked
  transdx_cond_cohens_d.csv
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
try:
    import vtk
    import vtk.util.numpy_support as ns
except ImportError as _e:
    print(f"[crossdx_maps] VTK not available ({_e}).\n"
          f"  Brain-surface rendering requires the venv with VTK/enigmatoolbox installed.\n"
          f"  Run this script from the venv to generate the surface maps.")
    raise SystemExit(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# OUTPUT DIRECTORY  (named after this script)
# =============================================================================
OUTPUT_DIR = '../results/crossdiagnostic/transdx_condition_surface_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}/\n")


from statsmodels.stats.multitest import multipletests
from enigmatoolbox.datasets import load_fc

# =============================================================================
# VTK 9.x PATCHES
# =============================================================================
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable
from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds
import enigmatoolbox.plotting.surface_plotting as _splot

def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        vtk_arr = ns.numpy_to_vtk(table.astype(np.uint8), deep=True,
                                   array_type=vtk.VTK_UNSIGNED_CHAR)
        self.VTKObject.SetTable(vtk_arr)
    elif hasattr(table, 'VTKObject'):
        self.VTKObject.SetTable(table.VTKObject)
    else:
        self.VTKObject.SetTable(table)

BSLookupTable.SetTable = _fixed_SetTable

_PKG = os.path.dirname(__import__('enigmatoolbox').__file__)

def _build_surface(path):
    gii    = nib.load(path)
    coords = gii.darrays[0].data.astype(np.float64)
    faces  = gii.darrays[1].data.astype(np.int64)
    pts    = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(coords))
    n      = len(faces)
    carr   = np.hstack([np.full((n, 1), 3, dtype=np.int64), faces]).flatten()
    cells  = vtk.vtkCellArray()
    cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_obj = vtk.vtkPolyData()
    pd_obj.SetPoints(pts)
    pd_obj.SetPolys(cells)
    return BSPolyData(pd_obj)

_SURF_LH = _build_surface(os.path.join(_PKG, 'datasets/surfaces/fsa5_lh.surf.gii'))
_SURF_RH = _build_surface(os.path.join(_PKG, 'datasets/surfaces/fsa5_rh.surf.gii'))

def _patched_load_fsa5(as_sphere=False):
    return _SURF_LH, _SURF_RH

_eds.load_fsa5   = _patched_load_fsa5
_splot.load_fsa5 = _patched_load_fsa5

print(f"VTK patches OK — LH: {_SURF_LH.n_points} pts, RH: {_SURF_RH.n_points} pts\n")

from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from data_utils import load_crossdx_matched

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = load_crossdx_matched().copy()



males   = df[df['sex'] == 'Male'].copy()
females = df[df['sex'] == 'Female'].copy()

GROUPS = [
    ('all',    'Whole group', df),
    ('male',   'Males',       males),
    ('female', 'Females',     females),
]

df['Condition_type'] = pd.Categorical(
    df['Condition_type'], categories=['Control', 'Clinical'], ordered=False)

print(f"Loaded {len(df)} subjects | {df['study_site'].nunique()} sites")
print(df['Condition_type'].value_counts().to_string(), '\n')

# =============================================================================
# 2. COLUMN DEFINITIONS
# =============================================================================
_, fc_ctx_labels, _, _ = load_fc()

def enigma_label_to_col(label, prefix):
    h, region = label.split('_', 1)
    return f"{'lh' if h == 'L' else 'rh'}.{prefix}.{region}"

MEASURES = {
    'CT':  [enigma_label_to_col(l, 'CT') for l in fc_ctx_labels],
    'GMV': [enigma_label_to_col(l, 'GM') for l in fc_ctx_labels],
    'SA':  [enigma_label_to_col(l, 'SA') for l in fc_ctx_labels],
}

ALL_SCTX  = [c for c in df.columns if c.startswith('SubcGM.')]
SCTX_COLS = [c for c in ALL_SCTX if 'Ventricle' not in c and 'Vent' not in c and 'Cereb' not in c]

_SCTX_TARGETS = ['Accumbens.area', 'Amygdala', 'Caudate', 'Hippocampus',
                  'Pallidum', 'Putamen', 'Thalamus', 'Ventricle']

def _match_sctx(side, target, cols):
    for c in cols:
        if side.lower() in c.lower() and target.lower() in c.lower():
            return c
    return None

SCTX_16 = ([_match_sctx('Left',  t, df.columns) for t in _SCTX_TARGETS] +
           [_match_sctx('Right', t, df.columns) for t in _SCTX_TARGETS])

# =============================================================================
# 3. PRINT RESULTS TABLE UTILITY
# =============================================================================

def print_results_table(res: pd.DataFrame, title: str = '') -> None:
    """Print t, p_unc, p_fdr for all regions."""
    cols_show = [c for c in ['t_val', 'p_unc', 'p_fdr', 'beta'] if c in res.columns]
    tbl = res[cols_show].reset_index().rename(columns={
        'col': 'region', 'index': 'region',
        't_val': 't', 'p_unc': 'p_unc', 'p_fdr': 'p_fdr', 'beta': 'beta'
    })
    if title:
        print(f"\n{'─'*65}\n  {title}\n{'─'*65}")
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(tbl.to_string(index=False))
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_rows')

# =============================================================================
# 4. FIT LINEAR MODEL
#    One model per region on the target group (whole / males / females).
#    Data is subset to CN + target dx so the model is a clean two-group
#    comparison with full covariate adjustment.
#    sex term is included only when both sexes are present.
# =============================================================================

def fit_lm_condition(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Model per region on the passed group:
        centile ~ age_z + [C(sex)] + C(study_site)
               + C(Condition_type, Treatment(reference="Control"))
               + C(Condition_type):age_z
               + [C(Condition_type):C(sex)]
    sex terms included only when both sexes present.
    Returns DataFrame with beta, t_val, p_unc, p_fdr, n indexed by col.
    """
    records = []
    for col in [c for c in cols if c is not None and c in data.columns]:
        safe = col.replace('.', '_')
        tmp  = data[[col, 'age', 'sex', 'study_site', 'Condition_type']].copy().dropna()
        tmp['age_z'] = (tmp['age'] - tmp['age'].mean()) / tmp['age'].std(ddof=1)
        tmp = tmp.rename(columns={col: safe})
        # Drop sites that don't have both Condition_type levels
        valid_sites = tmp.groupby('study_site')['Condition_type'].nunique()
        tmp = tmp[tmp['study_site'].isin(valid_sites[valid_sites > 1].index)]
        if tmp['Condition_type'].nunique() < 2 or len(tmp) < 10:
            continue
        n_sexes  = tmp['sex'].nunique()
        sex_main = '+ C(sex)' if n_sexes > 1 else ''
        sex_int  = '+ C(Condition_type, Treatment(reference="Control")):C(sex)' if n_sexes > 1 else ''
        model = smf.ols(
            f'{safe} ~ age_z {sex_main} + C(study_site)'
            f' + C(Condition_type, Treatment(reference="Control"))'
            f' + C(Condition_type, Treatment(reference="Control")):age_z'
            f' {sex_int}',
            data=tmp
        ).fit()
        coef_key = [k for k in model.params.index if 'Clinical' in k and ':' not in k]
        if not coef_key:
            continue
        ck = coef_key[0]
        records.append({
            'col':   col,
            'beta':  model.params[ck],
            't_val': model.tvalues[ck],
            'p_unc': model.pvalues[ck],
            'n':     int(model.nobs),
        })
    if not records:
        return pd.DataFrame(columns=['beta','t_val','p_unc','p_fdr','n'])
    res = pd.DataFrame(records).set_index('col')
    _, res['p_fdr'], _, _ = multipletests(res['p_unc'], method='fdr_bh')
    return res

# =============================================================================
# 5. MODEL-DERIVED COHEN'S D
#    d = t / sqrt(n), NaN for non-significant regions (p_fdr >= 0.05)
# =============================================================================

def cohens_d_from_model(res: pd.DataFrame, cols: list) -> np.ndarray:
    out = []
    for col in cols:
        if col is None or col not in res.index:
            out.append(np.nan)
        else:
            row = res.loc[col]
            d   = row['t_val'] / np.sqrt(row['n']) if row['p_fdr'] < 0.05 else np.nan
            out.append(d)
    return np.array(out)

# =============================================================================
# 6. FIT ALL MODELS
# =============================================================================
print("Fitting models …")

# results[dx][measure][group_key] = DataFrame
results  = {m: {} for m in MEASURES}
d_ctx    = {m: {} for m in MEASURES}
sig_ctx  = {m: {} for m in MEASURES}
sctx_res = {}
d_sctx   = {}
sig_sctx = {}

for measure, cols in MEASURES.items():
    for g, label, data in GROUPS:
        results[measure][g] = fit_lm_condition(data, cols)
        d_ctx[measure][g]   = cohens_d_from_model(results[measure][g], cols)
        sig_ctx[measure][g] = ~np.isnan(d_ctx[measure][g])
    print(f"  {measure}: done")

for g, label, data in GROUPS:
    sctx_res[g] = fit_lm_condition(data, SCTX_COLS)
    d16 = np.full(16, np.nan)
    for i, col in enumerate(SCTX_16):
        if col and col in sctx_res[g].index:
            row = sctx_res[g].loc[col]
            d16[i] = row['t_val'] / np.sqrt(row['n']) if row['p_fdr'] < 0.05 else np.nan
    d_sctx[g]   = d16
    sig_sctx[g] = ~np.isnan(d16)
print("  SubcGM: done")

# Print results tables
for measure, cols in MEASURES.items():
    for g, label, _ in GROUPS:
        print_results_table(results[measure][g], f'{measure} | {label}')
for g, label, _ in GROUPS:
    print_results_table(sctx_res[g], f'SubcGM | {label}')

# %%
#core model of the bootstrap
def fit_lm_condition(data: pd.DataFrame, cols: list) -> pd.DataFrame:

    records = []

    for col in [c for c in cols if c in data.columns]:

        safe = col.replace('.', '_')

        tmp = data[[col, 'age', 'sex', 'study_site', 'Condition_type']].copy().dropna()

        tmp['age_z'] = (tmp['age'] - tmp['age'].mean()) / tmp['age'].std(ddof=1)

        tmp = tmp.rename(columns={col: safe})

        # keep only sites with both groups
        valid_sites = tmp.groupby('study_site')['Condition_type'].nunique()
        tmp = tmp[tmp['study_site'].isin(valid_sites[valid_sites > 1].index)]

        if tmp['Condition_type'].nunique() < 2 or len(tmp) < 10:
            continue

        n_sexes = tmp['sex'].nunique()
        sex_main = '+ C(sex)' if n_sexes > 1 else ''
        sex_int  = '+ C(Condition_type, Treatment(reference="Control")):C(sex)' if n_sexes > 1 else ''

        model = smf.ols(
            f'{safe} ~ age_z {sex_main} + C(study_site)'
            f' + C(Condition_type, Treatment(reference="Control"))'
            f' + C(Condition_type, Treatment(reference="Control")):age_z'
            f' {sex_int}',
            data=tmp
        ).fit()

        key = [k for k in model.params.index
               if 'Condition_type' in k and 'Clinical' in k and ':' not in k]

        if not key:
            continue

        records.append({
            'col': col,
            'beta': model.params[key[0]]
        })

    if not records:
        return pd.DataFrame(columns=['beta'])

    return pd.DataFrame(records).set_index('col')

# %%
#bootstrap region-wise matrix 
def bootstrap_lm_condition(data, cols, n_iter=1000, seed=42):

    np.random.seed(seed)

    boot_results = []

    for i in range(n_iter):

        if i % 100 == 0:
            print(f"Bootstrap {i}/{n_iter}")

        boot = []

        for g in ["Control", "Clinical"]:

            df_g = data[data["Condition_type"] == g]

            boot.append(
                resample(
                    df_g,
                    replace=True,
                    n_samples=len(df_g)
                )
            )

        boot_df = pd.concat(boot)

        res = fit_lm_condition(boot_df, cols)

        if len(res):

            tmp = res[["beta"]].copy()
            tmp.columns = [f"iter_{i}"]

            boot_results.append(tmp)

    return pd.concat(boot_results, axis=1)

# %%
#summary w mean, ci and fdr sig
def summarize_bootstrap_regions(boot_mat):

    out = pd.DataFrame(index=boot_mat.index)

    out["beta_mean"] = boot_mat.mean(axis=1)

    out["ci_lower"] = boot_mat.quantile(0.025, axis=1)

    out["ci_upper"] = boot_mat.quantile(0.975, axis=1)

    out["sd"] = boot_mat.std(axis=1)

    # two-sided bootstrap p-value
    out["p_boot"] = [
        2 * min(
            np.mean(boot_mat.loc[r] <= 0),
            np.mean(boot_mat.loc[r] >= 0)
        )
        for r in boot_mat.index
    ]

    _, out["p_boot_fdr"], _, _ = multipletests(
        out["p_boot"],
        method="fdr_bh"
    )

    out["boot_sig"] = (
        (out["ci_lower"] > 0) |
        (out["ci_upper"] < 0)
    )

    return out

# %%
def summarize_bootstrap(boot_mat):

    out = pd.DataFrame(index=boot_mat.index)

    # mean effect across bootstrap iterations
    out["mean"] = boot_mat.mean(axis=1)

    # 95% CI
    out["ci_low"] = boot_mat.quantile(0.025, axis=1)
    out["ci_high"] = boot_mat.quantile(0.975, axis=1)

    # variability
    out["sd"] = boot_mat.std(axis=1)

    # bootstrap p-value (two-sided sign test)
    out["p_boot"] = [
        2 * min(
            np.mean(boot_mat.loc[r] <= 0),
            np.mean(boot_mat.loc[r] >= 0)
        )
        for r in boot_mat.index
    ]

    # FDR across regions
    _, out["p_fdr"], _, _ = multipletests(out["p_boot"], method="fdr_bh")

    # CI-based significance
    out["sig"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)

    return out

# %%
#forest plot 
def plot_forest(summary, measure, output_dir):

    dfp = summary.sort_values("mean")

    y = np.arange(len(dfp))

    plt.figure(figsize=(8, max(6, len(dfp) * 0.2)))

    plt.errorbar(
        dfp["mean"],
        y,
        xerr=[
            dfp["mean"] - dfp["ci_low"],
            dfp["ci_high"] - dfp["mean"]
        ],
        fmt="o",
        capsize=2
    )

    plt.axvline(0, linestyle="--")

    plt.yticks(y, dfp.index, fontsize=7)

    plt.xlabel("Bootstrap effect (Clinical vs Control)")
    plt.title(measure)

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"forest_{measure}.png"),
        dpi=300
    )

    plt.close()

# %%
print("=== START BOOTSTRAP PIPELINE ===")

for measure, cols in MEASURES.items():

    print(f"\nProcessing: {measure}")

    # bootstrap  (BUGFIX: use the full matched sample `df`, not the leftover
    # females-only `data` from the earlier `for g, label, data in GROUPS` loop)
    boot_mat = bootstrap_lm_condition(
        df,
        cols,
        n_iter=1000
    )

    if boot_mat.shape[0] == 0:
        print(f"{measure}: no valid results")
        continue

    # summary
    summary = summarize_bootstrap(boot_mat)

    # save summary CSV
    out_csv = os.path.join(
        OUTPUT_DIR,
        f"bootstrap_{measure}.csv"
    )
    summary.to_csv(out_csv)
    print("Saved:", out_csv)

    # save full bootstrap distribution
    dist_csv = os.path.join(
        OUTPUT_DIR,
        f"bootstrap_distribution_{measure}.csv"
    )
    boot_mat.to_csv(dist_csv)
    print("Saved:", dist_csv)

    # forest plot
    plot_forest(summary, measure, OUTPUT_DIR)
    print("Plot saved")

print("\n=== DONE ===")

# %%
# =============================================================================
# 7. PARCEL → VERTEX MAPPING
# =============================================================================
_lh_annot, _, _lh_names = nib.freesurfer.read_annot(
    os.path.join(_PKG, 'permutation_testing/annot/fsa5_lh_aparc.annot'))
_rh_annot, _, _rh_names = nib.freesurfer.read_annot(
    os.path.join(_PKG, 'permutation_testing/annot/fsa5_rh_aparc.annot'))
_lh_names = [n.decode() if isinstance(n, bytes) else n for n in _lh_names]
_rh_names = [n.decode() if isinstance(n, bytes) else n for n in _rh_names]

def ctx_to_vertices(d68: np.ndarray) -> np.ndarray:
    """Map 68 parcel values → 20484-vertex array. NaN = non-sig → grey."""
    lh = np.full(10242, np.nan)
    rh = np.full(10242, np.nan)
    for i, lbl in enumerate(fc_ctx_labels):
        h, region = str(lbl).split('_', 1)
        if h == 'L' and region in _lh_names:
            lh[_lh_annot == _lh_names.index(region)] = d68[i]
        elif h == 'R' and region in _rh_names:
            rh[_rh_annot == _rh_names.index(region)] = d68[i]
    return np.concatenate([lh, rh])

# =============================================================================
# 8. SURFACE MAPS
#    Per dx: 3 rows (Whole / Males / Females) × 4 cortical views
#            3 rows subcortical stacked
# =============================================================================
NAN_COLOR = (0.85, 0.85, 0.85, 1)
CMAP      = 'RdBu_r'
ZOOM      = 1.25
SURF_SIZE = (1600, 400)

# Global vmax
_all_ctx  = [d_ctx[m][g]  for m in MEASURES for g, *_ in GROUPS]
_all_sctx = [d_sctx[g] for g, *_ in GROUPS]
_ctx_vals  = [np.nanmax(np.abs(a)) for a in _all_ctx  if not np.all(np.isnan(a))]
_sctx_vals = [np.nanmax(np.abs(a)) for a in _all_sctx if not np.all(np.isnan(a))]
GLOBAL_VMAX      = np.ceil((max(_ctx_vals)  if _ctx_vals  else 0.5) * 10) / 10
GLOBAL_SCTX_VMAX = np.ceil((max(_sctx_vals) if _sctx_vals else 0.5) * 10) / 10
print(f'Global cortical vmax:    {GLOBAL_VMAX}')
print(f'Global subcortical vmax: {GLOBAL_SCTX_VMAX}')

print("\nRendering surface maps …")

summary_rows = []

for measure, cols in MEASURES.items():
    vmax = GLOBAL_VMAX

    # ── Cortical ─────────────────────────────────────────────────────────
    row_imgs = []
    for g, label, _ in GROUPS:
        tex = ctx_to_vertices(d_ctx[measure][g])
        tmp = f'_tmp_ctx_{measure}_{g}.png'
        plot_cortical(
            array_name=tex, surface_name='fsa5',
            color_bar=True, color_range=(-vmax, vmax),
            label_text={'left': [label]},
            cmap=CMAP, nan_color=NAN_COLOR,
            zoom=ZOOM, size=SURF_SIZE,
            interactive=False, screenshot=True,
            filename=tmp, transparent_bg=False,
        )
        row_imgs.append(plt.imread(tmp))
        os.remove(tmp)
        print(f"  Cortical {measure} | {label} done")

    h = sum(im.shape[0] for im in row_imgs)
    w = max(im.shape[1] for im in row_imgs)
    canvas = np.ones((h, w, row_imgs[0].shape[2]), dtype=row_imgs[0].dtype)
    y = 0
    for im in row_imgs:
        canvas[y:y+im.shape[0], :im.shape[1]] = im
        y += im.shape[0]

    fig, ax = plt.subplots(figsize=(w/100, h/100))
    ax.imshow(canvas); ax.axis('off')
    fig.suptitle(
        f"Cohen's d – {measure} | Clinical vs Control\n"
        "Bright = FDR-significant (p < 0.05)   •   Grey = non-significant",
        fontsize=13, y=1.01
    )
    fig.tight_layout(pad=0)
    fname_ctx = os.path.join(OUTPUT_DIR, f'transdx_cond_cortical_{measure}.png')
    fig.savefig(fname_ctx, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname_ctx}")

    # ── Subcortical ───────────────────────────────────────────────────────
    sctx_vmax = GLOBAL_SCTX_VMAX

    sctx_rows = []
    for g, label, _ in GROUPS:
        d16 = d_sctx[g].copy()
        tmp = f'_tmp_sctx_{measure}_{g}.png'
        plot_subcortical(
            array_name=d16, ventricles=True,
            color_bar=True, color_range=(-sctx_vmax, sctx_vmax),
            label_text={'left': [label]},
            cmap=CMAP, nan_color=NAN_COLOR,
            zoom=ZOOM, size=SURF_SIZE,
            interactive=False, screenshot=True,
            filename=tmp, transparent_bg=False,
        )
        sctx_rows.append(plt.imread(tmp))
        os.remove(tmp)
        print(f"  Subcortical {measure} | {label} done")

    sh = sum(im.shape[0] for im in sctx_rows)
    sw = max(im.shape[1] for im in sctx_rows)
    s_canvas = np.ones((sh, sw, sctx_rows[0].shape[2]), dtype=sctx_rows[0].dtype)
    y = 0
    for im in sctx_rows:
        s_canvas[y:y+im.shape[0], :im.shape[1]] = im
        y += im.shape[0]

    fig2, ax2 = plt.subplots(figsize=(sw/100, sh/100))
    ax2.imshow(s_canvas); ax2.axis('off')
    fig2.suptitle(
        f"Cohen's d – Subcortical | {measure} | Clinical vs Control\n"
        "Bright = FDR-significant (p < 0.05)   •   Grey = non-significant",
        fontsize=13, y=1.01
    )
    fig2.tight_layout(pad=0)
    fname_sctx = os.path.join(OUTPUT_DIR, f'transdx_cond_subcortical_{measure}.png')
    fig2.savefig(fname_sctx, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {fname_sctx}")

# =============================================================================
# 9. EXPORT SUMMARY CSV
# =============================================================================
rows = []
for measure, cols in MEASURES.items():
    for col, lbl in zip(cols, fc_ctx_labels):
        row = {'measure': measure, 'region': col, 'enigma_label': str(lbl)}
        for g, label, _ in GROUPS:
            i = cols.index(col)
            res = results[measure][g]
            row[f'd_{g}']     = d_ctx[measure][g][i]
            row[f't_{g}']     = res.loc[col, 't_val'] if col in res.index else np.nan
            row[f'p_unc_{g}'] = res.loc[col, 'p_unc'] if col in res.index else np.nan
            row[f'p_fdr_{g}'] = res.loc[col, 'p_fdr'] if col in res.index else np.nan
            row[f'sig_{g}']   = int(sig_ctx[measure][g][i])
        rows.append(row)
for i, col in enumerate(SCTX_16):
    row = {'measure': 'SubcGM', 'region': col or 'unknown', 'enigma_label': ''}
    for g, label, _ in GROUPS:
        res = sctx_res[g]
        row[f'd_{g}']     = d_sctx[g][i]
        row[f't_{g}']     = res.loc[col, 't_val'] if col and col in res.index else np.nan
        row[f'p_unc_{g}'] = res.loc[col, 'p_unc'] if col and col in res.index else np.nan
        row[f'p_fdr_{g}'] = res.loc[col, 'p_fdr'] if col and col in res.index else np.nan
        row[f'sig_{g}']   = int(sig_sctx[g][i])
    rows.append(row)

pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'transdx_cond_cohens_d.csv'), index=False)

print("\nAll files saved:")
for measure in MEASURES:
    print(f"  transdx_cond_cortical_{measure}.png  |  "
          f"transdx_cond_subcortical_{measure}.png")
print("  transdx_cond_cohens_d.csv")

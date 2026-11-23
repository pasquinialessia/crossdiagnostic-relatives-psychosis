# %%
"""
Surface Map – Cohen's d for CT, GMV, SA  |  per diagnosis (whole / males / females)
==============================================================
Each diagnosis is rendered on its own against controls — schizophrenia (SCZ) and
schizoaffective disorder (SAD) — with no pooled group. Run with no arguments to
render both in turn, or pass --diagnosis {scz,sad} for one. For each brain measure
(CT, GMV, SA) it produces:
  • One cortical figure   : 3 rows (Whole / Males / Females) × 4 views
  • One subcortical figure: 3 rows × bar chart

Significance: p-FDR < 0.05 from linear model (centile ~ age + Condition_type)
  - Significant regions  : full RdBu_r colour
  - Non-significant       : light grey (nan_color)

Outputs (per diagnosis, under ../results/casecontrol/{scz,sad}_surface_plots/)
-------
  {dx}_cortical_CT.png   /  {dx}_subcortical_CT.png
  {dx}_cortical_GMV.png  /  {dx}_subcortical_GMV.png
  {dx}_cortical_SA.png   /  {dx}_subcortical_SA.png
  cohens_d_summary.csv

Requirements
------------
  pip install vtk==9.3.1
  pip install git+https://github.com/MICA-MNI/ENIGMA.git
  pip install statsmodels nilearn matplotlib nibabel pandas numpy
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
import vtk
import vtk.util.numpy_support as ns
import warnings
warnings.filterwarnings('ignore')

from PIL import Image

# =============================================================================
# DIAGNOSIS SELECTION (each diagnosis rendered separately; never pooled)
# =============================================================================
DIAGNOSES = {
    "scz": {"dx": "SCZ",                      "label": "Schizophrenia"},
    "sad": {"dx": "Schizoaffective", "label": "Schizoaffective disorder"},
}
# Parse --diagnosis; with none given, re-run this script once per diagnosis.
_arg = None
if "--diagnosis" in sys.argv:
    _arg = sys.argv[sys.argv.index("--diagnosis") + 1]
if _arg is None:
    for _k in DIAGNOSES:
        print(f"\n########## rendering {_k} ##########")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--diagnosis", _k], check=True)
    sys.exit(0)
DXKEY = _arg
DX_VALUE = DIAGNOSES[DXKEY]["dx"]
DXLABEL = DIAGNOSES[DXKEY]["label"]

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# OUTPUT DIRECTORY  (per diagnosis)
# =============================================================================
OUTPUT_DIR = f'../results/casecontrol/{DXKEY}_surface_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Diagnosis: {DXLABEL} (dx == '{DX_VALUE}')")
print(f"Output directory: {OUTPUT_DIR}/\n")


import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# =============================================================================
# VTK 9.x PATCHES  (must run before any enigmatoolbox import)
# =============================================================================

# Patch 1 – LUT SetTable: convert numpy → vtkUnsignedCharArray
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable

def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        vtk_arr = ns.numpy_to_vtk(
            table.astype(np.uint8), deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR)
        self.VTKObject.SetTable(vtk_arr)
    elif hasattr(table, 'VTKObject'):
        self.VTKObject.SetTable(table.VTKObject)
    else:
        self.VTKObject.SetTable(table)

BSLookupTable.SetTable = _fixed_SetTable

# Patch 2 – load_fsa5: rebuild surface from gifti using raw VTK API
from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds
import enigmatoolbox.plotting.surface_plotting as _splot

_PKG = os.path.dirname(__import__('enigmatoolbox').__file__)

def _build_surface(path: str) -> BSPolyData:
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

# Safe to import plotting now
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.datasets import load_fc

# =============================================================================
# 1. LOAD DATA
# =============================================================================
_raw = load_scz_dataset().copy()
# Sites that recruited this diagnosis
case_sites = _raw.loc[_raw['dx'] == DX_VALUE, 'study_site'].unique()
_sub = _raw[_raw['study_site'].isin(case_sites) & _raw['dx'].isin(['CN', DX_VALUE])].copy()

# Site×sex-match CN to cases (same approach as BD)
_cases  = _sub[_sub['dx'] == DX_VALUE]
_cns    = _sub[_sub['dx'] == 'CN']
_sampled = []
for (_site, _sex), _cell in _cases.groupby(['study_site', 'sex']):
    _cn_cell = _cns[(_cns['study_site'] == _site) & (_cns['sex'] == _sex)]
    if len(_cn_cell) == 0:
        continue
    _sampled.append(_cn_cell.sample(n=min(len(_cell), len(_cn_cell)), random_state=SEED))
_matched_cn = pd.concat(_sampled) if _sampled else _cns.iloc[:0]
df = pd.concat([_cases, _matched_cn]).reset_index(drop=True)
df['Condition_type'] = np.where(df['dx'] == DX_VALUE, 'Clinical', 'Control')

print(f'N subjects after site×sex matching: {len(df)}')
print(df.groupby(['study_site','dx']).size())

df['Condition_type'] = pd.Categorical(df['Condition_type'], categories=['Control', 'Clinical'], ordered=False)


males   = df[df['sex'] == 'Male'].copy()
females = df[df['sex'] == 'Female'].copy()

GROUPS = [
    ('all',    'Whole group', df),
    ('male',   'Males',       males),
    ('female', 'Females',     females),
]

_, fc_ctx_labels, _, _ = load_fc()

def enigma_label_to_col(label: str, prefix: str) -> str:
    hemi, region = label.split('_', 1)
    h = 'lh' if hemi == 'L' else 'rh'
    return f'{h}.{prefix}.{region}'

# Column sets per measure
MEASURES = {
    'CT':  [enigma_label_to_col(l, 'CT')  for l in fc_ctx_labels],
    'GMV': [enigma_label_to_col(l, 'GM')  for l in fc_ctx_labels],
    'SA':  [enigma_label_to_col(l, 'SA')  for l in fc_ctx_labels],
}

# Subcortical columns (no ventricles, for LM)
ALL_SCTX  = [c for c in df.columns if c.startswith('SubcGM.')]
SCTX_COLS = [c for c in ALL_SCTX if 'Ventricle' not in c and 'Vent' not in c and 'Cereb' not in c]

# Subcortical 16-slot order for plot_subcortical (ventricles included → NaN)
_SCTX_TARGETS = [
    'Accumbens.area', 'Amygdala', 'Caudate',    'Hippocampus',
    'Pallidum',       'Putamen',  'Thalamus',    'Ventricle',
]
def _match_sctx(side, target, cols):
    for c in cols:
        if side.lower() in c.lower() and target.lower() in c.lower():
            return c
    return None

SCTX_16 = [_match_sctx('Left',  t, df.columns) for t in _SCTX_TARGETS] + \
          [_match_sctx('Right', t, df.columns) for t in _SCTX_TARGETS]

print(f"Cortical cols   — CT: {len(MEASURES['CT'])}, GMV: {len(MEASURES['GMV'])}, SA: {len(MEASURES['SA'])}")
print(f"Subcortical cols: {len(SCTX_COLS)} (non-ventricle) + ventricles for plot_subcortical\n")

# =============================================================================
# 2. LINEAR MODEL + COHEN'S D
# =============================================================================

def fit_lm(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    OLS per region:
        centile ~ age_z + C(sex) + C(study_site)
               + C(Condition_type, Treatment(reference='Control'))
               + C(Condition_type):age_z
               + C(Condition_type):C(sex)
    age_z      : age z-scored within the passed group (mean=0, sd=1)
    study_site : categorical fixed effect (dummy-coded, first site = reference)
    """
    records = []
    for col in [c for c in cols if c is not None and c in data.columns]:
        safe = col.replace('.', '_')
        tmp  = data[[col, 'age', 'sex', 'study_site', 'Condition_type']].copy().dropna()
        tmp['age_z'] = (tmp['age'] - tmp['age'].mean()) / tmp['age'].std(ddof=1)
        tmp = tmp.rename(columns={col: safe})
        m = smf.ols(
            f'{safe} ~ age_z + C(sex) + C(study_site)'
            f' + C(Condition_type, Treatment(reference="Control"))'
            f' + C(Condition_type, Treatment(reference="Control")):age_z'
            f' + C(Condition_type, Treatment(reference="Control")):C(sex)',
            data=tmp
        ).fit()
        coef_key = [k for k in m.params.index if 'Clinical' in k and ':' not in k][0]
        records.append({'col': col, 'beta': m.params[coef_key],
                        't_val': m.tvalues[coef_key],
                        'p_unc': m.pvalues[coef_key],
                        'n':     int(m.nobs)})
    res = pd.DataFrame(records).set_index('col')
    _, res['p_fdr'], _, _ = multipletests(res['p_unc'], method='fdr_bh')
    return res

def cohens_d_from_model(res: pd.DataFrame, cols: list) -> np.ndarray:
    """
    Compute Cohen's d from model t-statistic and n:
        d = t / sqrt(n)
    Only significant regions (p_fdr < 0.05) retain their d value;
    non-significant regions are set to NaN for map display.
    """
    out = []
    for col in cols:
        if col not in res.index:
            out.append(np.nan)
        else:
            row = res.loc[col]
            d   = row['t_val'] / np.sqrt(row['n']) if row['p_fdr'] < 0.05 else np.nan
            out.append(d)
    return np.array(out)


print("Fitting models …")

# Results dict: results[measure][group_key] = DataFrame
results = {}
d_vals  = {}   # Cohen's d arrays
sig     = {}   # significance masks


def print_results_table(res: pd.DataFrame, title: str = '') -> None:
    """Print t-values, p-values and FDR-corrected p-values for all regions."""
    cols_to_show = [c for c in ['t_val', 'p_unc', 'p_fdr', 'beta'] if c in res.columns]
    tbl = res[cols_to_show].copy()
    tbl = tbl.reset_index().rename(columns={
        'col': 'region', 'index': 'region',
        't_val': 't', 'p_unc': 'p_unc', 'p_fdr': 'p_fdr', 'beta': 'beta'
    })
    if title:
        print(f"\n{'─'*65}")
        print(f"  {title}")
        print(f"{'─'*65}")
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(tbl.to_string(index=False))
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_rows')

for measure, cols in MEASURES.items():
    results[measure] = {g: fit_lm(data, cols) for g, _, data in GROUPS}
    d_vals[measure]  = {g: cohens_d_from_model(results[measure][g], cols) for g, _, _ in GROUPS}
    sig[measure]     = {g: ~np.isnan(d_vals[measure][g]) for g, _, _ in GROUPS}

# Subcortical (same for all 3 measures — SubcGM is volume only)
sctx_results = {g: fit_lm(data, SCTX_COLS)        for g, _, data in GROUPS}
sctx_d = {}
for g, _, _ in GROUPS:
    sctx_d[g] = np.array([
        (sctx_results[g].loc[c, 't_val'] / np.sqrt(sctx_results[g].loc[c, 'n'])
         if c and c in sctx_results[g].index and sctx_results[g].loc[c, 'p_fdr'] < 0.05
         else np.nan)
        for c in SCTX_16
    ])

# Print results tables
for measure in MEASURES:
    for g, label, _ in GROUPS:
        print_results_table(results[measure][g], f'{measure} | {label}')
for g, label, _ in GROUPS:
    print_results_table(sctx_results[g], f'SubcGM | {label}')
sctx_sig16   = {}
for g, _, _ in GROUPS:
    mask = np.zeros(16, dtype=bool)
    for i, c in enumerate(SCTX_16):
        if c is not None and c in sctx_results[g].index:
            mask[i] = sctx_results[g].loc[c, 'p_fdr'] < 0.05
    sctx_sig16[g] = mask

for measure in MEASURES:
    for g, label, _ in GROUPS:
        n = sig[measure][g].sum()
        print(f"  {measure} [{label:<12}]: {n:2d} FDR-sig cortical regions")

# =============================================================================
# 3. VERTEX MAPPING HELPERS
# =============================================================================
_lh_annot, _, _lh_names = nib.freesurfer.read_annot(
    os.path.join(_PKG, 'permutation_testing/annot/fsa5_lh_aparc.annot'))
_rh_annot, _, _rh_names = nib.freesurfer.read_annot(
    os.path.join(_PKG, 'permutation_testing/annot/fsa5_rh_aparc.annot'))
_lh_names = [n.decode() if isinstance(n, bytes) else n for n in _lh_names]
_rh_names = [n.decode() if isinstance(n, bytes) else n for n in _rh_names]

def ctx_to_vertices(d68: np.ndarray, sig_mask: np.ndarray) -> np.ndarray:
    """68 parcel values → 20484-vertex array. Non-sig → NaN."""
    lh = np.full(10242, np.nan)
    rh = np.full(10242, np.nan)
    for i, lbl in enumerate(fc_ctx_labels):
        h, region = str(lbl).split('_', 1)
        val = d68[i] if sig_mask[i] else np.nan
        if h == 'L' and region in _lh_names:
            lh[_lh_annot == _lh_names.index(region)] = val
        elif h == 'R' and region in _rh_names:
            rh[_rh_annot == _rh_names.index(region)] = val
    return np.concatenate([lh, rh])

def sctx_masked(g: str) -> np.ndarray:
    """16-value subcortical array; non-sig → NaN."""
    d = sctx_d[g].copy()
    d[~sctx_sig16[g]] = np.nan
    return d

# =============================================================================
# 4. RENDER FUNCTION
#    Produces one cortical PNG and one subcortical PNG per measure.
#    The cortical PNG is assembled from 3 × 4-view screenshots stacked
#    vertically using matplotlib (since ENIGMA's renderer can't do multi-row).
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from data_utils import load_scz_dataset

NAN_COLOR   = (0.85, 0.85, 0.85, 1)
CMAP        = 'RdBu_r'
ZOOM        = 1.25
SURF_SIZE   = (1600, 400)   # per-row screenshot size
GROUP_LABELS = {g: lbl for g, lbl, _ in GROUPS}

def render_measure(measure, groups, suffix=""):
    cols  = MEASURES[measure]
    vmax  = GLOBAL_VMAX
    crange = (-vmax, vmax)

    # ── Cortical: save one temp image per group, then stack ──────────────────
    print(f"\n  [{measure}] Rendering cortical …")
    row_imgs = []
    for g, label, _ in groups:
        tex  = ctx_to_vertices(d_vals[measure][g], sig[measure][g])
        tmp  = f'_tmp_ctx_{measure}_{g}.png'
        plot_cortical(
            array_name   = tex,
            surface_name = 'fsa5',
            color_bar    = True,
            color_range  = crange,
            label_text   = {'left': [label]},
            cmap         = CMAP,
            nan_color    = NAN_COLOR,
            zoom         = ZOOM,
            size         = SURF_SIZE,
            interactive  = False,
            screenshot   = True,
            filename     = tmp,
            transparent_bg = False,
        )
        row_imgs.append(plt.imread(tmp))
        os.remove(tmp)
        print(f"    {label} done")

    # Stack 3 rows into one figure
    fig_h = sum(im.shape[0] for im in row_imgs)
    fig_w = max(im.shape[1] for im in row_imgs)
    canvas_ctx = np.ones((fig_h, fig_w, row_imgs[0].shape[2]), dtype=row_imgs[0].dtype)
    y = 0
    for im in row_imgs:
        canvas_ctx[y:y+im.shape[0], :im.shape[1]] = im
        y += im.shape[0]

    fig_ctx, ax_ctx = plt.subplots(figsize=(fig_w/100, fig_h/100))
    ax_ctx.imshow(canvas_ctx)
    ax_ctx.axis('off')
    fig_ctx.suptitle(
        f"Cohen's d – {measure} | Condition Effect (Clinical vs Control)\n"
        "Bright = FDR-significant (p < 0.05)   •   Grey = non-significant",
        fontsize=18, y=1.01
    )
    fig_ctx.tight_layout(pad=0)
    fig_ctx.savefig(os.path.join(OUTPUT_DIR, f'{DXKEY}_cortical_{measure}{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_ctx)
    print(f"    Saved: {DXKEY}_cortical_{measure}{suffix}.png")

    # ── Subcortical: plot_subcortical screenshots stacked (one row per group) ─
    print(f"  [{measure}] Rendering subcortical surface maps …")
    sctx_vmax  = GLOBAL_SCTX_VMAX
    sctx_crange = (-sctx_vmax, sctx_vmax)

    sctx_row_imgs = []
    for g, label, _ in groups:
        d16_masked = sctx_d[g].copy()
        d16_masked[~sctx_sig16[g]] = np.nan   # non-sig → grey via nan_color
        tmp = f'_tmp_sctx_{measure}_{g}.png'
        plot_subcortical(
            array_name   = d16_masked,
            ventricles   = True,
            color_bar    = True,
            color_range  = sctx_crange,
            label_text   = {'left': [label]},
            cmap         = CMAP,
            nan_color    = NAN_COLOR,
            zoom         = ZOOM,
            size         = SURF_SIZE,
            interactive  = False,
            screenshot   = True,
            filename     = tmp,
            transparent_bg = False,
        )
        sctx_row_imgs.append(plt.imread(tmp))
        os.remove(tmp)
        print(f"    {label} done")

    # Stack 3 rows into one figure
    s_h = sum(im.shape[0] for im in sctx_row_imgs)
    s_w = max(im.shape[1] for im in sctx_row_imgs)
    canvas_sctx = np.ones((s_h, s_w, sctx_row_imgs[0].shape[2]),
                         dtype=sctx_row_imgs[0].dtype)
    y = 0
    for im in sctx_row_imgs:
        canvas_sctx[y:y+im.shape[0], :im.shape[1]] = im
        y += im.shape[0]

    fig_s, ax_s = plt.subplots(figsize=(s_w/100, s_h/100))
    ax_s.imshow(canvas_sctx)
    ax_s.axis('off')
    fig_s.suptitle(
        f"Cohen's d – Subcortical | {measure} | Condition Effect (Clinical vs Control)\n"
        "Bright = FDR-significant (p < 0.05)   •   Grey = non-significant",
        fontsize=18, y=1.01
    )
    fig_s.tight_layout(pad=0)
    fig_s.savefig(os.path.join(OUTPUT_DIR, f'{DXKEY}_subcortical_{measure}{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_s)
    print(f"    Saved: {DXKEY}_subcortical_{measure}{suffix}.png")

# =============================================================================
# 5. RUN FOR ALL MEASURES
# =============================================================================
# Global vmax across all measures and groups
_all_ctx  = [d_vals[m][g]  for m in MEASURES for g, *_ in GROUPS]
_all_sctx = [sctx_d[g] for g, *_ in GROUPS]
GLOBAL_VMAX      = np.ceil(max(np.nanmax(np.abs(a)) for a in _all_ctx
                               if not np.all(np.isnan(a))) * 10) / 10
GLOBAL_SCTX_VMAX = np.ceil(max(np.nanmax(np.abs(a)) for a in _all_sctx
                               if not np.all(np.isnan(a))) * 10) / 10
print(f'Global cortical vmax:    {GLOBAL_VMAX}')
print(f'Global subcortical vmax: {GLOBAL_SCTX_VMAX}')

# Two variants of every figure: with the whole group, and without it (sex rows only)
GROUPS_NOWHOLE = [g for g in GROUPS if g[0] != 'all']
VARIANTS = [("", GROUPS), ("_no_whole", GROUPS_NOWHOLE)]
for suffix, groups in VARIANTS:
    for measure in ['CT', 'GMV', 'SA']:
        print(f"\n{'='*55}")
        print(f"  Measure: {measure}{suffix}")
        print(f"{'='*55}")
        render_measure(measure, groups, suffix)

# =============================================================================
# 6. SUMMARY CSV
# =============================================================================
print("\nExporting summary CSV …")
rows = []
for measure, cols in MEASURES.items():
    for col, lbl in zip(cols, fc_ctx_labels):
        row = {'measure': measure, 'region': col, 'enigma_label': str(lbl)}
        for g, label, _ in GROUPS:
            i = cols.index(col)
            row[f'd_{g}']     = d_vals[measure][g][i]
            row[f't_{g}']     = (results[measure][g].loc[col, 't_val']
                                  if col in results[measure][g].index else np.nan)
            row[f'p_unc_{g}'] = (results[measure][g].loc[col, 'p_unc']
                                  if col in results[measure][g].index else np.nan)
            row[f'p_fdr_{g}'] = (results[measure][g].loc[col, 'p_fdr']
                                  if col in results[measure][g].index else np.nan)
            row[f'sig_{g}']   = int(sig[measure][g][i])
        rows.append(row)

# Subcortical
for i, col in enumerate(SCTX_16):
    row = {'measure': 'SubcGM', 'region': col or 'unknown', 'enigma_label': ''}
    for g, label, _ in GROUPS:
        row[f'd_{g}']     = sctx_d[g][i]
        row[f't_{g}']     = (sctx_results[g].loc[col, 't_val']
                              if col and col in sctx_results[g].index else np.nan)
        row[f'p_unc_{g}'] = (sctx_results[g].loc[col, 'p_unc']
                              if col and col in sctx_results[g].index else np.nan)
        row[f'p_fdr_{g}'] = (sctx_results[g].loc[col, 'p_fdr']
                              if col and col in sctx_results[g].index else np.nan)
        row[f'sig_{g}']   = int(sctx_sig16[g][i])
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUTPUT_DIR, 'cohens_d_summary.csv'), index=False)
print("  Saved: cohens_d_summary.csv")

print("\nAll done. Files produced:")
for measure in ['CT', 'GMV', 'SA']:
    print(f"  {DXKEY}_cortical_{measure}.png  |  {DXKEY}_subcortical_{measure}.png")
print("  cohens_d_summary.csv")

# =============================================================================
# 11. GRAND SUMMARY FIGURE
# =============================================================================
print("\nCreating grand summary figure...")

# Composite laid out HORIZONTALLY: measures placed side by side (normalize height).
TARGET_HEIGHT = 1400
for suffix in ["", "_no_whole"]:
    summary_imgs = []
    for measure in MEASURES:
        fname = os.path.join(OUTPUT_DIR, f'{DXKEY}_cortical_{measure}{suffix}.png')
        if os.path.exists(fname):
            img = Image.open(fname)
            w, h = img.size
            img = img.resize((int(w * TARGET_HEIGHT / h), TARGET_HEIGHT), Image.Resampling.LANCZOS)
            summary_imgs.append(np.array(img))
    gmv_file = os.path.join(OUTPUT_DIR, f'{DXKEY}_subcortical_GMV{suffix}.png')
    if os.path.exists(gmv_file):
        img = Image.open(gmv_file)
        w, h = img.size
        img = img.resize((int(w * TARGET_HEIGHT / h), TARGET_HEIGHT), Image.Resampling.LANCZOS)
        summary_imgs.append(np.array(img))
    if len(summary_imgs) > 0:
        total_width = sum(im.shape[1] for im in summary_imgs)
        max_height = max(im.shape[0] for im in summary_imgs)
        canvas = np.ones((max_height, total_width, summary_imgs[0].shape[2]), dtype=summary_imgs[0].dtype)
        x = 0
        for im in summary_imgs:
            hh, ww = im.shape[:2]
            canvas[:hh, x:x+ww] = im
            x += ww
        fig, ax = plt.subplots(figsize=(total_width / 80, max_height / 80))
        ax.imshow(canvas)
        ax.axis('off')
        fig.tight_layout(pad=0)
        out_file = os.path.join(OUTPUT_DIR, f"{DXKEY}_complete_brain_summary{suffix}.png")
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_file}")

# %%
"""
BD vs CN only
=========================================

Model per region:
    centile ~ age_z + [C(sex)] + C(study_site)
           + C(dx, Treatment(reference="CN"))
           + C(dx):age_z + [C(dx):C(sex)]

Outputs in results/casecontrol/
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import vtk
import vtk.util.numpy_support as ns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


from statsmodels.stats.multitest import multipletests
from scipy import stats
from enigmatoolbox.datasets import load_fc
from enigmatoolbox.permutation_testing import spin_test, shuf_test
from PIL import Image

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
SEED = 42
np.random.seed(SEED)

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
    pts    = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(coords))
    n      = len(faces)
    carr   = np.hstack([np.full((n,1),3,dtype=np.int64), faces]).flatten()
    cells  = vtk.vtkCellArray()
    cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_obj = vtk.vtkPolyData(); pd_obj.SetPoints(pts); pd_obj.SetPolys(cells)
    return BSPolyData(pd_obj)

_SURF_LH = _build_surface(os.path.join(_PKG, 'datasets/surfaces/fsa5_lh.surf.gii'))
_SURF_RH = _build_surface(os.path.join(_PKG, 'datasets/surfaces/fsa5_rh.surf.gii'))

def _patched_load_fsa5(as_sphere=False): return _SURF_LH, _SURF_RH
_eds.load_fsa5 = _splot.load_fsa5 = _patched_load_fsa5

from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from data_utils import load_bd_matched

# =============================================================================
# OUTPUT DIRECTORY  (named after this script)
# =============================================================================
SURF_DIR = '../results/casecontrol/bd_surface_plots'          # cohen's d surface maps (mirrors scz_surface_plots/)
SEX_DIR  = '../results/casecontrol/bd_lm_by_sex_results'      # spin/shuffle male-vs-female (mirrors scz_lm_by_sex_results/)
os.makedirs(SURF_DIR, exist_ok=True)
os.makedirs(SEX_DIR, exist_ok=True)
print(f"Output directories: {SURF_DIR}/  {SEX_DIR}/\n")


print(f"VTK patches OK — LH: {_SURF_LH.n_points} pts, RH: {_SURF_RH.n_points} pts\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = load_bd_matched().copy()  # BD vs site/sex-matched CN
df['dx'] = pd.Categorical(df['dx'], categories=['CN', 'BD'], ordered=False)

print(f'Loaded matched BD sample — BD: {(df["dx"]=="BD").sum()}, '
      f'CN: {(df["dx"]=="CN").sum()}, Total: {len(df)}')
print(pd.crosstab(df['dx'], df['sex']))

DIAGNOSES = ['BD']   # CN = reference

males   = df[df['sex'] == 'Male'].copy()
females = df[df['sex'] == 'Female'].copy()
SEX_GROUPS = [('Whole', df), ('Male', males), ('Female', females)]


print(f"Loaded {df.shape[0]} subjects, {df['study_site'].nunique()} sites")
print(df['dx'].value_counts().to_string())
print()

# =============================================================================
# 2. COLUMN DEFINITIONS
# =============================================================================
_, fc_ctx_labels, _, _ = load_fc()

def enigma_label_to_col(label, prefix):
    h, region = label.split('_', 1)
    return f"{'lh' if h=='L' else 'rh'}.{prefix}.{region}"

MEASURES = {
    'CT':  [enigma_label_to_col(l, 'CT') for l in fc_ctx_labels],
    'GMV': [enigma_label_to_col(l, 'GM') for l in fc_ctx_labels],
    'SA':  [enigma_label_to_col(l, 'SA') for l in fc_ctx_labels],
}

ALL_SCTX  = [c for c in df.columns if c.startswith('SubcGM.')]
SCTX_COLS = [c for c in ALL_SCTX if 'Ventricle' not in c and 'Vent' not in c and 'Cereb' not in c]

# 16-slot subcortical order for plot_subcortical (ventricles = NaN)
_SCTX_TARGETS = ['Accumbens.area','Amygdala','Caudate','Hippocampus',
                  'Pallidum','Putamen','Thalamus','Ventricle']
def _match(side, target, cols):
    for c in cols:
        if side.lower() in c.lower() and target.lower() in c.lower():
            return c
    return None
SCTX_16 = [_match('Left',t,df.columns) for t in _SCTX_TARGETS] + \
          [_match('Right',t,df.columns) for t in _SCTX_TARGETS]


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

# =============================================================================
# 3. MODEL: per region, one dx at a time vs CN
# =============================================================================

def fit_lm_full(data: pd.DataFrame, cols: list) -> dict:
    """
    Fit ONE model on the full sample per region:
        centile ~ age_z + C(sex) + C(study_site)
               + C(dx, Treatment(reference="CN"))
               + C(dx):age_z
               + C(dx):C(sex)    [only when both sexes present]
    Returns dict with:
      - one key per dx (BD, SCZ, Schizoaffective): DataFrame with beta/t/p per region
      - '_all_params': DataFrame with ALL model coefficients per region
    Sex terms are omitted when data is sex-stratified (one sex only).
    """
    raw      = {dx: [] for dx in DIAGNOSES}
    raw_full = []   # all params from every model fit

    for col in [c for c in cols if c in data.columns]:
        safe = col.replace('.', '_')
        tmp  = data[[col, 'age', 'sex', 'study_site', 'dx']].copy().dropna()
        tmp['age_z'] = (tmp['age'] - tmp['age'].mean()) / tmp['age'].std(ddof=1)
        tmp = tmp.rename(columns={col: safe})
        n_sexes  = tmp['sex'].nunique()
        sex_main = '+ C(sex)' if n_sexes > 1 else ''
        sex_int  = '+ C(dx):C(sex)' if n_sexes > 1 else ''
        model = smf.ols(
            f'{safe} ~ age_z {sex_main} + C(study_site)'
            f' + C(dx, Treatment(reference="CN"))'
            f' + C(dx):age_z'
            f' {sex_int}',
            data=tmp
        ).fit()

        # Extract per-dx coefficients
        for dx in DIAGNOSES:
            coef_keys = [k for k in model.params.index if dx in k and ':' not in k]
            if not coef_keys:
                continue
            coef_key = coef_keys[0]
            raw[dx].append({
                'col':   col,
                'beta':  model.params[coef_key],
                't_val': model.tvalues[coef_key],
                'p_unc': model.pvalues[coef_key],
                'n':     int(model.nobs),
            })

        # Store ALL model parameters for this region
        full_row = {'col': col, 'n': int(model.nobs), 'r2': model.rsquared}
        for param in model.params.index:
            # clean param name for use as column key
            key = (param.replace('[', '_').replace(']', '')
                       .replace('(', '').replace(')', '')
                       .replace(',', '').replace('"', '')
                       .replace(' ', '_').replace('.', '_'))
            full_row[f'beta_{key}']  = model.params[param]
            full_row[f't_{key}']     = model.tvalues[param]
            full_row[f'p_unc_{key}'] = model.pvalues[param]
        raw_full.append(full_row)

    # Build per-dx DataFrames with FDR correction
    out = {}
    for dx in DIAGNOSES:
        if not raw[dx]:
            out[dx] = pd.DataFrame(columns=['beta','t_val','p_unc','p_fdr','n'])
            continue
        res = pd.DataFrame(raw[dx]).set_index('col')
        _, res['p_fdr'], _, _ = multipletests(res['p_unc'], method='fdr_bh')
        out[dx] = res

    # Full params table (no FDR — too many params; report raw)
    if raw_full:
        out['_all_params'] = pd.DataFrame(raw_full).set_index('col')
    else:
        out['_all_params'] = pd.DataFrame()

    return out


def fit_lm_dx(data: pd.DataFrame, cols: list, dx: str) -> pd.DataFrame:
    """
    Wrapper used by spin/shuffle tests (within-sex):
    Fit full-sample model and return results for a specific dx.
    """
    return fit_lm_full(data, cols)[dx]

def cohens_d_from_model(res: pd.DataFrame, cols: list) -> np.ndarray:
    """
    Model-derived Cohen's d = t / sqrt(n).
    Non-significant regions (p_fdr >= 0.05) are set to NaN.
    """
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
# 4. FIT MODELS FOR ALL DX × MEASURES
# =============================================================================
print("Fitting linear models …")

# results[measure][dx] = DataFrame (beta, t_val, p_unc, p_fdr)
results = {m: {} for m in MEASURES}
d_ctx   = {m: {} for m in MEASURES}   # Cohen's d cortical
sig_ctx = {m: {} for m in MEASURES}   # FDR significance mask

d_sctx   = {}   # Cohen's d subcortical
sig_sctx = {}

# Fit one model per region on the full sample (all dx vs CN, sex as covariate)
for measure, cols in MEASURES.items():
    print(f"  Fitting {measure} …")
    full_res = fit_lm_full(df, cols)
    results[measure]['_all_params'] = full_res.get('_all_params', pd.DataFrame())
    for dx in DIAGNOSES:
        results[measure][dx] = full_res[dx]
        d_ctx[measure][dx]   = cohens_d_from_model(results[measure][dx], cols)
        sig_ctx[measure][dx] = ~np.isnan(d_ctx[measure][dx])

# Print cortical results
for measure in MEASURES:
    for dx in DIAGNOSES:
        print_results_table(results[measure][dx], f'{measure} | {dx} vs CN')

# Subcortical — same single model
print("  Fitting SubcGM …")
sctx_full_res = fit_lm_full(df, SCTX_COLS)
for dx in DIAGNOSES:
    print_results_table(sctx_full_res[dx], f'SubcGM | {dx} vs CN')
    sctx_res     = sctx_full_res[dx]
    d_sctx[dx]   = cohens_d_from_model(sctx_res, [c for c in SCTX_16 if c])
    # Pad back to 16 slots (ventricles → NaN)
    d16 = np.full(16, np.nan)
    valid_idx = [i for i, c in enumerate(SCTX_16) if c]
    for k, i in enumerate(valid_idx):
        d16[i] = d_sctx[dx][k]
    d_sctx[dx]   = d16
    sig_sctx[dx] = ~np.isnan(d16)
    print(f"  {dx}: done")

# =============================================================================
# 5. VERTEX MAPPING HELPERS
# =============================================================================
_lh_annot,_,_lh_names = nib.freesurfer.read_annot(
    os.path.join(_PKG,'permutation_testing/annot/fsa5_lh_aparc.annot'))
_rh_annot,_,_rh_names = nib.freesurfer.read_annot(
    os.path.join(_PKG,'permutation_testing/annot/fsa5_rh_aparc.annot'))
_lh_names=[n.decode() if isinstance(n,bytes) else n for n in _lh_names]
_rh_names=[n.decode() if isinstance(n,bytes) else n for n in _rh_names]

def ctx_to_vertices(d68, sig_mask):
    lh = np.full(10242, np.nan); rh = np.full(10242, np.nan)
    for i, lbl in enumerate(fc_ctx_labels):
        h, region = str(lbl).split('_',1)
        val = d68[i] if sig_mask[i] else np.nan
        if h=='L' and region in _lh_names: lh[_lh_annot==_lh_names.index(region)] = val
        elif h=='R' and region in _rh_names: rh[_rh_annot==_rh_names.index(region)] = val
    return np.concatenate([lh, rh])

# =============================================================================
# 6. BRAIN MAPS – one combined figure per measure
#    Rows = BD / SCZ / SA  |  Columns = 4 cortical views (or subcortical)
# =============================================================================
NAN_COLOR = (0.85, 0.85, 0.85, 1)
CMAP      = 'RdBu_r'
ZOOM      = 1.25
SURF_SIZE = (1600, 400)
DX_COLORS     = {'BD': '#E8851B'}
MEASURE_COLORS = {'CT': '#6B3FA0', 'GMV': '#A8221C', 'SA': '#324F7D'}  # CT=purple, GMV=red, SA=blue

print("\nRendering brain maps …")

# Fit models per sex group for map rendering
# d_ctx_sex[sex_label][measure] = Cohen's d array (68,)
# d_sctx_sex[sex_label]         = Cohen's d array (16,)
d_ctx_sex  = {}
d_sctx_sex = {}
sig_ctx_sex  = {}
sig_sctx_sex = {}

for sex_label, sex_data in SEX_GROUPS:
    d_ctx_sex[sex_label]  = {}
    sig_ctx_sex[sex_label] = {}
    for measure, cols in MEASURES.items():
        res = fit_lm_full(sex_data, cols)['BD']
        d_ctx_sex[sex_label][measure]  = cohens_d_from_model(res, cols)
        sig_ctx_sex[sex_label][measure] = ~np.isnan(d_ctx_sex[sex_label][measure])
    # Subcortical
    sctx_res = fit_lm_full(sex_data, SCTX_COLS)['BD']
    d16 = np.full(16, np.nan)
    for i, col in enumerate(SCTX_16):
        if col and col in sctx_res.index:
            row = sctx_res.loc[col]
            d16[i] = row['t_val'] / np.sqrt(row['n']) if row['p_fdr'] < 0.05 else np.nan
    d_sctx_sex[sex_label]  = d16
    sig_sctx_sex[sex_label] = ~np.isnan(d16)
    print(f"  Maps fitted: {sex_label}")

SEX_ROW_LABELS = [('Whole', 'Whole group'), ('Male', 'Males'), ('Female', 'Females')]

# Global vmax across all measures and sex groups (computed after model fits)
# Will be set after d_ctx_sex and d_sctx_sex are populated

# Global vmax
_all_ctx  = [d_ctx_sex[sl][m]  for sl, _ in SEX_ROW_LABELS for m in MEASURES]
_all_sctx = [d_sctx_sex[sl] for sl, _ in SEX_ROW_LABELS]
_ctx_vals  = [np.nanmax(np.abs(a)) for a in _all_ctx  if not np.all(np.isnan(a))]
_sctx_vals = [np.nanmax(np.abs(a)) for a in _all_sctx if not np.all(np.isnan(a))]
GLOBAL_VMAX      = np.ceil((max(_ctx_vals)  if _ctx_vals  else 0.5) * 10) / 10
GLOBAL_SCTX_VMAX = np.ceil((max(_sctx_vals) if _sctx_vals else 0.5) * 10) / 10
print(f'Global cortical vmax:    {GLOBAL_VMAX}')
print(f'Global subcortical vmax: {GLOBAL_SCTX_VMAX}')

def render_bd_measure(measure, sex_rows, suffix):
    rowtxt = " / ".join(disp for _, disp in sex_rows)
    vmax = GLOBAL_VMAX

    # ── Cortical rows ────────────────────────────────────────────────────────
    row_imgs = []
    for sex_label, sex_display in sex_rows:
        tex = ctx_to_vertices(d_ctx_sex[sex_label][measure],
                              sig_ctx_sex[sex_label][measure])
        tmp = f'_tmp_ctx_{measure}_{sex_label}.png'
        plot_cortical(
            array_name=tex, surface_name='fsa5',
            color_bar=True, color_range=(-vmax, vmax),
            label_text={'left': [sex_display]},
            cmap=CMAP, nan_color=NAN_COLOR,
            zoom=ZOOM, size=SURF_SIZE,
            interactive=False, screenshot=True,
            filename=tmp, transparent_bg=False,
        )
        row_imgs.append(plt.imread(tmp)); os.remove(tmp)
        print(f"  Cortical {measure} [{sex_display}]{suffix} done")

    h = sum(im.shape[0] for im in row_imgs)
    w = max(im.shape[1] for im in row_imgs)
    canvas = np.ones((h, w, row_imgs[0].shape[2]), dtype=row_imgs[0].dtype)
    y = 0
    for im in row_imgs:
        canvas[y:y+im.shape[0], :im.shape[1]] = im; y += im.shape[0]

    fig, ax = plt.subplots(figsize=(w/80, h/80))
    ax.imshow(canvas); ax.axis('off')
    fig.suptitle(
        f"Cohen's d – {measure} | BD vs CN\n"
        f"Rows: {rowtxt}  •  Bright = FDR-sig (p<0.05)",
        fontsize=13, y=1.01
    )
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(SURF_DIR, f'bd_cortical_{measure}{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: bd_cortical_{measure}{suffix}.png")

    # ── Subcortical rows ─────────────────────────────────────────────────────
    sctx_vmax = GLOBAL_SCTX_VMAX
    sctx_rows = []
    for sex_label, sex_display in sex_rows:
        d16 = d_sctx_sex[sex_label].copy()
        tmp = f'_tmp_sctx_{measure}_{sex_label}.png'
        plot_subcortical(
            array_name=d16, ventricles=True,
            color_bar=True, color_range=(-sctx_vmax, sctx_vmax),
            label_text={'left': [sex_display]},
            cmap=CMAP, nan_color=NAN_COLOR,
            zoom=ZOOM, size=SURF_SIZE,
            interactive=False, screenshot=True,
            filename=tmp, transparent_bg=False,
        )
        sctx_rows.append(plt.imread(tmp)); os.remove(tmp)
        print(f"  Subcortical {measure} [{sex_display}]{suffix} done")

    sh = sum(im.shape[0] for im in sctx_rows)
    sw = max(im.shape[1] for im in sctx_rows)
    s_canvas = np.ones((sh, sw, sctx_rows[0].shape[2]), dtype=sctx_rows[0].dtype)
    y = 0
    for im in sctx_rows:
        s_canvas[y:y+im.shape[0], :im.shape[1]] = im; y += im.shape[0]

    fig2, ax2 = plt.subplots(figsize=(sw/100, sh/100))
    ax2.imshow(s_canvas); ax2.axis('off')
    fig2.suptitle(
        f"Cohen's d – Subcortical | {measure} | BD vs CN\n"
        f"Rows: {rowtxt}  •  Bright = FDR-sig (p<0.05)",
        fontsize=13, y=1.01
    )
    fig2.tight_layout(pad=0)
    fig2.savefig(os.path.join(SURF_DIR, f'bd_subcortical_{measure}{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: bd_subcortical_{measure}{suffix}.png")

# Two variants: with the whole group, and without it (sex rows only)
BD_VARIANTS = [("", SEX_ROW_LABELS), ("_no_whole", [r for r in SEX_ROW_LABELS if r[0] != 'Whole'])]
for suffix, sex_rows in BD_VARIANTS:
    for measure, cols in MEASURES.items():
        render_bd_measure(measure, sex_rows, suffix)

# =============================================================================
# 7. SPIN TEST – Male vs Female t-stat maps  (BD vs CN effect)
#    Compares whether the spatial pattern of the BD effect is similar
#    in males and females (cortical, per measure)
# =============================================================================
print("\nRunning cortical spin tests: Male vs Female (BD effect) …")

spin_rows = []

for measure, cols in MEASURES.items():
    # Fit model separately within each sex group
    t_male   = fit_lm_full(males,   cols)['BD'].reindex(cols)['t_val'].values
    t_female = fit_lm_full(females, cols)['BD'].reindex(cols)['t_val'].values

    p_spin, null_dist = spin_test(
        t_male, t_female,
        surface_name='fsa5',
        parcellation_name='aparc',
        type='spearman',
        n_rot=1000,
        null_dist=True,
    )
    r_emp = float(stats.spearmanr(t_male, t_female)[0])
    spin_rows.append({
        'measure': measure, 'comparison': 'Male vs Female',
        'r': r_emp, 'p_spin': p_spin,
    })
    print(f"  {measure} | Male vs Female: r={r_emp:+.3f}, p={p_spin:.3f}")

    # Null distribution plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_dist, bins=50, density=True,
            color=MEASURE_COLORS[measure], edgecolor='white', lw=0.4, alpha=0.88)
    ax.axvline(r_emp, lw=2, ls='--', color='k', dashes=(2, 3),
               label=f'$\\rho$={r_emp:+.3f}\n$p_{{spin}}$={p_spin:.3f}')
    ax.set_xlabel('Null correlations (spun)'); ax.set_ylabel('Density')
    ax.set_title(f'BD vs CN | {measure} | Male vs Female map', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(SEX_DIR, f'spin_BD_{measure}_Male_vs_Female.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Scatter plot: male vs female t-stats per region
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(t_male, t_female, c=MEASURE_COLORS[measure], alpha=0.75,
                edgecolors='white', linewidths=0.4, s=60)
    for col, tm, tf in zip(cols, t_male, t_female):
        if abs(tm) > np.percentile(np.abs(t_male), 90) or \
           abs(tf) > np.percentile(np.abs(t_female), 90):
            short = col.split('.', 2)[-1]
            ax2.annotate(short, (tm, tf), fontsize=6, alpha=0.8,
                         xytext=(4, 2), textcoords='offset points')
    lims = [min(t_male.min(), t_female.min()) - 0.3,
            max(t_male.max(), t_female.max()) + 0.3]
    ax2.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='identity')
    ax2.axhline(0, color='grey', linewidth=0.5)
    ax2.axvline(0, color='grey', linewidth=0.5)
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel('t-statistic – Males (BD vs CN)', fontsize=10)
    ax2.set_ylabel('t-statistic – Females (BD vs CN)', fontsize=10)
    ax2.set_title(
        f'BD vs CN | {measure} | Male vs Female\n'
        f'$\\rho$ = {r_emp:+.3f},  $p_{{spin}}$ = {p_spin:.3f}',
        fontsize=11)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.legend(frameon=False, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(SEX_DIR, f'spin_BD_{measure}_scatter_Male_vs_Female.png'),
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)

spin_df = pd.DataFrame(spin_rows)
spin_df.to_csv(os.path.join(SEX_DIR, 'spin_bd_male_vs_female_summary.csv'), index=False)
print("\nSpin test summary (Male vs Female):")
print(spin_df.to_string(index=False))

# =============================================================================
# 8. SHUFFLE TEST – Male vs Female subcortical t-maps  (BD vs CN effect)
# =============================================================================
print("\nRunning subcortical shuffle test: Male vs Female (BD effect) …")

shuf_rows = []

t_male_sctx   = fit_lm_full(males,   SCTX_COLS)['BD'].reindex(SCTX_COLS)['t_val'].values
t_female_sctx = fit_lm_full(females, SCTX_COLS)['BD'].reindex(SCTX_COLS)['t_val'].values

p_shuf, null_dist = shuf_test(
    t_male_sctx, t_female_sctx, n_rot=1000, type='spearman', null_dist=True)
r_emp = float(stats.spearmanr(t_male_sctx, t_female_sctx)[0])
shuf_rows.append({
    'comparison': 'Male vs Female',
    'r': r_emp, 'p_shuf': p_shuf,
})
print(f"  Subcortical | Male vs Female: r={r_emp:+.3f}, p={p_shuf:.3f}")

# Null distribution plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(null_dist, bins=50, density=True,
        color='#2A7F62', edgecolor='white', lw=0.4, alpha=0.88)  # green for shuffle
ax.axvline(r_emp, lw=2, ls='--', color='k', dashes=(2, 3),
           label=f'$\\rho$={r_emp:+.3f}\n$p_{{shuf}}$={p_shuf:.3f}')
ax.set_xlabel('Null correlations (shuffled)'); ax.set_ylabel('Density')
ax.set_title('BD vs CN | Subcortical | Male vs Female map', fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(SEX_DIR, 'shuf_BD_subcortical_Male_vs_Female.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)

# Scatter plot: male vs female subcortical t-stats
fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.scatter(t_male_sctx, t_female_sctx, c='#2A7F62', alpha=0.80,  # green for shuffle
            edgecolors='white', linewidths=0.4, s=80)
for col, tm, tf in zip(SCTX_COLS, t_male_sctx, t_female_sctx):
    short = col.replace('SubcGM.', '').replace('.', ' ')
    ax3.annotate(short, (tm, tf), fontsize=6.5, alpha=0.85,
                 xytext=(4, 2), textcoords='offset points')
lims3 = [min(t_male_sctx.min(), t_female_sctx.min()) - 0.5,
         max(t_male_sctx.max(), t_female_sctx.max()) + 0.5]
ax3.plot(lims3, lims3, 'k--', linewidth=0.8, alpha=0.5, label='identity')
ax3.axhline(0, color='grey', linewidth=0.5)
ax3.axvline(0, color='grey', linewidth=0.5)
ax3.set_xlim(lims3); ax3.set_ylim(lims3)
ax3.set_xlabel('t-statistic – Males (BD vs CN)', fontsize=10)
ax3.set_ylabel('t-statistic – Females (BD vs CN)', fontsize=10)
ax3.set_title(
    f'BD vs CN | Subcortical | Male vs Female\n'
    f'$\\rho$ = {r_emp:+.3f},  $p_{{shuf}}$ = {p_shuf:.3f}',
    fontsize=11)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.legend(frameon=False, fontsize=8)
fig3.tight_layout()
fig3.savefig(os.path.join(SEX_DIR, 'shuf_BD_subcortical_scatter_Male_vs_Female.png'),
             dpi=300, bbox_inches='tight')
plt.close(fig3)

shuf_df = pd.DataFrame(shuf_rows)
shuf_df.to_csv(os.path.join(SEX_DIR, 'shuf_bd_male_vs_female_summary.csv'), index=False)
print("\nShuffle test summary (Male vs Female):")
print(shuf_df.to_string(index=False))

# =============================================================================
# 9. EXPORT SUMMARY TABLES
# =============================================================================

# ── 9a. Cohen's d + dx stats (per dx × region) ───────────────────────────────
rows = []
for measure, cols in MEASURES.items():
    for col, lbl in zip(cols, fc_ctx_labels):
        row = {'measure': measure, 'region': col, 'enigma_label': str(lbl)}
        for dx in DIAGNOSES:
            i = cols.index(col)
            row[f'd_{dx}']     = d_ctx[measure][dx][i]
            row[f't_{dx}']     = (results[measure][dx].loc[col,'t_val']
                                   if col in results[measure][dx].index else np.nan)
            row[f'p_unc_{dx}'] = (results[measure][dx].loc[col,'p_unc']
                                   if col in results[measure][dx].index else np.nan)
            row[f'p_fdr_{dx}'] = (results[measure][dx].loc[col,'p_fdr']
                                   if col in results[measure][dx].index else np.nan)
            row[f'sig_{dx}']   = int(sig_ctx[measure][dx][i])
        rows.append(row)
for i, col in enumerate(SCTX_16):
    row = {'measure': 'SubcGM', 'region': col or 'unknown', 'enigma_label': ''}
    for dx in DIAGNOSES:
        row[f'd_{dx}']     = d_sctx[dx][i]
        row[f't_{dx}']     = (sctx_full_res[dx].loc[col, 't_val']
                               if col and col in sctx_full_res[dx].index else np.nan)
        row[f'p_unc_{dx}'] = (sctx_full_res[dx].loc[col, 'p_unc']
                               if col and col in sctx_full_res[dx].index else np.nan)
        row[f'p_fdr_{dx}'] = (sctx_full_res[dx].loc[col, 'p_fdr']
                               if col and col in sctx_full_res[dx].index else np.nan)
        row[f'sig_{dx}']   = int(sig_sctx[dx][i])
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(SURF_DIR, 'transdx_cohens_d.csv'), index=False)

# ── 9b. Full model params table (ALL predictors: age_z, sex, site, dx, interactions)
full_rows = []
for measure, cols in MEASURES.items():
    ap = results[measure].get('_all_params', pd.DataFrame())
    if ap.empty:
        continue
    for col, lbl in zip(cols, fc_ctx_labels):
        if col not in ap.index:
            continue
        row = {'measure': measure, 'region': col, 'enigma_label': str(lbl)}
        row.update(ap.loc[col].to_dict())
        full_rows.append(row)

# Subcortical full params
for measure in MEASURES:
    # refit subcortical to get full params (sctx_full_res only stores dx keys)
    break  # sctx already stored below

if full_rows:
    pd.DataFrame(full_rows).to_csv(
        os.path.join(SURF_DIR, 'transdx_all_params.csv'), index=False)
    print(f"  Saved: transdx_all_params.csv "
          f"({len(full_rows)} rows × {len(pd.DataFrame(full_rows).columns)} cols)")

# ── 9c. Subcortical full params ───────────────────────────────────────────────
sctx_full_rows = []
for col in SCTX_COLS:
    if '_all_params' not in sctx_full_res or sctx_full_res['_all_params'].empty:
        break
    if col not in sctx_full_res['_all_params'].index:
        continue
    row = {'measure': 'SubcGM', 'region': col, 'enigma_label': ''}
    row.update(sctx_full_res['_all_params'].loc[col].to_dict())
    sctx_full_rows.append(row)
if sctx_full_rows:
    pd.DataFrame(sctx_full_rows).to_csv(
        os.path.join(SURF_DIR, 'transdx_sctx_all_params.csv'), index=False)
    print(f"  Saved: transdx_sctx_all_params.csv")

# =============================================================================
# 11. GRAND SUMMARY FIGURE
# =============================================================================
print("\nCreating grand summary figure...")

# Composite laid out HORIZONTALLY: measures placed side by side (normalize height).
TARGET_HEIGHT = 1400
for suffix in ["", "_no_whole"]:
    summary_imgs = []
    for measure in MEASURES:
        fname = os.path.join(SURF_DIR, f'bd_cortical_{measure}{suffix}.png')
        if os.path.exists(fname):
            img = Image.open(fname)
            w, h = img.size
            img = img.resize((int(w * TARGET_HEIGHT / h), TARGET_HEIGHT), Image.Resampling.LANCZOS)
            summary_imgs.append(np.array(img))
    gmv_file = os.path.join(SURF_DIR, f'bd_subcortical_GMV{suffix}.png')
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
        out_file = os.path.join(SURF_DIR, f"bd_complete_brain_summary{suffix}.png")
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_file}")
# =============================================================================
# 10. COMBINED SUMMARY TABLE
# =============================================================================
summary_rows = []
for row in spin_rows:
    summary_rows.append({
        'test': 'spin', 'modality': 'cortical',
        'measure': row['measure'], 'comparison': row['comparison'],
        'r': row['r'], 'p': row['p_spin'],
    })
for row in shuf_rows:
    summary_rows.append({
        'test': 'shuffle', 'modality': 'subcortical',
        'measure': 'SubcGM', 'comparison': row['comparison'],
        'r': row['r'], 'p': row['p_shuf'],
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(SEX_DIR, 'bd_spin_shuf_summary.csv'), index=False)
print('\nCombined spin/shuffle summary:')
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.width', 100)
print(summary_df.to_string(index=False))
pd.reset_option('display.float_format')

print("\nAll files saved:")
for measure in MEASURES:
    print(f"  bd_cortical_{measure}.png  |  bd_subcortical_{measure}.png")
print("  spin_bd_male_vs_female_summary.csv  |  shuf_bd_male_vs_female_summary.csv")
print("  bd_spin_shuf_summary.csv")
print("  transdx_cohens_d.csv")
print("  + null distribution and scatter PNGs per measure")

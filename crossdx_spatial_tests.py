# %%
"""
Spin & Shuffle Tests  |  SCZ vs Schizoaffective vs BD
====================================================================================
Fits one OLS model per region on the full sample (CN as reference):
    centile ~ age_z + [C(sex)] + C(study_site)
           + C(dx, Treatment(reference="CN"))
           + C(dx):age_z + [C(dx):C(sex)]

Extracts t-statistics for each dx vs CN (raw, no FDR masking).
Then runs:
  • Spin test  (cortical)   : compare t-maps between dx pairs
  • Shuffle test (subcortical): same cross-dx comparisons

Pairs: BD–SCZ, BD–Schizoaffective, SCZ–Schizoaffective
Groups: Whole sample, Males, Females

Outputs (in transdx_spin_shuf_unthresh/)
-----------------------------------------
  spin_{measure}_{sex}_{dx1}_{dx2}.png
  shuf_{sex}_{dx1}_{dx2}.png
  spin_summary.csv
  shuf_summary.csv
  transdx_tvals.csv   (t-value per region × dx × measure × sex group)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.multitest import multipletests
from scipy import stats
try:
    from enigmatoolbox.datasets import load_fc
    from enigmatoolbox.permutation_testing import spin_test, shuf_test
    _HAS_ENIGMA = True
except ImportError as _e:
    _HAS_ENIGMA = False
    print(f"[crossdx_spatial_tests] enigmatoolbox not available ({_e}).\n"
          f"  Spin/shuffle spatial tests require the venv with enigmatoolbox installed.\n"
          f"  Run this script from the venv to get full output.")
    raise SystemExit(1)
from data_utils import load_crossdx_matched

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR = '../results/crossdiagnostic/transdx_spin_shuf_unthresh'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}/\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = load_crossdx_matched().copy()

df['dx'] = pd.Categorical(
    df['dx'], categories=['CN', 'BD', 'SCZ', 'Schizoaffective'], ordered=False)

males   = df[df['sex'] == 'Male'].copy()
females = df[df['sex'] == 'Female'].copy()

SEX_GROUPS = [('Whole', df), ('Male', males), ('Female', females)]
DIAGNOSES  = ['BD', 'SCZ', 'Schizoaffective']
DX_PAIRS   = [('BD', 'SCZ'), ('BD', 'Schizoaffective'), ('SCZ', 'Schizoaffective')]
DX_COLORS  = {'BD': '#E8851B', 'SCZ': '#A8221C', 'Schizoaffective': '#324F7D'}

print(f"Loaded {len(df)} subjects | {df['study_site'].nunique()} sites")
print(df['dx'].value_counts().to_string(), '\n')
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

# %%
# =============================================================================
# 3. MODEL — raw t-values
#    Same one model per region, but t-stats are NOT masked by FDR.
# =============================================================================

def fit_lm_tvals(data: pd.DataFrame, cols: list) -> dict:
    """
    Fit one model per region and return raw t-values.
    """

    raw = {dx: [] for dx in DIAGNOSES}

    for col in [c for c in cols if c in data.columns]:

        safe = col.replace('.', '_')

        tmp = data[[col, 'age', 'sex', 'study_site', 'dx']].copy().dropna()
        tmp['age_z'] = (
            tmp['age'] - tmp['age'].mean()
        ) / tmp['age'].std(ddof=1)

        tmp = tmp.rename(columns={col: safe})

        n_sexes = tmp['sex'].nunique()

        sex_main = '+ C(sex)' if n_sexes > 1 else ''

        model = smf.ols(
            f'{safe} ~ age_z'
            f' {sex_main}'
            f' + C(study_site)'
            f' + C(dx, Treatment(reference="CN"))'
            f' + C(dx):age_z',
            data=tmp
        ).fit()

        for dx in DIAGNOSES:

            key = f'C(dx, Treatment(reference="CN"))[T.{dx}]'

            if key not in model.tvalues:
                raw[dx].append({
                    'col': col,
                    't_val': np.nan
                })
            else:
                raw[dx].append({
                    'col': col,
                    't_val': model.tvalues[key]
                })

    out = {}

    for dx in DIAGNOSES:
        out[dx] = (
            pd.DataFrame(raw[dx])
            .set_index('col')['t_val']
        )

    return out

# %%
# =============================================================================
# 4. FIT MODELS PER GROUP
# =============================================================================
print("Fitting models (raw t-values) …")

# t_ctx[sex_label][measure][dx] = np.array (68,)
# t_sctx[sex_label][dx]         = np.array (n_sctx,)
t_ctx  = {}
t_sctx = {}

for sex_label, sex_data in SEX_GROUPS:
    t_ctx[sex_label]  = {}
    t_sctx[sex_label] = {}

    for measure, cols in MEASURES.items():
        tvals = fit_lm_tvals(sex_data, cols)
        t_ctx[sex_label][measure] = {
            dx: tvals[dx].reindex(cols).values for dx in DIAGNOSES
        }

    sctx_tvals = fit_lm_tvals(sex_data, SCTX_COLS)
    t_sctx[sex_label] = {
        dx: sctx_tvals[dx].reindex(SCTX_COLS).values for dx in DIAGNOSES
    }
    print(f"  {sex_label}: done")

# %%
# =============================================================================
# 5. PRINT TABLE
# =============================================================================
def print_tval_table(t_arr, cols, title=''):
    tbl = pd.DataFrame({'region': cols, 't_val': t_arr})
    if title:
        print(f"\n{'─'*60}\n  {title}\n{'─'*60}")
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 100)
    print(tbl.to_string(index=False))
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_rows')

for sex_label, _ in SEX_GROUPS:
    for measure, cols in MEASURES.items():
        for dx in DIAGNOSES:
            print_tval_table(
                t_ctx[sex_label][measure][dx], cols,
                f'{measure} | {dx} vs CN | {sex_label}'
            )
    for dx in DIAGNOSES:
        print_tval_table(
            t_sctx[sex_label][dx], SCTX_COLS,
            f'SubcGM | {dx} vs CN | {sex_label}'
        )

# =============================================================================
# 6. SPIN TEST – cortical t-maps 
# =============================================================================
print("\nRunning cortical spin tests  …")

spin_rows = []

for measure, cols in MEASURES.items():
    for sex_label, _ in SEX_GROUPS:
        for dx1, dx2 in DX_PAIRS:
            t1 = t_ctx[sex_label][measure][dx1]
            t2 = t_ctx[sex_label][measure][dx2]

            # Replace NaN with 0 for spin test (NaN = model failed for that region)
            t1_clean = np.nan_to_num(t1, nan=0.0)
            t2_clean = np.nan_to_num(t2, nan=0.0)

            p_spin, null_dist = spin_test(
                t1_clean, t2_clean,
                surface_name='fsa5',
                parcellation_name='aparc',
                type='spearman',
                n_rot=1000,
                null_dist=True,
            )
            r_emp = float(stats.spearmanr(t1_clean, t2_clean)[0])
            spin_rows.append({
                'measure': measure, 'sex': sex_label,
                'dx1': dx1, 'dx2': dx2,
                'r': r_emp, 'p_spin': p_spin,
            })
            print(f"  {measure} | {sex_label} | {dx1}–{dx2}: "
                  f"r={r_emp:+.3f}, p={p_spin:.3f}")

            # Null distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(null_dist, bins=50, density=True,
                    color=DX_COLORS[dx1], edgecolor='white', lw=0.4, alpha=0.88)
            ax.axvline(r_emp, lw=2, ls='--', color='k', dashes=(2, 3),
                       label=f'$\\rho$={r_emp:+.3f}\n$p_{{spin}}$={p_spin:.3f}')
            ax.set_xlabel('Null correlations (spun)')
            ax.set_ylabel('Density')
            ax.set_title(f'{measure} | {sex_label} | {dx1} vs {dx2}\n'
                         f'(raw t-maps)', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR,
                        f'spin_{measure}_{sex_label}_{dx1}_{dx2}.png'),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

            # Scatter plot: t-map dx1 vs dx2
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(t1_clean, t2_clean, c=DX_COLORS[dx1],
                        alpha=0.75, edgecolors='white', linewidths=0.4, s=60)
            lims = [min(t1_clean.min(), t2_clean.min()) - 0.3,
                    max(t1_clean.max(), t2_clean.max()) + 0.3]
            ax2.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
            ax2.axhline(0, color='grey', linewidth=0.5)
            ax2.axvline(0, color='grey', linewidth=0.5)
            ax2.set_xlim(lims); ax2.set_ylim(lims)
            ax2.set_xlabel(f't-statistic – {dx1} vs CN', fontsize=10)
            ax2.set_ylabel(f't-statistic – {dx2} vs CN', fontsize=10)
            ax2.set_title(
                f'{measure} | {sex_label} | {dx1} vs {dx2}\n'
                f'$\\rho$ = {r_emp:+.3f},  $p_{{spin}}$ = {p_spin:.3f}',
                fontsize=11)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            fig2.tight_layout()
            fig2.savefig(os.path.join(OUTPUT_DIR,
                         f'spin_scatter_{measure}_{sex_label}_{dx1}_{dx2}.png'),
                         dpi=120, bbox_inches='tight')
            plt.close(fig2)

spin_df = pd.DataFrame(spin_rows)

# FDR correction within each measure (across sex groups × dx pairs)
spin_df['p_fdr'] = np.nan
for measure in MEASURES:
    mask = spin_df['measure'] == measure
    _, fdr_vals, _, _ = multipletests(
        spin_df.loc[mask, 'p_spin'], method='fdr_bh')
    spin_df.loc[mask, 'p_fdr'] = fdr_vals

spin_df.to_csv(
    os.path.join(OUTPUT_DIR, 'spin_summary.csv'),
    index=False
)
print("\nSpin test summary (with FDR within measure):")
print(spin_df.to_string(index=False))

# =============================================================================
# 7. SHUFFLE TEST – subcortical t-maps 
# =============================================================================
print("\nRunning subcortical shuffle tests  …")

shuf_rows = []

for sex_label, _ in SEX_GROUPS:
    for dx1, dx2 in DX_PAIRS:
        t1 = np.nan_to_num(t_sctx[sex_label][dx1], nan=0.0)
        t2 = np.nan_to_num(t_sctx[sex_label][dx2], nan=0.0)

        p_shuf, null_dist = shuf_test(
            t1, t2, n_rot=1000, type='spearman', null_dist=True)
        r_emp = float(stats.spearmanr(t1, t2)[0])
        shuf_rows.append({
            'sex': sex_label, 'dx1': dx1, 'dx2': dx2,
            'r': r_emp, 'p_shuf': p_shuf,
        })
        print(f"  Subcortical | {sex_label} | {dx1}–{dx2}: "
              f"r={r_emp:+.3f}, p={p_shuf:.3f}")

        # Null distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null_dist, bins=50, density=True,
                color=DX_COLORS[dx1], edgecolor='white', lw=0.4, alpha=0.88)
        ax.axvline(r_emp, lw=2, ls='--', color='k', dashes=(2, 3),
                   label=f'$\\rho$={r_emp:+.3f}\n$p_{{shuf}}$={p_shuf:.3f}')
        ax.set_xlabel('Null correlations (shuffled)')
        ax.set_ylabel('Density')
        ax.set_title(f'Subcortical | {sex_label} | {dx1} vs {dx2}\n'
                     f'(raw t-maps)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,
                    f'shuf_{sex_label}_{dx1}_{dx2}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        # Scatter plot: subcortical t-map dx1 vs dx2
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.scatter(t1, t2, c=DX_COLORS[dx1],
                    alpha=0.80, edgecolors='white', linewidths=0.4, s=80)
        # Label all subcortical regions
        for col, tm, tf in zip(SCTX_COLS, t1, t2):
            short = col.replace('SubcGM.', '').replace('.', ' ')
            ax3.annotate(short, (tm, tf), fontsize=6.5, alpha=0.85,
                         xytext=(4, 2), textcoords='offset points')
        lims3 = [min(t1.min(), t2.min()) - 0.5,
                 max(t1.max(), t2.max()) + 0.5]
        ax3.plot(lims3, lims3, 'k--', linewidth=0.8, alpha=0.5)
        ax3.axhline(0, color='grey', linewidth=0.5)
        ax3.axvline(0, color='grey', linewidth=0.5)
        ax3.set_xlim(lims3); ax3.set_ylim(lims3)
        ax3.set_xlabel(f't-statistic – {dx1} vs CN', fontsize=10)
        ax3.set_ylabel(f't-statistic – {dx2} vs CN', fontsize=10)
        ax3.set_title(
            f'Subcortical | {sex_label} | {dx1} vs {dx2}\n'
            f'$\\rho$ = {r_emp:+.3f},  $p_{{shuf}}$ = {p_shuf:.3f}',
            fontsize=11)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR,
                     f'shuf_scatter_{sex_label}_{dx1}_{dx2}.png'),
                     dpi=120, bbox_inches='tight')
        plt.close(fig3)

shuf_df = pd.DataFrame(shuf_rows)

# FDR correction within SubcGM (across sex groups × dx pairs)
_, shuf_df['p_fdr'], _, _ = multipletests(
    shuf_df['p_shuf'], method='fdr_bh')

shuf_df.to_csv(os.path.join(OUTPUT_DIR, 'shuf_summary.csv'), index=False)
print("\nShuffle test summary (with FDR):")
print(shuf_df.to_string(index=False))

# %%
# =============================================================================
# 8. EXPORT T-VALUE TABLE
# =============================================================================
rows = []
for sex_label, _ in SEX_GROUPS:
    for measure, cols in MEASURES.items():
        for col, lbl in zip(cols, fc_ctx_labels):
            row = {'sex': sex_label, 'measure': measure,
                   'region': col, 'enigma_label': str(lbl)}
            for dx in DIAGNOSES:
                i = cols.index(col)
                row[f't_{dx}'] = t_ctx[sex_label][measure][dx][i]
            rows.append(row)
    for col in SCTX_COLS:
        row = {'sex': sex_label, 'measure': 'SubcGM',
               'region': col, 'enigma_label': ''}
        for dx in DIAGNOSES:
            i = SCTX_COLS.index(col)
            row[f't_{dx}'] = t_sctx[sex_label][dx][i]
        rows.append(row)

pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, 'transdx_tvals.csv'), index=False)

# =============================================================================
# COMBINED SUMMARY TABLE
# =============================================================================
summary_rows = []

# Spin — one row per measure × sex × pair
# Merge FDR values into summary rows
spin_fdr_map = {(r['measure'], r['sex'], r['dx1'], r['dx2']): r['p_fdr']
                for _, r in spin_df.iterrows()}
shuf_fdr_map = {(r['sex'], r['dx1'], r['dx2']): r['p_fdr']
                for _, r in shuf_df.iterrows()}

for row in spin_rows:
    summary_rows.append({
        'test':       'spin',
        'modality':   'cortical',
        'measure':    row['measure'],
        'sex':        row['sex'],
        'dx1':        row['dx1'],
        'dx2':        row['dx2'],
        'r':          row['r'],
        'p':          row['p_spin'],
        'p_fdr':      spin_fdr_map.get((row['measure'], row['sex'],
                                        row['dx1'], row['dx2']), np.nan),
    })

# Shuffle — one row per sex × pair (subcortical only)
for row in shuf_rows:
    summary_rows.append({
        'test':       'shuffle',
        'modality':   'subcortical',
        'measure':    'SubcGM',
        'sex':        row['sex'],
        'dx1':        row['dx1'],
        'dx2':        row['dx2'],
        'r':          row['r'],
        'p':          row['p_shuf'],
        'p_fdr':      shuf_fdr_map.get((row['sex'], row['dx1'],
                                         row['dx2']), np.nan),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, 'spin_shuf_summary.csv'), index=False)

print('\nCombined spin/shuffle summary :')
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.width', 120)
print(summary_df.to_string(index=False))
pd.reset_option('display.float_format')

print(f"\nAll files saved in: {OUTPUT_DIR}/")
print("  spin_summary.csv  |  shuf_summary.csv")
print("  spin_shuf_summary.csv")
print("  transdx_tvals.csv")
print("  + individual null distribution PNGs per pair × sex × measure")

# =============================================================================
# GLOBAL FDR ACROSS ALL SPIN + SHUFFLE TESTS
# =============================================================================

# Gather all p-values
all_pvals = np.concatenate([
    spin_df['p_spin'].values,
    shuf_df['p_shuf'].values
])

# Single BH correction across every test
_, all_fdr, _, _ = multipletests(
    all_pvals,
    method='fdr_bh'
)

# Split corrected values back
n_spin = len(spin_df)

spin_df['p_fdr_global'] = all_fdr[:n_spin]
shuf_df['p_fdr_global'] = all_fdr[n_spin:]

# %%
# ==========================================
# DEBUG ONE REGION
# ==========================================

col = MEASURES['CT'][0]

safe = col.replace('.', '_')

tmp = df[[col, 'age', 'sex', 'study_site', 'dx']].copy().dropna()
tmp['age_z'] = (tmp['age'] - tmp['age'].mean()) / tmp['age'].std(ddof=1)
tmp = tmp.rename(columns={col: safe})

model = smf.ols(
    f'{safe} ~ age_z + C(sex) + C(study_site)'
    f' + C(dx, Treatment(reference="CN"))'
    f' + C(dx):age_z'
    f' + C(dx):C(sex)',
    data=tmp
).fit()


print("Condition number:", model.condition_number)
print(model.summary())

print("\nDiagnosis coefficients:")
for k in model.tvalues.index:
    if "Treatment(reference" in k and ":" not in k:
        print(k, model.tvalues[k])
print("\nCross-tab:")
print(pd.crosstab(df['dx'], df['sex']))
#I removed the sex:dx interaction as it was fucking up the model 
#male and female were making the design matrix rank deficient 

# %%
summary = pd.read_csv(
    os.path.join(OUTPUT_DIR, "spin_shuf_summary.csv")
)

def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
        
# Use global FDR if available
sig_col = "p_fdr_global" if "p_fdr_global" in summary.columns else "p_fdr"
summary["pair"] = summary["dx1"] + " vs " + summary["dx2"]

for sex in summary["sex"].unique():
    sub = summary[summary["sex"] == sex].copy()
    mat = sub.pivot(
        index="pair",
        columns="measure",
        values="r"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        mat.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)

    # Add r values + significance
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            row = sub[
                (sub["pair"] == mat.index[i]) &
                (sub["measure"] == mat.columns[j])
            ]
            if len(row):
                r = row["r"].iloc[0]
                p = row[sig_col].iloc[0]
                txt = f"{r:.2f}{stars(p)}" if p < 0.05 else ""
                ax.text(
                    j, i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold"
                )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation (ρ)")

    legend_text = (
        "*   FDR < 0.05\n"
        "**  FDR < 0.01\n"
        "*** FDR < 0.001"
    )
    fig.text(
        0.02,
        -0.08,
        legend_text,
        fontsize=15
    )

    ax.set_title(f"Crossdiagnostic similarity ({sex}) group")
    plt.tight_layout()

    out_file = os.path.join(
        OUTPUT_DIR,
        f"transdx_heatmap_{sex}.png"
    )
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {out_file}")

# %%
def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def composite_heatmap(groups, out_file):
    """Render a horizontal panel of crossdiagnostic-similarity heatmaps, one column
    per group in `groups` (e.g. ['Whole','Male','Female'] or ['Male','Female'])."""
    groups = [g for g in groups if g in set(summary["sex"].unique())]
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), squeeze=False,
                             constrained_layout=True)
    axes = axes[0]  # unwrap the single row

    reference_index = None
    im = None  # keep a handle for the shared colorbar
    for k, (ax, sex) in enumerate(zip(axes, groups)):
        sub = summary[summary["sex"] == sex].copy()
        mat = sub.pivot(index="pair", columns="measure", values="r")

        # Ensure rows line up across panels (shared y-labels depend on this)
        if reference_index is None:
            reference_index = list(mat.index)
        else:
            assert list(mat.index) == reference_index, (
                f"pair index for '{sex}' differs from the first panel"
            )

        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(np.arange(len(mat.index)))
        if k == 0:
            ax.set_yticklabels(mat.index)
        else:
            ax.set_yticklabels([])
        ax.set_title("Whole sample" if sex == "Whole" else sex)

        # Add r values + significance
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                row = sub[(sub["pair"] == mat.index[i]) &
                          (sub["measure"] == mat.columns[j])]
                if len(row):
                    r = row["r"].iloc[0]
                    p = row[sig_col].iloc[0]
                    txt = f"{r:.2f}{stars(p)}" if p < 0.05 else ""
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=10, fontweight="bold")

    # Single shared colorbar across all subplots
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Correlation (ρ)")

    # Single legend
    fig.text(0.02, -0.05,
             "*   FDR < 0.05\n**  FDR < 0.01\n*** FDR < 0.001", fontsize=10)
    fig.suptitle("Crossdiagnostic similarity", fontsize=14)

    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


# Two versions, mirroring the whole / no_whole convention used elsewhere:
#   * with the whole group (first column), and
#   * sex panels only.
composite_heatmap(["Whole", "Male", "Female"],
                  os.path.join(OUTPUT_DIR, "transdx_heatmap_all.png"))
composite_heatmap(["Male", "Female"],
                  os.path.join(OUTPUT_DIR, "transdx_heatmap_all_no_whole.png"))

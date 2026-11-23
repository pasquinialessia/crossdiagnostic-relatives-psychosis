"""
Supplementary Figure S2 — Sex-stratified regional map correlations
===================================================================
Panel A (top):    3 rows × 4 cols scatter plots
                  Rows: SCZ, SAD, BD  |  Cols: CT, GMV, SA, Sub
                  x = male Cohen's d, y = female Cohen's d
                  Annotated with r and sex-label permutation p-value.

Panel B (bottom): 3 rows × 4 cols permutation null distributions
                  Histogram of null SSE (total residual sum of squares)
                  under sex-label shuffling.
                  Vertical line = observed SSE; dotted = null mean.

Output: figures/figureS2_sex_correlations.png
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cc      = pd.read_csv(f'{BASE}/results/casecontrol/tables/_supp_cc.csv')
sexspin = pd.read_csv(f'{BASE}/results/supp_tables/_supp_sexmap_spin.csv')
NULL    = f'{BASE}/results/qc_sensitivity/map_reliability'

# ── palette ───────────────────────────────────────────────────────────────────
COL       = {'SCZ': '#2a78d6', 'SAD': '#eb6834', 'BD': '#1baf7a'}
MEASURES  = ['CT', 'GMV', 'SA', 'Subcortical']
MLbl      = {'CT': 'CT', 'GMV': 'GMV', 'SA': 'SA', 'Subcortical': 'Sub'}
NULL_KEY  = {'SCZ': 'SCZ_matched', 'SAD': 'SAD_matched', 'BD': 'BD_matched'}
DISORDERS = ['SCZ', 'SAD', 'BD']
BG        = 'white'

# ── pre-load NPZ data ─────────────────────────────────────────────────────────
data = {}
for dx in DISORDERS:
    for meas in MEASURES:
        npz_path = os.path.join(NULL, f'null_{NULL_KEY[dx]}_{meas}.npz')
        npz      = np.load(npz_path)
        null     = npz['null']
        null_mean = float(npz['null_mean_sse'])
        r_obs    = float(npz['r_observed'])
        sse_obs  = float(npz['sse_observed'])
        p_above  = float(npz['p_above'])
        sr = sexspin[(sexspin.disorder==dx) & (sexspin.measure==meas)]
        p_spin = float(sr['p_spin'].iloc[0]) if len(sr) else float('nan')
        data[(dx, meas)] = dict(null=null, null_mean=null_mean,
                                r_obs=r_obs, sse_obs=sse_obs,
                                p_val=p_above, p_spin=p_spin)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 17), facecolor=BG)

outer = gridspec.GridSpec(
    2, 1,
    figure=fig,
    height_ratios=[3, 1.6],
    hspace=0.38,
    left=0.07, right=0.98, top=0.96, bottom=0.06
)
gs_A = gridspec.GridSpecFromSubplotSpec(
    3, 4, subplot_spec=outer[0],
    hspace=0.38, wspace=0.32
)
gs_B = gridspec.GridSpecFromSubplotSpec(
    3, 4, subplot_spec=outer[1],
    hspace=0.38, wspace=0.32
)

fig.text(0.01, 0.975, 'A', fontsize=22, fontweight='bold', color='#0b0b0b',
         va='top', ha='left', transform=fig.transFigure)
fig.text(0.01, 0.415, 'B', fontsize=22, fontweight='bold', color='#0b0b0b',
         va='top', ha='left', transform=fig.transFigure)

# ── Panel A: scatter plots ────────────────────────────────────────────────────
for ri, disorder in enumerate(DISORDERS):
    col_cc = cc[cc['disorder'] == disorder]
    col    = COL[disorder]

    for ci, meas in enumerate(MEASURES):
        ax = fig.add_subplot(gs_A[ri, ci])
        ax.set_facecolor(BG)

        m = (col_cc[(col_cc['group'] == 'male')   & (col_cc['measure'] == meas)]
             [['region','hemisphere','cohen_d']].rename(columns={'cohen_d':'d_m'}))
        f = (col_cc[(col_cc['group'] == 'female') & (col_cc['measure'] == meas)]
             [['region','hemisphere','cohen_d']].rename(columns={'cohen_d':'d_f'}))
        df = m.merge(f, on=['region','hemisphere'])

        if df.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#898781')
        else:
            # r from Cohen's d maps (consistent with scatter)
            r_obs  = float(stats.spearmanr(df['d_m'], df['d_f'])[0])
            # spin p for cortical; enigma spatial shuffle p for subcortical
            # both sourced from _supp_sexmap_spin.csv
            p_use   = data[(disorder, meas)]['p_spin']
            p_label = 'p (shuffle)' if meas == 'Subcortical' else 'p (spin)'

            ax.scatter(df['d_m'], df['d_f'], s=16, color=col, alpha=0.45,
                       linewidth=0, zorder=2)
            lims = [min(df['d_m'].min(), df['d_f'].min()) - 0.05,
                    max(df['d_m'].max(), df['d_f'].max()) + 0.05]
            xfit = np.array(lims)
            ax.plot(xfit, np.polyval(np.polyfit(df['d_m'], df['d_f'], 1), xfit),
                    color=col, linewidth=1.2, alpha=0.75, zorder=3)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect('equal', adjustable='datalim')

            if np.isnan(p_use):
                p_str = f'{p_label} = –'
            elif p_use < 0.001:
                p_str = f'{p_label} < 0.001'
            else:
                p_str = f'{p_label} = {p_use:.3f}'

            ax.text(0.05, 0.97, f'ρ = {r_obs:.2f}\n{p_str}',
                    transform=ax.transAxes, fontsize=12, va='top', ha='left',
                    color=col, fontweight='600', linespacing=1.55)

        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#c3c2b7')
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(colors='#898781', labelsize=11, length=3)

        if ri == 0:
            ax.set_title(MLbl[meas], fontsize=15, fontweight='600',
                         color='#0b0b0b', pad=6)
        if ci == 0:
            ax.set_ylabel(f'{disorder}\nFemale', fontsize=13,
                          color='#52514e', labelpad=5)
        if ri == 2:
            ax.set_xlabel('Male', fontsize=12, color='#52514e', labelpad=3)

# ── Panel B: permutation distributions ───────────────────────────────────────
# Compute global x-range so all 12 panels share the same axis
_all_vals = np.concatenate([
    np.append(data[(dx, m)]['null'], data[(dx, m)]['sse_obs'])
    for dx in DISORDERS for m in MEASURES
])
_B_xmin, _B_xmax = float(_all_vals.min()), float(_all_vals.max())
_B_pad = (_B_xmax - _B_xmin) * 0.04
_B_xlim = (_B_xmin - _B_pad, _B_xmax + _B_pad)

for ri, disorder in enumerate(DISORDERS):
    col = COL[disorder]
    for ci, meas in enumerate(MEASURES):
        ax = fig.add_subplot(gs_B[ri, ci])
        ax.set_facecolor(BG)

        rec      = data[(disorder, meas)]
        null     = rec['null']
        null_mean = rec['null_mean']
        sse_obs  = rec['sse_obs']

        ax.hist(null, bins=30, color='#c3c2b7', edgecolor='none', alpha=0.85)
        ax.set_xlim(_B_xlim)
        ax.axvline(null_mean, color='#898781', linewidth=1.0, linestyle=':', zorder=3)
        ax.axvline(sse_obs,   color=col,       linewidth=1.8, zorder=4)

        p_val = rec['p_val']
        if np.isnan(p_val):
            p_str = 'p = –'
        elif p_val < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = f"p = {p_val:.3f}"
        ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                fontsize=10, va='top', ha='right', color=col, fontweight='600')

        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.set_yticks([])
        ax.tick_params(colors='#898781', labelsize=11, length=2)

        if ri == 0:
            ax.set_title(MLbl[meas], fontsize=15, fontweight='600',
                         color='#0b0b0b', pad=6)
        if ci == 0:
            ax.set_ylabel(f'{disorder}', fontsize=13, color='#52514e', labelpad=5)
        if ri == 2:
            ax.set_xlabel('SSE (null)', fontsize=12, color='#52514e', labelpad=3)

# ── legend ────────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_els = [
    Line2D([0],[0], color='#2a78d6', linewidth=2, label='SCZ'),
    Line2D([0],[0], color='#eb6834', linewidth=2, label='SAD'),
    Line2D([0],[0], color='#1baf7a', linewidth=2, label='BD'),
    Line2D([0],[0], color='#898781', linewidth=1, linestyle=':', label='Null mean (SSE)'),
    Patch(facecolor='#c3c2b7', edgecolor='none', label='Permutation null'),
]
fig.legend(handles=legend_els, loc='lower center', fontsize=12,
           frameon=False, ncol=5, bbox_to_anchor=(0.5, 0.005))

out = os.path.join(BASE, 'figures', 'figureS2_sex_correlations.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print('Saved:', out)

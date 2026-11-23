"""
Supplementary Figure S1 — Hemispheric normative centile distributions
======================================================================
2 rows (Left / Right hemisphere) × 4 cols (CT, GMV, SA, Sub).
Groups: CN (grey), SCZ (blue), SAD (orange), BD (green).
Output: ../figures/figureS1_hemispheric.png

Run from project root or figures/ sibling:
    python code/figS_hemispheric.py
No enigmatoolbox dependency.
"""
import os, numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data   = pd.read_csv(f'{BASE}/data/dataset.csv')

data['dx'] = data['dx'].replace({'Schizoaffective': 'SAD'})

COL   = {'CN': '#9b9b97', 'SCZ': '#2a78d6', 'SAD': '#eb6834', 'BD': '#1baf7a'}
ORDER = ['CN', 'SCZ', 'SAD', 'BD']
BG    = 'white'

panels = [
    ('CT',          'lh', 'HEMI.left.cortical.thickness'),
    ('CT',          'rh', 'HEMI.right.cortical.thickness'),
    ('GMV',         'lh', 'HEMI.left.greymatter.volume'),
    ('GMV',         'rh', 'HEMI.right.greymatter.volume'),
    ('SA',          'lh', 'HEMI.left.surface.area'),
    ('SA',          'rh', 'HEMI.right.surface.area'),
    ('Subcortical', 'lh', 'HEMI.left.subcortex'),
    ('Subcortical', 'rh', 'HEMI.right.subcortex'),
]

# map (measure, hemi) → (row, col)
col_order  = ['CT', 'GMV', 'SA', 'Subcortical']
hemi_label = {'lh': 'Left', 'rh': 'Right'}

fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
fig.patch.set_facecolor(BG)

for meas, hemi, col_name in panels:
    row = 0 if hemi == 'lh' else 1
    col_idx = col_order.index(meas)
    ax = axes[row, col_idx]
    ax.set_facecolor(BG)

    if col_name not in data.columns:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='#898781', fontsize=8)
        continue

    rng = np.random.default_rng(42)
    for xi, grp in enumerate(ORDER):
        vals = data.loc[data['dx'] == grp, col_name].dropna().values
        if len(vals) == 0:
            continue
        c = COL[grp]
        # violin
        parts = ax.violinplot(vals, positions=[xi], widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c); pc.set_alpha(0.30); pc.set_edgecolor('none')
        # jitter
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, s=3, color=c, alpha=0.25, linewidth=0, zorder=2)
        # median + IQR
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.vlines(xi, q1, q3, color=c, linewidth=2.5, zorder=4)
        ax.scatter([xi], [med], s=36, zorder=5,
                   facecolors='white', edgecolors=c, linewidths=1.8)

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=11, color='#52514e')
    ax.tick_params(colors='#898781', length=3, left=True, bottom=False)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.axhline(0.5, color='#c3c2b7', linewidth=0.6, linestyle='--', zorder=0)
    ax.set_ylim(-0.02, 1.02)

    if col_idx == 0:
        ax.set_ylabel(f'{hemi_label[hemi]}\nhemisphere',
                      fontsize=11, color='#52514e', labelpad=5)
    if row == 0:
        ax.set_title(meas if meas != 'Subcortical' else 'Sub',
                     fontsize=13, fontweight='600', color='#0b0b0b', pad=6)

# legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=COL[g], alpha=0.6, label=g) for g in ORDER]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=2.5, w_pad=1.0)
out = os.path.join(BASE, 'figures', 'figureS1_hemispheric.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print('Saved:', out)

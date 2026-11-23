"""
Figure 2 — SWA hemispheric: age of peak effect
=============================================================
Brain surface maps colored by the MEAN AGE of the age window
with the highest (most negative) case-vs-control Cohen's d.

Rows:   SCZ, SAD, BD
Cols:   CT, GMV, SA, Subcortical

Data:  data/dataset.csv  (raw subject-level, Condition_type = Clinical/Control)
Color: viridis 18–50 years; grey = window analysis undefined (too few subjects)

Run from project root:
    python code/fig_swa_hemispheric.py
Output: figures/fig_swa_hemispheric.png
"""
import os, sys, io, warnings
import numpy as np, pandas as pd, nibabel as nib, vtk
import vtk.util.numpy_support as ns
import matplotlib, matplotlib.pyplot as plt, matplotlib.colors as mcolors
matplotlib.use('Agg')
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as _fm
warnings.filterwarnings('ignore')

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR  = os.path.join(BASE, 'figures')
TMP      = os.path.join(BASE, '_tmp_swa_cells')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# ── swa_lib (in code/) ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(BASE, 'code'))
from swa_lib import (
    prepare_arrays, window_mean_ages, _peak_all_np,
    START_AGES, SIGN, WINDOW_WIDTH, MIN_WINDOW_N
)

# ── VTK 9 patch ───────────────────────────────────────────────────────────────
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable
def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        self.VTKObject.SetTable(ns.numpy_to_vtk(table.astype(np.uint8), deep=True,
                                                  array_type=vtk.VTK_UNSIGNED_CHAR))
    elif hasattr(table, 'VTKObject'): self.VTKObject.SetTable(table.VTKObject)
    else: self.VTKObject.SetTable(table)
BSLookupTable.SetTable = _fixed_SetTable

from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds, enigmatoolbox.plotting.surface_plotting as _splot
_PKG = os.path.dirname(__import__('enigmatoolbox').__file__)

def _build(path):
    g = nib.load(path)
    coords = g.darrays[0].data.astype(np.float64)
    faces  = g.darrays[1].data.astype(np.int64)
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(coords))
    n = len(faces)
    carr  = np.hstack([np.full((n,1),3,dtype=np.int64), faces]).flatten()
    cells = vtk.vtkCellArray()
    cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_ = vtk.vtkPolyData(); pd_.SetPoints(pts); pd_.SetPolys(cells)
    return BSPolyData(pd_)

_LH  = _build(os.path.join(_PKG,'datasets/surfaces/fsa5_lh.surf.gii'))
_RH  = _build(os.path.join(_PKG,'datasets/surfaces/fsa5_rh.surf.gii'))
_SLH = _build(os.path.join(_PKG,'datasets/surfaces/sctx_lh.gii'))
_SRH = _build(os.path.join(_PKG,'datasets/surfaces/sctx_rh.gii'))
_eds.load_fsa5        = lambda as_sphere=False: (_LH, _RH)
_splot.load_fsa5      = _eds.load_fsa5
_eds.load_subcortical = lambda with_normals=False, join=False: (_SLH, _SRH)
_splot.load_subcortical = _eds.load_subcortical

from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.datasets import load_fc
from enigmatoolbox.utils.parcellation import parcel_to_surface

_, fc_ctx_labels, _, _ = load_fc()  # 68 ctx labels

# ── freesurfer annot for vertex mapping ───────────────────────────────────────
_la,_,_ln = nib.freesurfer.read_annot(os.path.join(_PKG,'permutation_testing/annot/fsa5_lh_aparc.annot'))
_ra,_,_rn = nib.freesurfer.read_annot(os.path.join(_PKG,'permutation_testing/annot/fsa5_rh_aparc.annot'))
_ln = [n.decode() if isinstance(n,bytes) else n for n in _ln]
_rn = [n.decode() if isinstance(n,bytes) else n for n in _rn]

def ctx_vert(d68):
    lh = np.full(10242, np.nan); rh = np.full(10242, np.nan)
    for i, lbl in enumerate(fc_ctx_labels):
        h, reg = str(lbl).split('_', 1)
        v = d68[i]
        if h == 'L' and reg in _ln: lh[_la == _ln.index(reg)] = v
        elif h == 'R' and reg in _rn: rh[_ra == _rn.index(reg)] = v
    return np.concatenate([lh, rh])

# ── ENIGMA subcortical order (from subcorticalvertices() docstring) ───────────
# L_accumbens, L_amygdala, L_caudate, L_hippocampus, L_pallidus,
# L_putamen, L_thalamus, L_ventricles, then R_ same order
SCTX_COLS = [
    'SubcGM.Left.Accumbens.area', 'SubcGM.Left.Amygdala', 'SubcGM.Left.Caudate',
    'SubcGM.Left.Hippocampus',    'SubcGM.Left.Pallidum',  'SubcGM.Left.Putamen',
    'SubcGM.Left.Thalamus.Proper','SubcGM.Left.Lateral.Ventricle',
    'SubcGM.Right.Accumbens.area','SubcGM.Right.Amygdala', 'SubcGM.Right.Caudate',
    'SubcGM.Right.Hippocampus',   'SubcGM.Right.Pallidum', 'SubcGM.Right.Putamen',
    'SubcGM.Right.Thalamus.Proper','SubcGM.Right.Lateral.Ventricle',
]

# ── sliding-window peak-age computation ───────────────────────────────────────
raw = pd.read_csv(os.path.join(DATA_DIR, 'dataset.csv'))
raw['dx'] = raw['dx'].replace({'Schizoaffective': 'SAD'})

DISORDER_DX = {'SCZ': 'SCZ', 'SAD': 'SAD', 'BD': 'BD'}

def peak_ages_for_disorder(dx_label):
    """Return dict {region_col: peak_window_mean_age} for a disorder."""
    sub = raw[raw['dx'].isin([dx_label, 'CN'])].copy()
    print(f"  {dx_label}: n={len(sub)} (clinical={len(sub[sub.dx==dx_label])}, CN={len(sub[sub.dx=='CN'])})")
    out = {}
    for mtype_prefix, col_prefix in [('CT','CT'), ('GM','GM'), ('SA','SA'), ('SubcGM','SubcGM')]:
        if col_prefix == 'SubcGM':
            region_cols = [c for c in raw.columns if c.startswith('SubcGM.')]
            region_cols = [c for c in region_cols if c in SCTX_COLS]
        else:
            region_cols = [c for c in raw.columns
                           if c.startswith(f'lh.{col_prefix}.') or c.startswith(f'rh.{col_prefix}.')]
        if not region_cols:
            continue
        arr = prepare_arrays(sub, region_cols)
        mage_lookup = window_mean_ages(arr['age0'])
        best_signed, argwin = _peak_all_np(
            arr['Y'], arr['grp'], arr['age0'], arr['sex'], arr['site_full'], SIGN)
        for j, reg in enumerate(region_cols):
            wi = int(argwin[j])
            if wi >= 0 and np.isfinite(best_signed[j]):
                start = START_AGES[wi]
                out[reg] = mage_lookup.get(start, np.nan)
            else:
                out[reg] = np.nan
    return out

print("Computing peak ages ...")
peak_age_data = {}
for dx in ['SCZ', 'SAD', 'BD']:
    peak_age_data[dx] = peak_ages_for_disorder(dx)

# ── build vectors ─────────────────────────────────────────────────────────────
def build_ctx_vec(peak_ages, mtype):
    """68-element vector in fc_ctx_labels order."""
    vec = []
    for lbl in fc_ctx_labels:
        h, reg = str(lbl).split('_', 1)
        hemi = 'lh' if h == 'L' else 'rh'
        col = f'{hemi}.{mtype}.{reg}'
        vec.append(peak_ages.get(col, np.nan))
    return np.array(vec)

def build_sctx_vec(peak_ages):
    """16-element vector in ENIGMA subcortical order."""
    return np.array([peak_ages.get(c, np.nan) for c in SCTX_COLS])

# ── rendering helpers ─────────────────────────────────────────────────────────
NAN_COLOR = (0.85, 0.85, 0.85, 1.0)
AGE_MIN, AGE_MAX = 18.0, 50.0

def render_ctx(d68, tag, size=(1400, 360)):
    fp = os.path.join(TMP, f'{tag}.png')
    surf = ctx_vert(d68)
    plot_cortical(array_name=surf, surface_name='fsa5',
                  color_bar=False, color_range=(AGE_MIN, AGE_MAX),
                  cmap='viridis', nan_color=NAN_COLOR, zoom=1.25,
                  size=size, interactive=False, screenshot=True,
                  filename=fp, transparent_bg=False)
    return fp

def render_sctx(d16, tag, size=(1400, 360)):
    fp = os.path.join(TMP, f'{tag}.png')
    plot_subcortical(array_name=d16, ventricles=True,
                     color_bar=False, color_range=(AGE_MIN, AGE_MAX),
                     cmap='viridis', nan_color=NAN_COLOR, zoom=1.25,
                     size=size, interactive=False, screenshot=True,
                     filename=fp, transparent_bg=False)
    return fp

# ── render all cells + assemble figS5-style panels ──────────────────────────
import matplotlib.font_manager as _fm
_FONTB = _fm.findfont(_fm.FontProperties(family='DejaVu Sans', weight='bold'))
def font(sz):
    try: return ImageFont.truetype(_FONTB, int(sz))
    except Exception: return ImageFont.load_default()

CORT_MAP = [('CT', 'CT'), ('GMV', 'GM'), ('SA', 'SA')]
DISORDERS = ['SCZ', 'SAD', 'BD']

def build_panel(dx_label):
    pa = peak_age_data[dx_label]
    # cortical renders
    cort_imgs = {}
    for col_label, mtype in CORT_MAP:
        tag = f'{dx_label}_{col_label}'
        d68 = build_ctx_vec(pa, mtype)
        fp  = render_ctx(d68, tag)
        cort_imgs[col_label] = Image.open(fp).convert('RGB')
        print(f'  rendered {tag}')
    # subcortical render
    d16  = build_sctx_vec(pa)
    fp_s = render_sctx(d16, f'{dx_label}_Subcortical')
    sctx_raw = Image.open(fp_s).convert('RGB')
    print(f'  rendered {dx_label}_Subcortical')

    cw = max(v.width  for v in cort_imgs.values())
    ch = max(v.height for v in cort_imgs.values())
    cort_imgs = {m: (v if v.size==(cw,ch) else v.resize((cw,ch), Image.Resampling.LANCZOS))
                 for m, v in cort_imgs.items()}

    lbl_w   = 0                         # row label added by outer assembly
    hdr_h   = int(ch * 0.22)
    col_gap = int(cw * 0.06)
    sw = int(cw * 1.0)
    sh = int(sw * sctx_raw.height / sctx_raw.width)
    subimg = sctx_raw.resize((sw, sh), Image.Resampling.LANCZOS)

    colx = lambda mi: mi * (cw + col_gap)
    W = cw * 3 + col_gap * 2
    panel = Image.new('RGB', (W, hdr_h + ch + hdr_h + sh), 'white')
    d = ImageDraw.Draw(panel)

    for mi, (col_label, _) in enumerate(CORT_MAP):
        d.text((colx(mi) + cw//2, hdr_h//2), col_label, fill='black',
               font=font(int(hdr_h * 0.78)), anchor='mm')
        panel.paste(cort_imgs[col_label], (colx(mi), hdr_h))

    gmv_cx = colx(1) + cw // 2
    d.text((gmv_cx, hdr_h + ch + hdr_h//2), 'Subcortical', fill='black',
           font=font(int(hdr_h * 0.78)), anchor='mm')
    panel.paste(subimg, (gmv_cx - sw//2, hdr_h + ch + hdr_h))
    return panel

print("Computing peak ages ...")
peak_age_data = {}
for dx in DISORDERS:
    peak_age_data[dx] = peak_ages_for_disorder(dx)
    print(f"  {dx}: done")

print("Rendering panels ...")
panels = {dx: build_panel(dx) for dx in DISORDERS}

# ── colorbar ──────────────────────────────────────────────────────────────────
W_panel = max(p.width for p in panels.values())

def colorbar_strip(width):
    fig = plt.figure(figsize=(7, 0.55))
    ax  = fig.add_axes([0.05, 0.45, 0.90, 0.45])
    cb  = plt.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(AGE_MIN, AGE_MAX), cmap='viridis'),
        cax=ax, orientation='horizontal')
    cb.set_label("Mean age of peak-effect window (years)", fontsize=12)
    cb.set_ticks(np.arange(18, 51, 4))
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    im = Image.open(buf).convert('RGB')
    tw = int(width * 0.65)
    return im.resize((tw, int(im.height * tw / im.width)), Image.Resampling.LANCZOS)

lw  = int(W_panel * 0.13)
cb  = colorbar_strip(W_panel)
pad = 20

H = sum(p.height for p in panels.values()) + pad * (len(DISORDERS) - 1) + pad + cb.height + 8
canvas = Image.new('RGB', (lw + W_panel, H), 'white')
draw   = ImageDraw.Draw(canvas)
y = 0
for dx_label, panel in panels.items():
    draw.text((int(lw * 0.10), y + panel.height // 2), dx_label, fill='black',
              font=font(int(W_panel * 0.028)), anchor='lm')
    canvas.paste(panel, (lw, y))
    y += panel.height + pad

canvas.paste(cb, (lw + (W_panel - cb.width) // 2, y))
out = os.path.join(FIG_DIR, 'fig_swa_hemispheric.png')
canvas.save(out, dpi=(300, 300))
print('Saved:', out)

import shutil; shutil.rmtree(TMP, ignore_errors=True)

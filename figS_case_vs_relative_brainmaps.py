#!/usr/bin/env python3
"""
Supplementary Figure S6 — Case-vs-Relative brain maps (Cohen's d, FDR-thresholded)
===================================================================
Companion to fig_relatives_brainmaps.py (Relative-vs-Control).
Same VTK/enigmatoolbox rendering; identical panel layout: one row per
disorder × columns CT | GMV | SA | Subcortical (Sub under GMV).
Grey = not FDR-significant (q > 0.05).

age_z note: z-scored within each analysis sample (case_vs_relative.py)
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)

Output: ../figures/figureS6_case_vs_relative.png
"""
import os, io
import numpy as np
import pandas as pd
import nibabel as nib
import vtk
import vtk.util.numpy_support as ns
import warnings; warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as _fm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "figures");        os.makedirs(FIGDIR, exist_ok=True)
TMP    = os.path.join(FIGDIR, "_cvr_tmp");     os.makedirs(TMP, exist_ok=True)
RESDIR = os.path.join(BASE, "results", "08_relatives", "case_vs_relative")

# ── VTK 9 patches (identical to fig_regional_brainmaps.py) ───────────────────────
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable
def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        self.VTKObject.SetTable(ns.numpy_to_vtk(table.astype(np.uint8), deep=True,
                                                array_type=vtk.VTK_UNSIGNED_CHAR))
    elif hasattr(table, "VTKObject"): self.VTKObject.SetTable(table.VTKObject)
    else: self.VTKObject.SetTable(table)
BSLookupTable.SetTable = _fixed_SetTable
from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds
import enigmatoolbox.plotting.surface_plotting as _splot
_PKG = os.path.dirname(__import__("enigmatoolbox").__file__)

def _build(path):
    g = nib.load(path)
    coords = g.darrays[0].data.astype(np.float64)
    faces  = g.darrays[1].data.astype(np.int64)
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(coords))
    n = len(faces)
    carr = np.hstack([np.full((n,1),3,dtype=np.int64), faces]).flatten()
    cells = vtk.vtkCellArray(); cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_ = vtk.vtkPolyData(); pd_.SetPoints(pts); pd_.SetPolys(cells)
    return BSPolyData(pd_)

_LH = _build(os.path.join(_PKG,"datasets/surfaces/fsa5_lh.surf.gii"))
_RH = _build(os.path.join(_PKG,"datasets/surfaces/fsa5_rh.surf.gii"))
_eds.load_fsa5   = lambda as_sphere=False: (_LH, _RH)
_splot.load_fsa5 = _eds.load_fsa5
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.datasets import load_fc


# ── parcellation labels & annot ────────────────────────────────────────────────
_, fc_ctx_labels, _, _ = load_fc()
_la, _, _ln = nib.freesurfer.read_annot(
    os.path.join(_PKG,"permutation_testing/annot/fsa5_lh_aparc.annot"))
_ra, _, _rn = nib.freesurfer.read_annot(
    os.path.join(_PKG,"permutation_testing/annot/fsa5_rh_aparc.annot"))
_ln = [n.decode() if isinstance(n,bytes) else n for n in _ln]
_rn = [n.decode() if isinstance(n,bytes) else n for n in _rn]

def enigma_col(label, pref):
    h, r = label.split("_", 1)
    return f"{'lh' if h=='L' else 'rh'}.{pref}.{r}"

MEAS_COLS = {
    "CT":  [enigma_col(l, "CT") for l in fc_ctx_labels],
    "GMV": [enigma_col(l, "GM") for l in fc_ctx_labels],
    "SA":  [enigma_col(l, "SA") for l in fc_ctx_labels],
}
# Subcortical: 16-slot array (L×8 + R×8); Ventricle & Cereb slots = NaN
SCTX_TARGETS = ["Accumbens.area","Amygdala","Caudate","Hippocampus",
                 "Pallidum","Putamen","Thalamus","Ventricle"]

def ctx_vert(d68, fdr_mask):
    """Map 68 Cohen's d values to vertex array; NaN where not FDR-sig."""
    lh = np.full(10242, np.nan)
    rh = np.full(10242, np.nan)
    for i, lab in enumerate(fc_ctx_labels):
        if not fdr_mask[i]:
            continue
        h, reg = str(lab).split("_", 1)
        if h == "L" and reg in _ln:
            lh[_la == _ln.index(reg)] = d68[i]
        elif h == "R" and reg in _rn:
            rh[_ra == _rn.index(reg)] = d68[i]
    return np.concatenate([lh, rh])

def load_subc_cols_order():
    """Return the column names in the order 08b saved them (subc function)."""
    df = pd.read_csv(os.path.join(BASE, "data", "bsnip_relatives_dataset.csv"))
    return [c for c in df.columns
            if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

SUBC_COLS = load_subc_cols_order()   # 14 columns, L first then R

def build_sctx16(d14, fdr_mask14):
    """Map 14 Cohen's d values → 16-slot SCTX array (Vent slots = NaN)."""
    d16 = np.full(16, np.nan)
    for ki, target in enumerate(SCTX_TARGETS):
        if target == "Ventricle":
            continue   # keep NaN for ventricle (slots 7 & 15)
        for side_offset, side in enumerate(["Left","Right"]):
            slot = ki + side_offset * 8
            match = [i for i,c in enumerate(SUBC_COLS)
                     if side in c and target.split(".")[0].lower() in c.lower()]
            if match and fdr_mask14[match[0]]:
                d16[slot] = d14[match[0]]
    return d16

# ── rendering helpers ──────────────────────────────────────────────────────────
VMAX = 0.50
NAN_COLOR = (0.85, 0.85, 0.85, 1)
SIZE_CTX  = (1400, 360)
SIZE_SCTX = (1400, 360)

def render_ctx(d68, tag):
    fp = os.path.join(TMP, f"{tag}.png")
    plot_cortical(array_name=d68, surface_name="fsa5",
                  color_bar=False, color_range=(-VMAX, VMAX),
                  cmap="RdBu_r", nan_color=NAN_COLOR,
                  zoom=1.25, size=SIZE_CTX,
                  interactive=False, screenshot=True,
                  filename=fp, transparent_bg=False)
    return fp

def render_sctx(d16, tag):
    fp = os.path.join(TMP, f"{tag}.png")
    plot_subcortical(array_name=d16, ventricles=True,
                     color_bar=False, color_range=(-VMAX, VMAX),
                     cmap="RdBu_r", nan_color=NAN_COLOR,
                     zoom=1.25, size=SIZE_SCTX,
                     interactive=False, screenshot=True,
                     filename=fp, transparent_bg=False)
    return fp

# ── typography (same as fig3a) ────────────────────────────────────────────────
_FONT  = _fm.findfont(_fm.FontProperties(family="DejaVu Sans"))
_FONTB = _fm.findfont(_fm.FontProperties(family="DejaVu Sans", weight="bold"))

def font(sz, bold=True):
    try: return ImageFont.truetype(_FONTB if bold else _FONT, int(sz))
    except Exception: return ImageFont.load_default()

# ── colorbar ──────────────────────────────────────────────────────────────────
def colorbar_img(width, label="Cohen's d · grey = n.s."):
    fig = plt.figure(figsize=(6, 0.5))
    ax  = fig.add_axes([0.05, 0.5, 0.9, 0.4])
    cb  = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap("RdBu_r"),
                                    norm=Normalize(-VMAX, VMAX),
                                    orientation="horizontal")
    cb.set_label(label, fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return im.resize((int(width * 0.5),
                      int(im.height * width * 0.5 / im.width)),
                     Image.Resampling.LANCZOS)

# ── panel builder (same layout as fig3a) ─────────────────────────────────────
def build_row(d_ctx, fdr_ctx, d_sub, fdr_sub, tag):
    """
    One row: CT | GMV | SA cortical + Subcortical map centred under GMV column.
    Consistent with fig3a layout.
    """
    CORT = ["CT", "GMV", "SA"]
    imgs = {}
    for meas in CORT:
        d68  = d_ctx[meas]
        mask = fdr_ctx[meas]
        vert = ctx_vert(d68, mask)
        imgs[meas] = Image.open(render_ctx(vert, f"{tag}_{meas}")).convert("RGB")

    d16 = build_sctx16(d_sub, fdr_sub)
    imgs["SCTX"] = Image.open(render_sctx(d16, f"{tag}_SCTX")).convert("RGB")

    cw = max(imgs[m].width  for m in CORT)
    ch = max(imgs[m].height for m in CORT)
    cort = {m: (imgs[m] if imgs[m].size==(cw,ch)
                else imgs[m].resize((cw,ch), Image.Resampling.LANCZOS))
            for m in CORT}

    lblw     = int(cw * 0.20)
    hdr      = int(ch * 0.20)
    col_gap  = int(cw * 0.06)
    colx     = lambda mi: lblw + mi * (cw + col_gap)
    sw       = int(cw * 1.0)
    raw_s    = imgs["SCTX"]
    sh       = int(sw * raw_s.height / raw_s.width)
    sublbl_h = int(sh * 0.28)
    subimg   = raw_s.resize((sw, sh), Image.Resampling.LANCZOS)

    W      = lblw + cw * 3 + col_gap * 2
    canvas = Image.new("RGB", (W, hdr + ch + hdr + sh + sublbl_h), "white")
    d      = ImageDraw.Draw(canvas)

    # column headers
    for mi, m in enumerate(CORT):
        d.text((colx(mi) + cw//2, hdr//2), m,
               fill="black", font=font(int(hdr * 0.62)), anchor="mm")
        canvas.paste(cort[m], (colx(mi), hdr))

    # subcortical panel centred under GMV column
    gmv_cx = colx(1) + cw // 2
    d.text((gmv_cx, hdr + ch + hdr//2), "Subcortical",
           fill="black", font=font(int(hdr * 0.62)), anchor="mm")
    canvas.paste(subimg, (gmv_cx - sw//2, hdr + ch + hdr))

    return canvas

# ── main ──────────────────────────────────────────────────────────────────────
DISORDERS = ["SCZ", "SAD", "BD"]
DIS_LABELS = {"SCZ": "SCZ\nclinical\nvs relatives",
              "SAD": "SAD\nclinical\nvs relatives",
              "BD":  "BD\nclinical\nvs relatives"}
MEAS_LIST  = ["CT", "GMV", "SA", "Subcortical"]

# load CSV for FDR masks
res = pd.read_csv(os.path.join(RESDIR, "_supp_case_vs_relative.csv"))

rows_img = []
for dx in DISORDERS:
    d_ctx  = {}
    fdr_ctx = {}
    for meas in ["CT", "GMV", "SA"]:
        npy = os.path.join(RESDIR, f"{dx}_{meas}_cohen_d.npy")
        d68 = np.load(npy)
        sub = res[(res.disorder==dx) & (res.measure==meas)]
        fdr = (sub["q_fdr"].values < 0.05)
        d_ctx[meas]  = d68
        fdr_ctx[meas] = fdr

    npy_s = os.path.join(RESDIR, f"{dx}_Subcortical_cohen_d.npy")
    d_sub = np.load(npy_s)
    sub_s = res[(res.disorder==dx) & (res.measure=="Subcortical")]
    fdr_sub = (sub_s["q_fdr"].values < 0.05)

    print(f"Rendering {dx} …")
    row = build_row(d_ctx, fdr_ctx, d_sub, fdr_sub, dx)
    rows_img.append((dx, row))

# assemble full figure
W      = max(r.width  for _, r in rows_img)
lw     = int(W * 0.18)
cb     = colorbar_img(W)
pad    = 20
H      = sum(r.height + pad for _, r in rows_img) + cb.height + 8
fig    = Image.new("RGB", (lw + W, H), "white")
dr     = ImageDraw.Draw(fig)

y = 0
for dx, row in rows_img:
    lab = DIS_LABELS[dx]
    dr.multiline_text((lw // 2, y + row.height // 2),
                      lab, fill="black",
                      font=font(int(W * 0.015)),
                      anchor="mm", align="center", spacing=6)
    fig.paste(row, (lw, y))
    y += row.height + pad

fig.paste(cb, (lw + (W - cb.width)//2, y))

out_path = os.path.join(FIGDIR, "figureS6_case_vs_relative.png")
fig.save(out_path)
print(f"\nSaved: {out_path}  size={fig.size}")

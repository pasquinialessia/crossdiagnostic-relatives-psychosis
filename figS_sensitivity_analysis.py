# %%
"""
Supplementary Figure S4 — sensitivity analysis (region-specificity) (region minus hemispheric mean).
=============================================================================
Cortical maps of regions that remained FDR-significant after re-expressing each
region as its deviation from the hemispheric mean. Per diagnosis (SCZ, SAD, BD;
rows) × measure (CT, GMV, SA; columns). Significant regions are coloured by
Cohen's d (case-vs-control, from _supp_cc.csv group=whole), the rest grey.

Reads:
  _supp_spec.csv  — significance mask (q_corrected < 0.05)
  _supp_cc.csv    — Cohen's d values (group='whole')

Run from project root:
  DISPLAY=:99 python code/figS_sensitivity_analysis.py
Output: figures/figureS4_region_specificity.png
"""
import os, io, numpy as np, pandas as pd, nibabel as nib, vtk
import vtk.util.numpy_support as ns
import warnings; warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw, ImageFont
import matplotlib, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.colors import Normalize
matplotlib.use('Agg')

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "figures")
TMP    = os.path.join(BASE, "figures", "_fS4tmp")
os.makedirs(TMP, exist_ok=True)

SPEC      = os.path.join(BASE, "results", "02_casecontrol", "tables", "_supp_spec.csv")
SPEC_SUBC = os.path.join(BASE, "results", "02_casecontrol", "tables", "_supp_spec_subcortical.csv")
CC        = os.path.join(BASE, "results", "02_casecontrol", "tables", "_supp_cc.csv")
VMAX = 0.35   # Cohen's d symmetric scale

# ── VTK 9 patch ──────────────────────────────────────────────────────────────
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable
def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        self.VTKObject.SetTable(ns.numpy_to_vtk(table.astype(np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    elif hasattr(table, "VTKObject"): self.VTKObject.SetTable(table.VTKObject)
    else: self.VTKObject.SetTable(table)
BSLookupTable.SetTable = _fixed_SetTable
from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds, enigmatoolbox.plotting.surface_plotting as _splot
_PKG=os.path.dirname(__import__("enigmatoolbox").__file__)
def _build(path):
    g=nib.load(path); coords=g.darrays[0].data.astype(np.float64); faces=g.darrays[1].data.astype(np.int64)
    pts=vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(coords)); n=len(faces)
    carr=np.hstack([np.full((n,1),3,dtype=np.int64),faces]).flatten()
    cells=vtk.vtkCellArray(); cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_=vtk.vtkPolyData(); pd_.SetPoints(pts); pd_.SetPolys(cells); return BSPolyData(pd_)
_LH=_build(os.path.join(_PKG,"datasets/surfaces/fsa5_lh.surf.gii"))
_RH=_build(os.path.join(_PKG,"datasets/surfaces/fsa5_rh.surf.gii"))
_eds.load_fsa5=lambda as_sphere=False:(_LH,_RH); _splot.load_fsa5=_eds.load_fsa5
_SLH=_build(os.path.join(_PKG,"datasets/surfaces/sctx_lh.gii"))
_SRH=_build(os.path.join(_PKG,"datasets/surfaces/sctx_rh.gii"))
_eds.load_subcortical=lambda with_normals=False,join=False:(_SLH,_SRH)
_splot.load_subcortical=_eds.load_subcortical
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.datasets import load_fc
_, fc_ctx_labels, _, _ = load_fc()

# ── annotation lookup ─────────────────────────────────────────────────────────
_la,_,_ln=nib.freesurfer.read_annot(os.path.join(_PKG,"permutation_testing/annot/fsa5_lh_aparc.annot"))
_ra,_,_rn=nib.freesurfer.read_annot(os.path.join(_PKG,"permutation_testing/annot/fsa5_rh_aparc.annot"))
_ln=[n.decode() if isinstance(n,bytes) else n for n in _ln]; _rn=[n.decode() if isinstance(n,bytes) else n for n in _rn]
def ctx_vert(d68):
    lh=np.full(10242,np.nan); rh=np.full(10242,np.nan)
    for i,l in enumerate(fc_ctx_labels):
        h,reg=str(l).split("_",1); v=d68[i]
        if h=="L" and reg in _ln: lh[_la==_ln.index(reg)]=v
        elif h=="R" and reg in _rn: rh[_ra==_rn.index(reg)]=v
    return np.concatenate([lh,rh])
NAN=(0.85,0.85,0.85,1)
def render_ctx(d68, tag, size=(1400,360)):
    fp=os.path.join(TMP,f"{tag}.png")
    plot_cortical(array_name=ctx_vert(d68), surface_name="fsa5", color_bar=False,
                  color_range=(-VMAX,VMAX), cmap="RdBu_r", nan_color=NAN,
                  zoom=1.25, size=size, interactive=False, screenshot=True,
                  filename=fp, transparent_bg=False); return fp

# ── load data ─────────────────────────────────────────────────────────────────
spec = pd.read_csv(SPEC)
cc   = pd.read_csv(CC)
cc_whole = cc[cc['group']=='whole'].copy()

def vec(dx, meas):
    """Cohen's d where q_corrected<0.05 (region-specificity analysis), else NaN."""
    spec_dx  = spec[(spec.disorder==dx) & (spec.measure==meas)]
    cc_dx    = cc_whole[(cc_whole.disorder==dx) & (cc_whole.measure==meas)]
    sig_mask = {(r.hemisphere, r.region): (r.q_corrected < 0.05)
                for r in spec_dx.itertuples()}
    cohen_d  = {(r.hemisphere, r.region): r.cohen_d
                for r in cc_dx.itertuples()}
    out = []
    for l in fc_ctx_labels:
        h, reg = str(l).split("_", 1)
        key = ("Left" if h=="L" else "Right", reg)
        if sig_mask.get(key, False):
            out.append(cohen_d.get(key, np.nan))
        else:
            out.append(np.nan)
    return np.array(out)


# ── subcortical helpers ──────────────────────────────────────────────────────
_SCTX_TARGETS_L = ["Accumbens.area","Amygdala","Caudate","Hippocampus",
                    "Pallidum","Putamen","Thalamus.Proper","Lateral.Ventricle"]

def sctx_vec_dx(dx):
    """Cohen's d for subcortical regions significant in the normalized analysis (SPEC_SUBC),
    else NaN — consistent with the cortical region-specificity masking."""
    spec_s  = pd.read_csv(SPEC_SUBC)
    spec_dx = spec_s[(spec_s.disorder==dx) & (spec_s.measure=="Subcortical")]
    sig_mask = {(r.hemisphere, r.region): (r.q_corrected < 0.05)
                for r in spec_dx.itertuples()}
    cc_dx = cc_whole[(cc_whole.disorder==dx) & (cc_whole.measure=="Subcortical")]
    cd_map = {(r.hemisphere, r.region): r.cohen_d for r in cc_dx.itertuples()}
    vec = []
    for side in ["Left","Right"]:
        for t in _SCTX_TARGETS_L:
            key = (side, t)
            if sig_mask.get(key, False):
                vec.append(cd_map.get(key, float("nan")))
            else:
                vec.append(float("nan"))
    return np.array(vec)

def render_sctx(d16, tag, size=(1400, 360)):
    fp = os.path.join(TMP, f"{tag}.png")
    plot_subcortical(array_name=d16, ventricles=True, color_bar=False,
                     color_range=(-VMAX, VMAX), cmap="RdBu_r", nan_color=NAN,
                     zoom=1.25, size=size, interactive=False, screenshot=True,
                     filename=fp, transparent_bg=False)
    return fp

# ── render cells ──────────────────────────────────────────────────────────────
import matplotlib.font_manager as _fm
_FONTB=_fm.findfont(_fm.FontProperties(family="DejaVu Sans",weight="bold"))
def font(sz):
    try: return ImageFont.truetype(_FONTB,int(sz))
    except Exception: return ImageFont.load_default()

DX=["SCZ","SAD","BD"]; CORT=["CT","GMV","SA"]
imgs={(dx,m):Image.open(render_ctx(vec(dx,m),f"{dx}_{m}")).convert("RGB")
      for dx in DX for m in CORT}
print("All cells rendered")


# ── figS5-style build_panel (CT | GMV | SA cortical + subcortical under GMV) ─
CORT = ["CT", "GMV", "SA"]

def build_panel(dx):
    cort_imgs = {m: imgs[(dx, m)] for m in CORT}
    d16      = sctx_vec_dx(dx)
    sctx_raw = Image.open(render_sctx(d16, f"{dx}_SCTX")).convert("RGB")

    cw = max(v.width  for v in cort_imgs.values())
    ch = max(v.height for v in cort_imgs.values())
    cort_imgs = {m: (v if v.size==(cw,ch) else v.resize((cw,ch), Image.Resampling.LANCZOS))
                 for m, v in cort_imgs.items()}

    hdr_h   = int(ch * 0.22)
    col_gap = int(cw * 0.06)
    sw = int(cw * 1.0)
    sh = int(sw * sctx_raw.height / sctx_raw.width)
    subimg  = sctx_raw.resize((sw, sh), Image.Resampling.LANCZOS)

    colx  = lambda mi: mi * (cw + col_gap)
    W     = cw * 3 + col_gap * 2
    panel = Image.new("RGB", (W, hdr_h + ch + hdr_h + sh), "white")
    d = ImageDraw.Draw(panel)

    for mi, m in enumerate(CORT):
        d.text((colx(mi)+cw//2, hdr_h//2), m, fill="black",
               font=font(int(hdr_h*0.62)), anchor="mm")
        panel.paste(cort_imgs[m], (colx(mi), hdr_h))

    gmv_cx = colx(1) + cw//2
    d.text((gmv_cx, hdr_h+ch+hdr_h//2), "Subcortical", fill="black",
           font=font(int(hdr_h*0.62)), anchor="mm")
    panel.paste(subimg, (gmv_cx - sw//2, hdr_h + ch + hdr_h))
    return panel

print("Building panels with subcortical ...")
DX     = ["SCZ", "SAD", "BD"]
panels = {dx: build_panel(dx) for dx in DX}

W_panel = max(p.width for p in panels.values())
lw      = int(W_panel * 0.13)

def colorbar_final(width):
    fig=plt.figure(figsize=(6,0.5)); ax=fig.add_axes([0.05,0.5,0.9,0.4])
    cb2=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap("RdBu_r"),
                                  norm=Normalize(-VMAX,VMAX),orientation="horizontal")
    cb2.set_label("Cohen's d (FDR-significant region-specificity) | grey = n.s.", fontsize=8)
    buf=io.BytesIO(); fig.savefig(buf,dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); buf.seek(0)
    im=Image.open(buf).convert("RGB")
    tw=int(width*0.6)
    return im.resize((tw, int(im.height*tw/im.width)), Image.Resampling.LANCZOS)

cb  = colorbar_final(W_panel)
pad = 20
H   = sum(p.height for p in panels.values()) + pad*(len(DX)-1) + pad + cb.height + 8
canvas = Image.new("RGB",(lw+W_panel,H),"white"); d2=ImageDraw.Draw(canvas)
y=0
for dx, panel in panels.items():
    d2.text((int(lw*0.10), y+panel.height//2), dx, fill="black",
            font=font(int(W_panel*0.022)), anchor="lm")
    canvas.paste(panel,(lw,y)); y+=panel.height+pad

canvas.paste(cb,(lw+(W_panel-cb.width)//2,y))
canvas.save(os.path.join(FIGDIR,"figureS4_region_specificity.png"))
print("Saved figureS4_region_specificity.png", canvas.size)

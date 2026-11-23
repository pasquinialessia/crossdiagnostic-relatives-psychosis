# %%
"""
Figure 1 (paper) — case-vs-control Cohen's d, whole/males/females; SCZ (A), SAD (B) and BD (C), each vs controls (no pooled SCZ+SAD group).
=====================================================================================
ENIGMA/VTK rendering (same look as 02b/04). Per panel: 3 rows (whole/males/females)
x 3 cortical columns (CT, GMV, SA); the SUBCORTICAL map is rendered small and placed
UNDER the GMV column. One shared colorbar on top; no per-map titles; whole/male/female
labels only on the leftmost (CT) column; small CT/GMV/SA headers; SCZ/SAD/BD row labels.

Run in the venv:  python fig_regional_brainmaps.py
Output: ../figures/figure1_paper.png
"""
import os, io, numpy as np, pandas as pd, nibabel as nib, vtk
import vtk.util.numpy_support as ns
import warnings; warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw, ImageFont
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
np.random.seed(42)

FIGDIR = os.path.join("..", "figures"); os.makedirs(FIGDIR, exist_ok=True)
TMP = os.path.join("..", "figures", "_f1tmp"); os.makedirs(TMP, exist_ok=True)

# ---- VTK 9 patches (identical to 02b) -------------------------------------- #
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable
def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        self.VTKObject.SetTable(ns.numpy_to_vtk(table.astype(np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    elif hasattr(table, "VTKObject"): self.VTKObject.SetTable(table.VTKObject)
    else: self.VTKObject.SetTable(table)
BSLookupTable.SetTable = _fixed_SetTable
from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds, enigmatoolbox.plotting.surface_plotting as _splot
_PKG = os.path.dirname(__import__("enigmatoolbox").__file__)
def _build(path):
    g=nib.load(path); coords=g.darrays[0].data.astype(np.float64); faces=g.darrays[1].data.astype(np.int64)
    pts=vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(coords)); n=len(faces)
    carr=np.hstack([np.full((n,1),3,dtype=np.int64),faces]).flatten()
    cells=vtk.vtkCellArray(); cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_=vtk.vtkPolyData(); pd_.SetPoints(pts); pd_.SetPolys(cells); return BSPolyData(pd_)
_LH=_build(os.path.join(_PKG,"datasets/surfaces/fsa5_lh.surf.gii")); _RH=_build(os.path.join(_PKG,"datasets/surfaces/fsa5_rh.surf.gii"))
_eds.load_fsa5=lambda as_sphere=False:(_LH,_RH); _splot.load_fsa5=_eds.load_fsa5
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.datasets import load_fc

# ---- data / model ---------------------------------------------------------- #
_, fc_ctx_labels, _, _ = load_fc()
def enigma_col(l, pref): h,r=l.split("_",1); return f"{'lh' if h=='L' else 'rh'}.{pref}.{r}"
MEAS={"CT":[enigma_col(l,"CT") for l in fc_ctx_labels],
      "GMV":[enigma_col(l,"GM") for l in fc_ctx_labels],
      "SA":[enigma_col(l,"SA") for l in fc_ctx_labels]}
_SCTX_TARGETS=["Accumbens.area","Amygdala","Caudate","Hippocampus","Pallidum","Putamen","Thalamus","Ventricle"]
def _match(side,t,cols):
    for c in cols:
        if side.lower() in c.lower() and t.lower() in c.lower(): return c
    return None

def fit_group(data, cols, whole):
    recs=[]
    for col in [c for c in cols if c and c in data.columns]:
        safe=col.replace(".","_"); tmp=data[[col,"age","sex","study_site","Condition_type"]].dropna().copy()
        tmp["age_z"]=(tmp["age"]-tmp["age"].mean())/tmp["age"].std(ddof=1); tmp=tmp.rename(columns={col:safe})
        f=f"{safe} ~ age_z + C(study_site) + C(Condition_type, Treatment(reference='Control'))"
        if whole: f=f"{safe} ~ age_z + C(sex) + C(study_site) + C(Condition_type, Treatment(reference='Control'))"
        m=smf.ols(f,data=tmp).fit()
        k=[i for i in m.params.index if "Clinical" in i and ":" not in i][0]
        recs.append({"col":col,"t":m.tvalues[k],"p":m.pvalues[k],"n":int(m.nobs)})
    r=pd.DataFrame(recs).set_index("col"); _,r["pf"],_,_=multipletests(r["p"],method="fdr_bh"); return r
def dvec(r, cols):
    return np.array([ (r.loc[c,"t"]/np.sqrt(r.loc[c,"n"]) if (c in r.index and r.loc[c,"pf"]<.05) else np.nan) for c in cols])

def maps_for(dsfile, clin=None):
    from data_utils import load_scz_dataset, load_bd_matched
    _loaders = {"dataset.csv": load_scz_dataset, "bd": load_bd_matched}
    loader = _loaders.get(dsfile)
    if loader is not None:
        df = loader(os.path.join("..", "data")).copy()
    else:
        df = pd.read_csv(os.path.join("..", "data", dsfile))
    df["Condition_type"] = df["Condition_type"].astype(str)
    if clin is not None:   # separate diagnosis vs CN (SCZ vs CN, SAD vs CN)
        df=df[df["dx"].isin(["CN",clin])].copy()
        sites=set(df[df["dx"]==clin]["study_site"]); df=df[df["study_site"].isin(sites)].copy()
        df["Condition_type"]=np.where(df["dx"]==clin,"Clinical","Control")
    SCTX16=[_match("Left",t,df.columns) for t in _SCTX_TARGETS]+[_match("Right",t,df.columns) for t in _SCTX_TARGETS]
    SCTX_NV=[c for c in SCTX16 if c and "Vent" not in c]
    groups={"Whole":df,"Male":df[df.sex.astype(str).str.startswith(("M","m"))],"Female":df[df.sex.astype(str).str.startswith(("F","f"))]}
    out={}
    for g,gd in groups.items():
        out[g]={}
        for meas,cols in MEAS.items(): out[g][meas]=dvec(fit_group(gd,cols,g=="Whole"),cols)
        rs=fit_group(gd,SCTX_NV,g=="Whole")
        out[g]["SCTX"]=np.array([ (rs.loc[c,"t"]/np.sqrt(rs.loc[c,"n"]) if (c and c in rs.index and rs.loc[c,"pf"]<.05) else np.nan) for c in SCTX16])
    return out

# ---- vertex mapping -------------------------------------------------------- #
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

VMAX=0.5; NAN=(0.85,0.85,0.85,1)
def render_ctx(d68, tag, size=(1400,360)):
    fp=os.path.join(TMP,f"{tag}.png")
    plot_cortical(array_name=ctx_vert(d68), surface_name="fsa5", color_bar=False, color_range=(-VMAX,0),
                  cmap="Blues_r", nan_color=NAN, zoom=1.25, size=size, interactive=False, screenshot=True,
                  filename=fp, transparent_bg=False); return fp
def render_sctx(d16, tag, size=(1400,360)):
    fp=os.path.join(TMP,f"{tag}.png")
    plot_subcortical(array_name=d16, ventricles=True, color_bar=False, color_range=(-VMAX,0), cmap="Blues_r",
                     nan_color=NAN, zoom=1.25, size=size, interactive=False, screenshot=True, filename=fp,
                     transparent_bg=False); return fp

import matplotlib.font_manager as _fm
_FONT=_fm.findfont(_fm.FontProperties(family="DejaVu Sans"))
_FONTB=_fm.findfont(_fm.FontProperties(family="DejaVu Sans",weight="bold"))
def font(sz,bold=True):
    try: return ImageFont.truetype(_FONTB if bold else _FONT,int(sz))
    except Exception: return ImageFont.load_default()

def build_panel(M):
    GROUPS=["Whole","Male","Female"]; CORT=["CT","GMV","SA"]
    imgs={}
    for g in GROUPS:
        for m in CORT: imgs[(g,m)]=Image.open(render_ctx(M[g][m],f"{id(M)}_{g}_{m}")).convert("RGB")
        imgs[(g,"SCTX")]=Image.open(render_sctx(M[g]["SCTX"],f"{id(M)}_{g}_SCTX")).convert("RGB")
    cw=max(imgs[(g,m)].width for g in GROUPS for m in CORT); ch=max(imgs[(g,m)].height for g in GROUPS for m in CORT)
    cort={k:(v if v.size==(cw,ch) else v.resize((cw,ch),Image.Resampling.LANCZOS)) for k,v in imgs.items() if k[1]!="SCTX"}
    lblw=int(cw*0.20); hdr=int(ch*0.20)
    col_gap=int(cw*0.06); row_gap=int(ch*0.05)            # space the measures apart
    colx=lambda mi: lblw+mi*(cw+col_gap)                  # left x of column mi
    # subcortical: three BIGGER maps (whole/male/female), spaced, centred under the GMV column
    sub_gap=int(cw*0.04); sw=int(cw*1.0)
    raw=imgs[(GROUPS[0],"SCTX")]; sh=int(sw*raw.height/raw.width)
    sub={g:imgs[(g,"SCTX")].resize((sw,sh),Image.Resampling.LANCZOS) for g in GROUPS}
    sublbl_h=int(sh*0.30)
    view_hdr=0
    W=lblw+cw*3+col_gap*2; Hc=ch*3+row_gap*2
    canvas=Image.new("RGB",(W,view_hdr+hdr+Hc+hdr+sh+sublbl_h),"white"); d=ImageDraw.Draw(canvas)

    for mi,m in enumerate(CORT):
        d.text((colx(mi)+cw//2,view_hdr+hdr//2),m,fill="black",font=font(int(hdr*0.90)),anchor="mm")
    for gi,g in enumerate(GROUPS):
        y=view_hdr+hdr+gi*(ch+row_gap)
        d.text((int(lblw*0.06),y+ch//2),g,fill="black",font=font(int(ch*0.16)),anchor="lm")
        for mi,m in enumerate(CORT): canvas.paste(cort[(g,m)],(colx(mi),y))
    # "Subcortical" header + the three maps, centred on the GMV (middle) column
    gmv_cx=colx(1)+cw//2
    d.text((gmv_cx,view_hdr+hdr+Hc+hdr//2),"Subcortical",fill="black",font=font(int(hdr*0.62)),anchor="mm")
    ysub=view_hdr+hdr+Hc+hdr; roww=sw*3+sub_gap*2; x0=gmv_cx-roww//2
    for gi,g in enumerate(GROUPS):
        x=x0+gi*(sw+sub_gap); canvas.paste(sub[g],(x,ysub))
        d.text((x+sw//2,ysub+sh+sublbl_h//2),{"Whole":"Whole","Male":"Male","Female":"Female"}[g],
               fill="black",font=font(int(sublbl_h*0.6)),anchor="mm")
    return canvas

def top_colorbar(width):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import Normalize
    fig=plt.figure(figsize=(6,0.5)); ax=fig.add_axes([0.05,0.5,0.9,0.4])
    cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap("Blues_r"),norm=Normalize(-VMAX,0),orientation="horizontal")
    cb.set_label("Cohen's d (Clinical vs Control) · grey = n.s.",fontsize=9)
    buf=io.BytesIO(); fig.savefig(buf,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig); buf.seek(0)
    im=Image.open(buf).convert("RGB"); return im.resize((int(width*0.5),int(im.height*width*0.5/im.width)),Image.Resampling.LANCZOS)

print("rendering SCZ…"); A=build_panel(maps_for("dataset.csv","SCZ"))
print("rendering SAD…"); B=build_panel(maps_for("dataset.csv","Schizoaffective"))
print("rendering BD…");  C=build_panel(maps_for("bd"))
W=max(A.width,B.width,C.width); lw=int(W*0.12); cb=top_colorbar(W); pad=90
H=A.height+pad+B.height+pad+C.height+pad+cb.height+8
canvas=Image.new("RGB",(lw+W,H),"white"); d=ImageDraw.Draw(canvas)
y=0
for lab,panel in [("SCZ",A),("SAD",B),("BD",C)]:
    d.text((int(lw*0.12),y+panel.height//2),lab,fill="black",font=font(int(W*0.022)),anchor="lm")
    canvas.paste(panel,(lw,y))
    y+=panel.height+pad
canvas.paste(cb,(lw+(W-cb.width)//2,y))
canvas.save(os.path.join(FIGDIR,"figure1_paper.png"))
print("Saved ../figures/figure1_paper.png (rows: SCZ, SAD, BD)", canvas.size)

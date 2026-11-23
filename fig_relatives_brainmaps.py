# %%
"""
Figure 4A (paper) — relative-vs-control Cohen's d brain maps, FDR-corrected, whole group.
=========================================================================================
Same VTK/enigmatoolbox surface look as fig_regional_brainmaps.py, but for the BSNIP relatives:
one ROW of maps per diagnosis (SCZ, SAD, BD) x cortical columns (CT, GMV, SA) with the
SUBCORTICAL map placed under the GMV column. No sex-stratified rows. Grey = not FDR-sig.

Relative groups (each vs CN, from bsnip_relatives_dataset.csv, dx column):
  SCZ : 'SCZ-Relative'
  SAD : 'Schizoaffective Disorder-Relative'
  BD  : 'BD-Relative' + 'Bipolar/Psych-Relative'  (merged psychosis-spectrum relatives)

Then composes 3A (top) over figure3B_scatter.png (the scatter grid) into figure3_paper.png,
with A / B panel letters.

Run in the venv:  python fig_relatives_brainmaps.py
Output: ../figures/figure3A_relatives.png and ../figures/figure3_paper.png
"""
import os, io, numpy as np, pandas as pd, nibabel as nib, vtk
import vtk.util.numpy_support as ns
import warnings; warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw, ImageFont
import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests
np.random.seed(42)

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"); os.makedirs(FIGDIR, exist_ok=True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = os.path.join(REPO, "figures", "_f3atmp"); os.makedirs(TMP, exist_ok=True)

# ---- VTK 9 patches (identical to fig_regional_brainmaps.py) ---------------------------- #
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

def fit_group(data, cols):
    """OLS per region: value ~ age_z + C(sex) + C(study_site) + Relative(vs Control).
    Relatives are coded Condition_type='Clinical' so the model matches fig_regional_brainmaps.py."""
    recs=[]
    for col in [c for c in cols if c and c in data.columns]:
        safe=col.replace(".","_"); tmp=data[[col,"age","sex","study_site","Condition_type"]].dropna().copy()
        tmp["age_z"]=(tmp["age"]-tmp["age"].mean())/tmp["age"].std(ddof=1); tmp=tmp.rename(columns={col:safe})
        f=f"{safe} ~ age_z + C(sex) + C(study_site) + C(Condition_type, Treatment(reference='Control'))"
        m=smf.ols(f,data=tmp).fit()
        k=[i for i in m.params.index if "Clinical" in i and ":" not in i][0]
        recs.append({"col":col,"t":m.tvalues[k],"p":m.pvalues[k],"n":int(m.nobs)})
    r=pd.DataFrame(recs).set_index("col"); _,r["pf"],_,_=multipletests(r["p"],method="fdr_bh"); return r
def val(r, c, metric):
    """FDR-thresholded value for region c: Cohen's d (t/sqrt(n)) or t."""
    if not (c and c in r.index and r.loc[c,"pf"]<.05): return np.nan
    return r.loc[c,"t"]/np.sqrt(r.loc[c,"n"]) if metric=="d" else r.loc[c,"t"]

REL={"SCZ":["SCZ-Relative"],
     "SAD":["Schizoaffective Disorder-Relative"],
     "BD" :["BD-Relative","Bipolar/Psych-Relative"]}
def maps_for(rel_labels, metric):
    df=pd.read_csv(os.path.join(REPO,"data","bsnip_relatives_dataset.csv"))
    df=df[df["dx"].isin(["CN"]+rel_labels)].copy()
    sites=set(df[df["dx"].isin(rel_labels)]["study_site"]); df=df[df["study_site"].isin(sites)].copy()
    df["Condition_type"]=np.where(df["dx"].isin(rel_labels),"Clinical","Control")
    SCTX16=[_match("Left",t,df.columns) for t in _SCTX_TARGETS]+[_match("Right",t,df.columns) for t in _SCTX_TARGETS]
    SCTX_NV=[c for c in SCTX16 if c and "Vent" not in c]
    out={}
    for meas,cols in MEAS.items():
        r=fit_group(df,cols); out[meas]=np.array([val(r,c,metric) for c in cols])
    rs=fit_group(df,SCTX_NV)
    out["SCTX"]=np.array([val(rs,c,metric) for c in SCTX16])
    return out

# ---- vertex mapping (identical to fig_regional_brainmaps.py) --------------------------- #
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
    plot_cortical(array_name=ctx_vert(d68), surface_name="fsa5", color_bar=False, color_range=(-VMAX,VMAX),
                  cmap="RdBu_r", nan_color=NAN, zoom=1.25, size=size, interactive=False, screenshot=True,
                  filename=fp, transparent_bg=False); return fp
def render_sctx(d16, tag, size=(1400,360)):
    fp=os.path.join(TMP,f"{tag}.png")
    plot_subcortical(array_name=d16, ventricles=True, color_bar=False, color_range=(-VMAX,VMAX), cmap="RdBu_r",
                     nan_color=NAN, zoom=1.25, size=size, interactive=False, screenshot=True, filename=fp,
                     transparent_bg=False); return fp

import matplotlib.font_manager as _fm
_FONT=_fm.findfont(_fm.FontProperties(family="DejaVu Sans"))
_FONTB=_fm.findfont(_fm.FontProperties(family="DejaVu Sans",weight="bold"))
def font(sz,bold=True):
    try: return ImageFont.truetype(_FONTB if bold else _FONT,int(sz))
    except Exception: return ImageFont.load_default()

def build_panel(M, tagbase):
    """One row: CT, GMV, SA cortical columns + subcortical map under GMV."""
    CORT=["CT","GMV","SA"]
    imgs={m:Image.open(render_ctx(M[m],f"{tagbase}_{m}")).convert("RGB") for m in CORT}
    imgs["SCTX"]=Image.open(render_sctx(M["SCTX"],f"{tagbase}_SCTX")).convert("RGB")
    cw=max(imgs[m].width for m in CORT); ch=max(imgs[m].height for m in CORT)
    cort={m:(v if v.size==(cw,ch) else v.resize((cw,ch),Image.Resampling.LANCZOS)) for m,v in imgs.items() if m!="SCTX"}
    lblw=int(cw*0.20); hdr=int(ch*0.20); col_gap=int(cw*0.06)
    colx=lambda mi: lblw+mi*(cw+col_gap)
    sw=int(cw*1.0); raw=imgs["SCTX"]; sh=int(sw*raw.height/raw.width); sublbl_h=int(sh*0.28)
    subimg=imgs["SCTX"].resize((sw,sh),Image.Resampling.LANCZOS)
    W=lblw+cw*3+col_gap*2; Hc=ch
    canvas=Image.new("RGB",(W,hdr+Hc+hdr+sh+sublbl_h),"white"); d=ImageDraw.Draw(canvas)
    for mi,m in enumerate(CORT):
        d.text((colx(mi)+cw//2,hdr//2),m,fill="black",font=font(int(hdr*0.62)),anchor="mm")
        canvas.paste(cort[m],(colx(mi),hdr))
    gmv_cx=colx(1)+cw//2
    d.text((gmv_cx,hdr+Hc+hdr//2),"Subcortical",fill="black",font=font(int(hdr*0.62)),anchor="mm")
    ysub=hdr+Hc+hdr; canvas.paste(subimg,(gmv_cx-sw//2,ysub))
    return canvas

def top_colorbar(width, label):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import Normalize
    fig=plt.figure(figsize=(6,0.5)); ax=fig.add_axes([0.05,0.5,0.9,0.4])
    cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap("RdBu_r"),norm=Normalize(-VMAX,VMAX),orientation="horizontal")
    cb.set_label(label,fontsize=9)
    buf=io.BytesIO(); fig.savefig(buf,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig); buf.seek(0)
    im=Image.open(buf).convert("RGB"); return im.resize((int(width*0.5),int(im.height*width*0.5/im.width)),Image.Resampling.LANCZOS)

# metric -> (colour-scale limit, colorbar label, scatter panel suffix)
METRICS={"d":(0.5,"Cohen's d · grey = n.s.","d"),
         "t":(5.0,"t (Relative vs Control) · grey = n.s.","t")}

def run(metric):
    global VMAX
    VMAX,label,scat=METRICS[metric]
    print(f"=== metric = {metric} (VMAX={VMAX}) ===")
    A=build_panel(maps_for(REL["SCZ"],metric),f"SCZ_{metric}")
    B=build_panel(maps_for(REL["SAD"],metric),f"SAD_{metric}")
    C=build_panel(maps_for(REL["BD"], metric),f"BD_{metric}")
    W=max(A.width,B.width,C.width); lw=int(W*0.11); cb=top_colorbar(W,label); pad=20
    H=A.height+pad+B.height+pad+C.height+pad+cb.height+8
    p3a=Image.new("RGB",(lw+W,H),"white"); d=ImageDraw.Draw(p3a)
    y=0
    for lab,panel in [("SCZ\nrelatives",A),("SAD\nrelatives",B),("BD\nrelatives",C)]:
        d.multiline_text((int(lw*0.10),y+panel.height//2),lab,fill="black",font=font(int(W*0.020)),anchor="lm",align="left",spacing=6)
        p3a.paste(panel,(lw,y)); y+=panel.height+pad
    p3a.paste(cb,(lw+(W-cb.width)//2,y))
    p3a.save(os.path.join(FIGDIR,f"figure4A_relatives_{metric}.png"))
    # Panel A = clinical-vs-relative thresholded maps (figS_case_vs_relative_brainmaps.py)
    a_path=os.path.join(FIGDIR,"figureS6_case_vs_relative.png")
    if not os.path.exists(a_path):
        print("WARNING: figureS6_case_vs_relative.png not found — run figS_case_vs_relative_brainmaps.py first; saved relatives map only."); return
    b_path=os.path.join(FIGDIR,f"figure4B_scatter_{scat}.png")
    if not os.path.exists(b_path):
        print(f"WARNING: {os.path.basename(b_path)} not found — run fig_relatives_scatter.py first; saved 4A only."); return
    pA=Image.open(a_path).convert("RGB")
    pB=Image.open(b_path).convert("RGB")
    panels=[(pA,"A"),(pB,"B")]
    Wc=max(im.width for im,_ in panels)
    fit=lambda im: im if im.width==Wc else im.resize((Wc,int(im.height*Wc/im.width)),Image.Resampling.LANCZOS)
    letter=font(int(Wc*0.03)); head=int(Wc*0.04); gap=int(Wc*0.02)
    total_H=sum(head+fit(im).height for im,_ in panels)+gap*(len(panels)-1)
    comp=Image.new("RGB",(Wc,total_H),"white"); dd=ImageDraw.Draw(comp); y=0
    for idx,(im,lbl) in enumerate(panels):
        fi=fit(im); dd.text((int(Wc*0.005),y+2),lbl,fill="black",font=letter)
        comp.paste(fi,(0,y+head)); y+=head+fi.height+(gap if idx<len(panels)-1 else 0)
    comp.save(os.path.join(FIGDIR,f"figure4_paper_{metric}.png"))
    print(f"Saved figure4_paper_{metric}.png", comp.size)

for _m in ("d","t"):
    run(_m)
# default embed = the Cohen's d version
_dpath=os.path.join(FIGDIR,"figure4_paper_d.png")
if os.path.exists(_dpath):
    Image.open(_dpath).save(os.path.join(FIGDIR,"figure4_paper.png"))
    print("figure4_paper.png := figure4_paper_d.png")

# %%
"""
Supplementary Figure S5 — UNTHRESHOLDED relative-versus-control Cohen's d maps.
============================================================================
Complementary to Figure 4A (which is FDR-thresholded): here every region is coloured
by its relative-versus-control Cohen's d, with no significance threshold, so the
generalised, sub-threshold reduction carried by relatives is visible. One row per
diagnosis (SCZ, SAD, BD) x cortical columns (CT, GMV, SA), subcortical under GMV.

Relative groups (each vs CN, from bsnip_relatives_dataset.csv, dx column):
  SCZ : 'SCZ-Relative'
  SAD : 'Schizoaffective Disorder-Relative'
  BD  : 'BD-Relative' + 'Bipolar/Psych-Relative'

Run in the venv:  python figS_relatives_unthresholded.py
Output: ../figures/figureS5_relatives_unthresholded.png
"""
import os, io, numpy as np, pandas as pd, nibabel as nib, vtk
import vtk.util.numpy_support as ns
import warnings; warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw, ImageFont
import statsmodels.formula.api as smf

np.random.seed(42)
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIGDIR=os.path.join(BASE,"figures"); TMP=os.path.join(BASE,"figures","_fSrtmp"); os.makedirs(TMP,exist_ok=True)
VMAX=0.25  # unthresholded d is small; tighter scale than the case-control maps

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
_PKG=os.path.dirname(__import__("enigmatoolbox").__file__)
def _build(path):
    g=nib.load(path); coords=g.darrays[0].data.astype(np.float64); faces=g.darrays[1].data.astype(np.int64)
    pts=vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(coords)); n=len(faces)
    carr=np.hstack([np.full((n,1),3,dtype=np.int64),faces]).flatten()
    cells=vtk.vtkCellArray(); cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_=vtk.vtkPolyData(); pd_.SetPoints(pts); pd_.SetPolys(cells); return BSPolyData(pd_)
_LH=_build(os.path.join(_PKG,"datasets/surfaces/fsa5_lh.surf.gii")); _RH=_build(os.path.join(_PKG,"datasets/surfaces/fsa5_rh.surf.gii"))
_eds.load_fsa5=lambda as_sphere=False:(_LH,_RH); _splot.load_fsa5=_eds.load_fsa5
_SLH=_build(os.path.join(_PKG,'datasets/surfaces/sctx_lh.gii'))
_SRH=_build(os.path.join(_PKG,'datasets/surfaces/sctx_rh.gii'))
_eds.load_subcortical=lambda with_normals=False,join=False:(_SLH,_SRH)
_splot.load_subcortical=_eds.load_subcortical
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.datasets import load_fc

_, fc_ctx_labels, _, _ = load_fc()
def enigma_col(l, pref): h,r=l.split("_",1); return f"{'lh' if h=='L' else 'rh'}.{pref}.{r}"
MEAS={"CT":[enigma_col(l,"CT") for l in fc_ctx_labels],"GMV":[enigma_col(l,"GM") for l in fc_ctx_labels],"SA":[enigma_col(l,"SA") for l in fc_ctx_labels]}
_SCTX_TARGETS=["Accumbens.area","Amygdala","Caudate","Hippocampus","Pallidum","Putamen","Thalamus","Ventricle"]
def _match(side,t,cols):
    for c in cols:
        if side.lower() in c.lower() and t.lower() in c.lower(): return c
    return None
def fit_group(data, cols):
    recs=[]
    for col in [c for c in cols if c and c in data.columns]:
        safe=col.replace(".","_"); tmp=data[[col,"age","sex","study_site","Condition_type"]].dropna().copy()
        tmp["age_z"]=(tmp["age"]-tmp["age"].mean())/tmp["age"].std(ddof=1); tmp=tmp.rename(columns={col:safe})
        m=smf.ols(f"{safe} ~ age_z + C(sex) + C(study_site) + C(Condition_type, Treatment(reference='Control'))",data=tmp).fit()
        k=[i for i in m.params.index if "Clinical" in i and ":" not in i][0]
        recs.append({"col":col,"t":m.tvalues[k],"n":int(m.nobs)})
    return pd.DataFrame(recs).set_index("col")
def dvec(r, cols):  # UNTHRESHOLDED: every region coloured by Cohen's d = t/sqrt(n)
    return np.array([ (r.loc[c,"t"]/np.sqrt(r.loc[c,"n"]) if c in r.index else np.nan) for c in cols])
REL={"SCZ":["SCZ-Relative"],"SAD":["Schizoaffective Disorder-Relative"],"BD":["BD-Relative","Bipolar/Psych-Relative"]}
def maps_for(rel_labels):
    df=pd.read_csv(os.path.join(BASE,"data","bsnip_relatives_dataset.csv"))
    df=df[df["dx"].isin(["CN"]+rel_labels)].copy()
    sites=set(df[df["dx"].isin(rel_labels)]["study_site"]); df=df[df["study_site"].isin(sites)].copy()
    df["Condition_type"]=np.where(df["dx"].isin(rel_labels),"Clinical","Control")
    SCTX16=[_match("Left",t,df.columns) for t in _SCTX_TARGETS]+[_match("Right",t,df.columns) for t in _SCTX_TARGETS]
    SCTX_NV=[c for c in SCTX16 if c and "Vent" not in c]
    out={m:dvec(fit_group(df,cols),cols) for m,cols in MEAS.items()}
    rs=fit_group(df,SCTX_NV)
    out["SCTX"]=np.array([ (rs.loc[c,"t"]/np.sqrt(rs.loc[c,"n"]) if (c and c in rs.index) else np.nan) for c in SCTX16])
    return out
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
    plot_cortical(array_name=ctx_vert(d68), surface_name="fsa5", color_bar=False, color_range=(-VMAX,VMAX),
                  cmap="RdBu_r", nan_color=NAN, zoom=1.25, size=size, interactive=False, screenshot=True, filename=fp, transparent_bg=False); return fp
def render_sctx(d16, tag, size=(1400,360)):
    fp=os.path.join(TMP,f"{tag}.png")
    plot_subcortical(array_name=d16, ventricles=True, color_bar=False, color_range=(-VMAX,VMAX), cmap="RdBu_r",
                     nan_color=NAN, zoom=1.25, size=size, interactive=False, screenshot=True, filename=fp, transparent_bg=False); return fp
import matplotlib.font_manager as _fm
_FONTB=_fm.findfont(_fm.FontProperties(family="DejaVu Sans",weight="bold"))
def font(sz):
    try: return ImageFont.truetype(_FONTB,int(sz))
    except Exception: return ImageFont.load_default()
def build_panel(M, tag):
    CORT=["CT","GMV","SA"]
    imgs={m:Image.open(render_ctx(M[m],f"{tag}_{m}")).convert("RGB") for m in CORT}
    imgs["SCTX"]=Image.open(render_sctx(M["SCTX"],f"{tag}_SCTX")).convert("RGB")
    cw=max(imgs[m].width for m in CORT); ch=max(imgs[m].height for m in CORT)
    cort={m:(v if v.size==(cw,ch) else v.resize((cw,ch),Image.Resampling.LANCZOS)) for m,v in imgs.items() if m!="SCTX"}
    lblw=int(cw*0.20); hdr=int(ch*0.20); col_gap=int(cw*0.06)
    colx=lambda mi: lblw+mi*(cw+col_gap)
    sw=int(cw*1.0); raw=imgs["SCTX"]; sh=int(sw*raw.height/raw.width); sublbl_h=int(sh*0.28)
    subimg=imgs["SCTX"].resize((sw,sh),Image.Resampling.LANCZOS)
    W=lblw+cw*3+col_gap*2; Hc=ch
    canvas=Image.new("RGB",(W,hdr+Hc+hdr+sh+sublbl_h),"white"); d=ImageDraw.Draw(canvas)
    for mi,m in enumerate(CORT):
        d.text((colx(mi)+cw//2,hdr//2),m,fill="black",font=font(int(hdr*0.62)),anchor="mm"); canvas.paste(cort[m],(colx(mi),hdr))
    gmv_cx=colx(1)+cw//2
    d.text((gmv_cx,hdr+Hc+hdr//2),"Subcortical",fill="black",font=font(int(hdr*0.62)),anchor="mm")
    canvas.paste(subimg,(gmv_cx-sw//2,hdr+Hc+hdr))
    return canvas
def top_colorbar(width):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import Normalize
    fig=plt.figure(figsize=(6,0.5)); ax=fig.add_axes([0.05,0.5,0.9,0.4])
    cb=mpl.colorbar.ColorbarBase(ax,cmap=plt.get_cmap("RdBu_r"),norm=Normalize(-VMAX,VMAX),orientation="horizontal")
    cb.set_label("Cohen's d • grey = n.s.",fontsize=9)
    buf=io.BytesIO(); fig.savefig(buf,dpi=150,bbox_inches="tight",facecolor="white"); plt.close(fig); buf.seek(0)
    im=Image.open(buf).convert("RGB"); return im.resize((int(width*0.5),int(im.height*width*0.5/im.width)),Image.Resampling.LANCZOS)
A=build_panel(maps_for(REL["SCZ"]),"SCZ"); B=build_panel(maps_for(REL["SAD"]),"SAD"); C=build_panel(maps_for(REL["BD"]),"BD")
W=max(A.width,B.width,C.width); lw=int(W*0.20); cb=top_colorbar(W); pad=20
H=A.height+pad+B.height+pad+C.height+pad+cb.height+8
out=Image.new("RGB",(lw+W,H),"white"); d=ImageDraw.Draw(out)
y=0
for lab, panel in [("SCZ relatives \nvs controls", A), ("SAD relatives \nvs controls", B), ("BD relatives \nvs controls", C)]:
    d.multiline_text((int(lw*0.10), y+panel.height//2), lab, fill="black", font=font(int(W*0.020)), anchor="lm", align="left", spacing=6)
    out.paste(panel, (lw, y)); y += panel.height + pad
out.paste(cb,(lw+(W-cb.width)//2,y))
# save unthresholded-only (Panel A of the merged figure)
_unth_path=os.path.join(FIGDIR,"figureS5_unthresholded_panelA.png")
out.save(_unth_path)
print("Saved figureS5_unthresholded_panelA.png (panel A, unthresholded)", out.size)

# ── Merge Panel A (unthresholded) + Panel B (FDR-thresholded from Figure 4A) ──
thresh_path=os.path.join(FIGDIR,"figure4A_relatives_d.png")
if os.path.exists(thresh_path):
    pA=out; pB=Image.open(thresh_path).convert("RGB")
    Wc=max(pA.width,pB.width)
    fit=lambda im: im if im.width==Wc else im.resize((Wc,int(im.height*Wc/im.width)),Image.Resampling.LANCZOS)
    A,B=fit(pA),fit(pB)
    ltr=font(int(Wc*0.025)); head=int(Wc*0.035); gap=int(Wc*0.02)
    merged=Image.new("RGB",(Wc,head+A.height+gap+head+B.height),"white"); dm=ImageDraw.Draw(merged)
    dm.text((int(Wc*0.005),2),"A",fill="black",font=ltr); merged.paste(A,(0,head))
    yb=head+A.height+gap; dm.text((int(Wc*0.005),yb),"B",fill="black",font=ltr); merged.paste(B,(0,yb+head))
    merged.save(os.path.join(FIGDIR,"figureS5_relatives_unthresholded.png"))
    print("Saved figureS5_relatives_unthresholded.png (merged A=unthresholded + B=FDR-thresholded)", merged.size)
else:
    out.save(os.path.join(FIGDIR,"figureS5_relatives_unthresholded.png"))
    print("WARNING: figure4A_relatives_d.png not found — run fig_relatives_brainmaps.py first for the merged figure.")
    print("Saved figureS5_relatives_unthresholded.png (unthresholded only)", out.size)

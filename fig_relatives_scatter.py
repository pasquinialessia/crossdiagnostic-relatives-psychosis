#!/usr/bin/env python3
"""Figure 4B. A: regional clinical-vs-relative scatterplots (t clinical-vs-CN
on x, t relative-vs-CN on y) per diagnosis and measure, with fit line and recapitulation
r (spin/shuffle FDR q). B: recapitulation matrix."""
import os, io, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.stats.multitest import multipletests
from PIL import Image
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES=os.path.join(REPO,"results/relatives"); FIG=os.path.join(REPO,"figures"); OUT=os.path.join(REPO,"results","supp_tables")
DX=["SCZ","SAD","BD"]; ME=["CT","GMV","SA","Subcortical"]
s=pd.read_csv(os.path.join(RES,"bsnip_relatives_summary.csv"))
sp=pd.read_csv(os.path.join(RES,"recap_spatial_null.csv"))
mrg=s.merge(sp[["contrast","measure","p_spatial"]],on=["contrast","measure"])
mrg["q"]=multipletests(mrg["p_spatial"].values,method="fdr_bh")[1]
def star(q): return "***" if q<.001 else "**" if q<.01 else "*" if q<.05 else "n.s."

# ---------- Panel 3B: scatter grid, one file per metric (t and d) ----------
# Cohen's d columns live in maps_{dx}_{m}_d.csv (run compute_relatives_d.py first).
# r and its FDR q come from the recap spatial null (identical for t and d, since d is a
# monotonic rescale of t); only the axis scale and the fit slope β change.
def make_scatter(metric):
    xcol="t_proband_vs_CN" if metric=="t" else "d_proband_vs_CN"
    ycol="t_relative_vs_CN" if metric=="t" else "d_relative_vs_CN"
    suffix="" if metric=="t" else "_d"
    figA=plt.figure(figsize=(11,8.2))
    gs=GridSpec(len(DX),len(ME),figure=figA,hspace=0.42,wspace=0.30,left=0.08,right=0.98,top=0.93,bottom=0.08)
    for i,dx in enumerate(DX):
        for j,m in enumerate(ME):
            ax=figA.add_subplot(gs[i,j])
            mp=pd.read_csv(os.path.join(RES,f"maps_{dx}_{m}{suffix}.csv"))
            x=mp[xcol].values; y=mp[ycol].values
            row=mrg[(mrg.contrast==dx)&(mrg.measure==m)].iloc[0]; r=row["r_recap_rel_vs_proband"]; q=row["q"]
            ax.axhline(0,color="0.85",lw=0.7); ax.axvline(0,color="0.85",lw=0.7)
            ax.scatter(x,y,s=12,alpha=0.7,edgecolor="none",color="#2c6fbb")
            xs=np.linspace(x.min(),x.max(),50); b,a=np.polyfit(x,y,1); ax.plot(xs,b*xs+a,color="#c0392b",lw=1.4)
            ax.set_title(f"{dx} · {m}",fontsize=13,pad=3)
            ax.text(0.04,0.93,f"ρ = {r:.2f} {star(q)}\nβ = {b:.2f}",transform=ax.transAxes,fontsize=11,va="top",
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="0.8",alpha=0.8))
            ax.tick_params(labelsize=10)
            if i==len(DX)-1: ax.set_xlabel(f"clinical vs control ({metric})",fontsize=11)
            if j==0: ax.set_ylabel(f"relative vs control ({metric})",fontsize=11)
    for d_ in (FIG,OUT):
        figA.savefig(os.path.join(d_,f"figure4B_scatter_{metric}.png"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(figA)
    from PIL import Image as _Im
    print(f"saved figure4B_scatter_{metric}.png",_Im.open(os.path.join(FIG,f"figure4B_scatter_{metric}.png")).size)

for _metric in ("t","d"):
    make_scatter(_metric)

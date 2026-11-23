#!/usr/bin/env python3
"""Figure 3 — cross-diagnostic similarity of case-vs-control maps (whole sample).
Whole-sample matrix only. No titles; single colorbar; significance stars kept on cells."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(REPO,"results","supp_tables"); FIG=os.path.join(REPO,"figures")
d=pd.read_csv(os.path.join(REPO,"results/crossdiagnostic/transdx_spin_shuf_unthresh/spin_shuf_summary.csv"))
PAIRS=[("SCZ","Schizoaffective"),("BD","SCZ"),("BD","Schizoaffective")]; PAIR_LBL=["SCZ – SAD","SCZ – BD","SAD – BD"]
MEAS=["CT","GMV","SA","SubcGM"]; MEAS_LBL=["CT","GMV","SA","Subcortical"]
def star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
def matrix(sex):
    R=np.full((3,4),np.nan); S=[["" for _ in range(4)] for _ in range(3)]
    for i,(a,b) in enumerate(PAIRS):
        for j,m in enumerate(MEAS):
            row=d[(d.sex==sex)&(d.measure==m)&(d.dx1==a)&(d.dx2==b)]
            if len(row):
                R[i,j]=row.r.iloc[0]
                if row.p_fdr.iloc[0]<0.05: S[i][j]=f"{row.r.iloc[0]:.2f}{star(row.p_fdr.iloc[0])}"  # blank n.s.
    return R,S
def draw(ax,R,S,ylabels=True):
    im=ax.imshow(R,cmap="RdBu_r",vmin=-1,vmax=1,aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels(MEAS_LBL,fontsize=14)
    ax.set_yticks(range(3)); ax.set_yticklabels(PAIR_LBL if ylabels else [],fontsize=14)
    for i in range(3):
        for j in range(4):
            if not np.isnan(R[i,j]):
                ax.text(j,i,S[i][j],ha="center",va="center",fontsize=14,color="white" if abs(R[i,j])>.6 else "black")
    ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which="minor",color="white",lw=1.5); ax.tick_params(which="minor",length=0)
    return im

# --- figure 3: whole sample ---
figA,axA=plt.subplots(figsize=(6.4,4.2))
im=draw(axA,*matrix("Whole"))
cb=figA.colorbar(im,ax=axA,fraction=0.046,pad=0.03); cb.set_label("correlation ρ",fontsize=15)
figA.tight_layout()
figA.savefig(os.path.join(FIG,"figure3_crossdx.png"),dpi=200,bbox_inches="tight")
figA.savefig(os.path.join(OUT,"figure3_crossdx.png"),dpi=200,bbox_inches="tight")


#!/usr/bin/env python3
"""Figure 5 — cross-diagnostic familial sharing: relative-of-X (rows) vs clinical-of-Y
(cols) map correlation per measure; boxed diagonal = within-disorder. No title; single
colorbar; stars from spin/shuffle FDR. Single panel (no A/B)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(REPO,"results","supp_tables"); FIG=os.path.join(REPO,"figures")
d=pd.read_csv(os.path.join(REPO,"results/relatives_reviewer/crossdisorder_familial_sharing_spin.csv"))
DX=["SCZ","SAD","BD"]; MEAS=["CT","GMV","SA","Subcortical"]
def star(p):
    return "" if pd.isna(p) else ("***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "")
fig,axes=plt.subplots(1,4,figsize=(15,4.0),constrained_layout=True)
im=None
for ax,m in zip(axes,MEAS):
    sub=d[d.measure==m]
    M=sub.pivot(index="relative_of",columns="proband_of",values="r_rel_vs_prob").reindex(index=DX,columns=DX)
    P=sub.pivot(index="relative_of",columns="proband_of",values="p_fdr_rel_vs_prob").reindex(index=DX,columns=DX)
    im=ax.imshow(M.values.astype(float),cmap="RdBu_r",vmin=-1,vmax=1,aspect="equal")
    ax.set_xticks(range(3)); ax.set_xticklabels(DX,fontsize=14)
    ax.set_yticks(range(3)); ax.set_yticklabels(DX if m=="CT" else [],fontsize=14)
    if m=="CT": ax.set_xlabel("clinical of",fontsize=14)
    if m=="CT": ax.set_ylabel("relative of",fontsize=15)
    for i in range(3):
        for j in range(3):
            v=M.values[i,j]; p=P.values[i,j]
            if not np.isnan(v) and not pd.isna(p) and p<0.05:   # only label significant cells
                ax.text(j,i,f"{v:.2f}{star(p)}",ha="center",va="center",
                        fontsize=13,fontweight="normal",
                        color="white" if abs(v)>.6 else "black")
        ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor="black",lw=1.8))
    ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which="minor",color="white",lw=1.2); ax.tick_params(which="minor",length=0)
    ax.set_title(m,fontsize=15)   # measure label only (column identifier, not a figure title)
cb=fig.colorbar(im,ax=axes,fraction=0.02,pad=0.01); cb.set_label("correlation ρ",fontsize=15)
fig.savefig(os.path.join(FIG,"figure5_familial.png"),dpi=200,bbox_inches="tight")
fig.savefig(os.path.join(OUT,"figure5_familial.png"),dpi=200,bbox_inches="tight")
print("saved figure5")

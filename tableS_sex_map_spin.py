#!/usr/bin/env python3
"""Spin (cortical) / shuffle (subcortical) significance of the male-vs-female
case-control effect-size maps, per diagnosis. Tests whether the two sex-specific
maps are correlated MORE than expected under spatial autocorrelation. FDR across
the 12 diagnosis x measure tests. Self-contained (reads ENIGMA fsa5 sphere+annot)."""
import os, numpy as np, pandas as pd, nibabel as nb
from scipy.stats import spearmanr as _spearmanr, rankdata as _rankdata
from scipy.spatial.distance import cdist
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(REPO,"results","supp_tables")
EPKG=os.path.join(REPO,"lib/python3.12/site-packages/enigmatoolbox")
ANNOT=os.path.join(EPKG,"permutation_testing","annot"); SURF=os.path.join(EPKG,"datasets","surfaces")
N_ROT=1000; np.random.seed(42)
SKIP={"unknown","corpus","corpuscallosum","medialwall","Background+FreeSurfer_Defined_Medial_Wall"}
def sphere_pts(s): return nb.load(os.path.join(SURF,f"fsa5_sphere_{s}.gii")).darrays[0].data.astype(float)
def centroids_and_names(s):
    lab,ctab,names=nb.freesurfer.io.read_annot(os.path.join(ANNOT,f"fsa5_{s}_aparc.annot"),orig_ids=True)
    pts=sphere_pts(s); cents=[]; keep=[]
    for ic in range(ctab.shape[0]):
        nm=names[ic].decode()
        if nm in SKIP: continue
        cents.append(pts[lab==ctab[ic,-1]].mean(0)); keep.append(nm)
    return np.vstack(cents),keep
def rotate_parcellation(cl,cr,nrot=N_ROT):
    nl,nr=cl.shape[0],cr.shape[0]; nroi=nl+nr; perm_id=np.zeros((nroi,nrot),int); r=0; I1=np.diag([-1.,1.,1.])
    while r<nrot:
        A=np.random.normal(0,1,(3,3)); TL,tmp=np.linalg.qr(A); TL=TL@np.diag(np.sign(np.diag(tmp)))
        if np.linalg.det(TL)<0: TL[:,0]=-TL[:,0]
        TR=I1@TL@I1; dl=cdist(cl,cl@TL); dr=cdist(cr,cr@TR)
        def assign(d,n):
            d=d.copy(); ref=np.zeros(n,int); rot=np.zeros(n,int)
            for k in range(n):
                mins=np.nanmin(d,axis=1); ri=int(np.argwhere(mins==np.nanmax(mins))[0])
                ci=int(np.argwhere(d[ri,:]==np.nanmin(d[ri,:]))[0]); ref[k]=ri; rot[k]=ci; d[:,ci]=np.nan; d[ri,:]=np.nan
            return ref,rot
        rl,rlr=assign(dl,nl); rr,rrr=assign(dr,nr)
        ref=np.append(rl,nl+rr); rot=np.append(rlr,nl+rrr); rs=rot[np.argsort(ref)]
        if not np.all(rs==np.arange(nroi)): perm_id[:,r]=rs; r+=1
    return perm_id
def _colcorr(M,v):
    Mc=M-M.mean(0); vc=v-v.mean(); return (Mc*vc[:,None]).sum(0)/(np.sqrt((Mc**2).sum(0))*np.sqrt((vc**2).sum()))
def perm_p(x,y,perm_id):
    rho=float(_spearmanr(x,y)[0])  # Spearman rho
    rx=_rankdata(x); ry=_rankdata(y)  # rank for consistent Spearman null
    rxp=rx[perm_id]; ryp=ry[perm_id]
    nxy=_colcorr(rxp,ry); nyx=_colcorr(ryp,rx)
    p=(np.mean(nxy>rho)+np.mean(nyx>rho))/2 if rho>=0 else (np.mean(nxy<rho)+np.mean(nyx<rho))/2
    return rho,float(p)
def fdr(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); r=np.empty(n); c=p[o]*n/(np.arange(n)+1)
    r[o]=np.minimum.accumulate(c[::-1])[::-1]; return np.clip(r,0,1)

cl,names_lh=centroids_and_names("lh"); cr,names_rh=centroids_and_names("rh")
perm_cort=rotate_parcellation(cl,cr,N_ROT)
cc=pd.read_csv(os.path.join(OUT,"_supp_cc.csv"))
def cort_vec(dx,meas,grp):
    d=cc[(cc.disorder==dx)&(cc.measure==meas)&(cc.group==grp)]
    m={(r.hemisphere,r.region):r.t for r in d.itertuples()}
    return np.array([m[("Left",n)] for n in names_lh]+[m[("Right",n)] for n in names_rh])
def sub_vec(dx,grp):
    d=cc[(cc.disorder==dx)&(cc.measure=="Subcortical")&(cc.group==grp)].sort_values(["hemisphere","region"])
    return d.t.values
rows=[]
for dx in ["SCZ","SAD","BD"]:
    for meas in ["CT","GMV","SA"]:
        x=cort_vec(dx,meas,"male"); y=cort_vec(dx,meas,"female"); rho,p=perm_p(x,y,perm_cort)
        rows.append(dict(disorder=dx,measure=meas,rho=round(rho,3),p_spin=round(p,4)))
    xs=sub_vec(dx,"male"); ys=sub_vec(dx,"female"); n=len(xs)
    perm_sub=np.array([np.random.permutation(n) for _ in range(N_ROT)]).T
    rho,p=perm_p(xs,ys,perm_sub); rows.append(dict(disorder=dx,measure="Subcortical",rho=round(rho,3),p_spin=round(p,4)))
res=pd.DataFrame(rows); res["q_fdr"]=np.round(fdr(res.p_spin.values),4)
res.to_csv(os.path.join(OUT,"_supp_sexmap_spin.csv"),index=False)
print(res.to_string(index=False))

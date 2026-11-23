#!/usr/bin/env python3
"""Case-control statistics for SCZ, SAD and BD SEPARATELY (each vs CN) — canonical
script behind Table 1 and Supplementary Tables S1-S3, S7.
Computes:
regional (whole/M/F), hemispheric, region-specificity, sex interaction, sex-permutation
ceiling. Writes tidy CSVs consumed by the supplementary-table builders."""
import os, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from data_utils import load_bd_matched, load_scz_dataset
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data"); OUT=os.path.join(HERE,"..","results","02_casecontrol","tables"); os.makedirs(OUT,exist_ok=True)
rng=np.random.default_rng(42)
DIS=[("SCZ","_scz","SCZ"),("SAD","_scz","Schizoaffective"),("BD","_bd",None)]
_scz_df=load_scz_dataset(D); _bd_df=load_bd_matched(D)
_loaders={"_scz":_scz_df,"_bd":_bd_df}
def parse(col):
    if col.startswith(("lh.","rh.")): h,m,r=col.split(".",2); return ("Left" if h=="lh" else "Right"),r
    p=col.split("."); return p[1],".".join(p[2:])
def sample(dsfile,clin):
    df=_loaders[dsfile].copy(); df["Condition_type"]=df["Condition_type"].astype(str)
    if clin is not None:
        df=df[df.dx.isin(["CN",clin])].copy(); sites=set(df[df.dx==clin].study_site); df=df[df.study_site.isin(sites)].copy()
        cond=(df.dx==clin).astype(float).values
    else:
        cond=(df.Condition_type=="Clinical").astype(float).values
    df=df.reset_index(drop=True); df["cond"]=cond; return df
def cols_of(df):
    return {"CT":[c for c in df if c.startswith(("lh.CT.","rh.CT."))],
            "GMV":[c for c in df if c.startswith(("lh.GM.","rh.GM."))],
            "SA":[c for c in df if c.startswith(("lh.SA.","rh.SA."))],
            "Subcortical":[c for c in df if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]}
def design(g,whole,extra=None):
    site=pd.get_dummies(g.study_site.astype(str),drop_first=True).astype(float)
    base=[np.ones(len(g)),g.age_z.values]
    if whole: base.append((g.sex.astype(str).str.startswith(("M","m"))).astype(float).values)
    cols=base+[site.values]+([extra] if extra is not None else [])
    return np.column_stack(cols)
def tmap(X,Y,jc):
    Xi=np.linalg.pinv(X.T@X);B=Xi@(X.T@Y);ssr=((Y-X@B)**2).sum(0);dof=X.shape[0]-X.shape[1]
    return B[jc]/np.sqrt(Xi[jc,jc]*ssr/dof),dof

cc=[];hemi=[];spec=[];permrows=[]
for name,dsfile,clin in DIS:
    df=sample(dsfile,clin); groups={"whole":df,"male":df[df.sex.astype(str).str.startswith(("M","m"))],"female":df[df.sex.astype(str).str.startswith(("F","f"))]}
    MC=cols_of(df)
    # ---- regional (whole/M/F) ----
    for gname,g in groups.items():
        g=g.copy()
        for meas,cols in MC.items():
            d=g.dropna(subset=cols).copy(); d["age_z"]=(d.age-d.age.mean())/d.age.std(ddof=1)
            X=design(d,gname=="whole",d["cond"].values); jc=X.shape[1]-1; Y=d[cols].values.astype(float)
            t,dof=tmap(X,Y,jc); p=2*stats.t.sf(np.abs(t),dof); q=multipletests(p,method="fdr_bh")[1]; dc=t/np.sqrt(len(d))
            for i,c in enumerate(cols):
                he,reg=parse(c); cc.append(dict(disorder=name,group=gname,measure=meas,hemisphere=he,region=reg,n=len(d),
                    t=round(t[i],3),cohen_d=round(dc[i],3),p_unc=float(p[i]),q_fdr=float(q[i]),sig_fdr=int(q[i]<.05)))
    # ---- hemispheric ----
    HM={"Cortical thickness":("HEMI.left.cortical.thickness","HEMI.right.cortical.thickness"),
        "Grey-matter volume":("HEMI.left.greymatter.volume","HEMI.right.greymatter.volume"),
        "Surface area":("HEMI.left.surface.area","HEMI.right.surface.area"),
        "Subcortical volume":("HEMI.left.subcortex","HEMI.right.subcortex")}
    for gname,g in groups.items():
        recs=[]
        for mlab,(lc,rc) in HM.items():
            for side,col in [("Left",lc),("Right",rc)]:
                d=g.dropna(subset=[col]).copy(); d["age_z"]=(d.age-d.age.mean())/d.age.std(ddof=1)
                X=design(d,gname=="whole",d["cond"].values); jc=X.shape[1]-1; Y=d[col].values.astype(float)
                t,dof=tmap(X,Y[:,None],jc); t=float(t[0]); p=2*stats.t.sf(abs(t),dof); recs.append((mlab,side,t,p,t/np.sqrt(len(d))))
        q=multipletests([r[3] for r in recs],method="fdr_bh")[1]  # FDR across the 8 hemispheric tests (4 measures x L/R)
        for (mlab,side,t,p,dc),qq in zip(recs,q): hemi.append(dict(disorder=name,group=gname,measure=mlab,hemisphere=side,t=round(t,3),cohen_d=round(dc,3),p_unc=float(p),q_fdr=float(qq),sig_fdr=int(qq<.05)))
    # ---- region-specificity (region - hemisphere), whole only ----
    g=df.copy()
    for meas,cols in MC.items():
        if meas=="Subcortical": continue
        d=g.dropna(subset=cols).copy(); d["age_z"]=(d.age-d.age.mean())/d.age.std(ddof=1)
        X=design(d,True,d["cond"].values); jc=X.shape[1]-1
        Y=d[cols].values.astype(float)
        # raw
        t0,dof=tmap(X,Y,jc); q0=multipletests(2*stats.t.sf(np.abs(t0),dof),method="fdr_bh")[1]
        # region - hemisphere mean (per hemi)
        lh=[i for i,c in enumerate(cols) if c.startswith("lh.")]; rh=[i for i,c in enumerate(cols) if c.startswith("rh.")]
        Yc=Y.copy(); Yc[:,lh]=Y[:,lh]-Y[:,lh].mean(1,keepdims=True); Yc[:,rh]=Y[:,rh]-Y[:,rh].mean(1,keepdims=True)
        tc,_=tmap(X,Yc,jc); qc=multipletests(2*stats.t.sf(np.abs(tc),dof),method="fdr_bh")[1]
        for i,c in enumerate(cols):
            he,reg=parse(c); spec.append(dict(disorder=name,measure=meas,hemisphere=he,region=reg,
                t_uncorrected=round(t0[i],3),q_uncorrected=float(q0[i]),t_corrected=round(tc[i],3),q_corrected=float(qc[i]),
                sig_after_correction=int(qc[i]<.05),interpretation=("" if qc[i]>=.05 else ("regionally-specific reduction" if tc[i]<0 else "relatively spared"))))
    # ---- sex interaction (count) + sex-permutation ceiling ----
    for meas,cols in MC.items():
        d=df.dropna(subset=cols).copy(); d["age_z"]=(d.age-d.age.mean())/d.age.std(ddof=1)
        male=(d.sex.astype(str).str.startswith(("M","m"))).astype(float).values
        site=pd.get_dummies(d.study_site.astype(str),drop_first=True).astype(float)
        # interaction model: ... + cond + male + cond:male
        inter=d["cond"].values*male
        Xf=np.column_stack([np.ones(len(d)),d.age_z.values,male,site.values,d["cond"].values,inter]); jint=Xf.shape[1]-1
        Y=d[cols].values.astype(float); tI,dof=tmap(Xf,Y,jint); qI=multipletests(2*stats.t.sf(np.abs(tI),dof),method="fdr_bh")[1]
        n_int=int((qI<.05).sum())
        # sex-perm ceiling
        def dxmap(mask):
            g2=d[mask]; X=design(g2,False,g2["cond"].values); jc=X.shape[1]-1
            t,_=tmap(X,g2[cols].values.astype(float),jc); return t
        rmap_m=dxmap(male==1); rmap_f=dxmap(male==0); r_obs=float(stats.spearmanr(rmap_m,rmap_f)[0])
        nM=int((male==1).sum()); N=1000; null=np.empty(N); idx=np.arange(len(d))
        for k in range(N):
            pm=rng.permutation(idx); a,b=pm[:nM],pm[nM:]
            ta=dxmap(np.isin(idx,a)); tb=dxmap(np.isin(idx,b)); null[k]=stats.spearmanr(ta,tb)[0]
        p_below=float((1+np.sum(null<=r_obs))/(1+N))
        permrows.append(dict(disorder=name,measure=meas,n_sexXdx_FDR=n_int,r_male_vs_female=round(r_obs,3),
            ceiling_null_mean=round(float(null.mean()),3),p_below_ceiling_1sided=round(p_below,4)))
    print(name,"done")
pd.DataFrame(cc).to_csv(os.path.join(OUT,"_supp_cc.csv"),index=False)
pd.DataFrame(hemi).to_csv(os.path.join(OUT,"_supp_hemi.csv"),index=False)
pd.DataFrame(spec).to_csv(os.path.join(OUT,"_supp_spec.csv"),index=False)
pd.DataFrame(permrows).to_csv(os.path.join(OUT,"_supp_permcc.csv"),index=False)
# summary
h=pd.DataFrame(hemi); h=h[h.group=="whole"]
print("\n=== GLOBAL (whole) Cohen d, q ==="); print(h[["disorder","measure","cohen_d","q_fdr"]].round(3).to_string(index=False))
c=pd.DataFrame(cc); c=c[c.group=="whole"]
print("\n=== regional FDR-sig counts (whole) ===")
print(c.groupby(["disorder","measure"]).sig_fdr.sum().to_string())
print("\n=== sex interaction / ceiling ==="); print(pd.DataFrame(permrows)[["disorder","measure","n_sexXdx_FDR","r_male_vs_female","ceiling_null_mean","p_below_ceiling_1sided"]].to_string(index=False))

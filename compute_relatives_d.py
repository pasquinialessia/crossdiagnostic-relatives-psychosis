#!/usr/bin/env python3
"""Add per-region Cohen's d (= stored t / sqrt(n)) to the relatives maps, for the
d-scale version of Figure 3. Keeps the stored t (so recap r/q are unchanged); writes
maps_{dx}_{meas}_d.csv with d_relative_vs_CN and d_proband_vs_CN alongside the t columns.
n per region = number of non-missing subjects in each contrast sample."""
import os, numpy as np, pandas as pd
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES=os.path.join(REPO,"results/relatives")
df=pd.read_csv(os.path.join(REPO,"data","bsnip_relatives_dataset.csv"))
REL={"SCZ":["SCZ-Relative"],"SAD":["Schizoaffective Disorder-Relative"],"BD":["BD-Relative","Bipolar/Psych-Relative"]}
# proband (clinical) group by dx_group
PROB={"SCZ":"SCZ","SAD":"SAD","BD":"BD"}
MEAS={"CT":("lh.CT.","rh.CT."),"GMV":("lh.GM.","rh.GM."),"SA":("lh.SA.","rh.SA."),"Subcortical":("SubcGM.",)}
def region_key(col):
    # match maps_*.csv 'region' column formatting
    if col.startswith("SubcGM."): return col
    return col
def contrast_sample(labels, role_sites_from):
    sites=set(df[df.dx.isin(labels)].study_site)
    sub=df[(df.study_site.isin(sites))&(df.dx.isin(["CN"]+labels))]
    return sub
for dx in REL:
    rel_sub=df[df.study_site.isin(set(df[df.dx.isin(REL[dx])].study_site)) & df.dx.isin(["CN"]+REL[dx])]
    prob_labels=[l for l in df.dx.unique() if isinstance(l,str) and df[df.dx==l].dx_group.iloc[0]==PROB[dx] and "Relative" not in l and l!="CN"]
    prob_sub=df[df.study_site.isin(set(df[df.dx.isin(prob_labels)].study_site)) & df.dx.isin(["CN"]+prob_labels)]
    for meas,prefs in MEAS.items():
        f=os.path.join(RES,f"maps_{dx}_{meas}.csv")
        if not os.path.exists(f): continue
        m=pd.read_csv(f)
        cols=[c for c in df.columns if c.startswith(prefs) and ("Vent" not in c and "Cereb" not in c if meas=="Subcortical" else True)]
        # map region label in maps csv -> dataframe column
        def col_for(regname):
            for c in cols:
                if c.split(".",2)[-1]==regname or c.endswith(regname): return c
            return None
        nrel=[]; nprob=[]
        for reg in m.region:
            c=col_for(reg)
            nrel.append(rel_sub[c].notna().sum() if c else np.nan)
            nprob.append(prob_sub[c].notna().sum() if c else np.nan)
        m["d_relative_vs_CN"]=m["t_relative_vs_CN"]/np.sqrt(nrel)
        m["d_proband_vs_CN"]=m["t_proband_vs_CN"]/np.sqrt(nprob)
        m.to_csv(os.path.join(RES,f"maps_{dx}_{meas}_d.csv"),index=False)
    print(dx,"done: rel n≈",len(rel_sub),"prob n≈",len(prob_sub))
print("wrote maps_*_d.csv")

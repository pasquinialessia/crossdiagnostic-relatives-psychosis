#!/usr/bin/env python3
import os, pandas as pd
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(REPO,"results","supp_tables")
L=lambda n: pd.read_csv(os.path.join(OUT,f"_supp_{n}.csv"))
cc,hemi,spec,rel,cdx,cds,permcc,cdsx=[L(n) for n in ["cc","hemi","spec","rel","cdx","cds","permcc","cdsx"]]
xlsx=os.path.join(REPO,"reports","Supplementary_Tables.xlsx")
readme=pd.DataFrame({"Sheet":["S1_CaseControl_regional","S2_Hemispheric","S3_RegionalSpecificity","S4_Relatives_regional","S5_CrossDiagnostic","S6_CrossDisorder_familial","S7_Sex_perDiagnosis","S8_CrossDiag_sexpermutation"],
 "Contents":["Case-control per-region t, Cohen's d, p and FDR q for schizophrenia, schizoaffective disorder and bipolar disorder (each vs CN), whole sample and by sex, all measures.",
 "Case-control results for the whole-hemisphere measures, with left and right hemispheres analysed separately (not averaged), per diagnosis and group.",
 "Regional specificity: per-region case-control t and FDR q before/after region-minus-hemisphere correction (t<0 = specific reduction, t>0 = relatively spared).",
 "Relatives per-region liability-gradient F/q, clinical- and relative-vs-control t, clinical-relative contrast, monotonicity.",
 "Cross-diagnostic similarity of case-control maps: Spearman r with spin/shuffle p and FDR q, by measure and sex.",
 "Cross-disorder familial sharing: relative-of-X vs clinical-of-Y map correlation with spin/shuffle p and FDR q.",
 "Sex analyses per diagnosis: diagnosis-by-sex interaction count, male-vs-female map correlation vs reliability ceiling, one-sided p.",
 "Cross-diagnostic sex-permutation: cross-disorder similarity by sex, female-minus-male difference and its permutation p."]})
with pd.ExcelWriter(xlsx,engine="openpyxl") as xw:
    readme.to_excel(xw,sheet_name="README",index=False)
    cc.to_excel(xw,sheet_name="S1_CaseControl_regional",index=False)
    hemi.to_excel(xw,sheet_name="S2_Hemispheric",index=False)
    spec.to_excel(xw,sheet_name="S3_RegionalSpecificity",index=False)
    rel.to_excel(xw,sheet_name="S4_Relatives_regional",index=False)
    cdx.to_excel(xw,sheet_name="S5_CrossDiagnostic",index=False)
    cds.to_excel(xw,sheet_name="S6_CrossDisorder_familial",index=False)
    permcc.to_excel(xw,sheet_name="S7_Sex_perDiagnosis",index=False)
    cdsx.to_excel(xw,sheet_name="S8_CrossDiag_sexpermutation",index=False)
print("wrote",xlsx,"| cc disorders:",sorted(cc.disorder.unique()))

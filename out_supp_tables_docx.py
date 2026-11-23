#!/usr/bin/env python3
"""Word supplementary tables (full per-region + hemispheric + relatives + cross-dx)."""
import os, pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(REPO,"results","supp_tables")
def L(n): return pd.read_csv(os.path.join(OUT,f"_supp_{n}.csv"))
cc,hemi,spec,rel,cdx,cds=L("cc"),L("hemi"),L("spec"),L("rel"),L("cdx"),L("cds")
permcc,cdsx=L("permcc"),L("cdsx")
def fp(v):
    try: v=float(v)
    except Exception: return ""
    if pd.isna(v): return ""
    return "<.0001" if v<1e-4 else f"{v:.4f}"
for df,colz in [(cc,["p_unc","q_fdr"]),(hemi,["p_unc","q_fdr"]),
                (spec,["q_uncorrected","q_corrected"]),
                (rel,["p_group","q_group_fdr","q_clin_minus_rel_fdr"]),
                (cdx,["p_spin_shuf","q_fdr"]),(cds,["p_spin_shuf","q_fdr"]),
                (permcc,["p_below_ceiling_1sided"]),(cdsx,["p_female_vs_male_1sided"])]:
    for c in colz:
        if c in df.columns: df[c]=df[c].map(fp)

doc=Document()
for s in doc.styles:
    if s.name=="Normal": s.font.size=Pt(10); s.font.name="Calibri"

def h1(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(14); r.font.color.rgb=RGBColor(0x1F,0x4E,0x79)
def cap(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(9)
def note(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.italic=True; r.font.size=Pt(8)

def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:val'),'clear'); sh.set(qn('w:fill'),fill); tcPr.append(sh)

def table(df, cols, headers):
    t=doc.add_table(rows=1, cols=len(cols)); t.style="Table Grid"
    hdr=t.rows[0].cells
    for i,hname in enumerate(headers):
        hdr[i].text=""; run=hdr[i].paragraphs[0].add_run(hname); run.bold=True; run.font.size=Pt(7.5); shade(hdr[i],"D9D9D9")
    # bulk add
    for _,row in df.iterrows():
        cells=t.add_row().cells
        for i,c in enumerate(cols):
            v=row[c]; cells[i].text=""; run=cells[i].paragraphs[0].add_run("" if pd.isna(v) else str(v)); run.font.size=Pt(7.5)
    return t

# ===== S1 case-control regional =====
h1("Supplementary Table S1. Case–control differences by region")
cap("Per-region t, Cohen’s d, uncorrected p and FDR-corrected q for schizophrenia, schizoaffective disorder and bipolar disorder vs. controls, in the whole sample and separately in men and women, for cortical thickness, grey-matter volume, surface area (68 regions each) and subcortical volume (18 structures).")
cc2=cc.copy()
cc2["group"]=cc2["group"].map({"whole":"Whole","male":"Male","female":"Female"})
table(cc2, ["disorder","group","measure","hemisphere","region","n","t","cohen_d","p_unc","q_fdr","sig_fdr"],
      ["Disorder","Group","Measure","Hemi","Region","n","t","d","p","q","sig"])
note("sig = 1 where q < 0.05 (Benjamini–Hochberg, applied within each measure and group).")
doc.add_page_break()

# ===== S2 hemispheric =====
h1("Supplementary Table S2. Case–control differences in hemispheric (whole-hemisphere) measures")
cap("Case–control results for the whole-hemisphere measures, with the left and right hemispheres analysed separately (not averaged), by disorder and group.")
hemi2=hemi.copy(); hemi2["group"]=hemi2["group"].map({"whole":"Whole","male":"Male","female":"Female"})
table(hemi2, ["disorder","group","measure","hemisphere","t","cohen_d","p_unc","q_fdr","sig_fdr"],
      ["Disorder","Group","Measure","Hemisphere","t","d","p","q","sig"])
note("Left and right hemispheres are analysed separately. q = FDR across the eight hemispheric tests (four measures × two hemispheres) within each disorder and group.")
doc.add_page_break()

# ===== S3 regional specificity =====
h1("Supplementary Table S3. Regional specificity (region − hemispheric mean)")
cap("Per-region case–control t and FDR q before and after re-expressing each region as its deviation from the hemispheric mean. Regions with q(corrected) < 0.05 are regionally specific: a negative corrected t denotes a reduction beyond the global shift, a positive corrected t a region relatively spared.")
table(spec, ["disorder","measure","hemisphere","region","t_uncorrected","q_uncorrected","t_corrected","q_corrected","sig_after_correction","interpretation"],
      ["Disorder","Measure","Hemi","Region","t uncorr","q uncorr","t corr","q corr","sig","Interpretation"])
note("sig = 1 where q(corrected) < 0.05.")
doc.add_page_break()

# ===== S4 relatives regional =====
h1("Supplementary Table S4. Relatives: per-region liability-gradient results")
cap("For each diagnosis and measure: omnibus liability-gradient F and FDR q (control/relative/clinical factor); clinical- and relative-vs-control t; the clinical−relative contrast and its FDR q; and whether the covariate-adjusted means were strictly monotonic (control<relative<clinical).")
table(rel, ["contrast","measure","hemisphere","region","F_group","q_group_fdr","sig_gradient","t_clinical_vs_CN","t_relative_vs_CN","t_clinical_minus_relative","q_clin_minus_rel_fdr","monotonic"],
      ["Dx","Measure","Hemi","Region","F","q(F)","sig","t clin","t rel","t clin−rel","q(clin−rel)","mono"])
doc.add_page_break()

# ===== S5 cross-diagnostic =====
h1("Supplementary Table S5. Cross-diagnostic similarity of case–control maps")
cap("Spearman correlation between each disorder pair’s case-versus-control effect map, per measure and group, with spin (cortical) / shuffle (subcortical) p and FDR q.")
table(cdx, ["modality","measure","sex","disorder_1","disorder_2","spearman_r","p_spin_shuf","q_fdr"],
      ["Modality","Measure","Group","Dx 1","Dx 2","rho","p","q"])
doc.add_page_break()

# ===== S6 cross-disorder familial =====
h1("Supplementary Table S6. Cross-disorder familial sharing")
cap("Correlation between the relative-versus-control map of one disorder and the clinical-versus-control map of another, per measure, with spin/shuffle p and FDR q.")
table(cds, ["measure","relative_of","clinical_of","r","p_spin_shuf","q_fdr"],
      ["Measure","Relative of","Clinical of","r","p","q"])
doc.add_page_break()

# ===== S7 sex-permutation reliability ceiling =====
h1("Supplementary Table S7. Sex-permutation reliability ceiling")
cap("Sex analyses per diagnosis (schizophrenia, schizoaffective disorder, bipolar disorder). n_sexXdx_FDR: number of regions with a significant diagnosis-by-sex interaction after FDR. r (M vs F): observed correlation between the male and female effect-size maps; Ceiling: concordance expected from random same-size splits; p (below): one-sided lower-tail permutation p testing whether the observed correlation falls below the ceiling (male–female divergence).")
table(permcc, ["disorder","measure","n_sexXdx_FDR","r_male_vs_female","ceiling_null_mean","p_below_ceiling_1sided"],
      ["Disorder","Measure","sex×dx FDR","r (M vs F)","Ceiling","p (below)"])
note("A non-significant interaction (0 regions) together with r not below the ceiling indicates a sex-common signature. All permutations n = 1,000.")
doc.add_page_break()

# ===== S8 cross-diagnostic sex-permutation =====
h1("Supplementary Table S8. Cross-diagnostic sex-permutation")
cap("Cross-disorder map similarity computed separately in men and women, the female−male difference, and a permutation test (against random sex relabelling) of whether that difference exceeds chance.")
table(cdsx, ["measure","disorder_1","disorder_2","r_male","r_female","female_minus_male_diff","perm_ceiling_mean","p_female_vs_male_1sided"],
      ["Measure","Dx 1","Dx 2","r male","r female","F−M diff","Perm mean","p (F−M)"])
note("No female−male difference in cross-disorder sharing survived permutation testing (all p > .38).")

out=os.path.join(REPO,"reports","Supplementary_Tables.docx")
doc.save(out); print("saved",out)

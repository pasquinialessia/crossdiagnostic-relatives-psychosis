#!/usr/bin/env python3
"""Single combined Supplementary Material: Supplementary Figures S1-S5 + Supplementary
Tables S1-S10 (datasets, demographics, and all statistical tables), one coherent numbering."""
import os, pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(REPO,"results","supp_tables"); FIG=os.path.join(REPO,"figures")
def L(n): return pd.read_csv(os.path.join(OUT,f"_supp_{n}.csv"))
cc,hemi,spec,rel,cdx,cds,permcc,cdsx=[L(n) for n in ["cc","hemi","spec","rel","cdx","cds","permcc","cdsx"]]
sexspin=L("sexmap_spin")

def fp(v):
    try: v=float(v)
    except Exception: return ""
    if pd.isna(v): return ""
    return "<.0001" if v<1e-4 else f"{v:.4f}"

doc=Document()
for st in doc.styles:
    if st.name=="Normal": st.font.size=Pt(10); st.font.name="Calibri"
def H1(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(15); r.font.color.rgb=RGBColor(0x1F,0x4E,0x79)
def H2(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(12.5); r.font.color.rgb=RGBColor(0x2E,0x74,0xB5)
def cap(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(9)
def note(t):
    p=doc.add_paragraph(); r=p.add_run(t); r.italic=True; r.font.size=Pt(8)
def shade(c,f):
    tcPr=c._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:val'),'clear'); sh.set(qn('w:fill'),f); tcPr.append(sh)
def table(df, cols, headers, fmt=None, size=7.5):
    if fmt:
        df=df.copy()
        for c,fn in fmt.items():
            if c in df.columns: df[c]=df[c].map(fn)
    t=doc.add_table(rows=1, cols=len(cols)); t.style="Table Grid"
    for i,h in enumerate(headers):
        cell=t.rows[0].cells[i]; cell.text=""; run=cell.paragraphs[0].add_run(h); run.bold=True; run.font.size=Pt(size); shade(cell,"D9D9D9")
    for _,row in df.iterrows():
        cs=t.add_row().cells
        for i,c in enumerate(cols):
            v=row[c]; cs[i].text=""; run=cs[i].paragraphs[0].add_run("" if pd.isna(v) else str(v)); run.font.size=Pt(size)
    return t
def figure(png, width_in, label, legend):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(FIG,png), width=Inches(width_in))
    q=doc.add_paragraph(); r=q.add_run(label); r.bold=True; r.font.size=Pt(9); r2=q.add_run(legend); r2.font.size=Pt(9)

doc.add_heading("Supplementary Material",0)

# =========================== SUPPLEMENTARY FIGURES ===========================
H1("Supplementary Figures")
figure("figureS1_hemispheric.png", 6.5, "Figure S1. ",
  "Hemispheric centile scores. Left- and right-hemisphere aggregate centile scores per diagnostic group (SCZ, SAD, BD) and measure (CT, GMV, SA, subcortical), showing the bilateral extent of the case-control effect.")
doc.add_page_break()
figure("figureS2_sex_correlations.png", 6.2, "Figure S2. ",
  "Sex-stratified case-control brain maps and permutation. Panel A: scatter plots of male vs female Cohen's d per region, per diagnosis and measure, with observed map correlations (rho) and sex-label permutation p-values. Panel B: null distributions from 1,000 sex-label shuffles with the observed residual sum of squares marked.")
doc.add_page_break()
figure("figureS3_swa_sensitivity.png", 6.5, "Figure S3. ",
  "Peak-effect brain maps from the sliding-window age analysis. Brain surface maps coloured by the age of the case-control peak effect (mean age of the highest-Cohen's-d window), per disorder (SCZ, SAD, BD) and measure (CT, GMV, SA, subcortical); grey = insufficient subjects in that window.")
doc.add_page_break()
figure("figureS4_region_specificity.png", 6.2, "Figure S4. ",
  "Regional specificity (sensitivity analysis). Cortical regions remaining FDR-significant after re-expressing each region as its deviation from the hemispheric mean, per diagnosis (rows) and measure (columns); significant regions are coloured by their corrected case-versus-control t (blue = specific reduction, red = specific relative increase), the rest grey.")
doc.add_page_break()
figure("figureS5_relatives_unthresholded.png", 5.6, "Figure S5. ",
  "Unthresholded relative-versus-control Cohen's d maps (complementary to Figure 4A). Every region is coloured by its relative-versus-control Cohen's d with no significance threshold, per diagnosis (rows) and measure (columns; subcortical beneath the GMV column); blue denotes lower centiles than controls. Generalised, sub-threshold reductions in CT, GMV, and SA are visible across all relative groups despite only SCZ relatives reaching FDR significance in SA.")
doc.add_page_break()

# =========================== SUPPLEMENTARY TABLES ============================
H1("Supplementary Tables")

# ---- S1 datasets ----
cap("Supplementary Table S1. Datasets contributing to the final case-control sample (after quality control and the 18-50 age restriction).")
S1=pd.DataFrame([
 ["BSNIP",5,150,182,59,94,485,"50.5","33.0 ± 9.5"],
 ["Iowa",1,92,92,7,0,191,"30.4","29.7 ± 8.1"],
 ["PARDIP",3,17,0,0,46,63,"61.9","20.6 ± 1.6"],
 ["UCLA-CNP",2,70,47,0,49,166,"38.6","33.8 ± 9.3"],
 ["MCIC",3,71,94,0,0,165,"75.8","32.5 ± 10.3"],
 ["Total",14,400,415,66,189,1070,"49.6","31.8 ± 9.6"]],
 columns=["Dataset","Sites","CN","SCZ","SAD","BD","Total","%F","Age (M±SD)"])
table(S1,list(S1.columns),list(S1.columns),size=8)
note("CN, control; SCZ, schizophrenia; SAD, schizoaffective disorder; BD, bipolar disorder. SAD is drawn from four sites (BSNIP, Iowa) and should be read as lower-reliability.")
doc.add_paragraph()

# ---- S2 demographics by group ----
cap("Supplementary Table S2. Demographics of the final case-control sample, by diagnostic group.")
S2=pd.DataFrame([
 ["Controls",400,"207 (51.8)","31.3 ± 9.4"],
 ["Schizophrenia (SCZ)",415,"176 (42.4)","32.4 ± 9.6"],
 ["Schizoaffective (SAD)",66,"36 (54.5)","32.5 ± 8.8"],
 ["Bipolar (BD)",189,"112 (59.3)","31.0 ± 10.2"]],
 columns=["Group","n","Female n (%)","Age (M ± SD)"])
table(S2,list(S2.columns),list(S2.columns),size=8)
note("Unaffected first-degree relatives (n=410: 222 SCZ, 74 SAD, 114 BD/psychosis-spectrum) were analysed in the BSNIP relatives design.")
doc.add_page_break()

# ---- S3 case-control regional ----
cap("Supplementary Table S3. Case-control differences by region. Per-region t, Cohen's d, uncorrected p and FDR q for SCZ, SAD and BD vs controls, in the whole sample and separately by sex, all measures.")
cc2=cc.copy(); cc2["group"]=cc2["group"].map({"whole":"Whole","male":"Male","female":"Female"})
table(cc2,["disorder","group","measure","hemisphere","region","n","t","cohen_d","p_unc","q_fdr","sig_fdr"],
      ["Dx","Group","Measure","Hemi","Region","n","t","d","p","q","sig"],
      fmt={"p_unc":fp,"q_fdr":fp})
doc.add_page_break()

# ---- S4 hemispheric ----
cap("Supplementary Table S4. Case-control differences in whole-hemisphere measures, left and right hemispheres analysed separately.")
hemi2=hemi.copy(); hemi2["group"]=hemi2["group"].map({"whole":"Whole","male":"Male","female":"Female"})
table(hemi2,["disorder","group","measure","hemisphere","t","cohen_d","p_unc","q_fdr","sig_fdr"],
      ["Dx","Group","Measure","Hemisphere","t","d","p","q","sig"],fmt={"p_unc":fp,"q_fdr":fp})
note("q = FDR across the eight hemispheric tests (four measures x two hemispheres) within each disorder and group.")
doc.add_page_break()

# ---- S5 region specificity ----
cap("Supplementary Table S5. Regional specificity (region minus hemispheric mean). Per-region t and FDR q before and after correction; a negative corrected t denotes a specific reduction.")
table(spec,["disorder","measure","hemisphere","region","t_uncorrected","q_uncorrected","t_corrected","q_corrected","sig_after_correction","interpretation"],
      ["Dx","Measure","Hemi","Region","t uncorr","q uncorr","t corr","q corr","sig","Interpretation"],
      fmt={"q_uncorrected":fp,"q_corrected":fp})
doc.add_page_break()

# ---- S6 relatives gradient ----
cap("Supplementary Table S6. Relatives: per-region liability-gradient results. Omnibus group F and FDR q, clinical- and relative-vs-control t, clinical-minus-relative contrast and its FDR q, and strict monotonicity (control<relative<clinical).")
table(rel,["contrast","measure","hemisphere","region","F_group","q_group_fdr","sig_gradient","t_clinical_vs_CN","t_relative_vs_CN","t_clinical_minus_relative","q_clin_minus_rel_fdr","monotonic"],
      ["Dx","Measure","Hemi","Region","F","q(F)","sig","t clin","t rel","t clin-rel","q(clin-rel)","mono"],
      fmt={"q_group_fdr":fp,"q_clin_minus_rel_fdr":fp})
doc.add_page_break()

# ---- S7 cross-diagnostic ----
cap("Supplementary Table S7. Cross-diagnostic similarity of case-control maps. Spearman r with spin/shuffle p and FDR q, by measure and group.")
table(cdx,["modality","measure","sex","disorder_1","disorder_2","spearman_r","p_spin_shuf","q_fdr"],
      ["Modality","Measure","Group","Dx 1","Dx 2","rho","p","q"],fmt={"p_spin_shuf":fp,"q_fdr":fp})
doc.add_page_break()

# ---- S8 cross-disorder familial ----
cap("Supplementary Table S8. Cross-disorder familial sharing. Correlation between the relative-vs-control map of one disorder and the clinical-vs-control map of another, with spin/shuffle p and FDR q.")
table(cds,["measure","relative_of","clinical_of","r","p_spin_shuf","q_fdr"],
      ["Measure","Relative of","Clinical of","r","p","q"],fmt={"p_spin_shuf":fp,"q_fdr":fp})
doc.add_page_break()

# ---- S9 sex analyses (interaction + ceiling + spin) ----
cap("Supplementary Table S9. Sex analyses per diagnosis. Diagnosis-by-sex interaction count; male-vs-female map correlation (rho) with its spin/shuffle significance (p-spin, FDR q); and the reliability-ceiling permutation testing whether the observed rho falls below random same-size splits (one-sided p).")
sx=permcc.merge(sexspin[["disorder","measure","rho","p_spin","q_fdr"]],on=["disorder","measure"],how="left")
sx=sx.rename(columns={"q_fdr":"q_spin"})
table(sx,["disorder","measure","n_sexXdx_FDR","rho","p_spin","q_spin","ceiling_null_mean","p_below_ceiling_1sided"],
      ["Dx","Measure","sex x dx FDR","rho (M vs F)","p-spin","q-spin","Ceiling","p (below)"],
      fmt={"p_spin":fp,"q_spin":fp,"p_below_ceiling_1sided":fp})
note("All permutations n=1,000. A non-significant interaction with a significantly positive rho (p-spin) and rho not below the ceiling indicates a sex-common signature.")
doc.add_page_break()

# ---- S10 cross-diagnostic sex-permutation ----
cap("Supplementary Table S10. Cross-diagnostic sex-permutation. Cross-disorder similarity in men and women, female-minus-male difference and its permutation p.")
table(cdsx,["measure","disorder_1","disorder_2","r_male","r_female","female_minus_male_diff","perm_ceiling_mean","p_female_vs_male_1sided"],
      ["Measure","Dx 1","Dx 2","r male","r female","F-M diff","Perm mean","p (F-M)"],fmt={"p_female_vs_male_1sided":fp})
note("No female-minus-male difference in cross-disorder sharing survived permutation testing.")

out=os.path.join(REPO,"reports","Supplementary_Material.docx")
doc.save(out); print("saved",out)

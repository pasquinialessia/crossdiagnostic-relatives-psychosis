# Code

All analyses are plain Python (`.py`) scripts with Spyder `# %%` cells in this
single folder. **Set `code/` as the working directory** in Spyder (or `cd code`)
— every script reads inputs from `../data/` and writes outputs to
`../results/<topic>/` (figures to `../figures/`).

**Every modelling script writes its results to CSV** — per-region statistics
(coefficient/t or F, p, FDR) and a summary table — under `../results/<topic>/`.
Figure/table generation for the manuscript lives in the `fig*`/`out_*` scripts
(see the end of this file).

Helper modules: `swa_lib.py`, `data_utils.py`,
`compute_relatives_d.py`, `generate_separate_nulls.py`.

## 01 · Demographics

- `demographics_table.py` — per-group N / sex / age / sites + case-vs-control age
  (Welch t) and sex (χ²) tests → `../results/demographics/`.

## 02 · Case–control by diagnosis

Schizophrenia (SCZ), schizoaffective disorder (SAD), and bipolar disorder (BD)
are analysed separately against controls. Each script selects controls + one
diagnosis 

- `cohens_d_maps.py` — whole-group and sex-stratified Cohen's d surface maps +
  `cohens_d_summary.csv` (per-region d/t/p/FDR for all groups), per diagnosis →
  `../results/casecontrol/{scz,sad}_surface_plots/`.
- `bd_casecontrol.py` — Cohen's d maps + male-vs-female spin/shuffle for BD →
  `../results/casecontrol/`.
- `casecontrol_stats.py` — canonical case-control statistics behind Table 1 and
  Supplementary Tables S1–S3, S7: for SCZ, SAD and BD each vs CN — regional
  (whole/M/F), hemispheric, region-specificity, diagnosis-by-sex interaction count
  → `../results/casecontrol/tables/`.

## 03 · Sex-stratified analysis & permutation

- `sex_maps.py` — male-vs-female Cohen's d maps with cortical spin + subcortical
  shuffle permutation null, run per diagnosis →
  `../results/casecontrol/{scz,sad}_lm_by_sex/`.
- `sex_permutation.py` — male-vs-female reliability ceiling (1,000 sex perms,
  all four measures). **p is one-sided lower-tail** (`P(null ≤ r_obs)` = divergence
  below the ceiling); column `p_below_ceiling`.
- `sex_by_diagnosis_interaction.py` — sex×diagnosis interaction (SCZ, SAD & BD model forms).
- `region_specificity_qc.py` — `y = region − hemi` outcome, whole-group model with
  sex×dx and age×dx, both matched samples; per-region + summary CSVs.
  → all under `../results/qc_sensitivity/`.

## 04 · Sliding-window peak-age (SWA)

Sensitivity analysis of the age of peak case–control effect (each clinical group
vs. controls).

- `swa_lib.py` — shared analysis library (imported by `peak_age_ssd`,
  `peak_age_bd`, and figure scripts).
- `peak_age_ssd.py` — SCZ + SAD sliding-window peak-age analysis →
  `../results/swa_peak_age/`.
- `peak_age_bd.py` — BD sliding-window peak-age → `../results/swa_peak_age/bd/`.


## 05 · Cross-diagnostic

- `crossdx_maps.py` — per-region Clinical-vs-CN + bootstrap →
  `../results/crossdiagnostic/`.
- `crossdx_spatial_tests.py` — disorder-pair map correlations, spin/shuffle +
  global FDR.


## 06 · Familial analysis (BSNIP)

- `relatives_liability.py` — relative-vs-CN & clinical-vs-CN maps, recapitulation
  correlation (label-permutation null) 
- `case_vs_relative.py` — **clinical-vs-relative contrast**: OLS model per region
  with Clinical as reference category. Writes `_supp_case_vs_relative.csv` and per-disorder Cohen's d
  `.npy` arrays → `../results/relatives/case_vs_relative/`.

## 07 · Cross-diagnostic familial matrix

- `crossdx_familial_fdr.py` — ENIGMA spin (cortical) / shuffle (subcortical)
  spatial null + FDR for the cross-disorder familial-sharing matrix
  (self-contained; reads the toolbox's fsa5 `.gii` + aparc `.annot` without
  importing the VTK parts). Off-diagonal only for the rel-vs-rel family →
  `../results/relatives_reviewer/`.

---

## Manuscript figures & supplementary (final paper)

These scripts build the figures and supplementary material as they appear in
the manuscript. Brain-surface renders (`fig_regional_brainmaps`,
`fig_swa_hemispheric`, `fig_relatives_brainmaps`, `figS_hemispheric`,
`figS_relatives_unthresholded`, `figS_case_vs_relative_brainmaps`) require the
venv (ENIGMA/VTK). The matplotlib builders (`fig_crossdx_matrix`,
`fig_familial_matrix`, `figS_sex_correlations`, `figS_sensitivity_analysis`) run
anywhere.

**Note:** Two analysis scripts also require enigmatoolbox: `crossdx_maps.py`
(VTK surface plots) and `crossdx_spatial_tests.py` (spin/shuffle permutation
tests). Both give a clear message if the venv is not active. Run them from the
venv to get full output. `crossdx_familial_fdr.py` also needs enigmatoolbox's
data files (fsa5 sphere + aparc annot) and must be run from the venv. Tidy per-region stat tables are read from `../results/` (written by
the modelling scripts above).

- `fig_regional_brainmaps.py` — Figure 1: case-vs-control Cohen's d maps,
  SCZ/SAD/BD, whole/M/F + subcortical.
- `fig_swa_hemispheric.py` — Figure 2: brain surface maps coloured by the mean
  age of the peak-effect window (viridis, 18–50 yr), per disorder (SCZ/SAD/BD) ×
  measure (CT/GMV/SA/Subcortical).
- `fig_crossdx_matrix.py` — Figure 3: cross-diagnostic similarity matrix (and
  the sex-resolved version).
- `fig_relatives_brainmaps.py` — Figure 4: two-panel composed figure.
  Panel A: proband-vs-relative Cohen's d maps, FDR-thresholded (requires
  `figS_case_vs_relative_brainmaps.py` to be run first).
  Panel B: clinical-vs-relative scatter (read from `fig_relatives_scatter.py`
  output).
- `fig_relatives_scatter.py` — generates the scatter panel (Figure 4B), read
  by `fig_relatives_brainmaps.py`.
- `compute_relatives_d.py` — helper called by `fig_relatives_brainmaps` /
  `fig_relatives_scatter` to build the relatives Cohen's d arrays.
- `fig_familial_matrix.py` — Figure 5: cross-disorder familial overlap matrix.
- `figS_hemispheric.py` — **Figure S1**: hemispheric (L/R average) case-control
  centile scores, per disorder and measure → `figureS1_hemispheric.png`.
- `figS_sex_correlations.py` — **Figure S2**: sex-stratified brain maps and
  permutation null distributions (Panel A: male-vs-female Cohen's d scatter;
  Panel B: sex-label shuffle null) → `figureS2_sex_correlations.png`.
- `figS_swa_regional.py` — **Figure S3**: peak-effect brain maps (Cohen's d at
  peak window) from the SWA → `figureS3_swa_sensitivity.png`.
- `figS_sensitivity_analysis.py` — **Figure S4**: region-specificity
  (region-minus-hemisphere) sensitivity analysis →
  `figureS4_region_specificity.png`.
- `figS_relatives_unthresholded.py` — **Figure S5** (merged): Panel A =
  unthresholded relative-vs-control maps; Panel B = FDR-thresholded maps (from
  Figure 4A; requires `fig_relatives_brainmaps.py` to be run first) →
  `figureS5_relatives_unthresholded.png`.
- `figS_case_vs_relative_brainmaps.py` — generates the case-vs-relative brain
  maps (used as Figure 4A input); saves `figureS6_case_vs_relative.png` for use
  by `fig_relatives_brainmaps.py`.
- `tableS_sex_map_spin.py` — spin/shuffle significance of the male-vs-female
  maps (Table S9).
- `out_supp_tables_docx.py` — statistical supplementary tables → `.docx`.
- `out_tables_xlsx.py` — supplementary tables → `.xlsx`.
- `out_supplementary.py` — assembles the single **Supplementary Material**
  (Figures S1–S6 + Tables S1–S10) → `../reports/Supplementary_Material.docx`.


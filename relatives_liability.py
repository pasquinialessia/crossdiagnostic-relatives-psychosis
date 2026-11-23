#!/usr/bin/env python3
"""
Within-BSNIP relatives: familial-liability / endophenotype analysis (reviewer/extension)
========================================================================================
Question: do unaffected first-degree relatives carry an ATTENUATED version of the
proband brain signature? If the relative-vs-control map recapitulates the
proband-vs-control map, the signature reflects familial liability, not medication
or chronicity (relatives are unmedicated and unaffected).

Design: NDAR_BSNIP only (proband + relative + control at the same 6 sites).
Sample: bsnip_relatives_dataset.csv (loaded via load_bsnip_relatives() from data_utils).
No control resampling — site is a covariate (controls present at every site);
age and sex are covariates (relatives are older / more female).

Per diagnosis (SCZ, SAD, BD) x measure:
  * relative-vs-CN and proband-vs-CN t-maps (one shared model, CN reference):
        region ~ age_z + C(sex) + C(study_site) + C(role, ref=Control)
  * recapitulation r = corr(relative_map, proband_map) across regions;
  * attenuation slope beta (relative_map ~ proband_map): 0<beta<1 = same pattern, smaller;
  * sign concordance on proband-affected regions (|t_prob|>2);
  * label-permutation null: shuffle Relative/Control labels (proband map fixed),
    recompute the relative map, recorrelate -> p for the recapitulation;
  * n relative regions surviving FDR (relative-vs-CN);
  * cross-check: corr of the BSNIP relative map with the MAIN-pipeline disease map
    (SCZ and SAD from dataset.csv (via load_scz_dataset); BD from dataset.csv (via load_bd_matched)).

NOTE: p_recap here is a subject-relabeling null (accounts for finite-sample noise
in the relative map). A spatial-autocorrelation (spin/shuffle) null is complementary
and needs the ENIGMA toolbox — run in the venv if a spatial p is also wanted.

Outputs -> ../results/relatives/
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "08_relatives")
os.makedirs(OUT, exist_ok=True)

SEED, N_PERM = 42, 1000
rng = np.random.default_rng(SEED)

# diagnosis -> (proband dx_group, relative dx_group, main-disease file/label)
CONTRASTS = {
    "SCZ":     ("SCZ", "SCZ",     "_scz"),
    "SAD":     ("SAD", "SAD",     "_scz"),          # SAD analysed separately against controls
    "BD":      ("BD",  "BD",      "_bd"),
}

def cortical(df, pref):
    return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]
def subc(df):
    return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]
def measures_of(df):
    return [("CT", cortical(df, "CT")), ("GMV", cortical(df, "GM")),
            ("SA", cortical(df, "SA")), ("Subcortical", subc(df))]

def design(sub, extra_cols):
    """Intercept + age_z + sex + site dummies + the extra 0/1 columns (group dummies).
    Returns X and the indices of the extra columns."""
    site = pd.get_dummies(sub["study_site"].astype(str), drop_first=True).astype(float)
    sex  = (sub["sex"].astype(str).str.startswith(("M", "m"))).astype(float).values
    base = [np.ones(len(sub)), sub["age_z"].values, sex, site.values]
    X = np.column_stack(base + [c for c in extra_cols])
    start = 3 + site.shape[1]
    idx = list(range(start, start + len(extra_cols)))
    return X, idx

def tvals(X, Y, cols):
    XtXi = np.linalg.pinv(X.T @ X)
    B = XtXi @ (X.T @ Y)
    resid = Y - X @ B
    ssr = (resid ** 2).sum(0)
    dof = max(X.shape[0] - X.shape[1], 1)
    out = {}
    for j in cols:
        se = np.sqrt(XtXi[j, j] * ssr / dof)         # proper OLS SE
        out[j] = B[j] / se
    return out, dof

def fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    ranked = np.empty(n); c = p[order] * n / (np.arange(n) + 1)
    ranked[order] = np.minimum.accumulate(c[::-1])[::-1]
    return np.clip(ranked, 0, 1)

from scipy import stats
from data_utils import load_bd_matched, load_bsnip_relatives, load_scz_dataset
def corr(a, b): return float(stats.spearmanr(a, b)[0])

# ---- main-pipeline disease maps (for cross-check) --------------------------- #
def disease_map(dsfile, cols):
    _dis_loaders = {"_scz": load_scz_dataset(DATA), "_bd": load_bd_matched(DATA)}
    d = _dis_loaders[dsfile]
    d = d.dropna(subset=cols).reset_index(drop=True)
    d["age_z"] = (d["age"] - d["age"].mean()) / d["age"].std(ddof=1)
    cond = (d["Condition_type"].astype(str) == "Clinical").astype(float).values
    X, idx = design(d, [cond])
    Y = d[cols].values.astype(float)
    t, _ = tvals(X, Y, idx)
    return t[idx[0]]

df = load_bsnip_relatives(DATA)
df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)

rows = []
print("=== Within-BSNIP relatives: recapitulation of the proband map ===")
for dxlab, (prob_grp, rel_grp, mainfile) in CONTRASTS.items():
    for meas, cols in measures_of(df):
        sub = df[((df["role"] == "Control") |
                  ((df["role"] == "Proband")  & (df["dx_group"] == prob_grp)) |
                  ((df["role"] == "Relative") & (df["dx_group"] == rel_grp)))].copy()
        sub = sub.dropna(subset=cols).reset_index(drop=True)
        rel = (sub["role"] == "Relative").astype(float).values
        prob = (sub["role"] == "Proband").astype(float).values
        nR, nP, nC = int(rel.sum()), int(prob.sum()), int((sub["role"]=="Control").sum())
        Y = sub[cols].values.astype(float)
        X, idx = design(sub, [rel, prob])   # idx[0]=relative, idx[1]=proband
        t, dof = tvals(X, Y, idx)
        rel_map, prob_map = t[idx[0]], t[idx[1]]

        r_recap = corr(rel_map, prob_map)
        beta = np.polyfit(prob_map, rel_map, 1)[0]
        aff = np.abs(prob_map) > 2
        sign_conc = float((np.sign(rel_map[aff]) == np.sign(prob_map[aff])).mean()) if aff.any() else np.nan
        # relative-vs-CN FDR
        p_rel = 2 * stats.t.sf(np.abs(rel_map), dof)
        n_rel_fdr = int((fdr(p_rel) < 0.05).sum())

        # label-permutation null (shuffle Relative/Control; proband map fixed)
        cr = sub["role"].isin(["Control", "Relative"]).values
        Xc, Yc = X[cr], Y[cr]
        role_rc = rel[cr]                      # 1=relative among control+relative
        # build a design without the relative dummy, then swap in permuted dummy
        # (drop last column = relative dummy is idx[0]; rebuild per perm cheaply)
        base_c = np.delete(Xc, idx[0], axis=1)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            perm = rng.permutation(role_rc)
            Xp = np.column_stack([base_c[:, :idx[0]], perm, base_c[:, idx[0]:]])
            tp, _ = tvals(Xp, Yc, [idx[0]])
            null[i] = corr(tp[idx[0]], prob_map)
        p_recap = float((1 + np.sum(null >= r_recap)) / (1 + N_PERM))

        # cross-check vs main-pipeline disease map
        r_vs_main = corr(rel_map, disease_map(mainfile, cols))

        rows.append({"contrast": dxlab, "measure": meas,
                     "n_control": nC, "n_proband": nP, "n_relative": nR,
                     "r_recap_rel_vs_proband": round(r_recap, 3),
                     "p_recap_perm": round(p_recap, 4),
                     "attenuation_slope": round(float(beta), 3),
                     "sign_concordance_affected": round(sign_conc, 3) if sign_conc==sign_conc else np.nan,
                     "mean_abs_t_proband": round(float(np.abs(prob_map).mean()), 2),
                     "mean_abs_t_relative": round(float(np.abs(rel_map).mean()), 2),
                     "n_relative_regions_FDR": n_rel_fdr,
                     "r_rel_vs_main_disease_map": round(r_vs_main, 3)})
        pd.DataFrame({"region": cols, "t_relative_vs_CN": rel_map,
                      "t_proband_vs_CN": prob_map}).to_csv(
            os.path.join(OUT, f"maps_{dxlab}_{meas}.csv"), index=False)
        print(f"{dxlab:8} {meas:11} nR={nR:3} | r(rel,prob)={r_recap:+.2f} p={p_recap:.3f} "
              f"| slope={beta:+.2f} | signconc={sign_conc if sign_conc==sign_conc else float('nan'):.2f} "
              f"| relFDR={n_rel_fdr:2} | r vs main={r_vs_main:+.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "bsnip_relatives_summary.csv"), index=False)

# ============================================================================ #
# Liability-gradient test — CATEGORICAL group factor (endophenotype)           #
# region ~ age_z + C(sex) + C(study_site) + C(group, ref=Control)              #
# Per region: omnibus F-test of the 3-level group factor (2 df) + FDR; the      #
# Relative-vs-CN and Proband-vs-CN dummies, a strict-monotonicity flag, and the #
# Proband-Relative contrast localise the relatives on the axis.                 #
# ============================================================================ #
def ssr_of(X, Y):
    B = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
    return ((Y - X @ B) ** 2).sum(0)

def adjusted_means(sub, cols, mask_dict):
    """Covariate-adjusted (age+sex+site) group means per region: residualise on
    covariates, add the grand mean back, then take group means."""
    site = pd.get_dummies(sub["study_site"].astype(str), drop_first=True).astype(float)
    sex  = (sub["sex"].astype(str).str.startswith(("M", "m"))).astype(float).values
    Xc = np.column_stack([np.ones(len(sub)), sub["age_z"].values, sex, site.values])
    Y = sub[cols].values.astype(float)
    B = np.linalg.pinv(Xc.T @ Xc) @ (Xc.T @ Y)
    adj = (Y - Xc @ B) + Y.mean(0)
    return {g: adj[m].mean(0) for g, m in mask_dict.items()}

grad_rows = []
print("\n=== Liability gradient — categorical group factor (omnibus F) ===")
for dxlab, (prob_grp, rel_grp, _mainfile) in CONTRASTS.items():
    for meas, cols in measures_of(df):
        sub = df[((df["role"] == "Control") |
                  ((df["role"] == "Proband")  & (df["dx_group"] == prob_grp)) |
                  ((df["role"] == "Relative") & (df["dx_group"] == rel_grp)))].copy()
        sub = sub.dropna(subset=cols).reset_index(drop=True)
        Y = sub[cols].values.astype(float)
        rel_d  = (sub["role"] == "Relative").astype(float).values
        prob_d = (sub["role"] == "Proband").astype(float).values

        # full (categorical group factor, Control ref) vs reduced (covariates only)
        Xf, idxf = design(sub, [rel_d, prob_d])    # idxf[0]=Relative, idxf[1]=Proband vs CN
        Xr, _    = design(sub, [])                 # covariates only
        ssr_f = ssr_of(Xf, Y); ssr_r = ssr_of(Xr, Y)
        dof_f = Xf.shape[0] - Xf.shape[1]; q = 2
        Fstat = ((ssr_r - ssr_f) / q) / (ssr_f / dof_f)
        p_group = stats.f.sf(Fstat, q, dof_f)
        p_fdr = fdr(p_group)
        sig = p_fdr < 0.05                          # omnibus: groups differ at all

        tf, _ = tvals(Xf, Y, idxf)
        t_rel_cn, t_prob_cn = tf[idxf[0]], tf[idxf[1]]   # Relative-vs-CN, Proband-vs-CN

        m = adjusted_means(sub, cols, {
            "CN":  (sub["role"] == "Control").values,
            "Rel": (sub["role"] == "Relative").values,
            "Prob":(sub["role"] == "Proband").values})
        mono = ((m["CN"] < m["Rel"]) & (m["Rel"] < m["Prob"])) | \
               ((m["CN"] > m["Rel"]) & (m["Rel"] > m["Prob"]))
        frac_mono_sig = float(mono[sig].mean()) if sig.any() else np.nan

        # Proband - Relative contrast: same factor re-referenced to Relative
        ctrl_d = (sub["role"] == "Control").astype(float).values
        Xc, idxc = design(sub, [ctrl_d, prob_d])   # idxc[1] = Proband (vs Relative)
        tc, dofc = tvals(Xc, Y, idxc)
        t_pmr = tc[idxc[1]]
        p_pmr = 2 * stats.t.sf(np.abs(t_pmr), dofc)
        pfdr_pmr = fdr(p_pmr)
        n_pmr_fdr = int((pfdr_pmr < 0.05).sum())
        n_relative_like_proband = int((pfdr_pmr >= 0.05).sum())

        pd.DataFrame({"region": cols, "F_group": Fstat, "p_group": p_group,
                      "p_fdr": p_fdr, "sig_fdr": sig,
                      "t_relative_vs_CN": t_rel_cn, "t_proband_vs_CN": t_prob_cn,
                      "t_proband_minus_relative": t_pmr, "p_pmr_fdr": pfdr_pmr,
                      "adj_mean_CN": m["CN"], "adj_mean_Relative": m["Rel"],
                      "adj_mean_Proband": m["Prob"], "monotonic": mono}).to_csv(
            os.path.join(OUT, f"gradient_{dxlab}_{meas}.csv"), index=False)

        grad_rows.append({"contrast": dxlab, "measure": meas,
                          "n_control": int((sub["role"]=="Control").sum()),
                          "n_relative": int((sub["role"]=="Relative").sum()),
                          "n_proband": int((sub["role"]=="Proband").sum()),
                          "n_regions": len(cols),
                          "n_gradient_FDR": int(sig.sum()),   # omnibus group-difference regions
                          "frac_monotonic_among_sig": round(frac_mono_sig, 3) if frac_mono_sig==frac_mono_sig else np.nan,
                          "n_proband_minus_relative_FDR": n_pmr_fdr,
                          "n_relative_indistinct_from_proband": n_relative_like_proband,
                          "mean_F_group": round(float(Fstat.mean()), 2),
                          "max_F_group": round(float(Fstat.max()), 2)})
        print(f"{dxlab:8} {meas:11} group FDR {int(sig.sum()):2}/{len(cols):2} | "
              f"monotonic {frac_mono_sig if frac_mono_sig==frac_mono_sig else float('nan'):.2f} | "
              f"Prob-Rel FDR {n_pmr_fdr:2} | mean F {Fstat.mean():.2f}")

pd.DataFrame(grad_rows).to_csv(os.path.join(OUT, "bsnip_gradient_summary.csv"), index=False)

# ============================================================================ #
# Spatial (spin/shuffle) null for the recapitulation r  — needs the ENIGMA      #
# toolbox, so it is guarded and runs in the venv (like the surface-plot steps). #
# Complements the label-permutation p by accounting for spatial autocorrelation.#
# ============================================================================ #
N_ROT = 1000
try:
    from enigmatoolbox.permutation_testing import spin_test, shuf_test
    print("\n=== Spatial null (spin/shuffle) for recapitulation r ===")
    srows = []
    for dxlab, (prob_grp, rel_grp, _m) in CONTRASTS.items():
        for meas, _cols in measures_of(df):
            mp = pd.read_csv(os.path.join(OUT, f"maps_{dxlab}_{meas}.csv"))
            rmap = mp["t_relative_vs_CN"].values; pmap = mp["t_proband_vs_CN"].values
            np.random.seed(SEED)
            if meas == "Subcortical":
                p_spatial, _ = shuf_test(rmap, pmap, n_rot=N_ROT, type="spearman", null_dist=True)
                test = "shuf"
            else:
                p_spatial, _ = spin_test(rmap, pmap, surface_name="fsa5",
                                         parcellation_name="aparc", type="spearman",
                                         n_rot=N_ROT, null_dist=True)
                test = "spin"
            srows.append({"contrast": dxlab, "measure": meas,
                          "r_recap": round(float(stats.spearmanr(rmap, pmap)[0]), 3),
                          "p_spatial": round(float(np.asarray(p_spatial).ravel()[0]), 4),
                          "test": test, "n_rot": N_ROT})
            print(f"{dxlab:8} {meas:11} {test} p={srows[-1]['p_spatial']:.3f}")
    pd.DataFrame(srows).to_csv(os.path.join(OUT, "recap_spatial_null.csv"), index=False)
except Exception as e:
    print(f"\n[spin/shuffle skipped — run in the venv with enigmatoolbox: {e}]")

print("\nSaved ->", OUT)

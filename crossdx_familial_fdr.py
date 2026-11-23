#!/usr/bin/env python3
"""
Cross-disorder familial sharing — spatial (spin/shuffle) null + FDR (topic 11).
====================================================================================
Adds the ENIGMA spatial null and FDR correction to the cross-disorder sharing matrix
(relative-of-X vs proband-of-Y). Cortical measures use the spin test (Alexander-Bloch
rotation, spatial-autocorrelation-preserving); subcortical uses the label-shuffle test.
Implemented self-contained (vectorised) using the ENIGMA toolbox's own fsa5 sphere
(.gii) and aparc (.annot) data files — located without importing the VTK-dependent
parts — so it runs anywhere nibabel is available, and reproduces enigmatoolbox
spin_test/shuf_test exactly.

FDR (Benjamini-Hochberg) is applied across the family of relative-vs-proband tests
(all measures pooled) and, separately, the relative-vs-relative tests.

Sample: bsnip_relatives_dataset.csv (loaded via load_bsnip_relatives() from data_utils)
Outputs -> ../results/relatives_reviewer/crossdisorder_familial_sharing_spin.csv
"""
import os, importlib.util, warnings, numpy as np, pandas as pd, nibabel as nb
from scipy.spatial.distance import cdist
from scipy import stats
warnings.filterwarnings("ignore")

DATA = os.path.join("..", "data")
OUT  = os.path.join("..", "results", "11_relatives_reviewer")
os.makedirs(OUT, exist_ok=True)
N_ROT, SEED = 1000, 1234
DXES = ["SCZ", "SAD", "BD"]

# ---- locate the enigmatoolbox data files WITHOUT importing it (avoids VTK) --- #
import glob
from data_utils import load_bsnip_relatives
def _find_enigma():
    spec = importlib.util.find_spec("enigmatoolbox")
    if spec is not None and spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    # fallback: the venv layout (../lib/python*/site-packages/enigmatoolbox)
    for cand in glob.glob(os.path.join("..", "lib", "python*", "site-packages", "enigmatoolbox")):
        if os.path.isdir(cand):
            return cand
    raise SystemExit("enigmatoolbox package not found (needed for its fsa5 sphere + "
                     "aparc annot data). Run in the venv where it is installed.")
EPKG = _find_enigma()
ANNOT = os.path.join(EPKG, "permutation_testing", "annot")
SURF  = os.path.join(EPKG, "datasets", "surfaces")
SKIP = {"unknown", "corpus", "corpuscallosum", "medialwall",
        "Background+FreeSurfer_Defined_Medial_Wall"}

def sphere_pts(side):
    return nb.load(os.path.join(SURF, f"fsa5_sphere_{side}.gii")).darrays[0].data.astype(float)

def centroids_and_names(side):
    lab, ctab, names = nb.freesurfer.io.read_annot(
        os.path.join(ANNOT, f"fsa5_{side}_aparc.annot"), orig_ids=True)
    pts = sphere_pts(side)
    cents, keep = [], []
    for ic in range(ctab.shape[0]):
        nm = names[ic].decode()
        if nm in SKIP:
            continue
        cents.append(pts[lab == ctab[ic, -1]].mean(0)); keep.append(nm)
    return np.vstack(cents), keep

# ---- faithful rotate_parcellation (Alexander-Bloch/Vasa; author @saratheriver) - #
def rotate_parcellation(coord_l, coord_r, nrot=N_ROT):
    nl, nr = coord_l.shape[0], coord_r.shape[0]; nroi = nl + nr
    perm_id = np.zeros((nroi, nrot), int); r = 0
    I1 = np.diag([-1., 1., 1.])
    while r < nrot:
        A = np.random.normal(0, 1, (3, 3)); TL, tmp = np.linalg.qr(A)
        TL = TL @ np.diag(np.sign(np.diag(tmp)))
        if np.linalg.det(TL) < 0: TL[:, 0] = -TL[:, 0]
        TR = I1 @ TL @ I1
        dl = cdist(coord_l, coord_l @ TL); dr = cdist(coord_r, coord_r @ TR)
        def assign(d, n):
            d = d.copy(); ref = np.zeros(n, int); rot = np.zeros(n, int)
            for k in range(n):
                mins = np.nanmin(d, axis=1)
                ri = int(np.argwhere(mins == np.nanmax(mins))[0])
                ci = int(np.argwhere(d[ri, :] == np.nanmin(d[ri, :]))[0])
                ref[k] = ri; rot[k] = ci; d[:, ci] = np.nan; d[ri, :] = np.nan
            return ref, rot
        rl_ref, rl_rot = assign(dl, nl); rr_ref, rr_rot = assign(dr, nr)
        ref = np.append(rl_ref, nl + rr_ref); rot = np.append(rl_rot, nl + rr_rot)
        rot_sort = rot[np.argsort(ref)]
        if not np.all(rot_sort == np.arange(nroi)):
            perm_id[:, r] = rot_sort; r += 1
    return perm_id

def _colcorr(M, v):
    Mc = M - M.mean(0); vc = v - v.mean()
    return (Mc * vc[:, None]).sum(0) / (np.sqrt((Mc**2).sum(0)) * np.sqrt((vc**2).sum()))

def perm_p(x, y, perm_id):
    """Vectorised perm_sphere_p / shuf_test p (two-sided-by-sign, xy & yx averaged)."""
    rho = float(stats.spearmanr(x, y)[0])
    xr = stats.rankdata(x); yr = stats.rankdata(y)
    xp = x[perm_id]; yp = y[perm_id]                    # (nroi, nperm)
    xpr = stats.rankdata(xp, axis=0); ypr = stats.rankdata(yp, axis=0)
    nxy = _colcorr(xpr, yr); nyx = _colcorr(ypr, xr)
    if rho >= 0:
        p = (np.mean(nxy > rho) + np.mean(nyx > rho)) / 2
    else:
        p = (np.mean(nxy < rho) + np.mean(nyx < rho)) / 2
    return rho, float(p)

def fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    r = np.empty(n); c = p[o] * n / (np.arange(n) + 1)
    r[o] = np.minimum.accumulate(c[::-1])[::-1]; return np.clip(r, 0, 1)

# ---- disorder relative/proband t-maps (as in 11 section 3) ------------------ #
df = load_bsnip_relatives(DATA)
df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std(ddof=1)
def cortical(pref): return [c for c in df.columns if c.startswith(f"lh.{pref}.") or c.startswith(f"rh.{pref}.")]
def subc(): return [c for c in df.columns if c.startswith("SubcGM.") and "Vent" not in c and "Cereb" not in c]

def order_cortical(cols, names_lh, names_rh, pref):
    idx = {c: i for i, c in enumerate(cols)}
    order = [idx[f"lh.{pref}.{n}"] for n in names_lh] + [idx[f"rh.{pref}.{n}"] for n in names_rh]
    return np.array(order)

def rel_prob_maps(dx, cols):
    sub = df[(df["role"] == "Control") | (df["dx_group"] == dx)].dropna(subset=cols).reset_index(drop=True)
    rel  = ((sub["role"] == "Relative") & (sub["dx_group"] == dx)).astype(float).values
    prob = ((sub["role"] == "Proband") & (sub["dx_group"] == dx)).astype(float).values
    site = pd.get_dummies(sub["study_site"].astype(str), drop_first=True).astype(float)
    sex  = (sub["sex"].astype(str).str.startswith(("M", "m"))).astype(float).values
    X = np.column_stack([np.ones(len(sub)), sub["age_z"].values, sex, site.values, rel, prob])
    jr, jp = X.shape[1]-2, X.shape[1]-1
    XtXi = np.linalg.pinv(X.T @ X); B = XtXi @ (X.T @ sub[cols].values.astype(float))
    ssr = ((sub[cols].values.astype(float) - X @ B) ** 2).sum(0); dof = X.shape[0]-X.shape[1]
    tr = B[jr] / np.sqrt(XtXi[jr, jr] * ssr / dof); tp = B[jp] / np.sqrt(XtXi[jp, jp] * ssr / dof)
    return tr, tp

# ---- build permutations once ------------------------------------------------ #
np.random.seed(SEED)
cl, names_lh = centroids_and_names("lh"); cr, names_rh = centroids_and_names("rh")
print("rotating parcellation (spin null, cortical)...")
perm_cort = rotate_parcellation(cl, cr, N_ROT)
n_sub = len(subc()); perm_sub = np.array([np.random.permutation(n_sub) for _ in range(N_ROT)]).T

MEAS = [("CT", "CT"), ("GMV", "GM"), ("SA", "SA"), ("Subcortical", None)]
rows = []
print("computing spin/shuffle p for each cross-disorder cell...")
for meas, pref in MEAS:
    cols = subc() if meas == "Subcortical" else cortical(pref)
    perm = perm_sub if meas == "Subcortical" else perm_cort
    if meas != "Subcortical":
        ordr = order_cortical(cols, names_lh, names_rh, pref)
    relmap, probmap = {}, {}
    for dx in DXES:
        tr, tp = rel_prob_maps(dx, cols)
        if meas != "Subcortical":
            tr, tp = tr[ordr], tp[ordr]
        relmap[dx], probmap[dx] = tr, tp
    for x in DXES:
        for y in DXES:
            r_rp, p_rp = perm_p(relmap[x], probmap[y], perm)
            # rel-vs-rel: skip the diagonal (x==y is a map vs itself, r=1 trivially)
            if x == y:
                r_rr, p_rr = np.nan, np.nan
            else:
                r_rr, p_rr = perm_p(relmap[x], relmap[y], perm)
            rows.append({"measure": meas, "relative_of": x, "proband_of": y,
                         "r_rel_vs_prob": round(r_rp, 3), "p_spin_rel_vs_prob": round(p_rp, 4),
                         "r_rel_vs_rel": round(r_rr, 3) if r_rr == r_rr else np.nan,
                         "p_spin_rel_vs_rel": round(p_rr, 4) if p_rr == p_rr else np.nan,
                         "test": "shuf" if meas == "Subcortical" else "spin"})
tab = pd.DataFrame(rows)
def fdr_masked(p):                       # BH-FDR over finite entries only
    p = np.asarray(p, float); out = np.full(len(p), np.nan)
    m = np.isfinite(p)
    if m.any(): out[m] = fdr(p[m])
    return out
tab["p_fdr_rel_vs_prob"] = fdr(tab["p_spin_rel_vs_prob"].values)     # across all cells
tab["p_fdr_rel_vs_rel"]  = fdr_masked(tab["p_spin_rel_vs_rel"].values)  # off-diagonal only
tab.to_csv(os.path.join(OUT, "crossdisorder_familial_sharing_spin.csv"), index=False)
print("Saved -> crossdisorder_familial_sharing_spin.csv")
print(tab[["measure","relative_of","proband_of","r_rel_vs_prob","p_spin_rel_vs_prob","p_fdr_rel_vs_prob"]].to_string(index=False))

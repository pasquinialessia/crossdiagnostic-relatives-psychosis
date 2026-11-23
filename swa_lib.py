#!/usr/bin/env python3
"""
Sliding-window peak-reliability permutation test + ENIGMA DK plotting.
Parallelized (joblib). 

WHAT'S PLOTTED: by default the surface is colored by the MEAN AGE of each
significant region's peak window (a "peak-age map"), not by Cohen's d. Set
plot_metric="d" to color by effect size instead.

OUTPUTS (see run_pipeline args):
  - results CSV  (results_csv=...)   : per-region peaks, peak mean-ages, p, FDR
  - cortical PNG (cortical_png=...)  : DK surface, significant regions colored
  - subcortical PNG: run on SubcGM.* cols, then plot_subcortical_sig(...)
  - trajectory PNG: plot_trajectories(...)  d-vs-age curves for chosen regions

d = 2*t/sqrt(df_error)  (R effectsize::t_to_d). SIGN=-1 because you set Control
as reference so d is Clinical-Control (negative); the peak is the most negative
d. Reported/plotted d keeps the real (negative) sign.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# --- matplotlib<->enigmatoolbox compatibility shim -------------------------- #
# matplotlib >=3.9 removed cm.get_cmap(), which older enigmatoolbox still calls.
# Restore the alias so plotting works without downgrading matplotlib.
try:
    import matplotlib
    import matplotlib.cm as _mcm
    import matplotlib.pyplot as _plt
    if not hasattr(_mcm, "get_cmap"):
        _mcm.get_cmap = _plt.get_cmap
except Exception as _e:      # matplotlib not installed -> only stats will run
    print(f"[shim] matplotlib not patched: {_e}")

# ============================ CONFIG ========================================= #
GROUP_COL   = "Condition_type"
GROUP_REF   = "Control"
GROUP_COEF  = "Clinical"     # dummy=1; term Condition_typeClinical
AGE_COL     = "age"
SEX_COL     = "sex"
SITE_COL    = "study_site"

WINDOW_WIDTH = 10
START_AGES   = list(range(18, 42))   # 18..41 -> 24 windows
MIN_WINDOW_N = 10
SIGN         = -1.0
N_PERM       = 1000
FDR_ALPHA    = 0.05
RANDOM_SEED  = 1234

SUMMARY_REGION_COL = "measure"
SUMMARY_D_COL      = "Mean_Cohen_d_clinical"
SUMMARY_START_COL  = "Start_age"
SUMMARY_MEANAGE_COL = "Mean_age"

# ============================ CORE STATS ===================================== #
def prepare_arrays(df, region_cols, include_sex=None):
    """Extract numpy arrays once. include_sex=None auto-detects: if the subset
    contains only one sex, the sex term is DROPPED (a constant column would
    duplicate the intercept and make X'X singular, killing every window)."""
    grp = (df[GROUP_COL].values == GROUP_COEF).astype(np.float64)
    sex_raw = df[SEX_COL].astype(str).str.lower().str.startswith("m")
    sex = sex_raw.astype(np.float64).values
    if include_sex is None:
        include_sex = df[SEX_COL].nunique(dropna=True) > 1
    if not include_sex:
        print(f"  [design] single sex in subset -> dropping 'sex' from the model")
        sex = None
    age0 = df[AGE_COL].values.astype(np.float64)
    site_full = pd.get_dummies(df[SITE_COL].astype(str)).values.astype(np.float64)
    Y = df[region_cols].values.astype(np.float64)
    return dict(age0=age0, grp=grp, sex=sex, site_full=site_full, Y=Y)


def subset_by_sex(df, sex=None):
    """sex: None (all), 'Male', or 'Female'. Matches on first letter, case-insensitive."""
    if sex is None:
        return df
    first = sex.strip().lower()[0]
    keep = df[SEX_COL].astype(str).str.lower().str.startswith(first)
    out = df.loc[keep]
    print(f"  [subset] sex={sex}: {len(out)}/{len(df)} subjects")
    return out


def window_mean_ages(age_vec, start_ages=START_AGES, width=WINDOW_WIDTH):
    """Empirical mean subject age within each window (from real ages)."""
    out = {}
    for st in start_ages:
        m = (age_vec >= st) & (age_vec < st + width)
        out[st] = float(np.nanmean(age_vec[m])) if m.any() else np.nan
    return out


def _peak_all_np(Y, grp, age_vec, sex, site_full, sign,
                 start_ages=START_AGES, width=WINDOW_WIDTH, min_n=MIN_WINDOW_N,
                 return_traj=False):
    """Per-region peak of (sign*d) across windows. d = 2*t/sqrt(df_error)."""
    n_all, R = Y.shape
    W = len(start_ages)
    best = np.full(R, -np.inf)
    argwin = np.full(R, -1, dtype=int)
    traj = np.full((R, W), np.nan) if return_traj else None

    for wi, start in enumerate(start_ages):
        m = (age_vec >= start) & (age_vec < start + width)
        idx = np.where(m)[0]
        if idx.size < min_n:
            continue
        g = grp[idx]
        n1 = g.sum(); n0 = idx.size - n1
        if n1 < 2 or n0 < 2:
            continue
        a = age_vec[idx]
        sf = site_full[idx]
        sf = sf[:, sf.sum(0) > 0]
        if sf.shape[1] >= 1:
            sf = sf[:, 1:]
        cols = [np.ones(idx.size), g, a]
        if sex is not None:                 # dropped in single-sex subsets
            cols.append(sex[idx])
        if sf.shape[1]:
            cols.append(sf)
        X = np.column_stack(cols)
        gi = 1
        n, p = X.shape
        dof = n - p
        if dof <= 1:
            continue
        try:
            inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        Xt = X.T
        Ysub = Y[idx]
        finite = np.isfinite(Ysub).all(0)

        if finite.any():
            Yg = Ysub[:, finite]
            B = inv @ (Xt @ Yg)
            resid = Yg - X @ B
            sigma2 = (resid ** 2).sum(0) / dof
            se = np.sqrt(sigma2 * inv[gi, gi])
            with np.errstate(divide="ignore", invalid="ignore"):
                t = B[gi, :] / se
            d = sign * 2.0 * t / np.sqrt(dof)
            fi = np.where(finite)[0]
            if return_traj:
                traj[fi, wi] = d
            upd = d > best[fi]
            best[fi[upd]] = d[upd]
            argwin[fi[upd]] = wi

        for j in np.where(~finite)[0]:
            y = Ysub[:, j]
            good = np.isfinite(y)
            if good.sum() <= p + 1:
                continue
            Xg = X[good]; yg = y[good]
            gg = Xg[:, gi]
            if (gg == 1).sum() < 2 or (gg == 0).sum() < 2:
                continue
            try:
                invg = np.linalg.inv(Xg.T @ Xg)
            except np.linalg.LinAlgError:
                continue
            b = invg @ (Xg.T @ yg)
            r = yg - Xg @ b
            dofg = good.sum() - p
            sig2 = (r @ r) / dofg
            seg = np.sqrt(sig2 * invg[gi, gi])
            if seg == 0 or not np.isfinite(seg):
                continue
            dv = sign * 2.0 * (b[gi] / seg) / np.sqrt(dofg)
            if return_traj:
                traj[j, wi] = dv
            if dv > best[j]:
                best[j] = dv; argwin[j] = wi

    best[~np.isfinite(best)] = np.nan
    if return_traj:
        return best, traj, argwin
    return best, argwin


def _one_perm(seed, age0, Y, grp, sex, site_full, sign):
    rng = np.random.default_rng(seed)
    shuffled = age0.copy(); rng.shuffle(shuffled)
    peaks, _ = _peak_all_np(Y, grp, shuffled, sex, site_full, sign)
    return peaks


def run_permutation_parallel(arr, sign=SIGN, n_perm=N_PERM, n_jobs=-1,
                             seed=RANDOM_SEED, verbose=5):
    child_seeds = np.random.SeedSequence(seed).spawn(n_perm)
    out = Parallel(n_jobs=n_jobs, verbose=verbose, prefer="processes")(
        delayed(_one_perm)(cs, arr["age0"], arr["Y"], arr["grp"],
                           arr["sex"], arr["site_full"], sign)
        for cs in child_seeds)
    return np.vstack(out)


def perm_pvalue(observed_signed, null_col):
    v = null_col[np.isfinite(null_col)]
    if not np.isfinite(observed_signed) or v.size == 0:
        return np.nan
    return (1 + np.sum(v >= observed_signed)) / (1 + v.size)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = idx.size
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0, 1)
    return out

# ============================ ENIGMA CORTICAL (DK) =========================== #
DK34 = ["bankssts","caudalanteriorcingulate","caudalmiddlefrontal","cuneus",
        "entorhinal","fusiform","inferiorparietal","inferiortemporal",
        "isthmuscingulate","lateraloccipital","lateralorbitofrontal","lingual",
        "medialorbitofrontal","middletemporal","parahippocampal","paracentral",
        "parsopercularis","parsorbitalis","parstriangularis","pericalcarine",
        "postcentral","posteriorcingulate","precentral","precuneus",
        "rostralanteriorcingulate","rostralmiddlefrontal","superiorfrontal",
        "superiorparietal","superiortemporal","supramarginal","frontalpole",
        "temporalpole","transversetemporal","insula"]


def build_dk_vector(values_by_region, measure_prefix):
    vec = np.full(68, np.nan)
    for i, reg in enumerate(DK34):
        vec[i]      = values_by_region.get(f"lh.{measure_prefix}.{reg}", np.nan)
        vec[i + 34] = values_by_region.get(f"rh.{measure_prefix}.{reg}", np.nan)
    return vec


def _auto_age_ticks(color_range, step=2):
    lo, hi = color_range
    return list(np.arange(np.floor(lo), np.ceil(hi) + 0.001, step))


# ---- black outlines around significant parcels ----------------------------- #
def _get_triangles(surf):
    """Return (n_tri, 3) triangle array for a surface, trying several APIs.
    brainspace may not be installed; ENIGMA vendors its own mesh utilities."""
    # 1) ENIGMA's vendored copy
    try:
        from enigmatoolbox.mesh.mesh_elements import get_cells
        return np.asarray(get_cells(surf))
    except Exception:
        pass
    # 2) standalone brainspace, if present
    try:
        from brainspace.mesh.mesh_elements import get_cells
        return np.asarray(get_cells(surf))
    except Exception:
        pass
    # 3) BSPolyData attribute
    try:
        return np.asarray(surf.Polygons).reshape(-1, 3)
    except Exception:
        pass
    # 4) raw VTK
    from vtk.util.numpy_support import vtk_to_numpy
    raw = vtk_to_numpy(surf.GetPolys().GetData())
    return raw.reshape(-1, 4)[:, 1:]      # vtk stores [3, i, j, k] per triangle


def _n_points(surf):
    try:
        return int(surf.n_points)
    except Exception:
        return int(surf.GetNumberOfPoints())


def _adj_from_triangles(tri, n):
    """Symmetric vertex adjacency (csr) from a triangle list."""
    import scipy.sparse as sp
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    a = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr()


def _fsa5_adjacency():
    """Vertex adjacency for the fsa5 surface (lh+rh concatenated, 20484).
    Works with or without brainspace installed."""
    import scipy.sparse as sp
    from enigmatoolbox.datasets import load_fsa5
    surfs = load_fsa5()
    blocks = []
    # preferred: a ready-made adjacency helper, if either package provides one
    try:
        try:
            from enigmatoolbox.mesh.mesh_elements import get_immediate_adjacency as _gia
        except Exception:
            from brainspace.mesh.mesh_elements import get_immediate_adjacency as _gia
        blocks = [_gia(s) for s in surfs]
    except Exception:
        # fallback: build it ourselves from the triangles
        blocks = [_adj_from_triangles(_get_triangles(s), _n_points(s)) for s in surfs]
    return sp.block_diag(blocks).tocsr()


def _parcel_boundary_mask(vert_parcel, sig_parcel_ids, adj, width=1):
    """Boolean mask of vertices on the border of any significant parcel.
    A vertex qualifies if its parcel is significant AND at least one immediate
    neighbour belongs to a different parcel (so adjacent significant parcels
    still get their own outline).
    width>1 thickens the line inward by that many vertex rings (the outline is
    grown only into significant parcels, so it never bleeds into grey cortex)."""
    boundary = np.zeros(vert_parcel.shape[0], bool)
    sig_vert = np.isin(vert_parcel, list(sig_parcel_ids))
    for i in np.where(sig_vert)[0]:
        nb = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        if np.any(vert_parcel[nb] != vert_parcel[i]):
            boundary[i] = True
    # grow inward: each extra ring adds neighbours that sit in the same parcel
    for _ in range(max(0, int(width) - 1)):
        grow = boundary.copy()
        for i in np.where(boundary)[0]:
            nb = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
            same = nb[vert_parcel[nb] == vert_parcel[i]]
            grow[same] = True
        boundary = grow
    return boundary


def _make_outline_cmap(base_cmap="viridis", name="_swp_outline_cmap", n=256):
    """Colormap identical to base_cmap but with index 0 = black (the sentinel).
    Registered by name because brainspace/enigmatoolbox resolve cmaps by string."""
    import matplotlib as mpl
    base = mpl.colormaps[base_cmap].resampled(n)
    colors = base(np.linspace(0, 1, n))
    colors = np.vstack([np.array([0.0, 0.0, 0.0, 1.0]), colors])  # n+1 entries
    cmap = mpl.colors.ListedColormap(colors, name=name)
    try:
        mpl.colormaps.register(cmap, name=name, force=True)
    except Exception:
        pass
    return name, n


def _apply_outlines(vec_parcel, surf_vals, color_range, base_cmap, width=1,
                    outline_vec=None):
    """Paint boundary vertices black.
    outline_vec: 68-vector, finite where a parcel should be outlined. If None,
    outlines whatever is finite in vec_parcel (i.e. the coloured parcels)."""
    from enigmatoolbox.utils.parcellation import parcel_to_surface
    adj = _fsa5_adjacency()
    # vertexwise parcel ids (1..68); medial wall -> 0
    vert_parcel = parcel_to_surface(np.arange(1, 69, dtype=float), "aparc_fsa5")
    src = vec_parcel if outline_vec is None else outline_vec
    sig_ids = np.where(np.isfinite(src))[0] + 1
    bmask = _parcel_boundary_mask(vert_parcel, sig_ids, adj, width=width)
    lo, hi = color_range
    name, n = _make_outline_cmap(base_cmap)
    bin_w = (hi - lo) / n
    sentinel = lo - bin_w / 2.0          # lands in the black bin
    out = surf_vals.copy()
    out[bmask] = sentinel
    return out, name, (lo - bin_w, hi)


def _compose_colorbar(brain_png, out_png, color_range, cmap, ticks, label):
    """Attach a matplotlib colorbar (with many ticks) to a brain screenshot."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    img = plt.imread(brain_png)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100 + 1.4, h / 100), dpi=100)
    ax.imshow(img); ax.axis("off")
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(color_range[0], color_range[1]), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, ticks=ticks)
    cbar.set_label(label)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    import os
    if os.path.exists(brain_png):
        os.remove(brain_png)


_MEDIAL_MASK_CACHE = {}


def _medial_wall_mask():
    """Vertices that are NOT cortex (medial wall / corpus callosum / unknown).
    parcel_to_surface may fill these with 0 instead of NaN; with a colour range
    like (18, 50) a 0 clamps to the bottom colour and paints the medial 'hole'.
    Forcing them to NaN makes nan_color (grey) apply instead."""
    if "m" not in _MEDIAL_MASK_CACHE:
        from enigmatoolbox.utils.parcellation import parcel_to_surface
        vp = parcel_to_surface(np.arange(1, 69, dtype=float), "aparc_fsa5")
        _MEDIAL_MASK_CACHE["m"] = (~np.isfinite(vp)) | (vp <= 0)
        n = int(_MEDIAL_MASK_CACHE["m"].sum())
        print(f"  [medial wall] {n} non-cortex vertices forced to NaN (grey)")
    return _MEDIAL_MASK_CACHE["m"]


def _blank_medial_wall(surf):
    """Set non-cortex vertices to NaN so they render grey, not bottom-of-scale."""
    try:
        surf = surf.copy()
        surf[_medial_wall_mask()] = np.nan
    except Exception as e:
        print(f"  [medial wall mask skipped: {e}]")
    return surf


def plot_cortical_sig(values_by_region, measure_prefix, out_png,
                      color_range=None, cmap="viridis",
                      cbar_ticks=None, cbar_label="value", outline=True,
                      outline_width=2, outline_regions=None):
    """Color parcels by values_by_region (NaN/absent = grey).
    outline_regions: list of region column names to trace in black. If None,
    the coloured parcels are outlined. Pass the significant regions here while
    colouring ALL regions to get 'full map + significance delineated'.
    outline_width = thickness in vertex rings (1 thinnest, 2-3 print-visible)."""
    from enigmatoolbox.utils.parcellation import parcel_to_surface
    from enigmatoolbox.plotting import plot_cortical
    vec = build_dk_vector(values_by_region, measure_prefix)
    surf = parcel_to_surface(vec, "aparc_fsa5")
    if color_range is None:
        fin = vec[np.isfinite(vec)]
        color_range = (float(np.nanmin(fin)), float(np.nanmax(fin))) if fin.size else (0, 1)
    # the colorbar must keep the TRUE range/cmap; only the brain gets the
    # sentinel-extended range and black-bottom colormap
    bar_range, bar_cmap = color_range, cmap
    brain_range, brain_cmap = color_range, cmap
    if outline:
        try:
            ovec = None
            if outline_regions is not None:
                ovec = build_dk_vector({r: 1.0 for r in outline_regions}, measure_prefix)
            surf, brain_cmap, brain_range = _apply_outlines(
                vec, surf, color_range, cmap, width=outline_width, outline_vec=ovec)
        except Exception as e:
            print(f"  [outline skipped: {e}] plotting without borders")
    surf = _blank_medial_wall(surf)          # non-cortex -> grey, never coloured
    if cbar_ticks is None:
        plot_cortical(array_name=surf, surface_name="fsa5", size=(900, 400),
                      cmap=brain_cmap, color_bar=True, color_range=brain_range,
                      nan_color=(0.75, 0.75, 0.75, 1), screenshot=True, filename=out_png)
    else:
        tmp = out_png + ".brain.png"
        plot_cortical(array_name=surf, surface_name="fsa5", size=(900, 400),
                      cmap=brain_cmap, color_bar=False, color_range=brain_range,
                      nan_color=(0.75, 0.75, 0.75, 1), screenshot=True, filename=tmp)
        _compose_colorbar(tmp, out_png, bar_range, bar_cmap, cbar_ticks, cbar_label)
    print(f"  saved {out_png}")

# ============================ ENIGMA SUBCORTICAL ============================= #
# ENIGMA's subcortical vector is INTERLEAVED L/R pairs, ventricles FIRST:
#   LLatVent, RLatVent, Lthal, Rthal, Lcaud, Rcaud, Lput, Rput,
#   Lpal, Rpal, Lhippo, Rhippo, Lamyg, Ramyg, Laccumb, Raccumb
# ventricles=False drops the first pair -> 14 values.
# VERIFY with verify_subcortical_order() before trusting any subcortical figure.
SUBCORT_PAIR_ORDER = ["lateralventricle", "thalamus", "caudate", "putamen",
                      "pallidum", "hippocampus", "amygdala", "accumbens"]
SUBCORT_VENTRICLES = True

SUBCORT_MAP = {
    "accumbens":        ("SubcGM.Left.Accumbens.area",   "SubcGM.Right.Accumbens.area"),
    "amygdala":         ("SubcGM.Left.Amygdala",         "SubcGM.Right.Amygdala"),
    "caudate":          ("SubcGM.Left.Caudate",          "SubcGM.Right.Caudate"),
    "hippocampus":      ("SubcGM.Left.Hippocampus",      "SubcGM.Right.Hippocampus"),
    "pallidum":         ("SubcGM.Left.Pallidum",         "SubcGM.Right.Pallidum"),
    "putamen":          ("SubcGM.Left.Putamen",          "SubcGM.Right.Putamen"),
    "thalamus":         ("SubcGM.Left.Thalamus.Proper",  "SubcGM.Right.Thalamus.Proper"),
    "lateralventricle": ("SubcGM.Left.Lateral.Ventricle","SubcGM.Right.Lateral.Ventricle"),
}


def _subcort_order():
    """Structure names in ENIGMA order (one entry per L/R pair)."""
    return (SUBCORT_PAIR_ORDER if SUBCORT_VENTRICLES
            else [s for s in SUBCORT_PAIR_ORDER if s != "lateralventricle"])


def build_subcortical_vector(values_by_region):
    """Interleaved L,R,L,R... vector in ENIGMA order (16 with ventricles, else 14)."""
    vec = []
    for s in _subcort_order():
        l_col, r_col = SUBCORT_MAP[s]
        vec.append(values_by_region.get(l_col, np.nan))
        vec.append(values_by_region.get(r_col, np.nan))
    return np.array(vec, dtype=float)


def plot_subcortical_sig(results_df, subcort_png="subcortical_sig.png",
                         value_col="peak_win_pointest_meanage", sig_col="sig_pointest",
                         cmap="viridis", color_range=None, cbar_ticks=None,
                         cbar_label=None):
    """Plot significant subcortical regions on the ENIGMA subcortical surface.
    Default colors by peak mean-age; pass value_col='peak_d_pointest' for d."""
    from enigmatoolbox.plotting import plot_subcortical
    sig = results_df[results_df[sig_col]]
    mappable = set().union(*[{a, b} for a, b in SUBCORT_MAP.values()])
    plotted = [r for r in sig["region"] if r in mappable]
    dropped = [r for r in sig["region"] if r not in mappable]
    print(f"  subcortical: {len(sig)} significant | {len(plotted)} on ENIGMA surface"
          f" | {len(dropped)} not drawable")
    if dropped:
        print(f"    not drawable (tested, no ENIGMA geometry): {dropped}")
    if not plotted:
        print("  -> nothing to draw (no significant structure maps to the surface)")
        return
    vals = {row.region: getattr(row, value_col) for row in sig.itertuples()}
    vec = build_subcortical_vector(vals)
    if color_range is None:
        fin = vec[np.isfinite(vec)]
        color_range = (float(np.nanmin(fin)), float(np.nanmax(fin)))
    is_age = "meanage" in value_col
    if cbar_ticks is None and is_age:
        cbar_ticks = _auto_age_ticks(color_range)
    if cbar_label is None:
        cbar_label = "peak window mean age (years)" if is_age else "Cohen's d"
    if cbar_ticks is None:
        plot_subcortical(array_name=vec, ventricles=SUBCORT_VENTRICLES, size=(900, 400),
                         cmap=cmap, color_bar=True, color_range=color_range,
                         nan_color=(0.75, 0.75, 0.75, 1), screenshot=True, filename=subcort_png)
    else:
        tmp = subcort_png + ".brain.png"
        plot_subcortical(array_name=vec, ventricles=SUBCORT_VENTRICLES, size=(900, 400),
                         cmap=cmap, color_bar=False, color_range=color_range,
                         nan_color=(0.75, 0.75, 0.75, 1), screenshot=True, filename=tmp)
        _compose_colorbar(tmp, subcort_png, color_range, cmap, cbar_ticks, cbar_label)
    print(f"  saved {subcort_png}")


def verify_cortical_order(region="lh.CT.superiortemporal", measure_prefix="CT",
                          out_png="verify_cortical.png"):
    from enigmatoolbox.utils.parcellation import parcel_to_surface
    from enigmatoolbox.plotting import plot_cortical
    surf = parcel_to_surface(build_dk_vector({region: 1.0}, measure_prefix), "aparc_fsa5")
    plot_cortical(array_name=surf, surface_name="fsa5", cmap="Reds", color_bar=True,
                  color_range=(0, 1), nan_color=(0.8, 0.8, 0.8, 1),
                  screenshot=True, filename=out_png)
    print(f"  '{region}' should be the ONLY lit parcel -> {out_png}")


def verify_subcortical_order(structure="SubcGM.Left.Hippocampus",
                             out_png="verify_subcortical.png"):
    from enigmatoolbox.plotting import plot_subcortical
    vec = build_subcortical_vector({structure: 1.0})
    plot_subcortical(array_name=vec, ventricles=SUBCORT_VENTRICLES, cmap="Reds",
                     color_bar=True, color_range=(0, 1), nan_color=(0.8, 0.8, 0.8, 1),
                     screenshot=True, filename=out_png)
    print(f"  '{structure}' should be the ONLY lit structure -> {out_png}")

# ============================ TRAJECTORY DIAGNOSTIC ========================== #
def compute_trajectories(raw_path, region_cols, sign=SIGN):
    """Return (traj_df, meanages). traj_df: index=region, cols=window mean age,
    values = REAL Cohen's d per window. Lets you see where each region dips."""
    raw = _read(raw_path)
    arr = prepare_arrays(raw, region_cols)
    _, traj_signed, _ = _peak_all_np(arr["Y"], arr["grp"], arr["age0"],
                                     arr["sex"], arr["site_full"], sign,
                                     return_traj=True)
    real_d = sign * traj_signed                      # back to real (negative) d
    mage = window_mean_ages(arr["age0"])
    cols = [mage[st] for st in START_AGES]
    traj_df = pd.DataFrame(real_d, index=region_cols, columns=np.round(cols, 1))
    return traj_df, mage


def plot_trajectories(raw_path, region_cols, out_png="trajectories.png",
                      sign=SIGN, mark_peak=True, title=None):
    """Line plot of per-window Cohen's d vs window mean age, one line per region,
    with each region's peak (most negative d) marked."""
    import matplotlib.pyplot as plt
    traj_df, _ = compute_trajectories(raw_path, region_cols, sign)
    x = traj_df.columns.values.astype(float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for reg, row in traj_df.iterrows():
        y = row.values.astype(float)
        ax.plot(x, y, lw=1, alpha=0.6, label=reg.split(".")[-1])
        if mark_peak:
            j = np.nanargmin(y)                      # deepest deficit (real d)
            ax.plot(x[j], y[j], "o", ms=5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(np.arange(np.floor(x.min()), np.ceil(x.max()) + 0.001, 2))
    ax.set_xlabel("window mean age (years)")
    ax.set_ylabel("Cohen's d  (Clinical − Control)")
    ax.set_title(title or "Sliding-window effect trajectories")
    if len(region_cols) <= 12:
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved {out_png}")

# ============================ PIPELINE ======================================= #
def _read(path):
    return pd.read_csv(path, sep=None, engine="python")


def run_pipeline(raw_path, summary_path, measure_prefix="CT",
                 region_cols=None, n_perm=N_PERM, n_jobs=-1, sign=SIGN,
                 sex=None,
                 plot_pval="pointestimate", plot_metric="meanage",
                 cortical_png=None, results_csv=None, outline=True,
                 outline_width=2, show_all=False,
                 fdr_alpha=FDR_ALPHA, seed=RANDOM_SEED, do_cortical_plot=True,
                 summary_d_col=None, summary_region_col=None,
                 summary_start_col=None, summary_meanage_col=None):
    """Run the test for one measure. Returns (results_df, null_array).

    plot_metric : "meanage" (surface colored by peak window mean age, default)
                  or "d" (colored by peak Cohen's d).
    plot_pval   : "pointestimate" or "bootstrap" (which p decides significance).
    results_csv : if given, the results table is written there.
    summary_*_col : override the bootstrap-summary column names (defaults to the
                    module SUMMARY_* config). The d column may
                    also be passed as "auto" to pick the single 'Mean_Cohen_d_*'
                    column in the file (handy for BD-named summaries).
    """
    # resolve summary column names (fall back to module config)
    d_col   = summary_d_col       or SUMMARY_D_COL
    reg_col = summary_region_col  or SUMMARY_REGION_COL
    st_col  = summary_start_col   or SUMMARY_START_COL
    ma_col  = summary_meanage_col or SUMMARY_MEANAGE_COL

    raw = _read(raw_path)
    summ = _read(summary_path) if summary_path else None   # bootstrap optional
    if summ is not None and str(d_col).lower() == "auto":
        cand = [c for c in summ.columns if c.startswith("Mean_Cohen_d_")]
        if len(cand) != 1:
            raise SystemExit(f"summary_d_col='auto' needs exactly one "
                             f"'Mean_Cohen_d_*' column, found {cand}")
        d_col = cand[0]
        print(f"  [summary] auto-detected d column: {d_col}")

    if region_cols is None:
        region_cols = [c for c in raw.columns
                       if c.startswith(f"lh.{measure_prefix}.")
                       or c.startswith(f"rh.{measure_prefix}.")]
    if not region_cols:
        raise SystemExit(f"No region columns found for prefix {measure_prefix}.")
    print(f"{len(region_cols)} regions | {n_perm} permutations | SIGN={sign}"
          f" | sex={sex or 'all'}")

    raw = subset_by_sex(raw, sex)
    arr = prepare_arrays(raw, region_cols)
    mage_lookup = window_mean_ages(arr["age0"])       # start_age -> empirical mean age

    # observed point-estimate peak (same rule as null)
    obs_pe_signed, argwin_pe = _peak_all_np(arr["Y"], arr["grp"], arr["age0"],
                                            arr["sex"], arr["site_full"], sign)
    # observed bootstrap peak from summary (optional; skipped if no summary file)
    if summ is not None:
        ss = summ.copy()
        ss["_signed"] = sign * ss[d_col]
        boot_idx = ss.groupby(reg_col)["_signed"].idxmax()
        boot_peak = ss.loc[boot_idx].set_index(reg_col)
        obs_boot_signed = boot_peak["_signed"].to_dict()
        boot_start = boot_peak[st_col].to_dict() if st_col in ss else {}
        boot_mage = (boot_peak[ma_col].to_dict()
                     if ma_col in ss else {})
    else:
        obs_boot_signed, boot_start, boot_mage = {}, {}, {}

    null = run_permutation_parallel(arr, sign=sign, n_perm=n_perm,
                                    n_jobs=n_jobs, seed=seed)

    rows = []
    for j, r in enumerate(region_cols):
        nj = null[:, j]
        pe_s = obs_pe_signed[j]
        bt_s = obs_boot_signed.get(r, np.nan)
        pe_start = START_AGES[argwin_pe[j]] if argwin_pe[j] >= 0 else np.nan
        rows.append({
            "region": r,
            "peak_d_pointest": sign * pe_s if np.isfinite(pe_s) else np.nan,
            "peak_win_pointest_start": pe_start,
            "peak_win_pointest_meanage": mage_lookup.get(pe_start, np.nan),
            "p_pointest": perm_pvalue(pe_s, nj),
            "peak_d_boot": sign * bt_s if np.isfinite(bt_s) else np.nan,
            "peak_win_boot_start": boot_start.get(r, np.nan),
            "peak_win_boot_meanage": boot_mage.get(r, np.nan),
            "p_boot": perm_pvalue(bt_s, nj),
            "null_mean_d": sign * np.nanmean(nj),
        })
    res = pd.DataFrame(rows)
    res["p_pointest_fdr"] = bh_fdr(res["p_pointest"].values)
    res["p_boot_fdr"]     = bh_fdr(res["p_boot"].values)
    res["sig_pointest"] = res["p_pointest_fdr"] < fdr_alpha
    res["sig_boot"]     = res["p_boot_fdr"] < fdr_alpha

    which = "pointest" if plot_pval.startswith("point") else "boot"
    n_sig = int(res[f"sig_{which}"].sum())
    print(f"Significant ({plot_pval}, FDR<{fdr_alpha}): {n_sig}/{len(res)}")

    if results_csv:
        res.to_csv(results_csv, index=False)
        print(f"  wrote {results_csv}")

    if do_cortical_plot and cortical_png and n_sig:
        if plot_metric == "meanage":
            value_col = f"peak_win_{which}_meanage"
            cmap = "viridis"
            ages = [a for a in mage_lookup.values() if np.isfinite(a)]
            color_range = (float(np.nanmin(ages)), float(np.nanmax(ages)))
            cbar_ticks = _auto_age_ticks(color_range)     # every 2 years
            cbar_label = "peak window mean age (years)"
        else:
            value_col = f"peak_d_{which}"; cmap = "RdBu_r"
            lim = float(np.nanmax(np.abs(res.loc[res[f"sig_{which}"], value_col])))
            color_range = (-lim, lim)
            cbar_ticks = list(np.round(np.linspace(-lim, lim, 7), 2))
            cbar_label = "Cohen's d"
        sig_regions = [row.region for row in res.itertuples()
                       if getattr(row, f"sig_{which}")]
        if show_all:
            # colour EVERY region; trace black outlines around significant ones
            vals = {row.region: getattr(row, value_col) for row in res.itertuples()}
            outline_regions = sig_regions
        else:
            vals = {r: v for r, v in
                    ((row.region, getattr(row, value_col)) for row in res.itertuples())
                    if r in set(sig_regions)}
            outline_regions = None
        try:
            plot_cortical_sig(vals, measure_prefix, cortical_png,
                              color_range=color_range, cmap=cmap,
                              cbar_ticks=cbar_ticks, cbar_label=cbar_label,
                              outline=outline, outline_width=outline_width,
                              outline_regions=outline_regions)
        except Exception as e:
            print(f"[cortical plot skipped] {e}")

    return res, null


# ==================== MULTI-PANEL PEAK-AGE FIGURE ============================ #
def _region_cols_for(raw, measure):
    """Region columns for a measure. 'subcortical' -> SubcGM.*, else lh/rh.<PREFIX>.*"""
    if str(measure).lower().startswith(("subc", "sub")):
        return [c for c in raw.columns if c.startswith("SubcGM.")]
    return [c for c in raw.columns
            if c.startswith(f"lh.{measure}.") or c.startswith(f"rh.{measure}.")]


def run_all_combinations(raw_path, summary_path,
                         measures=("CT", "SA", "GM", "subcortical"),
                         sexes=(None, "Male", "Female"),
                         n_perm=N_PERM, n_jobs=-1, out_prefix="results", **kw):
    """Run every measure x group combination. Returns {(measure, group): results_df}
    and writes one CSV per combination.
    NOTE: p_boot is only meaningful for the whole-group runs, because your
    bootstrap summary was computed on the full sample. Use p_pointest for
    sex-stratified rows."""
    raw = _read(raw_path)
    out = {}
    for meas in measures:
        rc = _region_cols_for(raw, meas)
        if not rc:
            print(f"[skip] no columns for measure '{meas}'"); continue
        for sx in sexes:
            gname = sx or "All"
            print(f"\n===== {meas} | {gname} =====")
            res, _ = run_pipeline(raw_path, summary_path, measure_prefix=meas,
                                  region_cols=rc, sex=sx, n_perm=n_perm,
                                  n_jobs=n_jobs, do_cortical_plot=False,
                                  results_csv=f"{out_prefix}_{meas}_{gname}.csv", **kw)
            out[(meas, gname)] = res
    return out


def _render_cortical_panel(vals, prefix, out_png, color_range, outline_width=2,
                           cmap="viridis", outline=True):
    """One cortical brain image, no colorbar (for montage)."""
    from enigmatoolbox.utils.parcellation import parcel_to_surface
    from enigmatoolbox.plotting import plot_cortical
    vec = build_dk_vector(vals, prefix)
    surf = parcel_to_surface(vec, "aparc_fsa5")
    brain_cmap, brain_range = cmap, color_range
    if outline and np.isfinite(vec).any():
        try:
            surf, brain_cmap, brain_range = _apply_outlines(
                vec, surf, color_range, cmap, width=outline_width)
        except Exception as e:
            print(f"  [outline skipped: {e}]")
    surf = _blank_medial_wall(surf)          # non-cortex -> grey, never coloured
    plot_cortical(array_name=surf, surface_name="fsa5", size=(900, 400),
                  cmap=brain_cmap, color_bar=False, color_range=brain_range,
                  nan_color=(0.75, 0.75, 0.75, 1), screenshot=True, filename=out_png)


def _render_subcortical_panel(vals, out_png, color_range, cmap="viridis"):
    """One subcortical image, no colorbar (for montage)."""
    from enigmatoolbox.plotting import plot_subcortical
    vec = build_subcortical_vector(vals)
    plot_subcortical(array_name=vec, ventricles=SUBCORT_VENTRICLES, size=(900, 400),
                     cmap=cmap, color_bar=False, color_range=color_range,
                     nan_color=(0.75, 0.75, 0.75, 1), screenshot=True, filename=out_png)


def make_peak_age_panels(results, out_png="peak_age_panels.png",
                         measures=("CT", "SA", "GM", "subcortical"),
                         groups=("All", "Male", "Female"),
                         value_col="peak_win_pointest_meanage",
                         sig_col="sig_pointest", age_range=(18.0, 50.0),
                         outline_width=2, tick_step=2, cmap="viridis",
                         row_labels=None, dpi=200):
    """Grid figure: rows = measures, cols = groups. Each panel shows ONLY the
    regions significant in that run, coloured by peak window mean age.
    All panels share ONE colour scale so peak ages are comparable across them.
    age_range defaults to (18, 50) = the full span your windows cover; pass
    None to auto-fit to the observed peak ages instead."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os

    # ---- shared colour range across every panel ----
    if age_range is None:
        vals = []
        for r in results.values():
            v = r.loc[r[sig_col], value_col].values
            vals.extend(v[np.isfinite(v)].tolist())
        if not vals:
            print("No significant regions in ANY panel — nothing to plot."); return
        age_range = (float(np.min(vals)), float(np.max(vals)))
    print(f"shared colour scale: {age_range[0]:.1f} – {age_range[1]:.1f} years")

    tmp_files, panels = [], {}
    for meas in measures:
        for grp in groups:
            res = results.get((meas, grp))
            if res is None:
                panels[(meas, grp)] = None; continue
            sig = res[res[sig_col]]
            vals = {row.region: getattr(row, value_col) for row in sig.itertuples()}
            n_sig = len(vals)
            tmp = f"_panel_{meas}_{grp}.png"
            try:
                if str(meas).lower().startswith(("subc", "sub")):
                    _render_subcortical_panel(vals, tmp, age_range, cmap=cmap)
                else:
                    _render_cortical_panel(vals, meas, tmp, age_range,
                                           outline_width=outline_width, cmap=cmap)
                panels[(meas, grp)] = (tmp, n_sig)
                tmp_files.append(tmp)
            except Exception as e:
                print(f"  [panel {meas}/{grp} failed] {e}")
                panels[(meas, grp)] = None
            print(f"  {meas:12s} {grp:7s}: {n_sig} significant")

    # ---- assemble grid ----
    nr, nc = len(measures), len(groups)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.1 * nr))
    axes = np.atleast_2d(axes)
    if row_labels is None:
        row_labels = {m: ("Subcortical" if str(m).lower().startswith(("subc", "sub"))
                          else m) for m in measures}
    for i, meas in enumerate(measures):
        for j, grp in enumerate(groups):
            ax = axes[i, j]; ax.axis("off")
            p = panels.get((meas, grp))
            if p is not None and os.path.exists(p[0]):
                ax.imshow(plt.imread(p[0]))
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=8, color="0.6", transform=ax.transAxes)
            if i == 0:
                ax.set_title(grp, fontsize=12, pad=12)
            if j == 0:
                ax.text(-0.04, 0.5, row_labels.get(meas, meas), rotation=90,
                        va="center", ha="center", fontsize=12,
                        transform=ax.transAxes)

    # ---- one shared colourbar ----
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(*age_range), cmap=cmap)
    sm.set_array([])
    ticks = np.arange(np.floor(age_range[0]), np.ceil(age_range[1]) + .001, tick_step)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015,
                        ticks=ticks)
    cbar.set_label("peak window mean age (years)")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for f in tmp_files:
        try: os.remove(f)
        except OSError: pass
    print(f"saved {out_png}")



if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        print("Imported in a notebook; call run_pipeline(...) directly.")
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--raw", required=True)
        ap.add_argument("--summary", required=True)
        ap.add_argument("--measure-prefix", default="CT")
        ap.add_argument("--nperm", type=int, default=N_PERM)
        ap.add_argument("--njobs", type=int, default=-1)
        ap.add_argument("--plot-pval", default="pointestimate",
                        choices=["pointestimate", "bootstrap"])
        ap.add_argument("--plot-metric", default="meanage",
                        choices=["meanage", "d"])
        ap.add_argument("--out", default="peak_perm_results.csv")
        ap.add_argument("--cortical-png", default="dk_significant.png")
        a = ap.parse_args()
        run_pipeline(a.raw, a.summary, measure_prefix=a.measure_prefix,
                     n_perm=a.nperm, n_jobs=a.njobs, plot_pval=a.plot_pval,
                     plot_metric=a.plot_metric, cortical_png=a.cortical_png,
                     results_csv=a.out)
        print(f"Wrote {a.out}")

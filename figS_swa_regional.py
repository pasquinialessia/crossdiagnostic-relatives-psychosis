"""
Figure S3 — SWA regional: peak-window mean age brain maps
=============================================================
For each disorder × brain phenotype, every parcel is colored by the mean age
of the sliding-window in which the case-control effect was strongest (the
"when" of the peak effect). Colorscale: viridis, 18–50 yr.

Rows:  SCZ, SAD, BD
Cols:  CT, GMV, SA, Subcortical

Data loaded on the fly from ../data/dataset.csv via data_utils.
No pre-computed CSVs required.

Run via run_vtk_figures.sh:
    DISPLAY=:99 python code/figS_swa_regional.py
Output: figures/figureS3_swa_sensitivity.png
"""
import os, sys, io, warnings, importlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as _fm
from PIL import Image, ImageDraw, ImageFont
import vtk, vtk.util.numpy_support as ns, nibabel as nib
warnings.filterwarnings("ignore")

# ── swa_lib ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swa_lib; importlib.reload(swa_lib)
import swa_lib as swp

from data_utils import build_diagnosis_sample

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
FIG_DIR  = os.path.join(BASE, "figures")
TMP      = os.path.join(BASE, "_fS3tmp")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# ── VTK 9 patch ──────────────────────────────────────────────────────────────
from enigmatoolbox.vtk_interface.wrappers.lookup_table import BSLookupTable
def _fixed_SetTable(self, table):
    if isinstance(table, np.ndarray):
        self.VTKObject.SetTable(ns.numpy_to_vtk(
            table.astype(np.uint8), deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR))
    elif hasattr(table, "VTKObject"):
        self.VTKObject.SetTable(table.VTKObject)
    else:
        self.VTKObject.SetTable(table)
BSLookupTable.SetTable = _fixed_SetTable

from enigmatoolbox.vtk_interface.wrappers import BSPolyData
import enigmatoolbox.datasets as _eds
import enigmatoolbox.plotting.surface_plotting as _splot
_PKG = os.path.dirname(__import__("enigmatoolbox").__file__)

def _build_surf(path):
    g = nib.load(path)
    coords = g.darrays[0].data.astype(np.float64)
    faces  = g.darrays[1].data.astype(np.int64)
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(coords))
    n = len(faces)
    carr  = np.hstack([np.full((n, 1), 3, dtype=np.int64), faces]).flatten()
    cells = vtk.vtkCellArray()
    cells.SetCells(n, ns.numpy_to_vtkIdTypeArray(carr))
    pd_ = vtk.vtkPolyData()
    pd_.SetPoints(pts); pd_.SetPolys(cells)
    return BSPolyData(pd_)

_LH  = _build_surf(os.path.join(_PKG, "datasets/surfaces/fsa5_lh.surf.gii"))
_RH  = _build_surf(os.path.join(_PKG, "datasets/surfaces/fsa5_rh.surf.gii"))
_SLH = _build_surf(os.path.join(_PKG, "datasets/surfaces/sctx_lh.gii"))
_SRH = _build_surf(os.path.join(_PKG, "datasets/surfaces/sctx_rh.gii"))
_eds.load_fsa5          = lambda as_sphere=False: (_LH, _RH)
_splot.load_fsa5        = _eds.load_fsa5
_eds.load_subcortical   = lambda with_normals=False, join=False: (_SLH, _SRH)
_splot.load_subcortical = _eds.load_subcortical

# ── config ───────────────────────────────────────────────────────────────────
DISORDERS = [
    ("scz", "SCZ",             "Schizophrenia"),
    ("sad", "Schizoaffective", "Schizoaffective"),
    ("bd",  "BD",              "Bipolar Disorder"),
]
MEASURES = [
    ("CT",          "CT"),
    ("GMV",         "GM"),
    ("SA",          "SA"),
    ("Subcortical", "subcortical"),
]

CMAP        = "viridis"
COLOR_RANGE = (18.0, 50.0)   # years — spans entire age window range

# ── load data once ───────────────────────────────────────────────────────────
print("Loading dataset.csv ...")
_all = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
print(f"  {len(_all)} rows, dx: {dict(_all['dx'].value_counts())}")

# ── compute peak-window mean age per region ───────────────────────────────────
def compute_peak_mean_age(sub_df, meas_key):
    """
    For every region in `meas_key`, find the SWA window with the strongest
    case-control effect and return its empirical mean subject age.
    Returns: {col_name: mean_age_years}
    """
    rc = swp._region_cols_for(sub_df, meas_key)
    if not rc:
        return {}
    arr = swp.prepare_arrays(sub_df, rc)
    mage_lookup = swp.window_mean_ages(arr["age0"])
    _, argwin = swp._peak_all_np(
        arr["Y"], arr["grp"], arr["age0"],
        arr["sex"], arr["site_full"], swp.SIGN)
    return {
        col: (mage_lookup.get(swp.START_AGES[argwin[j]], np.nan)
              if argwin[j] >= 0 else np.nan)
        for j, col in enumerate(rc)
    }

# ── render one brain panel (no colorbar) ────────────────────────────────────
def render_panel(vals, meas_key, tag):
    fp = os.path.join(TMP, f"{tag}.png")
    if meas_key.lower().startswith(("subc", "sub")):
        swp._render_subcortical_panel(vals, fp, COLOR_RANGE, cmap=CMAP)
    else:
        swp._render_cortical_panel(vals, meas_key, fp, COLOR_RANGE,
                                   outline_width=0, cmap=CMAP, outline=False)
    return fp

# ── main loop ────────────────────────────────────────────────────────────────
print("\nComputing SWA and rendering panels ...")
panel_imgs = {}   # (dxkey, meas_label) -> PIL Image

for dxkey, dx_value, dx_label in DISORDERS:
    print(f"\n  [{dx_label}]")
    sub_df = build_diagnosis_sample(_all, dx_value)
    n_clin = (sub_df["Condition_type"] == "Clinical").sum()
    n_ctrl = (sub_df["Condition_type"] == "Control").sum()
    print(f"    n={len(sub_df)} (clinical={n_clin}, control={n_ctrl})")

    for meas_label, meas_key in MEASURES:
        print(f"    {meas_label} ...", end=" ", flush=True)
        vals = compute_peak_mean_age(sub_df, meas_key)
        fp   = render_panel(vals, meas_key, f"{dxkey}_{meas_label}")
        panel_imgs[(dxkey, meas_label)] = Image.open(fp).convert("RGB")
        print("done")

# ── assemble montage ─────────────────────────────────────────────────────────
PW, PH   = next(iter(panel_imgs.values())).size
ROWS     = len(DISORDERS)
COLS     = len(MEASURES)
HDR_H    = int(PH * 0.22)
ROW_LW   = int(PW * 0.14)
COL_GAP  = int(PW * 0.025)
ROW_GAP  = int(PH * 0.02)
CB_H     = int(PH * 0.28)

W = ROW_LW + COLS * PW + (COLS - 1) * COL_GAP
H = HDR_H + ROWS * PH + (ROWS - 1) * ROW_GAP + CB_H + 16

canvas = Image.new("RGB", (W, H), "white")
draw   = ImageDraw.Draw(canvas)

_FONTB = _fm.findfont(_fm.FontProperties(family="DejaVu Sans", weight="bold"))
def font(sz):
    try:    return ImageFont.truetype(_FONTB, int(sz))
    except: return ImageFont.load_default()

# column headers
for ci, (meas_label, _) in enumerate(MEASURES):
    cx = ROW_LW + ci * (PW + COL_GAP) + PW // 2
    draw.text((cx, HDR_H // 2), meas_label, fill="black",
              font=font(int(HDR_H * 0.65)), anchor="mm")

# panels + row labels
for ri, (dxkey, dx_value, dx_label) in enumerate(DISORDERS):
    y = HDR_H + ri * (PH + ROW_GAP)
    draw.text((ROW_LW // 2, y + PH // 2), dx_label, fill="black",
              font=font(int(PW * 0.022)), anchor="mm")
    for ci, (meas_label, _) in enumerate(MEASURES):
        x = ROW_LW + ci * (PW + COL_GAP)
        canvas.paste(panel_imgs[(dxkey, meas_label)], (x, y))

# colorbar
fig_cb = plt.figure(figsize=(8, 0.65))
ax_cb  = fig_cb.add_axes([0.05, 0.45, 0.90, 0.45])
sm = plt.cm.ScalarMappable(
    norm=mcolors.Normalize(COLOR_RANGE[0], COLOR_RANGE[1]), cmap=CMAP)
cb = plt.colorbar(sm, cax=ax_cb, orientation="horizontal")
cb.set_label("Mean age of peak-effect window (years)", fontsize=12)
cb.set_ticks(np.arange(20, 51, 5))
buf = io.BytesIO()
fig_cb.savefig(buf, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_cb)
buf.seek(0)
cb_img = Image.open(buf).convert("RGB")
cb_w   = int(W * 0.55)
cb_img = cb_img.resize((cb_w, int(cb_img.height * cb_w / cb_img.width)),
                        Image.Resampling.LANCZOS)
y_cb = HDR_H + ROWS * PH + (ROWS - 1) * ROW_GAP + 8
canvas.paste(cb_img, ((W - cb_w) // 2, y_cb))

out = os.path.join(FIG_DIR, "figureS3_swa_sensitivity.png")
canvas.save(out, dpi=(300, 300))
print(f"\nSaved: {out}")

# ============================================================
# STRONG SIMPEG 3D MAGNETIC SUSCEPTIBILITY INVERSION
# TIFF = observed RTP residual magnetic anomaly only
# Borehole susceptibility = depth-only constraint
# Goal: predicted magnetic anomaly closely matches observed residual
# ============================================================

import os
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm, colors as mcolors
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import plotly.graph_objects as go
from pathlib import Path

from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from discretize import TensorMesh

from simpeg import (
    maps, data, data_misfit, regularization,
    optimization, inverse_problem, inversion, directives
)

from simpeg.potential_fields import magnetics


# ============================================================
# ROBUST INPUT FILE PATH HANDLING
# ============================================================
# VS Code sometimes runs Python from its own installation folder instead of
# the folder where this .py file is saved. Relative filenames like
# "RBGUT_5_predicted_susceptibility_table (1).xlsx" then fail even when the
# files are beside the code. This block forces all relative input filenames
# to be resolved from the script folder first.

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
os.chdir(BASE_DIR)

print("Script folder:", BASE_DIR)
print("Current working directory:", Path.cwd())


def _canonical_filename(name):
    """Normalize file names for tolerant matching: spaces/case/05-vs-5."""
    s = Path(str(name)).name.lower()

    def _fix_bh_id(match):
        return f"rbgut_{int(match.group(1)):02d}"

    s = re.sub(r"rbgut[\s_\-]*(\d+)", _fix_bh_id, s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def get_file_path(filename, extra_dirs=None):
    """
    Return a real Path for an input file.

    Search order:
      1) absolute path if supplied
      2) same folder as this script
      3) current working directory
      4) optional extra directories
      5) /mnt/data fallback, useful only in ChatGPT/container runs

    It also performs case-insensitive and 05-vs-5 tolerant matching.
    """
    filename = str(filename)
    raw_path = Path(filename)

    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    search_dirs = [BASE_DIR, Path.cwd()]

    if extra_dirs is not None:
        if isinstance(extra_dirs, (str, Path)):
            extra_dirs = [extra_dirs]
        search_dirs.extend(Path(d) for d in extra_dirs)

    # ChatGPT/container fallback. Harmless on Windows if it does not exist.
    search_dirs.append(Path("/mnt/data"))

    # Remove duplicates while preserving order
    unique_dirs = []
    seen = set()
    for d in search_dirs:
        try:
            d = Path(d).resolve()
        except Exception:
            d = Path(d)
        if d not in seen and d.exists():
            unique_dirs.append(d)
            seen.add(d)

    # 1) direct relative match in known folders
    for d in unique_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate

    target_lower = Path(filename).name.lower()
    target_canon = _canonical_filename(filename)

    # 2) case-insensitive exact filename match
    for d in unique_dirs:
        for f in d.glob("*"):
            if f.is_file() and f.name.lower() == target_lower:
                return f

    # 3) tolerant canonical match, useful for RBGUT_5 vs RBGUT_05
    close_matches = []
    for d in unique_dirs:
        for f in d.glob("*"):
            if not f.is_file():
                continue
            if _canonical_filename(f.name) == target_canon:
                return f

            # Keep similar files for the error message
            if "rbgut" in f.name.lower() or f.suffix.lower() in [".xlsx", ".xls", ".tif", ".tiff"]:
                close_matches.append(f.name)

    suggestions = "\n".join(sorted(set(close_matches))[:80])
    raise FileNotFoundError(
        "\nInput file not found:\n"
        f"  {filename}\n\n"
        "Searched in:\n"
        + "\n".join(f"  {d}" for d in unique_dirs)
        + "\n\nPossible similar input files found:\n"
        + (suggestions if suggestions else "  None")
        + "\n\nFix: put the Excel/TIFF files in the same folder as this .py file, "
          "or edit the filename in the boreholes dictionary so it exactly matches."
    )


# ============================================================
# HISTOGRAM-EQUALIZED COLOUR SCALE WITH REAL VALUE COLORBARS
# ============================================================
# This is NOT 0-1 normalization.
# It creates quantile-spaced colour boundaries in the original data units.
# Therefore colour contrast is histogram-equalized, but the colourbar labels
# remain actual nT or actual susceptibility values.

def hist_equalized_boundaries(values, n_bins=64):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise ValueError("No finite values found for histogram-equalized colour scale.")

    if np.nanmin(vals) == np.nanmax(vals):
        v = float(np.nanmin(vals))
        return np.array([v - 1e-12, v + 1e-12])

    q = np.linspace(0.0, 100.0, n_bins + 1)
    boundaries = np.nanpercentile(vals, q)

    # Remove duplicate boundaries caused by repeated values.
    boundaries = np.unique(boundaries)

    if boundaries.size < 2:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        boundaries = np.linspace(vmin, vmax, 2)

    return boundaries


def hist_equalized_norm(values, n_bins=64, cmap_name="turbo"):
    boundaries = hist_equalized_boundaries(values, n_bins=n_bins)
    cmap = plt.get_cmap(cmap_name)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    return boundaries, norm


def plotly_hist_equalized_colorscale(values, cmap_name="turbo", n_bins=64):
    boundaries = hist_equalized_boundaries(values, n_bins=n_bins)
    vmin = float(boundaries[0])
    vmax = float(boundaries[-1])

    if vmax == vmin:
        vmax = vmin + 1e-12

    mpl_cmap = cm.get_cmap(cmap_name)
    positions = (boundaries - vmin) / (vmax - vmin)

    colorscale = []
    n = len(boundaries)

    for i, pos in enumerate(positions):
        rgba = mpl_cmap(i / max(n - 1, 1))
        rgb = mcolors.to_rgb(rgba)
        rgb255 = tuple(int(round(c * 255)) for c in rgb)
        colorscale.append([float(pos), f"rgb{rgb255}"])

    return colorscale, vmin, vmax, boundaries

# ============================================================
# FILES
# ============================================================

mag_tiff = get_file_path("RTP_residual_upward_1000m.tif")  # RTP residual anomaly from upward continuation at 1000 m
print("Using magnetic TIFF:", mag_tiff)

boreholes = {
    "RBGUT-05": {
        "file": "RBGUT_5_predicted_susceptibility_table (1).xlsx",
        "lon": 72.416023,
        "lat": 25.597145,
        "sand_bottom": 111.0,
        "drilled_depth": 140.0,
    },
    "RBGUT-04": {
        "file": "RBGUT_04_predicted_susceptibility_table (2).xlsx",
        "lon": 72.412617,
        "lat": 25.591939,
        "sand_bottom": 0.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-14": {
        "file": "RBGUT_14_predicted_susceptibility_table (1).xlsx",
        "lon": 72.423984,
        "lat": 25.595341,
        "sand_bottom": 11.5,
        "drilled_depth": 125.0,
    },
    "RBGUT-18": {
        "file": "RBGUT_18_predicted_susceptibility_table.xlsx",
        "lon": 72.428653,
        "lat": 25.595517,
        "sand_bottom": 0.0,
        "drilled_depth": 140.0,
    },
    "RBGUT-10": {
        "file": "RBGUT_10_predicted_susceptibility_table (2).xlsx",
        "lon": 72.419539,
        "lat": 25.596867,
        "sand_bottom": 71.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-11": {
        "file": "RBGUT_11_predicted_susceptibility_table (2).xlsx",
        "lon": 72.419996,
        "lat": 25.593534,
        "sand_bottom": 80.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-13": {
        "file": "RBGUT_13_predicted_susceptibility_table (2).xlsx",
        "lon": 72.423984,
        "lat": 25.597144,
        "sand_bottom": 69.9,
        "drilled_depth": 125.0,
    },
    "RBGUT-17": {
        "file": "RBGUT_17_predicted_susceptibility_table.xlsx",
        "lon": 72.427963,
        "lat": 25.597145,
        "sand_bottom": 17.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-19": {
        "file": "RBGUT_19_predicted_susceptibility_table.xlsx",
        "lon": 72.427963,
        "lat": 25.593534,
        "sand_bottom": 0.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-7": {
        "file": "RBGUT_7_predicted_susceptibility_table (2).xlsx",
        "lon": 72.415306,
        "lat": 25.593222,
        "sand_bottom": 29.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-01": {
        "file": "RBGUT_01_predicted_susceptibility_table (1).xlsx",
        "lon": 72.412046,
        "lat": 25.597145,
        "sand_bottom": 65.0,
        "drilled_depth": 125.0,
    },
    "RBGUT-03": {
        "file": "RBGUT_03_predicted_susceptibility_table (2).xlsx",
        "lon": 72.412030,
        "lat": 25.593534,
        "sand_bottom": 26.3,
        "drilled_depth": 125.0,
    },
    "RBGUT-16": {
        "file": "RBGUT_16_predicted_susceptibility_table (2).xlsx",
        "lon": 72.424050,
        "lat": 25.591844,
        "sand_bottom": 48.5,
        "drilled_depth": 125.0,
    },
}


sand_top = 0.0
obs_height_m = 50.0


# ============================================================
# LOCAL LON-LAT TO METRE CONVERSION
# ============================================================

ref_lon = np.mean([b["lon"] for b in boreholes.values()])
ref_lat = np.mean([b["lat"] for b in boreholes.values()])

meter_per_deg_lat = 111320.0
meter_per_deg_lon = 111320.0 * np.cos(np.deg2rad(ref_lat))


def lonlat_to_xy(lon, lat):
    x = (lon - ref_lon) * meter_per_deg_lon
    y = (lat - ref_lat) * meter_per_deg_lat
    return x, y


def xy_to_lonlat(x, y):
    lon = ref_lon + x / meter_per_deg_lon
    lat = ref_lat + y / meter_per_deg_lat
    return lon, lat


for name in boreholes:
    x, y = lonlat_to_xy(boreholes[name]["lon"], boreholes[name]["lat"])
    boreholes[name]["x"] = x
    boreholes[name]["y"] = y
    print(name, "X(m) =", x, "Y(m) =", y)


# ============================================================
# GUGROT BLOCK RECTANGLE
# ============================================================

gugrot_block_lonlat = [
    (72.410000, 25.598472),
    (72.430000, 25.598472),
    (72.430000, 25.590417),
    (72.410000, 25.590417),
    (72.410000, 25.598472)
]

gugrot_lon_plot = [p[0] for p in gugrot_block_lonlat]
gugrot_lat_plot = [p[1] for p in gugrot_block_lonlat]
gugrot_z = [0] * len(gugrot_block_lonlat)


# ============================================================
# MAGNETIC FIELD PARAMETERS
# ============================================================

field_strength = 42451.2
declination = 0.0
inclination = 90.0

print("Inclination:", inclination)
print("Declination:", declination)
print("Field strength:", field_strength, "nT")


# ============================================================
# READ BOREHOLE OBSERVED SUSCEPTIBILITY
# ============================================================

constraint_xyz = []
constraint_sus = []
bh_plot_data = {}


def find_col(df, keys):
    for c in df.columns:
        cl = str(c).lower()
        if all(k.lower() in cl for k in keys):
            return c
    return None


for bh_name, info in boreholes.items():

    excel_path = get_file_path(info["file"])
    print(f"{bh_name}: using Excel file -> {excel_path.name}")
    df = pd.read_excel(excel_path)

    depth_col = find_col(df, ["depth"])

    obs_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "observed" in cl and "sus" in cl:
            obs_col = c
            break

    if depth_col is None:
        depth_col = df.columns[0]

    if obs_col is None:
        obs_col = df.columns[1]

    depth = pd.to_numeric(df[depth_col], errors="coerce").values
    obs_sus = pd.to_numeric(df[obs_col], errors="coerce").values

    mask = (~np.isnan(depth)) & (~np.isnan(obs_sus))

    depth = np.abs(depth[mask])
    obs_sus = obs_sus[mask]

    idx = np.argsort(depth)
    depth = depth[idx]
    obs_sus = obs_sus[idx]

    rock_mask = depth >= info["sand_bottom"]

    bh_depth = depth[rock_mask]
    bh_sus = obs_sus[rock_mask]

    if len(bh_depth) == 0:
        raise ValueError(f"No observed susceptibility values below sand for {bh_name}")

    xbh = info["x"]
    ybh = info["y"]

    for d, s in zip(bh_depth, bh_sus):
        constraint_xyz.append([xbh, ybh, -d])
        constraint_sus.append(s)

    bh_plot_data[bh_name] = {
        "depth": bh_depth,
        "sus": bh_sus,
        "x": xbh,
        "y": ybh,
        "lon": info["lon"],
        "lat": info["lat"],
        "sand_bottom": info["sand_bottom"],
        "drilled_depth": info["drilled_depth"]
    }

    print(f"{bh_name}: {len(bh_depth)} borehole depth constraint points")


constraint_xyz = np.array(constraint_xyz)
constraint_sus = np.array(constraint_sus)

print("Total borehole constraint points:", len(constraint_sus))


# ============================================================
# READ AND CROP RESIDUAL MAGNETIC TIFF
# TIFF IS ONLY OBSERVED DATA, NOT A SUSCEPTIBILITY CONSTRAINT
# ============================================================

# For RTP residual anomaly, crop exactly to Gugrot block.
# This preserves the observed RTP residual map used in the 2D observed-vs-predicted comparison.
crop_half_width_deg = 0.0

# Stronger inversion needs more magnetic data points
downsample = 1
max_data_points = 8000

lon_min = 72.410000 - crop_half_width_deg
lon_max = 72.430000 + crop_half_width_deg
lat_min = 25.590417 - crop_half_width_deg
lat_max = 25.598472 + crop_half_width_deg


with rasterio.open(mag_tiff) as src:

    print("TIFF CRS:", src.crs)
    print("TIFF bounds:", src.bounds)

    if src.crs is None:
        raise ValueError("TIFF CRS missing.")

    if src.crs.to_epsg() == 4326:
        crop_bounds = (lon_min, lat_min, lon_max, lat_max)
    else:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        x1, y1 = transformer.transform(lon_min, lat_min)
        x2, y2 = transformer.transform(lon_max, lat_max)

        crop_bounds = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        )

    window = from_bounds(*crop_bounds, transform=src.transform)

    mag_grid = src.read(1, window=window).astype(float)
    transform = src.window_transform(window)

    nodata = src.nodata

    if nodata is not None:
        mag_grid[mag_grid == nodata] = np.nan

    nrows, ncols = mag_grid.shape

    row_grid, col_grid = np.meshgrid(
        np.arange(nrows),
        np.arange(ncols),
        indexing="ij"
    )

    x_grid = transform.c + col_grid * transform.a + row_grid * transform.b
    y_grid = transform.f + col_grid * transform.d + row_grid * transform.e

    if src.crs.to_epsg() == 4326:
        lon_grid = x_grid
        lat_grid = y_grid
    else:
        transformer_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon_grid, lat_grid = transformer_back.transform(x_grid, y_grid)


mag_grid = mag_grid[::downsample, ::downsample]
lon_grid = lon_grid[::downsample, ::downsample]
lat_grid = lat_grid[::downsample, ::downsample]

lon = lon_grid.ravel()
lat = lat_grid.ravel()
tmi = mag_grid.ravel()

valid = np.isfinite(tmi)

lon = lon[valid]
lat = lat[valid]
tmi = tmi[valid]

# Keep the RTP residual anomaly in its original nT polarity and baseline.
# Do NOT zero-mean here, otherwise the RTP residual map can artificially produce
# negative/blue zones inside the cropped block.
# tmi = tmi - np.nanmean(tmi)

if len(tmi) == 0:
    raise ValueError("No magnetic anomaly points found in crop window.")

if len(tmi) > max_data_points:
    rng = np.random.default_rng(42)
    choose = rng.choice(len(tmi), size=max_data_points, replace=False)
    lon = lon[choose]
    lat = lat[choose]
    tmi = tmi[choose]

n_data = len(tmi)

print("Magnetic data points used:", n_data)
print("Observed RTP residual range:", np.nanmin(tmi), np.nanmax(tmi))


x_obs = (lon - ref_lon) * meter_per_deg_lon
y_obs = (lat - ref_lat) * meter_per_deg_lat
z_obs = np.ones_like(x_obs) * obs_height_m

receiver_locations = np.c_[x_obs, y_obs, z_obs]


# ============================================================
# DENSE 3D MESH - BALANCED FOR FIT + CLOUD PLOT
# ============================================================

dx = 50.0
dy = 50.0
dz = 1.0

nx = 70
ny = 70
nz = 500

mesh = TensorMesh(
    [[(dx, nx)], [(dy, ny)], [(dz, nz)]],
    x0=[
        -nx * dx / 2,
        -ny * dy / 2,
        -nz * dz
    ]
)
cc = mesh.cell_centers

Xc = cc[:, 0]
Yc = cc[:, 1]
Zc = cc[:, 2]

depth_cells = -Zc


# ============================================================
# UNDULATING SAND-ROCK BOUNDARY
# ============================================================

bh_xy = np.array([[b["x"], b["y"]] for b in boreholes.values()])
bh_sand = np.array([b["sand_bottom"] for b in boreholes.values()])


def sand_bottom_surface(x, y):

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)

    base = np.zeros_like(x_flat, dtype=float)
    wsum = np.zeros_like(x_flat, dtype=float)

    for (xbh, ybh), sb in zip(bh_xy, bh_sand):

        d2 = (x_flat - xbh)**2 + (y_flat - ybh)**2

        w = 1.0 / (d2 + 25.0)

        base += w * sb
        wsum += w

    base = base / wsum

    undulation = (
        6.0 * np.sin(x_flat / 180.0)
        + 4.0 * np.cos(y_flat / 160.0)
        + 3.0 * np.sin((x_flat + y_flat) / 220.0)
    )

    result = base + undulation

    return result.reshape(np.shape(x))


sand_base = sand_bottom_surface(Xc, Yc)

active_cells = (
    (Zc < 0)
    & (depth_cells >= sand_base)
)

nC = int(active_cells.sum())

print("Mesh cells:", mesh.nC)
print("Active cells below sand:", nC)

if nC == 0:
    raise ValueError("No active cells below sand boundary.")


# ============================================================
# MAGNETIC SURVEY AND SIMULATION
# ============================================================

receivers = magnetics.receivers.Point(
    receiver_locations,
    components=["tmi"]
)

source_field = magnetics.sources.UniformBackgroundField(
    receiver_list=[receivers],
    amplitude=field_strength,
    inclination=inclination,
    declination=declination
)

survey = magnetics.survey.Survey(source_field)

# Use a clean, local sensitivity-cache folder.
# This fixes the common SimPEG error that appears at np.save(sens_name, kernel)
# during BetaEstimate_ByEig / simulation.fields(model), especially on Windows
# when the default sensitivity path is missing, locked, or too large.
sensitivity_dir = os.path.join(os.getcwd(), "simpeg_sensitivity_cache_rtp")

if os.path.isdir(sensitivity_dir):
    try:
        shutil.rmtree(sensitivity_dir)
    except PermissionError:
        print("WARNING: old sensitivity cache is locked. Using a new cache folder.")
        sensitivity_dir = os.path.join(os.getcwd(), "simpeg_sensitivity_cache_rtp_new")

os.makedirs(sensitivity_dir, exist_ok=True)
print("Sensitivity cache folder:", sensitivity_dir)

simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    active_cells=active_cells,
    chiMap=maps.IdentityMap(nP=nC),
    store_sensitivities="disk",
    sensitivity_path=sensitivity_dir,
    sensitivity_dtype=np.float32
)


# ============================================================
# BOREHOLE CONSTRAINT CELLS
# Exact borehole-depth cells only
# No XY spreading
# No cylinder constraint
# ============================================================

active_cc = cc[active_cells]

tree = cKDTree(active_cc)

_, bh_cell_ids = tree.query(constraint_xyz)

tmp = pd.DataFrame({
    "cell": bh_cell_ids,
    "sus": constraint_sus
})

tmp = tmp.groupby("cell", as_index=False)["sus"].mean()

fixed_cells = tmp["cell"].values.astype(int)
fixed_values = tmp["sus"].values.astype(float)

print("Unique borehole constrained cells:", len(fixed_cells))


# ============================================================
# TIFF-GUIDED STARTING MODEL AND NEUTRAL REFERENCE MODEL
# ============================================================
# TIFF is magnetic anomaly in nT, not susceptibility.
# So here we only use its spatial pattern to initialize m0.
# mref remains neutral, based on borehole median susceptibility.

median_sus = max(np.nanmedian(constraint_sus), 1e-8)

# Reference model should remain neutral/geological
mref = np.ones(nC) * median_sus

# ============================================================
# INTERPOLATE TIFF RTP RESIDUAL ANOMALY ONTO ACTIVE CELLS
# ============================================================

obs_xy = np.c_[x_obs, y_obs]
active_xy = active_cc[:, :2]

tiff_on_cells = griddata(
    obs_xy,
    tmi,
    active_xy,
    method="linear"
)

tiff_nearest = griddata(
    obs_xy,
    tmi,
    active_xy,
    method="nearest"
)

# Fill NaN cells from nearest interpolation
bad = ~np.isfinite(tiff_on_cells)
tiff_on_cells[bad] = tiff_nearest[bad]

if not np.all(np.isfinite(tiff_on_cells)):
    raise ValueError("TIFF-guided m0 failed: non-finite values after interpolation.")

# ============================================================
# NORMALIZE TIFF ANOMALY PATTERN FROM 0 TO 1
# ============================================================
# High RTP anomaly -> higher starting susceptibility
# Low RTP anomaly  -> lower starting susceptibility

p_low, p_high = np.nanpercentile(tiff_on_cells, [5, 95])

if p_high <= p_low:
    tiff_norm = np.ones(nC) * 0.5
else:
    tiff_norm = (tiff_on_cells - p_low) / (p_high - p_low)
    tiff_norm = np.clip(tiff_norm, 0.0, 1.0)

# gamma > 1 makes high-anomaly core sharper in starting model
gamma = 1.25
tiff_norm = tiff_norm ** gamma

# ============================================================
# CONVERT NORMALIZED TIFF PATTERN TO SUSCEPTIBILITY RANGE
# ============================================================

sus_min_start = max(1e-5, 0.25 * median_sus)

sus_max_start = max(
    np.nanpercentile(constraint_sus, 90),
    3.0 * median_sus,
    0.05
)

# Keep starting model inside a realistic range
sus_upper_cap = max(np.nanmax(constraint_sus) * 1.2, 1.0)
sus_max_start = min(sus_max_start, sus_upper_cap)

tiff_sus_model = sus_min_start + tiff_norm * (sus_max_start - sus_min_start)

# ============================================================
# BLEND TIFF-GUIDED MODEL WITH NEUTRAL BACKGROUND
# ============================================================
# 0.0 = fully neutral m0
# 1.0 = fully TIFF-shaped m0
# Recommended: 0.4 to 0.6

blend_strength = 0.50

m0 = (
    (1.0 - blend_strength) * median_sus
    + blend_strength * tiff_sus_model
)

# ============================================================
# HARD-FIX BOREHOLE CELLS IN BOTH m0 AND mref
# ============================================================

for cell_id, val in zip(fixed_cells, fixed_values):
    locked_val = max(val, 1e-8)
    m0[cell_id] = locked_val
    mref[cell_id] = locked_val

print("TIFF-guided starting model m0 created.")
print("Neutral borehole-median reference model mref created.")
print("TIFF is used only to initialize m0, not as hard susceptibility constraint.")
print("Borehole constraints are only along exact borehole depths.")
print("m0 range:", np.nanmin(m0), np.nanmax(m0))
print("mref range:", np.nanmin(mref), np.nanmax(mref))
print("TIFF normalized range:", np.nanmin(tiff_norm), np.nanmax(tiff_norm))
print("TIFF-guided susceptibility range:", np.nanmin(tiff_sus_model), np.nanmax(tiff_sus_model))


# ============================================================
# DATA MISFIT FOR CURRENT RTP RESIDUAL DATA COUNT
# ============================================================
# n_data is now computed directly from the cropped RTP residual TIFF.
# If the Gugrot-only RTP crop gives ~308 magnetic points, the stopping
# target below automatically becomes phi_d ~= 308 instead of the old 2296.
#
# Keep the standard deviation tied to the observed RTP residual spread.
# 0.01 = strong fitting; increase to 0.02-0.05 only if inversion becomes noisy.

n_data = len(tmi)
std_fraction = 0.02
std = std_fraction * np.nanstd(tmi) * np.ones_like(tmi)

if not np.all(np.isfinite(std)) or np.nanstd(tmi) == 0:
    std = np.ones_like(tmi)

print("Number of RTP magnetic data points n_data:", n_data)
print("Using standard deviation fraction:", std_fraction)

data_obj = data.Data(
    survey=survey,
    dobs=tmi,
    standard_deviation=std
)

dmis = data_misfit.L2DataMisfit(
    data=data_obj,
    simulation=simulation
)

# ============================================================
# BOUNDS: FREE BETWEEN BOREHOLES, HARD-FIXED AT BOREHOLES
# ============================================================

lower = np.zeros(nC)
upper = np.ones(nC) * max(np.nanmax(constraint_sus) * 1.2, 1.0)

for cell_id, val in zip(fixed_cells, fixed_values):
    locked_val = max(val, 1e-8)
    lower[cell_id] = locked_val
    upper[cell_id] = locked_val + 1e-12

# ============================================================
# DEPTH PENALTY / DEPTH WEIGHTING
# ============================================================
# Purpose:
# Magnetic inversion can place excessive susceptibility too deep because
# deep and shallow bodies can produce similar long-wavelength anomalies.
# This penalty makes deeper cells slightly more expensive unless the data
# really require deep susceptibility.

active_depth = -active_cc[:, 2]

# Start penalty near 1 at shallow levels and increase with depth.
# penalty_strength controls how strongly deep cells are discouraged.
# Use 1.0 to 2.0 for mild-to-moderate depth penalty.
penalty_strength = 2.0
depth_scale = 160.0   # metres; close to borehole depth scale

depth_penalty = 1.0 + penalty_strength * (active_depth / depth_scale) ** 1.5

# Avoid excessive punishment at very deep cells
depth_penalty = np.clip(depth_penalty, 1.0, 8.0)

# SimPEG weights enter squared in the regularization, so use sqrt.
depth_penalty_weight = np.sqrt(depth_penalty)

print("Depth penalty range:", depth_penalty.min(), depth_penalty.max())

# ============================================================
# REGULARIZATION WITH DEPTH PENALTY
# ============================================================

# For a small RTP data set (~308 points), use slightly stronger smoothness
# than the older large-data run. This avoids unstable multi-lobe splitting
# while still allowing the observed RTP residual high to be recovered.
alpha_s = 2.0
alpha_x = 1.0
alpha_y = 1.0
alpha_z = 1.0

reg = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    reference_model=mref,
    alpha_s=alpha_s,
    alpha_x=alpha_x,
    alpha_y=alpha_y,
    alpha_z=alpha_z
)

print("Regularization alphas:")
print("alpha_s =", alpha_s, "alpha_x =", alpha_x, "alpha_y =", alpha_y, "alpha_z =", alpha_z)

# Apply depth penalty to the smallness term.
# This discourages unnecessary deep susceptibility while still allowing it.
# if magnetic data require it.
reg.set_weights(depth_penalty=depth_penalty_weight)

# ============================================================
# OPTIMIZATION
# ============================================================

# Optimization tuned for the current smaller RTP residual data set.
# With ~308 magnetic points, there is no need to chase the old 2296-target run.
opt = optimization.ProjectedGNCG(
    maxIter=90,
    maxIterLS=30,
    maxIterCG=120,
    tolCG=1e-5,
    lower=lower,
    upper=upper
)

# ============================================================
# DIRECTIVES
# ============================================================
class StopAtPhiD(directives.InversionDirective):
    def __init__(self, target_phi_d=None):
        super().__init__()
        self.target_phi_d = float(n_data if target_phi_d is None else target_phi_d)

    def endIter(self):
        phi_d_now = self.invProb.phi_d

        if phi_d_now <= self.target_phi_d:
            print("\n===================================")
            print(f"Stopping inversion: phi_d = {phi_d_now:.3e}")
            print(f"Target phi_d reached: {self.target_phi_d:.3e}")
            print("===================================\n")

            self.opt.stopNextIteration = True


class InversionConvergenceRecorder(directives.InversionDirective):
    """
    Records inversion convergence values after every iteration.

    Output columns:
      iteration : inversion iteration number
      phi_d     : data misfit
      phi_m     : model regularization / model objective
      beta      : trade-off parameter
      objective : phi_d + beta * phi_m
      target_phi_d : desired chi-factor target, usually n_data
    """
    def __init__(self, target_phi_d=None):
        super().__init__()
        self.target_phi_d = float(n_data if target_phi_d is None else target_phi_d)
        self.history = []
        self.iteration = 0

    def initialize(self):
        self.history = []
        self.iteration = 0

    def endIter(self):
        phi_d_now = float(getattr(self.invProb, "phi_d", np.nan))
        phi_m_now = float(getattr(self.invProb, "phi_m", np.nan))
        beta_now = float(getattr(self.invProb, "beta", np.nan))
        objective_now = phi_d_now + beta_now * phi_m_now

        self.history.append({
            "iteration": self.iteration,
            "phi_d": phi_d_now,
            "phi_m": phi_m_now,
            "beta": beta_now,
            "objective": objective_now,
            "target_phi_d": self.target_phi_d
        })

        print(
            f"Convergence record | Iter {self.iteration:03d}: "
            f"phi_d={phi_d_now:.6e}, phi_m={phi_m_now:.6e}, "
            f"beta={beta_now:.6e}, objective={objective_now:.6e}"
        )

        self.iteration += 1

# Target misfit is now tied to the actual number of RTP magnetic data points.
# Example: if n_data = 308, target_phi_d = 308.
# This replaces the old fixed value 2296.
target_phi_d = float(n_data)
print("Dynamic target phi_d set to:", target_phi_d)

# Recorder object is kept outside directive_list so it can be used later
# for saving the convergence PNG and Excel/CSV table.
convergence_recorder = InversionConvergenceRecorder(target_phi_d=target_phi_d)

directive_list = [
    directives.BetaEstimate_ByEig(beta0_ratio=3.0),
    directives.BetaSchedule(coolingFactor=2, coolingRate=1),
    convergence_recorder,
    StopAtPhiD(target_phi_d=target_phi_d),
    directives.UpdatePreconditioner()
]
# ============================================================
# INVERSION
# ============================================================

inv_prob = inverse_problem.BaseInvProblem(
    dmis,
    reg,
    opt
)

inv = inversion.BaseInversion(
    inv_prob,
    directiveList=directive_list
)

recovered_model = inv.run(m0)

print("SimPEG inversion completed.")

# ============================================================
# INVERSION CONVERGENCE PLOT AND TABLE
# ============================================================
# This plot is generated immediately after the inversion finishes.
# It shows how phi_d, phi_m, beta, and total objective changed with iteration.

convergence_df = pd.DataFrame(convergence_recorder.history)

if len(convergence_df) > 0:
    convergence_excel = "Inversion_Convergence_History.xlsx"
    convergence_csv = "Inversion_Convergence_History.csv"
    convergence_png = "Inversion_Convergence_Plot.png"

    convergence_df.to_excel(convergence_excel, index=False)
    convergence_df.to_csv(convergence_csv, index=False)

    fig_conv, axes_conv = plt.subplots(
        2,
        2,
        figsize=(16, 11),
        constrained_layout=True
    )

    ax = axes_conv[0, 0]
    ax.plot(
        convergence_df["iteration"],
        convergence_df["phi_d"],
        marker="o",
        linewidth=2,
        label="Data misfit phi_d"
    )
    ax.axhline(
        target_phi_d,
        linestyle="--",
        linewidth=2,
        label=f"Target phi_d = {target_phi_d:.0f}"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("phi_d")
    ax.set_title("Data Misfit Convergence")
    ax.grid(True, alpha=0.35)
    ax.legend()

    ax = axes_conv[0, 1]
    ax.plot(
        convergence_df["iteration"],
        convergence_df["phi_m"],
        marker="o",
        linewidth=2,
        label="Model objective phi_m"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("phi_m")
    ax.set_title("Model Objective Convergence")
    ax.grid(True, alpha=0.35)
    ax.legend()

    ax = axes_conv[1, 0]
    ax.plot(
        convergence_df["iteration"],
        convergence_df["beta"],
        marker="o",
        linewidth=2,
        label="Beta"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("beta")
    ax.set_yscale("log")
    ax.set_title("Trade-off Parameter Cooling")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()

    ax = axes_conv[1, 1]
    ax.plot(
        convergence_df["iteration"],
        convergence_df["objective"],
        marker="o",
        linewidth=2,
        label="Objective = phi_d + beta*phi_m"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective function")
    ax.set_title("Total Objective Function Convergence")
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig_conv.suptitle(
        "SimPEG Magnetic Inversion Convergence",
        fontsize=20,
        fontweight="bold"
    )

    plt.savefig(convergence_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved inversion convergence plot:", convergence_png)
    print("Saved inversion convergence Excel table:", convergence_excel)
    print("Saved inversion convergence CSV table:", convergence_csv)
else:
    print("WARNING: convergence recorder is empty; no convergence plot was saved.")


# ============================================================
# MISFIT CHECK
# ============================================================
pred_tmi = simulation.dpred(recovered_model)
print("Observed anomaly range:")
print(np.nanmin(tmi), np.nanmax(tmi))
print("Predicted anomaly range:")
print(np.nanmin(pred_tmi), np.nanmax(pred_tmi))
normalized_residual = (tmi - pred_tmi) / std
chi_factor = np.sum(normalized_residual**2) / len(tmi)
print("Final chi-factor:", chi_factor)

# ============================================================
# AMPLITUDE-CORRECTED PREDICTION CHECK
# This does not change inversion.
# It only checks whether the recovered shape matches observed data.
# ============================================================

A = np.vstack([pred_tmi, np.ones_like(pred_tmi)]).T
scale, offset = np.linalg.lstsq(A, tmi, rcond=None)[0]
pred_tmi_scaled = scale * pred_tmi + offset
print("Post-fit display scale:", scale)
print("Post-fit display offset:", offset)
error_tmi = tmi - pred_tmi
error_tmi_scaled = tmi - pred_tmi_scaled
rmse_2d = np.sqrt(np.mean(error_tmi ** 2))
mae_2d = np.mean(np.abs(error_tmi))
corr_2d = np.corrcoef(tmi, pred_tmi)[0, 1]
rmse_scaled = np.sqrt(np.mean(error_tmi_scaled ** 2))
mae_scaled = np.mean(np.abs(error_tmi_scaled))
corr_scaled = np.corrcoef(tmi, pred_tmi_scaled)[0, 1]
print("Raw predicted anomaly:")
print("RMSE:", rmse_2d)
print("MAE:", mae_2d)
print("Corr:", corr_2d)
print("Amplitude-corrected predicted anomaly:")
print("RMSE:", rmse_scaled)
print("MAE:", mae_scaled)
print("Corr:", corr_scaled)

# ============================================================
# 2D OBSERVED VS PREDICTED MAPS
# ============================================================

grid_lon = np.linspace(np.nanmin(lon), np.nanmax(lon), 180)
grid_lat = np.linspace(np.nanmin(lat), np.nanmax(lat), 120)
GLON, GLAT = np.meshgrid(grid_lon, grid_lat)
obs_grid = griddata(
    np.c_[lon, lat],
    tmi,
    (GLON, GLAT),
    method="linear"
)
pred_grid = griddata(
    np.c_[lon, lat],
    pred_tmi,
    (GLON, GLAT),
    method="linear"
)
pred_scaled_grid = griddata(
    np.c_[lon, lat],
    pred_tmi_scaled,
    (GLON, GLAT),
    method="linear"
)
obs_nearest = griddata(
    np.c_[lon, lat],
    tmi,
    (GLON, GLAT),
    method="nearest"
)
pred_nearest = griddata(
    np.c_[lon, lat],
    pred_tmi,
    (GLON, GLAT),
    method="nearest"
)
pred_scaled_nearest = griddata(
    np.c_[lon, lat],
    pred_tmi_scaled,
    (GLON, GLAT),
    method="nearest"
)
obs_grid[np.isnan(obs_grid)] = obs_nearest[np.isnan(obs_grid)]
pred_grid[np.isnan(pred_grid)] = pred_nearest[np.isnan(pred_grid)]
pred_scaled_grid[np.isnan(pred_scaled_grid)] = pred_scaled_nearest[np.isnan(pred_scaled_grid)]
err_grid = obs_grid - pred_grid
err_scaled_grid = obs_grid - pred_scaled_grid

# ============================================================
# 2D OBSERVED VS PREDICTED MAPS
# HISTOGRAM-EQUALIZED COLOUR SCALE WITH ACTUAL nT COLORBARS
# ============================================================

combined_mag_values = np.concatenate([
    obs_grid[np.isfinite(obs_grid)].ravel(),
    pred_scaled_grid[np.isfinite(pred_scaled_grid)].ravel()
])
# Quantile-spaced levels: histogram-equalized visual contrast,
# but levels are actual magnetic anomaly values in nT.
mag_levels = hist_equalized_boundaries(
    combined_mag_values,
    n_bins=64
)
err_levels = hist_equalized_boundaries(
    err_scaled_grid[np.isfinite(err_scaled_grid)],
    n_bins=64
)
fig2, ax2 = plt.subplots(
    1,
    3,
    figsize=(21, 6),
    constrained_layout=True
)
# ---------------- OBSERVED HIST-EQ, REAL nT ----------------
im0 = ax2[0].contourf(
    GLON,
    GLAT,
    obs_grid,
    levels=mag_levels,
    cmap="turbo",
    extend="both"
)
ax2[0].plot(gugrot_lon_plot, gugrot_lat_plot, "k-", linewidth=2)
ax2[0].set_title("Observed RTP Residual Magnetic Anomaly")
ax2[0].set_xlabel("Longitude")
ax2[0].set_ylabel("Latitude")
ax2[0].axis("equal")
plt.colorbar(
    im0,
    ax=ax2[0],
    label="Observed RTP residual magnetic anomaly (nT)"
)
# ---------------- PREDICTED HIST-EQ, REAL nT ----------------
im1 = ax2[1].contourf(
    GLON,
    GLAT,
    pred_scaled_grid,
    levels=mag_levels,
    cmap="turbo",
    extend="both"
)
ax2[1].plot(gugrot_lon_plot, gugrot_lat_plot, "k-", linewidth=2)
ax2[1].set_title(
    "Predicted Magnetic Anomaly\n"
    "from 3D Recovered Susceptibility"
)
ax2[1].set_xlabel("Longitude")
ax2[1].set_ylabel("Latitude")
ax2[1].axis("equal")
plt.colorbar(
    im1,
    ax=ax2[1],
    label="Amplitude-corrected predicted magnetic anomaly (nT)"
)
# ---------------- ERROR HIST-EQ, REAL nT ----------------
im2 = ax2[2].contourf(
    GLON,
    GLAT,
    err_scaled_grid,
    levels=err_levels,
    cmap="seismic",
    extend="both"
)
ax2[2].plot(gugrot_lon_plot, gugrot_lat_plot, "k-", linewidth=2)
ax2[2].set_title(
    f"Error Map: Observed - Predicted\n"
    f"RMSE={rmse_scaled:.3f}, MAE={mae_scaled:.3f}, Corr={corr_scaled:.3f}"
)
ax2[2].set_xlabel("Longitude")
ax2[2].set_ylabel("Latitude")
ax2[2].axis("equal")

plt.colorbar(
    im2,
    ax=ax2[2],
    label="Error: observed - predicted (nT)"
)

plt.savefig(
    "Observed_vs_Strongly_Predicted_Magnetic_Anomaly_HistEq_Actual_nT_Colorbar.png",
    dpi=300
)

plt.show()

comparison_2d = pd.DataFrame({
    "Longitude": lon,
    "Latitude": lat,
    "Observed_RTP_Residual_Magnetic_Anomaly": tmi,
    "Raw_Predicted_Magnetic_Anomaly": pred_tmi,
    "Amplitude_Corrected_Predicted_Magnetic_Anomaly": pred_tmi_scaled,
    "Raw_Error_Observed_minus_Predicted": error_tmi,
    "Amplitude_Corrected_Error": error_tmi_scaled
})

comparison_2d.to_excel(
    "Observed_vs_Strongly_Predicted_Magnetic_Anomaly_Error_Table.xlsx",
    index=False
)

print("Saved map:")
print("Observed_vs_Strongly_Predicted_Magnetic_Anomaly_HistEq_Actual_nT_Colorbar_sample.png")

print("Saved table:")
print("Observed_vs_Strongly_Predicted_Magnetic_Anomaly_Error_Table_sample.xlsx")


# ============================================================
# BOREHOLE COMPARISON TABLE
# ============================================================

recovered_at_bh = recovered_model[fixed_cells]

comparison = pd.DataFrame({
    "Cell_ID": fixed_cells,
    "Observed_Borehole_Susceptibility": fixed_values,
    "Recovered_Model_Value": recovered_at_bh,
    "Difference": fixed_values - recovered_at_bh
})

comparison.to_excel(
    "Observed_vs_Recovered_Borehole_Depth_Constraints_sample.xlsx",
    index=False
)

print("Saved borehole comparison table.")

# ============================================================
# OBSERVED vs DEPTH, PREDICTED vs DEPTH, AND % ERROR vs DEPTH
# Depth on X-axis, susceptibility / percentage error on Y-axis
# One plot per borehole, saved in separate Downloads folders
# ============================================================

obs_depth_dir = r"C:\Users\subhr\Downloads\Observed_Susceptibility_vs_Depth_Plots"
pred_depth_dir = r"C:\Users\subhr\Downloads\Predicted_Susceptibility_vs_Depth_Plots"
percent_error_depth_dir = r"C:\Users\subhr\Downloads\Percentage_Error_vs_Depth_Plots"

os.makedirs(obs_depth_dir, exist_ok=True)
os.makedirs(pred_depth_dir, exist_ok=True)
os.makedirs(percent_error_depth_dir, exist_ok=True)

table_output_dir = r"C:\Users\subhr\Downloads\Borehole_Observed_Predicted_Percentage_Error_Tables"
os.makedirs(table_output_dir, exist_ok=True)

for bh_name, pdata in bh_plot_data.items():

    depth = np.asarray(pdata["depth"], dtype=float)
    obs_sus = np.asarray(pdata["sus"], dtype=float)

    valid = np.isfinite(depth) & np.isfinite(obs_sus)
    depth = depth[valid]
    obs_sus = obs_sus[valid]

    if len(depth) < 2:
        print(f"Skipping {bh_name}: not enough observed depth points")
        continue

    idx = np.argsort(depth)
    depth = depth[idx]
    obs_sus = obs_sus[idx]

    # ========================================================
    # EXTRACT PREDICTED SUSCEPTIBILITY AT SAME BOREHOLE DEPTHS
    # ========================================================

    bh_xyz = np.c_[
        np.ones_like(depth) * pdata["x"],
        np.ones_like(depth) * pdata["y"],
        -depth
    ]

    _, pred_cell_ids = tree.query(bh_xyz)

    pred_sus = recovered_model[pred_cell_ids]
    pred_sus = np.asarray(pred_sus, dtype=float)

    valid2 = np.isfinite(depth) & np.isfinite(obs_sus) & np.isfinite(pred_sus)

    depth = depth[valid2]
    obs_sus = obs_sus[valid2]
    pred_sus = pred_sus[valid2]

    if len(depth) < 2:
        print(f"Skipping {bh_name}: not enough valid predicted values")
        continue

    # ========================================================
    # ERROR AND PERCENTAGE ERROR
    # ========================================================

    error_sus = obs_sus - pred_sus

    eps = 1e-12
    percent_error = np.where(
        np.abs(obs_sus) > eps,
        (error_sus / obs_sus)*100,
        np.nan
    )

    valid_err = np.isfinite(percent_error)

    percent_rmse = np.sqrt(
        np.nanmean(percent_error[valid_err] ** 2)
    )

    percent_mae = np.nanmean(
        np.abs(percent_error[valid_err])
    )

    if np.std(obs_sus) > 0 and np.std(pred_sus) > 0:
        corr = np.corrcoef(obs_sus, pred_sus)[0, 1]
    else:
        corr = np.nan

    # ========================================================
    # SAVE COMPARISON TABLE
    # ========================================================

    out_table = pd.DataFrame({
        "Depth_m": depth,
        "Observed_Susceptibility_SI": obs_sus,
        "Predicted_Susceptibility_SI": pred_sus,
        "Error_Observed_minus_Predicted": error_sus,
        "Percentage_Error_Observed_minus_Predicted": percent_error
    })

    table_file = os.path.join(
        table_output_dir,
        f"{bh_name}_Observed_Predicted_Percentage_Error_vs_Depth.xlsx"
    )

    out_table.to_excel(table_file, index=False)

    # ========================================================
    # BAR WIDTH / TEXT OFFSET
    # ========================================================

    if len(depth) > 1:
        spacing = np.median(np.diff(np.sort(depth)))
        bar_width = spacing * 0.75
    else:
        bar_width = 0.5

    obs_offset = 0.025 * (np.nanmax(obs_sus) - np.nanmin(obs_sus))
    if obs_offset == 0:
        obs_offset = 0.001

    pred_offset = 0.025 * (np.nanmax(pred_sus) - np.nanmin(pred_sus))
    if pred_offset == 0:
        pred_offset = 0.001

    # ========================================================
    # 1) OBSERVED SUSCEPTIBILITY vs DEPTH
    # ========================================================

    plt.figure(figsize=(16, 7))

    plt.bar(
        depth,
        obs_sus,
        width=bar_width,
        color="black",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5
    )

    plt.plot(
        depth,
        obs_sus,
        color="black",
        linewidth=1.6,
        marker="o",
        markersize=3
    )

    for x, y in zip(depth, obs_sus):
        plt.text(
            x,
            y + obs_offset,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90
        )

    plt.xlabel("Depth (m)", fontsize=13)
    plt.ylabel("Observed Susceptibility (SI)", fontsize=13)

    plt.title(
        f"{bh_name}\nObserved Susceptibility vs Depth",
        fontsize=16,
        fontweight="bold"
    )

    plt.xticks(
        depth,
        [f"{d:.1f}" for d in depth],
        rotation=90,
        fontsize=6
    )

    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()

    obs_file = os.path.join(
        obs_depth_dir,
        f"{bh_name}_Observed_Susceptibility_vs_Depth.png"
    )

    plt.savefig(obs_file, dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 2) PREDICTED SUSCEPTIBILITY vs DEPTH
    # ========================================================

    plt.figure(figsize=(16, 7))

    plt.bar(
        depth,
        pred_sus,
        width=bar_width,
        color="darkorange",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5
    )

    plt.plot(
        depth,
        pred_sus,
        color="darkorange",
        linewidth=1.6,
        linestyle="--",
        marker="o",
        markersize=3
    )

    for x, y in zip(depth, pred_sus):
        plt.text(
            x,
            y + pred_offset,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90
        )

    plt.xlabel("Depth (m)", fontsize=13)
    plt.ylabel("Predicted Susceptibility (SI)", fontsize=13)

    plt.title(
        f"{bh_name}\nPredicted Susceptibility vs Depth",
        fontsize=16,
        fontweight="bold"
    )

    plt.xticks(
        depth,
        [f"{d:.1f}" for d in depth],
        rotation=90,
        fontsize=6
    )

    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()

    pred_file = os.path.join(
        pred_depth_dir,
        f"{bh_name}_Predicted_Susceptibility_vs_Depth.png"
    )

    plt.savefig(pred_file, dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # 3) PERCENTAGE ERROR vs DEPTH
    # ========================================================

    plt.figure(figsize=(16, 7))

    plt.plot(
        depth,
        percent_error,
        color="crimson",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Percentage Error"
    )

    plt.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.3
    )

    plt.xlabel("Depth (m)", fontsize=13)
    plt.ylabel("Percentage Error", fontsize=13)

    plt.title(
        f"{bh_name}\nError vs Depth: "
        f"((Observed - Predicted) / Observed) *100\n"
        f"Percent RMSE={percent_rmse:.2f}%, "
        f"Percent MAE={percent_mae:.2f}%, Corr={corr:.3f}",
        fontsize=15,
        fontweight="bold"
    )

    plt.xticks(
        depth,
        [f"{d:.1f}" for d in depth],
        rotation=90,
        fontsize=6
    )

    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    percent_error_file = os.path.join(
        percent_error_depth_dir,
        f"{bh_name}_Percentage_Error_vs_Depth.png"
    )

    plt.savefig(percent_error_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots for {bh_name}")
    print("  Observed:", obs_file)
    print("  Predicted:", pred_file)
    print("  Percentage error:", percent_error_file)
    print("  Table:", table_file)

print("\nDone. Observed, predicted, and percentage-error-vs-depth plots saved separately.")

# ============================================================
# OBSERVED + PREDICTED KDE WITH DIFFERENCE PANEL
# Left panel: observed and predicted KDE together
# Right panel: KDE difference = observed density - predicted density
# One PNG per borehole
# ============================================================

from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

combined_kde_output_dir = (
    r"C:\Users\subhr\Downloads"
    r"\Observed_vs_Predicted_KDE_with_Difference_Plots"
)

os.makedirs(combined_kde_output_dir, exist_ok=True)

for bh_name, pdata in bh_plot_data.items():

    # ========================================================
    # OBSERVED SUSCEPTIBILITY
    # ========================================================

    obs_sus = pd.to_numeric(
        pd.Series(pdata["sus"]),
        errors="coerce"
    ).dropna().values

    # ========================================================
    # PREDICTED SUSCEPTIBILITY AT SAME BOREHOLE DEPTH POINTS
    # ========================================================

    bh_xyz = np.c_[
        np.ones_like(pdata["depth"]) * pdata["x"],
        np.ones_like(pdata["depth"]) * pdata["y"],
        -pdata["depth"]
    ]

    _, pred_cell_ids = tree.query(bh_xyz)

    pred_sus = recovered_model[pred_cell_ids]

    pred_sus = pd.to_numeric(
        pd.Series(pred_sus),
        errors="coerce"
    ).dropna().values

    if len(obs_sus) < 2:
        print(f"Skipping {bh_name}: Not enough observed values")
        continue

    if len(pred_sus) < 2:
        print(f"Skipping {bh_name}: Not enough predicted values")
        continue

    # ========================================================
    # COMMON X GRID FOR BOTH KDE CURVES
    # ========================================================

    xmin = min(np.nanmin(obs_sus), np.nanmin(pred_sus))
    xmax = max(np.nanmax(obs_sus), np.nanmax(pred_sus))

    x_pad = max(0.01, 0.08 * (xmax - xmin))

    x_grid = np.linspace(
        xmin - x_pad,
        xmax + x_pad,
        3000
    )

    kde_obs = gaussian_kde(obs_sus)
    kde_pred = gaussian_kde(pred_sus)

    obs_density = kde_obs(x_grid)
    pred_density = kde_pred(x_grid)

    kde_difference = obs_density - pred_density

    # ========================================================
    # PEAK VALUES
    # ========================================================

    obs_peak_x = x_grid[np.argmax(obs_density)]
    obs_peak_y = np.max(obs_density)

    pred_peak_x = x_grid[np.argmax(pred_density)]
    pred_peak_y = np.max(pred_density)

    kde_mae = np.mean(np.abs(kde_difference))
    kde_rmse = np.sqrt(np.mean(kde_difference ** 2))
    kde_max_abs = np.max(np.abs(kde_difference))

    # ========================================================
    # TWO-PANEL PLOT
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True
    )

    ax1, ax2 = axes

    # ========================================================
    # LEFT PANEL: OBSERVED + PREDICTED KDE
    # ========================================================

    ax1.plot(
        x_grid,
        obs_density,
        color="black",
        linewidth=2.8,
        label=f"Observed Peak = {obs_peak_x:.4f} SI"
    )

    ax1.plot(
        x_grid,
        pred_density,
        color="darkorange",
        linewidth=2.8,
        linestyle="--",
        label=f"Predicted Peak = {pred_peak_x:.4f} SI"
    )

    ax1.scatter(
        obs_peak_x,
        obs_peak_y,
        color="black",
        s=70,
        zorder=5
    )

    ax1.scatter(
        pred_peak_x,
        pred_peak_y,
        color="darkorange",
        s=70,
        zorder=5
    )

    ax1.axvline(
        obs_peak_x,
        color="black",
        linestyle=":",
        linewidth=1.8
    )

    ax1.axvline(
        pred_peak_x,
        color="darkorange",
        linestyle=":",
        linewidth=1.8
    )

    ax1.text(
        obs_peak_x,
        obs_peak_y,
        f"Obs peak\n{obs_peak_x:.4f}",
        fontsize=9,
        ha="center",
        va="bottom",
        color="black"
    )

    ax1.text(
        pred_peak_x,
        pred_peak_y,
        f"Pred peak\n{pred_peak_x:.4f}",
        fontsize=9,
        ha="center",
        va="bottom",
        color="darkorange"
    )

    ax1.set_title(
        f"{bh_name}\nObserved vs Predicted Susceptibility KDE",
        fontsize=16,
        fontweight="bold"
    )

    ax1.set_xlabel("Susceptibility (SI)", fontsize=13)
    ax1.set_ylabel("Density", fontsize=13)

    ax1.xaxis.set_major_locator(MultipleLocator(0.01))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.002))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax1.tick_params(axis="x", which="major", labelsize=8, rotation=45)
    ax1.tick_params(axis="y", which="major", labelsize=9)

    ax1.grid(which="major", alpha=0.35)
    ax1.grid(which="minor", linestyle=":", alpha=0.20)

    ax1.legend(fontsize=10)

    # ========================================================
    # RIGHT PANEL: KDE DIFFERENCE
    # ========================================================

    ax2.plot(
        x_grid,
        kde_difference,
        color="crimson",
        linewidth=2.5,
        label="Observed KDE - Predicted KDE"
    )

    ax2.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.3
    )

    ax2.fill_between(
        x_grid,
        kde_difference,
        0,
        where=(kde_difference >= 0),
        color="red",
        alpha=0.25,
        label="Observed > Predicted"
    )

    ax2.fill_between(
        x_grid,
        kde_difference,
        0,
        where=(kde_difference < 0),
        color="blue",
        alpha=0.25,
        label="Predicted > Observed"
    )

    ax2.text(
        0.03,
        0.97,
        f"MAE = {kde_mae:.4f}\n"
        f"RMSE = {kde_rmse:.4f}\n"
        f"Max |Diff| = {kde_max_abs:.4f}",
        transform=ax2.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            edgecolor="black"
        )
    )

    ax2.set_title(
        "KDE Difference / Error",
        fontsize=16,
        fontweight="bold"
    )

    ax2.set_xlabel("Susceptibility (SI)", fontsize=13)
    ax2.set_ylabel("Density Difference", fontsize=13)

    ax2.xaxis.set_major_locator(MultipleLocator(0.01))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.002))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax2.tick_params(axis="x", which="major", labelsize=8, rotation=45)
    ax2.tick_params(axis="y", which="major", labelsize=9)

    ax2.grid(which="major", alpha=0.35)
    ax2.grid(which="minor", linestyle=":", alpha=0.20)

    ax2.legend(fontsize=9)

    output_file = os.path.join(
        combined_kde_output_dir,
        f"{bh_name}_Observed_vs_Predicted_KDE_with_Difference.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved KDE comparison plot: {output_file}")

print("\nDone. All observed-vs-predicted KDE difference plots saved.")

# ============================================================
# PUT MODEL BACK INTO FULL MESH
# ============================================================

recovered_full = np.zeros(mesh.nC)
recovered_full[active_cells] = recovered_model
recovered_full[~active_cells] = 0.0

C = recovered_full.copy()

# ============================================================

# SUSCEPTIBILITY THRESHOLD-EVOLUTION / TIMELAPSE PLOTS
# ============================================================
# Instead of using one fixed BODY_SUS_MIN, this section loops through a
# sequence of susceptibility minimum thresholds. BODY_SUS_MAX remains fixed.
# This shows how the recovered body cloud collapses from broad/halo zones
# into compact high-susceptibility cores as the threshold increases.
#
# Outputs produced:
#   1) One 3D HTML cloud for every susceptibility threshold
#   2) One horizontal-slice PNG for every susceptibility threshold
#   3) One fixed-longitude vertical-section PNG for every threshold
#   4) One fixed-latitude vertical-section PNG for every threshold
#   5) One animated 3D HTML with a slider over susceptibility thresholds
#
# IMPORTANT:
# This does NOT rerun the inversion. It only re-filters the already recovered
# susceptibility model C = recovered_full.copy().

BODY_SUS_MAX = 0.47

# Fine threshold sequence from 0.06 to 0.35 SI
# Change step to 0.02 if you want fewer files.
sus_min_values = np.round(np.arange(0.06, 0.351, 0.01), 3)

# Optional: use a KDE-guided sparse sequence instead of the fine one above.
# sus_min_values = np.array([0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.32, 0.35])

print("\nSusceptibility threshold-evolution run")
print("BODY_SUS_MAX =", BODY_SUS_MAX, "SI")
print("Thresholds:", sus_min_values)

# Output folders
threshold_root_dir = "Susceptibility_Threshold_Evolution"
html_3d_dir = os.path.join(threshold_root_dir, "3D_HTML_Frames")
horizontal_dir = os.path.join(threshold_root_dir, "Horizontal_Slice_Frames")
vertical_lon_dir = os.path.join(threshold_root_dir, "Vertical_Longitude_Frames")
vertical_lat_dir = os.path.join(threshold_root_dir, "Vertical_Latitude_Frames")
summary_dir = os.path.join(threshold_root_dir, "Summary")

for _d in [threshold_root_dir, html_3d_dir, horizontal_dir, vertical_lon_dir, vertical_lat_dir, summary_dir]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# COMMON GEOMETRY FOR ALL THRESHOLD FRAMES
# ============================================================

unique_depths = np.unique(np.round(depth_cells, 6))
slice_depths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
max_depth_plot = 500.0

# Gugrot block plotting window
gugrot_lon_min = 72.410000
gugrot_lon_max = 72.430000
gugrot_lat_min = 25.590417
gugrot_lat_max = 25.598472

margin_lon = 0.002
margin_lat = 0.002

nx_plot = 260
ny_plot = 220

lon_slice_grid = np.linspace(gugrot_lon_min - margin_lon, gugrot_lon_max + margin_lon, nx_plot)
lat_slice_grid = np.linspace(gugrot_lat_min - margin_lat, gugrot_lat_max + margin_lat, ny_plot)

LON2D, LAT2D = np.meshgrid(lon_slice_grid, lat_slice_grid)
XS2D, YS2D = lonlat_to_xy(LON2D, LAT2D)

inside_gugrot_grid = (
    (LON2D >= gugrot_lon_min) &
    (LON2D <= gugrot_lon_max) &
    (LAT2D >= gugrot_lat_min) &
    (LAT2D <= gugrot_lat_max)
)

# Section locations inside Gugrot block
n_lon_sections = 5
n_lat_sections = 5

lon_sections = np.linspace(gugrot_lon_min + 0.001, gugrot_lon_max - 0.001, n_lon_sections)
lat_sections = np.linspace(gugrot_lat_min + 0.001, gugrot_lat_max - 0.001, n_lat_sections)

Lon_cells, Lat_cells = xy_to_lonlat(Xc, Yc)
lon_tolerance_m = dx / 2.0
lat_tolerance_m = dy / 2.0

# Sand-rock surface for 3D HTML plots
sand_lon_vec = np.linspace(gugrot_lon_min, gugrot_lon_max, 180)
sand_lat_vec = np.linspace(gugrot_lat_min, gugrot_lat_max, 180)
Sand_lon, Sand_lat = np.meshgrid(sand_lon_vec, sand_lat_vec)
XS, YS = lonlat_to_xy(Sand_lon, Sand_lat)
sand_surface = sand_bottom_surface(XS, YS)

# Layer mask for dense cloud display
LAYER_STEP = 1
shown_depths = unique_depths[::LAYER_STEP]
layer_mask = np.isin(np.round(depth_cells, 6), shown_depths)

# A fixed colour scale for all threshold frames, so the timelapse is comparable.
# We build it from all active cells within the full threshold-evolution envelope.
full_evolution_mask = (
    active_cells
    & np.isfinite(C)
    & (C >= np.nanmin(sus_min_values))
    & (C <= BODY_SUS_MAX)
)

full_evolution_vals = C[full_evolution_mask]
if full_evolution_vals.size == 0:
    raise ValueError(
        "No cells found inside the full evolution envelope. "
        "Lower the minimum threshold or increase BODY_SUS_MAX."
    )

fixed_colorscale_3d, fixed_cmin_3d, fixed_cmax_3d, fixed_cbar_levels_3d = plotly_hist_equalized_colorscale(
    full_evolution_vals,
    cmap_name="turbo",
    n_bins=64
)

fixed_levels_2d, fixed_norm_2d = hist_equalized_norm(
    full_evolution_vals,
    n_bins=64,
    cmap_name="turbo"
)

# ============================================================
# HELPER: HORIZONTAL DEPTH-SLICE FRAME FOR ONE THRESHOLD
# ============================================================

def save_horizontal_slices_for_threshold(BODY_SUS_MIN, body_band_mask):

    slice_grids = []
    slice_cell_counts = []

    for target_depth in slice_depths:

        nearest_depth = unique_depths[np.argmin(np.abs(unique_depths - target_depth))]
        depth_mask = np.isclose(np.round(depth_cells, 6), nearest_depth)
        valid_mask = depth_mask & body_band_mask
        slice_cell_counts.append(int(valid_mask.sum()))

        if valid_mask.sum() < 5:
            slice_grids.append(None)
            continue

        x_layer = Xc[valid_mask]
        y_layer = Yc[valid_mask]
        c_layer = C[valid_mask]

        grid_layer = griddata(
            np.c_[x_layer, y_layer],
            c_layer,
            (XS2D, YS2D),
            method="linear"
        )

        nearest_layer = griddata(
            np.c_[x_layer, y_layer],
            c_layer,
            (XS2D, YS2D),
            method="nearest"
        )

        grid_layer[np.isnan(grid_layer)] = nearest_layer[np.isnan(grid_layer)]
        grid_layer[~inside_gugrot_grid] = np.nan
        slice_grids.append(grid_layer)

    fig_slice, axes = plt.subplots(
        3,
        4,
        figsize=(22, 14),
        constrained_layout=True
    )

    axes = axes.ravel()
    last_im = None

    for i, target_depth in enumerate(slice_depths):

        ax = axes[i]
        grid_layer = slice_grids[i]

        if grid_layer is None:
            ax.axis("off")
            ax.set_title(f"{target_depth} m\nNo cells", fontsize=12)
            continue

        cloud_mask = np.isfinite(grid_layer)

        last_im = ax.scatter(
            LON2D[cloud_mask],
            LAT2D[cloud_mask],
            c=grid_layer[cloud_mask],
            s=6,
            cmap="turbo",
            norm=fixed_norm_2d,
            alpha=0.85,
            edgecolors="none"
        )

        ax.plot(gugrot_lon_plot, gugrot_lat_plot, "k-", linewidth=2.5)

        for bh_name, pdata in bh_plot_data.items():
            ax.scatter(
                pdata["lon"],
                pdata["lat"],
                s=35,
                c="red",
                edgecolors="black",
                linewidths=0.6,
                zorder=10
            )

        ax.set_title(
            f"Depth = {target_depth} m\nCells = {slice_cell_counts[i]}",
            fontsize=13,
            fontweight="bold"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(gugrot_lon_min - margin_lon, gugrot_lon_max + margin_lon)
        ax.set_ylim(gugrot_lat_min - margin_lat, gugrot_lat_max + margin_lat)
        ax.set_aspect("equal")

    for j in range(len(slice_depths), len(axes)):
        axes[j].axis("off")

    fig_slice.suptitle(
        f"Threshold Evolution: Horizontal Susceptibility Depth Slices\n"
        f"Filtered band: {BODY_SUS_MIN:.3f}–{BODY_SUS_MAX:.3f} SI",
        fontsize=22,
        fontweight="bold"
    )

    if last_im is not None:
        fig_slice.colorbar(
            last_im,
            ax=axes,
            shrink=0.82,
            pad=0.02,
            label="Recovered Susceptibility (SI)"
        )

    out_png = os.path.join(
        horizontal_dir,
        f"Horizontal_Slices_susmin_{BODY_SUS_MIN:.3f}_susmax_{BODY_SUS_MAX:.3f}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig_slice)

    return out_png, slice_cell_counts

# ============================================================
# HELPER: VERTICAL SECTION FRAMES FOR ONE THRESHOLD
# ============================================================

def save_vertical_sections_for_threshold(BODY_SUS_MIN, body_band_mask):

    vertical_vals = C[body_band_mask]
    if vertical_vals.size == 0:
        return None, None, [], []

    # Use fixed norm across all thresholds so frames are comparable.
    vertical_norm = fixed_norm_2d

    # ---------------- Fixed-longitude sections ----------------
    fig_lon, axes_lon = plt.subplots(
        1,
        n_lon_sections,
        figsize=(5 * n_lon_sections, 7),
        constrained_layout=True
    )
    if n_lon_sections == 1:
        axes_lon = [axes_lon]

    lon_counts = []
    last_im_lon = None

    for i, lon_sec in enumerate(lon_sections):

        ax = axes_lon[i]
        x_sec, _ = lonlat_to_xy(lon_sec, ref_lat)

        section_mask = (
            body_band_mask
            & (np.abs(Xc - x_sec) <= lon_tolerance_m)
            & (Lat_cells >= gugrot_lat_min)
            & (Lat_cells <= gugrot_lat_max)
            & (depth_cells <= max_depth_plot)
        )

        lon_counts.append(int(section_mask.sum()))

        if section_mask.sum() < 5:
            ax.axis("off")
            ax.set_title(f"Lon {lon_sec:.5f}\nNo cells")
            continue

        last_im_lon = ax.scatter(
            Lat_cells[section_mask],
            depth_cells[section_mask],
            c=C[section_mask],
            s=15,
            cmap="turbo",
            norm=vertical_norm,
            alpha=1.0,
            edgecolors="none"
        )

        for bh_name, pdata in bh_plot_data.items():
            bh_x, _ = lonlat_to_xy(pdata["lon"], pdata["lat"])
            if abs(bh_x - x_sec) <= lon_tolerance_m:
                ax.plot(
                    [pdata["lat"], pdata["lat"]],
                    [0, pdata["drilled_depth"]],
                    "k-",
                    linewidth=1.5
                )
                ax.text(
                    pdata["lat"],
                    pdata["drilled_depth"] + 8,
                    bh_name,
                    fontsize=8,
                    rotation=90,
                    ha="center",
                    va="bottom"
                )

        ax.set_title(
            f"Longitude = {lon_sec:.5f}\nCells = {section_mask.sum()}",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Depth below z=0 (m)")
        ax.set_ylim(max_depth_plot, 0)
        ax.set_xlim(gugrot_lat_min, gugrot_lat_max)
        ax.grid(alpha=0.25)

    fig_lon.suptitle(
        f"Threshold Evolution: Fixed-Longitude Vertical Sections\n"
        f"Filtered band: {BODY_SUS_MIN:.3f}–{BODY_SUS_MAX:.3f} SI",
        fontsize=18,
        fontweight="bold"
    )

    if last_im_lon is not None:
        fig_lon.colorbar(
            last_im_lon,
            ax=axes_lon,
            shrink=0.85,
            pad=0.02,
            label="Recovered Susceptibility (SI)"
        )

    out_lon_png = os.path.join(
        vertical_lon_dir,
        f"Vertical_Longitudes_susmin_{BODY_SUS_MIN:.3f}_susmax_{BODY_SUS_MAX:.3f}.png"
    )
    plt.savefig(out_lon_png, dpi=300, bbox_inches="tight")
    plt.close(fig_lon)

    # ---------------- Fixed-latitude sections ----------------
    fig_lat, axes_lat = plt.subplots(
        1,
        n_lat_sections,
        figsize=(5 * n_lat_sections, 7),
        constrained_layout=True
    )
    if n_lat_sections == 1:
        axes_lat = [axes_lat]

    lat_counts = []
    last_im_lat = None

    for i, lat_sec in enumerate(lat_sections):

        ax = axes_lat[i]
        _, y_sec = lonlat_to_xy(ref_lon, lat_sec)

        section_mask = (
            body_band_mask
            & (np.abs(Yc - y_sec) <= lat_tolerance_m)
            & (Lon_cells >= gugrot_lon_min)
            & (Lon_cells <= gugrot_lon_max)
            & (depth_cells <= max_depth_plot)
        )

        lat_counts.append(int(section_mask.sum()))

        if section_mask.sum() < 5:
            ax.axis("off")
            ax.set_title(f"Lat {lat_sec:.5f}\nNo cells")
            continue

        last_im_lat = ax.scatter(
            Lon_cells[section_mask],
            depth_cells[section_mask],
            c=C[section_mask],
            s=15,
            cmap="turbo",
            norm=vertical_norm,
            alpha=1.0,
            edgecolors="none"
        )

        for bh_name, pdata in bh_plot_data.items():
            _, bh_y = lonlat_to_xy(pdata["lon"], pdata["lat"])
            if abs(bh_y - y_sec) <= lat_tolerance_m:
                ax.plot(
                    [pdata["lon"], pdata["lon"]],
                    [0, pdata["drilled_depth"]],
                    "k-",
                    linewidth=1.5
                )
                ax.text(
                    pdata["lon"],
                    pdata["drilled_depth"] + 8,
                    bh_name,
                    fontsize=8,
                    rotation=90,
                    ha="center",
                    va="bottom"
                )

        ax.set_title(
            f"Latitude = {lat_sec:.5f}\nCells = {section_mask.sum()}",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Depth below z=0 (m)")
        ax.set_ylim(max_depth_plot, 0)
        ax.set_xlim(gugrot_lon_min, gugrot_lon_max)
        ax.grid(alpha=0.25)

    fig_lat.suptitle(
        f"Threshold Evolution: Fixed-Latitude Vertical Sections\n"
        f"Filtered band: {BODY_SUS_MIN:.3f}–{BODY_SUS_MAX:.3f} SI",
        fontsize=18,
        fontweight="bold"
    )

    if last_im_lat is not None:
        fig_lat.colorbar(
            last_im_lat,
            ax=axes_lat,
            shrink=0.85,
            pad=0.02,
            label="Recovered Susceptibility (SI)"
        )

    out_lat_png = os.path.join(
        vertical_lat_dir,
        f"Vertical_Latitudes_susmin_{BODY_SUS_MIN:.3f}_susmax_{BODY_SUS_MAX:.3f}.png"
    )
    plt.savefig(out_lat_png, dpi=300, bbox_inches="tight")
    plt.close(fig_lat)

    return out_lon_png, out_lat_png, lon_counts, lat_counts

# ============================================================
# HELPER: 3D HTML FRAME FOR ONE THRESHOLD
# ============================================================

def add_static_3d_context(fig):
    """Add Gugrot boundary, surface, sand-rock boundary, and boreholes."""

    fig.add_trace(go.Scatter3d(
        x=gugrot_lon_plot,
        y=gugrot_lat_plot,
        z=gugrot_z,
        mode="lines",
        line=dict(color="black", width=8),
        name="Gugrot Block Boundary at z=0"
    ))

    fig.add_trace(go.Mesh3d(
        x=gugrot_lon_plot[:-1],
        y=gugrot_lat_plot[:-1],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        opacity=0.15,
        color="black",
        name="Gugrot Block Surface"
    ))

    fig.add_trace(go.Surface(
        x=Sand_lon,
        y=Sand_lat,
        z=-sand_surface,
        opacity=0.5,
        colorscale="YlOrBr",
        showscale=False,
        name="Undulating sand-rock boundary"
    ))

    for bh_name, pdata in bh_plot_data.items():

        lon_bh = pdata["lon"]
        lat_bh = pdata["lat"]
        sb = pdata["sand_bottom"]

        fig.add_trace(go.Scatter3d(
            x=[lon_bh],
            y=[lat_bh],
            z=[25],
            mode="markers+text",
            marker=dict(size=10, color="red", line=dict(color="yellow", width=2)),
            text=[bh_name],
            textposition="top center",
            name=bh_name,
            showlegend=True
        ))

        fig.add_trace(go.Scatter3d(
            x=[lon_bh, lon_bh],
            y=[lat_bh, lat_bh],
            z=[0, -pdata["drilled_depth"]],
            mode="lines",
            line=dict(color="black", width=5),
            name=f"{bh_name} trace",
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[lon_bh, lon_bh],
            y=[lat_bh, lat_bh],
            z=[-sand_top, -sb],
            mode="lines",
            line=dict(color="gold", width=8),
            name=f"{bh_name} sand cover",
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=np.ones_like(pdata["depth"]) * lon_bh,
            y=np.ones_like(pdata["depth"]) * lat_bh,
            z=-pdata["depth"],
            mode="markers",
            marker=dict(
                size=5,
                color=pdata["sus"],
                colorscale="Plasma",
                symbol="diamond",
                opacity=0.95,
                showscale=False
            ),
            name=f"{bh_name} observed susceptibility",
            showlegend=False,
            text=[
                f"{bh_name}<br>Longitude: {lon_bh:.6f}<br>Latitude: {lat_bh:.6f}<br>"
                f"Depth: {d:.1f} m<br>Observed Sus: {s:.6f}"
                for d, s in zip(pdata["depth"], pdata["sus"])
            ],
            hoverinfo="text"
        ))

        fig.add_trace(go.Scatter3d(
            x=[lon_bh],
            y=[lat_bh],
            z=[-pdata["drilled_depth"]],
            mode="markers+text",
            marker=dict(size=9, color="red", symbol="x"),
            text=[f"TD {pdata['drilled_depth']} m"],
            textposition="bottom center",
            name=f"{bh_name} total drilled depth",
            showlegend=False
        ))


def update_3d_layout(fig, title_text):
    fig.update_layout(
        title=title_text,
        width=1800,
        height=900,
        margin=dict(l=20, r=500, t=80, b=20),
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth (m)",
            xaxis=dict(range=[72.405, 72.435]),
            yaxis=dict(range=[25.586, 25.602]),
            zaxis=dict(range=[-mesh.h[2].sum(), 60]),
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.1, z=1.4)
        ),
        legend=dict(
            x=1.22,
            y=0.95,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11)
        )
    )


def save_3d_html_for_threshold(BODY_SUS_MIN, body_band_mask):

    plot_mask = body_band_mask & layer_mask

    if plot_mask.sum() < 5:
        return None, 0

    X_plot = Xc[plot_mask]
    Y_plot = Yc[plot_mask]
    Z_plot = Zc[plot_mask]
    C_plot = C[plot_mask]
    Lon_plot, Lat_plot = xy_to_lonlat(X_plot, Y_plot)

    fig = go.Figure()
    add_static_3d_context(fig)

    fig.add_trace(go.Scatter3d(
        x=Lon_plot,
        y=Lat_plot,
        z=Z_plot,
        mode="markers",
        marker=dict(
            size=2.4,
            color=C_plot,
            cmin=fixed_cmin_3d,
            cmax=fixed_cmax_3d,
            colorscale=fixed_colorscale_3d,
            opacity=0.75,
            colorbar=dict(
                title="Recovered Susceptibility (SI)",
                len=0.95,
                y=0.50,
                thickness=25,
                x=1.08
            )
        ),
        name=f"Cloud: χ ≥ {BODY_SUS_MIN:.3f} SI",
        text=[
            f"Depth: {-z:.1f} m<br>Sus: {c:.6f}<br>Threshold: {BODY_SUS_MIN:.3f} SI"
            for z, c in zip(Z_plot, C_plot)
        ],
        hoverinfo="text"
    ))

    update_3d_layout(
        fig,
        title_text=(
            "Susceptibility Threshold Evolution - 3D Cloud<br>"
            f"Only recovered susceptibility {BODY_SUS_MIN:.3f}–{BODY_SUS_MAX:.3f} SI shown"
        )
    )

    out_html = os.path.join(
        html_3d_dir,
        f"3D_Cloud_susmin_{BODY_SUS_MIN:.3f}_susmax_{BODY_SUS_MAX:.3f}.html"
    )
    fig.write_html(out_html)
    return out_html, int(plot_mask.sum())

# ============================================================
# HELPER: 3D ANIMATED HTML WITH SLIDER
# ============================================================

def save_animated_3d_threshold_html(frame_data):
    """Create one Plotly animation/slider showing the 3D cloud evolution."""

    valid_frames = [fd for fd in frame_data if fd["n_3d_cells"] >= 5]
    if len(valid_frames) == 0:
        print("No valid 3D frames for animated HTML.")
        return None

    # Initial frame data
    fd0 = valid_frames[0]
    fig_anim = go.Figure()
    add_static_3d_context(fig_anim)

    fig_anim.add_trace(go.Scatter3d(
        x=fd0["lon"],
        y=fd0["lat"],
        z=fd0["z"],
        mode="markers",
        marker=dict(
            size=2.4,
            color=fd0["c"],
            cmin=fixed_cmin_3d,
            cmax=fixed_cmax_3d,
            colorscale=fixed_colorscale_3d,
            opacity=0.75,
            colorbar=dict(
                title="Recovered Susceptibility (SI)",
                len=0.95,
                y=0.50,
                thickness=25,
                x=1.08
            )
        ),
        name="Animated threshold cloud",
        text=fd0["text"],
        hoverinfo="text"
    ))

    # The cloud trace is the last trace after static context.
    cloud_trace_index = len(fig_anim.data) - 1

    frames = []
    slider_steps = []

    for fd in valid_frames:
        frame_name = f"{fd['sus_min']:.3f}"
        frames.append(go.Frame(
            name=frame_name,
            data=[go.Scatter3d(
                x=fd["lon"],
                y=fd["lat"],
                z=fd["z"],
                mode="markers",
                marker=dict(
                    size=2.4,
                    color=fd["c"],
                    cmin=fixed_cmin_3d,
                    cmax=fixed_cmax_3d,
                    colorscale=fixed_colorscale_3d,
                    opacity=0.75,
                    colorbar=dict(
                        title="Recovered Susceptibility (SI)",
                        len=0.95,
                        y=0.50,
                        thickness=25,
                        x=1.08
                    )
                ),
                name=f"χ ≥ {fd['sus_min']:.3f} SI",
                text=fd["text"],
                hoverinfo="text"
            )],
            traces=[cloud_trace_index]
        ))

        slider_steps.append(dict(
            method="animate",
            label=f"{fd['sus_min']:.2f}",
            args=[
                [frame_name],
                dict(
                    mode="immediate",
                    frame=dict(duration=600, redraw=True),
                    transition=dict(duration=200)
                )
            ]
        ))

    fig_anim.frames = frames

    update_3d_layout(
        fig_anim,
        title_text=(
            "Animated Susceptibility Threshold Evolution - 3D Cloud<br>"
            f"χmin evolves from {valid_frames[0]['sus_min']:.3f} to {valid_frames[-1]['sus_min']:.3f} SI; "
            f"χmax = {BODY_SUS_MAX:.3f} SI"
        )
    )

    fig_anim.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=700, redraw=True),
                        transition=dict(duration=250),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate"
                    )]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="χmin threshold = ",
                suffix=" SI",
                visible=True
            ),
            pad=dict(t=50),
            steps=slider_steps
        )]
    )

    out_html = os.path.join(
        threshold_root_dir,
        "Animated_3D_Susceptibility_Threshold_Evolution.html"
    )
    fig_anim.write_html(out_html)
    return out_html

# ============================================================
# MAIN THRESHOLD-EVOLUTION LOOP
# ============================================================

summary_rows = []
animation_frame_data = []

for BODY_SUS_MIN in sus_min_values:

    body_band_mask = (
        active_cells
        & np.isfinite(C)
        & (C >= BODY_SUS_MIN)
        & (C <= BODY_SUS_MAX)
    )

    total_cells = int(body_band_mask.sum())

    print("\n============================================================")
    print(f"Threshold frame: {BODY_SUS_MIN:.3f}–{BODY_SUS_MAX:.3f} SI")
    print("Total body-band cells:", total_cells)

    if total_cells < 5:
        print("Skipping this threshold: too few cells.")
        summary_rows.append({
            "sus_min": BODY_SUS_MIN,
            "sus_max": BODY_SUS_MAX,
            "total_cells": total_cells,
            "n_3d_cells": 0,
            "horizontal_png": None,
            "vertical_longitude_png": None,
            "vertical_latitude_png": None,
            "html_3d": None
        })
        continue

    horizontal_png, horizontal_counts = save_horizontal_slices_for_threshold(BODY_SUS_MIN, body_band_mask)
    vertical_lon_png, vertical_lat_png, lon_counts, lat_counts = save_vertical_sections_for_threshold(BODY_SUS_MIN, body_band_mask)
    html_3d, n_3d_cells = save_3d_html_for_threshold(BODY_SUS_MIN, body_band_mask)

    # Prepare data for one animated HTML slider.
    plot_mask_anim = body_band_mask & layer_mask
    if plot_mask_anim.sum() >= 5:
        X_plot = Xc[plot_mask_anim]
        Y_plot = Yc[plot_mask_anim]
        Z_plot = Zc[plot_mask_anim]
        C_plot = C[plot_mask_anim]
        Lon_plot, Lat_plot = xy_to_lonlat(X_plot, Y_plot)

        animation_frame_data.append({
            "sus_min": float(BODY_SUS_MIN),
            "lon": Lon_plot,
            "lat": Lat_plot,
            "z": Z_plot,
            "c": C_plot,
            "n_3d_cells": int(plot_mask_anim.sum()),
            "text": [
                f"Depth: {-z:.1f} m<br>Sus: {c:.6f}<br>Threshold: {BODY_SUS_MIN:.3f} SI"
                for z, c in zip(Z_plot, C_plot)
            ]
        })

    summary_rows.append({
        "sus_min": float(BODY_SUS_MIN),
        "sus_max": float(BODY_SUS_MAX),
        "total_cells": total_cells,
        "n_3d_cells": int(n_3d_cells),
        "horizontal_png": horizontal_png,
        "vertical_longitude_png": vertical_lon_png,
        "vertical_latitude_png": vertical_lat_png,
        "html_3d": html_3d,
        "horizontal_slice_cell_counts": str(horizontal_counts),
        "longitude_section_cell_counts": str(lon_counts),
        "latitude_section_cell_counts": str(lat_counts)
    })

    print("Saved horizontal slices:", horizontal_png)
    print("Saved vertical longitude sections:", vertical_lon_png)
    print("Saved vertical latitude sections:", vertical_lat_png)
    print("Saved 3D HTML:", html_3d)

animated_html = save_animated_3d_threshold_html(animation_frame_data)

summary_df = pd.DataFrame(summary_rows)
summary_excel = os.path.join(summary_dir, "Susceptibility_Threshold_Evolution_Summary.xlsx")
summary_csv = os.path.join(summary_dir, "Susceptibility_Threshold_Evolution_Summary.csv")
summary_df.to_excel(summary_excel, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\n============================================================")
print("Threshold-evolution plotting completed.")
print("Root output folder:", threshold_root_dir)
print("Summary Excel:", summary_excel)
print("Summary CSV:", summary_csv)
if animated_html is not None:
    print("Animated 3D HTML:", animated_html)
print("============================================================")



# -*- coding: utf-8 -*-
"""
2D Combined Short-Range Order (SRO) vs Stability Mapping Script

Features:
- Separate display matrices for Surface vs Bulk pairwise descriptors (alpha_ij).
- Horizontal strip visualization at the bottom for local deviation metric (delta_j).
- Typography tuning: Uniform unbolded (normal weight) sans-serif labels.
- Layout: Balanced spacing using explicit structural grid adjustments.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Optional interpolation scaling fallback layer
try:
    from scipy.interpolate import Rbf, griddata
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    warnings.warn("Scipy not resolved in environment. Falling back to basic nearest-neighbor upsampling.")

# ==========================================
# Global Data Paths & Target Parameter Configurations
# ==========================================
BASE_PROJECT_DIR = "/path/to/your/project/data"

# Source metrics computed from previous structural extraction sweeps
CCSRO_CSV  = os.path.join(BASE_PROJECT_DIR, "descriptors", "ccsro_surface_bulk_radius_split.csv")
CSRO_CSV   = os.path.join(BASE_PROJECT_DIR, "descriptors", "csro_surface_bulk_radius_split.csv")
ENERGY_CSV = os.path.join(BASE_PROJECT_DIR, "thermodynamics", "sorted_energies_catalyst.csv")

# Output plot dump path
OUT_DIR = os.path.join(BASE_PROJECT_DIR, "visualizations_2d_sro")
os.makedirs(OUT_DIR, exist_ok=True)

# Selection filters
ALPHA_J_REGION = "surface"
PREFERRED_ALPHA_J = ["alpha_Bi", "alpha_La", "alpha_Sn"]

# Scoring filter logic: Requires both positive and negative values to gauge a trend line
MIN_GROUPS_REQUIRED = True

# Matrix upscaling properties for smoother field renders
INTERP_METHOD = "griddata-cubic"
GRID_UPSCALE = 10

# Direct color mappings
ALPHAIJ_CMAP = cm.get_cmap("jet")
ALPHAJ_CMAP  = cm.get_cmap("coolwarm")

# ==========================================
# Layout & Typographical Geometry Parameters
# ==========================================
FIGSIZE = (22.0, 11.0)
DPI = 300
FS_TITLE = 28
FS_TICK  = 28
FS_CBAR  = 28

# ==========================================
# Data Restructuring & Processing Helpers
# ==========================================

PAIR_PATTERN = re.compile(r"^alpha_([^_]+)_([^_]+)__(all|bulk|surface)$")

def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _get_target_alpha_j_columns(df, preferred=None):
    """Scans and extracts single-species indicators based on tracking flags."""
    pattern = re.compile(r"^alpha_([^_]+)__(surface|bulk)$")
    cols = [c for c in df.columns if isinstance(c, str) and pattern.match(c)]
    base_elements = sorted(set(["alpha_" + pattern.match(c).group(1) for c in cols]))
    
    if preferred:
        chosen = [c for c in preferred if c in base_elements]
        if chosen: 
            return chosen
    return base_elements


def _melt_alpha_j_profiles(df_wide, alpha_bases):
    """Melts wide single-species dataframes into organized long-form structures."""
    records = []
    for _, row in df_wide.iterrows():
        struct_id = row["structure_id"]
        energy_val = row["energy"]
        for base in alpha_bases:
            for region in ("bulk", "surface"):
                target_col = f"{base}__{region}"
                if target_col in df_wide.columns:
                    records.append((struct_id, energy_val, region, base.replace("alpha_", ""), float(row[target_col])))
    return pd.DataFrame(records, columns=["structure_id", "energy", "region", "j", "alpha"])


def _extract_unique_pairs(dataframe_columns):
    """Isolates paired index tags out of flat column strings."""
    detected_pairs = []
    for col in dataframe_columns:
        match = PAIR_PATTERN.match(str(col))
        if match:
            i, j = match.group(1), match.group(2)
            detected_pairs.append(f"{i}-{j}")
    return sorted(set(detected_pairs), key=lambda s: (s.split("-")[0], s.split("-")[1]))


def _melt_alpha_pair_profiles(df_wide, unique_pairs, regions=("all", "bulk", "surface")):
    """Converts multi-element pairwise arrays into grouped long-form entries."""
    records = []
    for _, row in df_wide.iterrows():
        struct_id = row["structure_id"]
        energy_val = row["energy"]
        for pair in unique_pairs:
            i, j = pair.split("-")
            for phase in regions:
                target_col = f"alpha_{i}_{j}__{phase}"
                if target_col in df_wide.columns:
                    records.append((struct_id, energy_val, phase, pair, float(row[target_col])))
    return pd.DataFrame(records, columns=["structure_id", "energy", "region", "pair", "alpha"])


def _compute_stability_scores(long_df, grouping_keys=("region", "j"), min_groups_required=True):
    """
    Evaluates stability profile energy splits based on environmental signals.
    
    Returns the average energy differential across environments containing
    diluted vs concentrated active configurations.
    """
    score_records = []
    for keys, sub_frame in long_df.groupby(list(grouping_keys)):
        x_vals = _to_numeric(sub_frame["alpha"]).to_numpy()
        y_vals = _to_numeric(sub_frame["energy"]).to_numpy()
        
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        
        if len(x_vals) == 0:
            score = 0.0
        else:
            positive_signals = x_vals > 0
            negative_signals = x_vals < 0
            
            if min_groups_required and not (np.any(positive_signals) and np.any(negative_signals)):
                score = 0.0
            elif np.any(positive_signals) and np.any(negative_signals):
                score = float(np.mean(y_vals[negative_signals]) - np.mean(y_vals[positive_signals]))
            else:
                score = 0.0
                
        score_records.append((*keys, score))
    return pd.DataFrame(score_records, columns=list(grouping_keys) + ["score"])


def _build_alpha_j_vector(df_scores, alpha_bases, region="surface"):
    """Extracts processed single-element trends into a neat sequence array."""
    element_order = [base.replace("alpha_", "") for base in alpha_bases]
    filtered_sub = df_scores[df_scores["region"] == region]
    series_out = pd.Series(index=element_order, dtype=float)
    
    for _, row in filtered_sub.iterrows():
        if row["j"] in series_out.index:
            series_out.loc[row["j"]] = row["score"]
    return series_out.fillna(0.0)


def _build_alpha_ij_matrix(df_scores_pairs, unique_pairs, region="surface"):
    """Maps computed pairing values onto a 2D matrix layout."""
    filtered_sub = df_scores_pairs[df_scores_pairs["region"] == region].copy()
    elements_i = sorted({p.split("-")[0] for p in unique_pairs})
    elements_j = sorted({p.split("-")[1] for p in unique_pairs})
    
    mapping_i = {element: idx for idx, element in enumerate(elements_i)}
    mapping_j = {element: idx for idx, element in enumerate(elements_j)}
    
    matrix_grid = np.full((len(elements_i), len(elements_j)), np.nan, dtype=float)
    for _, row in filtered_sub.iterrows():
        i, j = row["pair"].split("-")
        if i in mapping_i and j in mapping_j:
            matrix_grid[mapping_i[i], mapping_j[j]] = row["score"]
            
    return np.nan_to_num(matrix_grid, nan=0.0), elements_i, elements_j


def _interp_continuous_field(Z_matrix, upscale=10, method="rbf"):
    """Generates upscaled surface meshes using scipy multi-quadric paths."""
    num_x, num_y = Z_matrix.shape
    mesh_size_x = max(80, num_x * upscale)
    mesh_size_y = max(80, num_y * upscale)
    
    x_coords = np.linspace(0, num_x - 1, num_x)
    y_coords = np.linspace(0, num_y - 1, num_y)
    
    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
    
    xi_coords = np.linspace(0, num_x - 1, mesh_size_x)
    yi_coords = np.linspace(0, num_y - 1, mesh_size_y)
    XI_mesh, YI_mesh = np.meshgrid(xi_coords, yi_coords, indexing="ij")

    if method == "nearest" or not HAS_SCIPY:
        xi_indices = np.clip(np.round(XI_mesh).astype(int), 0, num_x - 1)
        yi_indices = np.clip(np.round(YI_mesh).astype(int), 0, num_y - 1)
        return XI_mesh, YI_mesh, Z_matrix[xi_indices, yi_indices]

    flat_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
    flat_values = Z_matrix.ravel()
    
    if method.startswith("griddata"):
        interpolation_mode = method.split("-", 1)[1] if "-" in method else "cubic"
        ZI_mesh = griddata(flat_points, flat_values, (XI_mesh, YI_mesh), method=interpolation_mode, fill_value=0.0)
    else:
        radial_basis_fn = Rbf(flat_points[:, 0], flat_points[:, 1], flat_values, function="multiquadric", smooth=0.12)
        ZI_mesh = radial_basis_fn(XI_mesh, YI_mesh)
        
    return XI_mesh, YI_mesh, ZI_mesh


def _strip_axis_bolding(axis):
    """Enforces non-bold styling across all text blocks on a designated plot axis."""
    target_elements = (
        [axis.title, axis.xaxis.label, axis.yaxis.label] +
        axis.get_xticklabels() + axis.get_yticklabels()
    )
    for element in target_elements:
        element.set_fontweight('normal')


# ==========================================
# Main Orchestration Loop
# ==========================================

def main():
    # ------------------------------------
    # 1. Data Ingestion & Formatting
    # ------------------------------------
    print("Ingesting analytical datasets...")
    df_ccsro_raw = pd.read_csv(CCSRO_CSV)
    df_csro_raw  = pd.read_csv(CSRO_CSV)
    df_energy    = pd.read_csv(ENERGY_CSV)
    
    # Harmonize structural ID header keys
    for frame in [df_energy, df_ccsro_raw, df_csro_raw]:
        if "name" in frame.columns: 
            frame.rename(columns={"name": "structure_id"}, inplace=True)
        if "key" in frame.columns: 
            frame.rename(columns={"key": "structure_id"}, inplace=True)
        frame["structure_id"] = frame["structure_id"].astype(str).str.strip()

    df_ccsro = pd.merge(df_ccsro_raw, df_energy[["structure_id", "energy"]], on="structure_id", how="inner")
    df_csro  = pd.merge(df_csro_raw, df_energy[["structure_id", "energy"]], on="structure_id", how="inner")

    # Evaluate single-species markers (delta_j trends)
    alpha_bases = _get_target_alpha_j_columns(df_ccsro, preferred=PREFERRED_ALPHA_J)
    if not alpha_bases:
        raise RuntimeError("Failed to detect single-species tracking columns in data source.")
        
    long_j_data = _melt_alpha_j_profiles(df_ccsro, alpha_bases)
    scores_j = _compute_scores(long_j_data, keys=("region", "j"), min_groups_required=MIN_GROUPS_REQUIRED)
    vector_j = _build_alpha_j_vector(scores_j, alpha_bases, region=ALPHA_J_REGION)

    # Evaluate pairwise configurations (alpha_ij fields)
    extracted_pairs = _extract_unique_pairs(df_csro.columns)
    if not extracted_pairs:
        raise RuntimeError("No structural matrix coordinate headers matched the required format.")

    # Process surface matrices
    long_ij_surf = _melt_alpha_pair_profiles(df_csro, extracted_pairs, regions=("surface",))
    scores_ij_surf = _compute_stability_scores(long_ij_surf, grouping_keys=("region", "pair"), min_groups_required=MIN_GROUPS_REQUIRED)
    Z_surf, labels_i_surf, labels_j_surf = _build_alpha_ij_matrix(scores_ij_surf, extracted_pairs, region="surface")

    # Process bulk matrices
    long_ij_bulk = _melt_alpha_pair_profiles(df_csro, extracted_pairs, regions=("bulk",))
    scores_ij_bulk = _compute_stability_scores(long_ij_bulk, grouping_keys=("region", "pair"), min_groups_required=MIN_GROUPS_REQUIRED)
    Z_bulk, labels_i_bulk, labels_j_bulk = _build_alpha_ij_matrix(scores_ij_bulk, extracted_pairs, region="bulk")

    # Interpolate discrete matrix configurations into clean fields
    print("Mapping matrix profiles onto interpolation fields...")
    XI_surf, YI_surf, ZI_surf = _interp_continuous_field(Z_surf, upscale=GRID_UPSCALE, method=INTERP_METHOD)
    XI_bulk, YI_bulk, ZI_bulk = _interp_continuous_field(Z_bulk, upscale=GRID_UPSCALE, method=INTERP_METHOD)

    # Balance color mapping ranges separately
    norm_surf = Normalize(vmin=float(Z_surf.min()), vmax=float(Z_surf.max()))
    norm_bulk = Normalize(vmin=float(Z_bulk.min()), vmax=float(Z_bulk.max()))
    norm_jbar = Normalize(vmin=float(vector_j.min()), vmax=float(vector_j.max()))

    # ------------------------------------
    # 2. Plot Generation & Formatting
    # ------------------------------------
    plt.close("all")
    
    # Enforce precise font overrides globally
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "mathtext.default": "regular",  # Stops LaTeX strings from forcing bold styles
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "figure.dpi": DPI,
    })

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)

    # Establish layouts using asymmetric row combinations
    grid_spec = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 0.4, 1.0], 
        height_ratios=[15, 1],
        wspace=0.0, hspace=0.25
    )

    ax_surf = fig.add_subplot(grid_spec[0, 0])  # Top-left quadrant
    ax_bulk = fig.add_subplot(grid_spec[0, 2])  # Top-right quadrant
    ax_jbar = fig.add_subplot(grid_spec[1, :])  # Spanned bottom track

    # --- Matrix Surface Plot: alpha_ij (Surface Phase) ---
    im_surf = ax_surf.imshow(
        ZI_surf.T, origin="lower",
        cmap=ALPHAIJ_CMAP, norm=norm_surf,
        extent=[-0.5, len(labels_i_surf) - 0.5, -0.5, len(labels_j_surf) - 0.5],
        aspect="auto", interpolation="bilinear"
    )
    ax_surf.set_xticks(np.arange(len(labels_i_surf)))
    ax_surf.set_xticklabels(labels_i_surf, rotation=0, ha="center")
    ax_surf.set_yticks(np.arange(len(labels_j_surf)))
    ax_surf.set_yticklabels(labels_j_surf)
    ax_surf.set_title(r"$\alpha_{{ij}}$" + " (surface)", pad=8)
    _strip_axis_bolding(ax_surf)

    cbar_surf = fig.colorbar(im_surf, ax=ax_surf, fraction=0.046, pad=0.02)
    cbar_surf.ax.tick_params(labelsize=FS_CBAR, width=0.5)
    for tick_label in cbar_surf.ax.yaxis.get_ticklabels(): 
        tick_label.set_fontweight("normal")
    cbar_surf.ax.set_title(r"$\mathcal{S}_{\mathit{stab}}$ (eV)", fontsize=FS_CBAR, pad=12, x=2.0)

    # --- Matrix Bulk Plot: alpha_ij (Bulk Phase) ---
    im_bulk = ax_bulk.imshow(
        ZI_bulk.T, origin="lower",
        cmap=ALPHAIJ_CMAP, norm=norm_bulk,
        extent=[-0.5, len(labels_i_bulk) - 0.5, -0.5, len(labels_j_bulk) - 0.5],
        aspect="auto", interpolation="bilinear"
    )
    ax_bulk.set_xticks(np.arange(len(labels_i_bulk)))
    ax_bulk.set_xticklabels(labels_i_bulk, rotation=0, ha="center")
    ax_bulk.set_yticks(np.arange(len(labels_j_bulk)))
    ax_bulk.set_yticklabels(labels_j_bulk)
    ax_bulk.set_title(r"$\alpha_{{ij}}$" + " (bulk)", pad=8)
    _strip_axis_bolding(ax_bulk)

    cbar_bulk = fig.colorbar(im_bulk, ax=ax_bulk, fraction=0.046, pad=0.02)
    cbar_bulk.ax.tick_params(labelsize=FS_CBAR, width=0.5)
    for tick_label in cbar_bulk.ax.yaxis.get_ticklabels(): 
        tick_label.set_fontweight("normal")
    cbar_bulk.ax.set_title(r"$\mathcal{S}_{\mathit{stab}}$ (eV)", fontsize=FS_CBAR, pad=12, x=2.0)

    # --- Strip Trend Plot: delta_j (Surface Matrix) ---
    row_vector_data = vector_j.values.reshape(1, -1)
    im_jbar = ax_jbar.imshow(
        row_vector_data, origin="lower",
        cmap=ALPHAJ_CMAP, norm=norm_jbar,
        aspect="auto", interpolation="nearest",
        extent=[-0.5, len(vector_j) - 0.5, -0.5, 0.5]
    )
    ax_jbar.set_xticks(np.arange(len(vector_j)))
    ax_jbar.set_xticklabels(vector_j.index.tolist(), rotation=0, ha="center")
    ax_jbar.set_yticks([])
    ax_jbar.set_title(r"$\delta_{{j}}$" + " (surface)", pad=6)
    _strip_axis_bolding(ax_jbar)

    # Anchor secondary colorbar onto inner coordinate transforms
    cax_inset = inset_axes(
        ax_jbar,
        width="1.5%",     
        height="190%",
        loc="center right",
        bbox_to_anchor=(0.04, 0.0, 1, 1), 
        bbox_transform=ax_jbar.transAxes,
        borderpad=0
    )
    cbar_jbar = plt.colorbar(im_jbar, cax=cax_inset, orientation="vertical")
    cbar_jbar.ax.tick_params(labelsize=FS_CBAR - 2, width=0.5)
    for tick_label in cbar_jbar.ax.yaxis.get_ticklabels(): 
        tick_label.set_fontweight("normal")
    cbar_jbar.ax.set_title(r"$\mathcal{S}_{\mathit{stab}}$ (eV)", fontsize=FS_CBAR - 2, pad=12, x=1.5)

    # Manual margins buffer distribution tuning
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.1, top=0.93)

    # Commit generated canvas frame layout out to file
    out_png_path = os.path.join(OUT_DIR, "sro_stability_mapping_separated.png")
    fig.savefig(out_png_path, dpi=DPI)
    print(f"[SUCCESS] Exported structural canvas -> {out_png_path}")


if __name__ == "__main__":
    main()
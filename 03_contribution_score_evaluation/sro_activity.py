# -*- coding: utf-8 -*-
"""
Unweighted Mean-Difference Heatmap Generator for SRO-Activity Profiling

Analytical Core:
- Target metric: Delta_log_tof = log_tof - max_log_tof (evaluates the offset from 
  the volcano peak at each distinct pH level).
- Score calculation: S_act = mean(Delta | alpha < 0) - mean(Delta | alpha >= 0).
- Column sorting sequence: Feature columns (alpha_ij pairs) are sorted by their overall 
  "importance", defined as the sum of absolute score values accrued across all evaluated pH levels.
- Edge case protection: If the database population for either sub-group dips below 
  MIN_GROUP_N, the activity metric automatically resets to 0.0.
- Layout setup: A multi-row heatmap matrix (spanning pH 1, 7, and 14) mapping 
  short-range order indicators (formatted as A--B pairs) with centered zero-norm bounds.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ----------------------------
# System Architecture Paths
# ----------------------------
PROJECT_ROOT_DIR = "/path/to/your/project/data"

# Standard file handles tracking raw database components
EXCEL_FILE = os.path.join(PROJECT_ROOT_DIR, "volcano_analysis", "catalyst_activity_profiles.xlsx")
CSRO_FILE  = os.path.join(PROJECT_ROOT_DIR, "three_elements", "OCHO", "OCHO_site_csro.csv")

PH_VALUES = ["pH=1", "pH=7", "pH=14"]
MIN_GROUP_N = 2   # Frequency floor cutoff parameter to filter statistical drift noise

OUTPUT_HEATMAP = os.path.join(PROJECT_ROOT_DIR, "volcano_analysis", "sro_activity_heatmap.png")
ALPHA_IJ_REGEX = r"^alpha_[A-Za-z0-9]+_[A-Za-z0-9]+$"

# Graphic Presentation Geometries
XTICK_FONTSIZE  = 16
YTICK_FONTSIZE  = 16
CBAR_FONTSIZE   = 16
DPI             = 300
BASE_W_MIN      = 5.5
W_PER_COL       = 0.22
W_MARGIN        = 2.0
BASE_H          = 1.2
H_PER_ROW       = 1.2
CMAP_NAME       = "YlGnBu_r"  # Alternate maps: 'coolwarm', 'viridis_r'


# ----------------------------
# System Parsing Helpers
# ----------------------------

def _to_numeric(series):
    """Forces string arrays into structural floating vectors for calculations."""
    return pd.to_numeric(series, errors="coerce")


def _available_alpha_ij_cols(df, pattern=ALPHA_IJ_REGEX):
    """Extracts dataframe columns that adhere to short-range order naming protocols."""
    compiled_regex = re.compile(pattern)
    return [c for c in df.columns if isinstance(c, str) and compiled_regex.match(c)]


def _alpha_label_pair(alpha_col):
    """Converts raw database string structures like 'alpha_La_Bi' into pair markers 'La--Bi'."""
    clean_tag = re.sub(r"^alpha_", "", str(alpha_col))
    components = clean_tag.split("_", 1)
    return f"{components[0]}--{components[1]}" if len(components) == 2 else clean_tag


def compute_delta_log_tof_per_ph(df, ph_values):
    """Calculates absolute volcano-peak deviation values for each structural configuration."""
    df_copy = df.copy()
    for ph in ph_values:
        df_copy[ph] = _to_numeric(df_copy[ph])
        max_val = df_copy[ph].max(skipna=True)
        df_copy[f"Delta_{ph}"] = np.abs(df_copy[ph] - max_val)
    return df_copy


# ----------------------------
# Core Processing Metrics
# ----------------------------

def mean_diff_unweighted(df, alpha_col, delta_col, min_group_n=MIN_GROUP_N):
    """
    Computes S_act = mean(Delta | alpha < 0) - mean(Delta | alpha >= 0).
    Enforces sample cutoff safety thresholds by returning 0.0 on limited pools.
    """
    if alpha_col not in df.columns or delta_col not in df.columns:
        return 0.0
        
    working_frame = df[[alpha_col, delta_col]].copy()
    working_frame[alpha_col] = _to_numeric(working_frame[alpha_col])
    working_frame[delta_col] = _to_numeric(working_frame[delta_col])
    working_frame.dropna(inplace=True)
    
    if working_frame.empty:
        return 0.0
        
    group_ordered = working_frame[working_frame[alpha_col] < 0]
    group_disordered = working_frame[working_frame[alpha_col] >= 0]
    
    if len(group_ordered) < min_group_n or len(group_disordered) < min_group_n:
        return 0.0
        
    return float(group_ordered[delta_col].mean() - group_disordered[delta_col].mean())


def sort_columns_by_importance(score_matrix, alpha_cols, method="abs_sum"):
    """
    Ranks column metrics by environmental cross-pH significance trends.
    
    Default setup ('abs_sum') calculates the absolute summation value across all rows.
    """
    matrix_array = np.asarray(score_matrix, dtype=float)
    if matrix_array.ndim != 2 or matrix_array.shape[1] == 0:
        return np.arange(matrix_array.shape[1]), list(alpha_cols)

    if method == "max_abs":
        calculated_importance = np.nanmax(np.abs(matrix_array), axis=0)
    elif method == "l2":
        calculated_importance = np.sqrt(np.nansum(matrix_array**2, axis=0))
    else:  # Standard cumulative path: abs_sum
        calculated_importance = np.nansum(np.abs(matrix_array), axis=0)

    descending_order = np.argsort(-calculated_importance)
    return descending_order, [alpha_cols[idx] for idx in descending_order]


# ----------------------------
# Graphics Rendering Engine
# ----------------------------

def plot_heatmap_matrix(score_matrix, alpha_labels, ph_values, title="", output_path=None):
    """Generates an aligned multi-row heatmap matrix using non-bold text boundaries."""
    if score_matrix.size == 0:
        print("[WARN] Target score array contains zero values. Terminating plot loop."); return
        
    calculated_width = max(BASE_W_MIN, W_PER_COL * len(alpha_labels) + W_MARGIN)
    calculated_height = BASE_H + H_PER_ROW * len(ph_values)
    
    fig, axis = plt.subplots(figsize=(calculated_width, calculated_height), dpi=DPI)
    heatmap_img = axis.imshow(score_matrix, cmap=CMAP_NAME, norm=TwoSlopeNorm(vcenter=0.0), aspect="auto")

    axis.set_xticks(range(len(alpha_labels)))
    axis.set_xticklabels(alpha_labels, rotation=90, fontsize=XTICK_FONTSIZE)
    
    axis.set_yticks(range(len(ph_values)))
    axis.set_yticklabels(ph_values, rotation=90, va='center', ha='center', fontsize=YTICK_FONTSIZE)
    axis.tick_params(axis='y', which='major', pad=10)

    colorbar = plt.colorbar(heatmap_img, ax=axis, shrink=0.85, pad=0.02)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontsize(CBAR_FONTSIZE)
        
    colorbar.ax.set_title(r"$\mathcal{S}_{\mathrm{act}}$ (a.u.)", fontsize=CBAR_FONTSIZE, pad=15, x=3.0, ha='center')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[INFO] Analysis visualization exported to file -> {output_path}")
    plt.show()


# ----------------------------
# Execution Control Logic
# ----------------------------

def main():
    if not os.path.isfile(EXCEL_FILE):
        raise FileNotFoundError(f"Missing core catalytic Excel source table tracking sheet: {EXCEL_FILE}")
    if not os.path.isfile(CSRO_FILE):
        raise FileNotFoundError(f"Missing short-range order reference descriptors matrix: {CSRO_FILE}")

    df_activity = pd.read_excel(EXCEL_FILE)
    df_csro = pd.read_csv(CSRO_FILE)

    if "material_id_a" not in df_activity.columns:
        raise KeyError("Source Excel data sheet must include an active configuration index column named 'material_id_a'.")
    if "material_id" not in df_csro.columns:
        raise KeyError("Source short-range order file is missing the relational registry column 'material_id'.")

    alpha_feature_cols = _available_alpha_ij_cols(df_csro, pattern=ALPHA_IJ_REGEX)
    if not alpha_feature_cols:
        print("[WARN] Completed search loop with zero active short-range order feature headers tracked."); return

    keep_headers = ["material_id"] + alpha_feature_cols
    merged_dataset = df_activity.merge(df_csro[keep_headers], left_on="material_id_a", right_on="material_id", how="left")
    merged_dataset = compute_delta_log_tof_per_ph(merged_dataset, PH_VALUES)

    # Compute unweighted activity metrics for each pH condition separately
    scores_by_condition = []
    for ph_condition in PH_VALUES:
        target_delta_col = f"Delta_{ph_condition}"
        condition_scores = [
            mean_diff_unweighted(merged_dataset, alpha_col, target_delta_col, min_group_n=MIN_GROUP_N) 
            for alpha_col in alpha_feature_cols
        ]
        scores_by_condition.append(condition_scores)

    score_matrix_raw = np.asarray(scores_by_condition, dtype=float)

    # Reorder parameters based on total cross-pH variance profiles
    sort_indices, sorted_feature_cols = sort_columns_by_importance(score_matrix_raw, alpha_feature_cols, method="abs_sum")
    score_matrix_sorted = score_matrix_raw[:, sort_indices]
    clean_display_labels = [_alpha_label_pair(col_name) for col_name in sorted_feature_cols]

    # Render output dataset visualization package
    runtime_plot_title = f"Unweighted split array summary tracking | Sample limit floor = {MIN_GROUP_N}"
    plot_heatmap_matrix(score_matrix_sorted, clean_display_labels, PH_VALUES, title=runtime_plot_title, output_path=OUTPUT_HEATMAP)


if __name__ == "__main__":
    main()
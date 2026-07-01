# -*- coding: utf-8 -*-
"""
Pointwise Linearly Weighted Mean-Difference Heatmap Generator for SRO Selectivity

This script calculates the selectivity scores (S_sel) across paired configurations 
of different adsorbates (OCHO vs COOH/H) based on an edge-emphasized linear weighting scheme.

Mathematical Logic:
- Base score: score = delta_G_OCHO - delta_G_Y (where Y = COOH or H).
- Weight center: c = (min(score) + max(score)) / 2.
- Pointwise linear weight (w) scales with distance from the center:
    range = max(score) - min(score)
    d = |score - c| / (range + eps) , where d falls in [0, 0.5]
    w = (1 - beta) + beta * (2 * d)
  The hyperparameter 'beta' ∈ [0, 1] controls the intensity of edge-tail weighting.
- Small-sample protection: If the count of data points in either sub-group (alpha < 0 
  vs alpha >= 0) is less than MIN_GROUP_N, the final selection score is forced to 0.0.
- Final output score (S_sel):
    S_sel = mean(score | alpha < 0, weighted) - mean(score | alpha >= 0, weighted)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ----------------------------
# System Environment & Paths
# ----------------------------
# Abstracted root directory placeholder. Replace with your workspace path.
PROJECT_BASE_DIR = "/path/to/your/project/root"

FOLDERS = {
    "COOH": os.path.join(PROJECT_BASE_DIR, "three_elements", "COOH"),
    "H":    os.path.join(PROJECT_BASE_DIR, "three_elements", "H"),
    "OCHO": os.path.join(PROJECT_BASE_DIR, "three_elements", "OCHO"),
}

# Dynamically resolve the shared parent path to keep the pipeline environment-agnostic
OUTPUT_DIR = os.path.join(os.path.commonpath(list(FOLDERS.values())), "csro_ccsro_joint_matching")

SIMILARITY_THRESHOLD = 0.90
PERC = int(round(SIMILARITY_THRESHOLD * 100))
OCHO_ALPHA_IJ_CSV = os.path.join(FOLDERS["OCHO"], "OCHO_site_csro.csv")
ALPHA_IJ_REGEX = r"^alpha_[A-Za-z0-9]+_[A-Za-z0-9]+$"

# Weighting & filtering tuning parameters
BETA = 0.8        # Tail amplification factor (0: uniform weights, 1: center weight hits absolute zero)
MIN_GROUP_N = 2   # Minimum sample floor required per group to prevent statistical noise instabilities


# ----------------------------
# Core Utility Routines
# ----------------------------

def _load_csv_or_warn(path, not_found_msg):
    """Loads target spreadsheet file while handling missing path issues cleanly."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{not_found_msg}: {path}")
    return pd.read_csv(path)


def _to_numeric(s):
    """Converts input series to numeric format, forcing non-parseable entries to NaN."""
    return pd.to_numeric(s, errors="coerce")


def _available_alpha_ij_cols(df, pattern=ALPHA_IJ_REGEX):
    """Scans and extracts column labels matching standard short-range order signatures."""
    pat = re.compile(pattern)
    return [c for c in df.columns if isinstance(c, str) and pat.match(c)]


def _merge_alpha_ij(matches_df, ocho_csro_df, alpha_cols):
    """Merges matched alignment records with central OCHO short-range order descriptors."""
    if "material_id" not in ocho_csro_df.columns:
        raise KeyError("Target OCHO reference table is missing the critical 'material_id' tracking column.")
    keep = ["material_id"] + [c for c in alpha_cols if c in ocho_csro_df.columns]
    merged = matches_df.merge(ocho_csro_df[keep], left_on="material_id_a", right_on="material_id", how="left")
    return merged


def _alpha_ij_to_pair(label):
    """Maps coding format names like 'alpha_Bi_Sn' onto pristine presentation strings 'Bi–Sn'."""
    t = re.sub(r"^alpha_", "", str(label))
    parts = t.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}–{parts[1]}"
    return t


# ----------------------------
# Mathematical Scoring Engine
# ----------------------------

def compute_score(df, x_col, y_col):
    """Evaluates raw energy split profile differences across paired entries."""
    return _to_numeric(df[y_col]) - _to_numeric(df[x_col])


def point_linear_weights(scores, beta=BETA):
    """
    Computes custom linear scaling weights for score arrays.
    Accentuates values distributed close to tail thresholds via mid-point balancing.
    """
    s = np.asarray(scores, dtype=float)
    s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
    c = 0.5 * (s_min + s_max)
    rng = max(s_max - s_min, 1e-12)
    d = np.abs(s - c) / rng          # Maps values onto interval bounding [0, ~0.5]
    w = (1.0 - beta) + beta * (2.0 * d)
    return w


def weighted_mean_diff_pointwise(df, x_col, y_col, alpha_col, beta=BETA, min_group_n=MIN_GROUP_N):
    """
    Evaluates final selectivity index splits between localized ordered vs disordered states.
    Drops the return metric down to flat zero if sub-group frequencies do not satisfy minimum floor limits.
    """
    use = df[[x_col, y_col, alpha_col]].copy()
    use[x_col] = _to_numeric(use[x_col])
    use[y_col] = _to_numeric(use[y_col])
    use[alpha_col] = _to_numeric(use[alpha_col])
    use = use.dropna(subset=[x_col, y_col, alpha_col])
    
    if use.empty:
        return 0.0

    use["score"] = compute_score(use, x_col, y_col)
    use["group"] = (use[alpha_col] >= 0).astype(int)  # Classifies into 0: alpha < 0, 1: alpha >= 0

    n0 = int((use["group"] == 0).sum())
    n1 = int((use["group"] == 1).sum())
    if n0 < min_group_n or n1 < min_group_n:
        return 0.0

    # Derive pointwise linear weight scalars
    w = point_linear_weights(use["score"].to_numpy(), beta=beta)
    use["w"] = w

    # Calculate separate group averages accounting for tail weights
    s0 = use.loc[use["group"] == 0, ["score", "w"]]
    s1 = use.loc[use["group"] == 1, ["score", "w"]]
    
    mean0 = float((s0["score"] * s0["w"]).sum() / s0["w"].sum()) if len(s0) else 0.0
    mean1 = float((s1["score"] * s1["w"]).sum() / s1["w"].sum()) if len(s1) else 0.0
    
    return mean0 - mean1


# ----------------------------
# Graphics Generation Engine
# ----------------------------

def plot_heat(values_df, title="Pointwise linearly weighted mean-diff (two panels)"):
    """Generates dual-track horizontal heatmap profiles mapping chemical indicators without cell values."""
    if values_df.empty:
        print("[WARN] Data matrix empty; aborting plot rendering loop."); return
        
    order = values_df.assign(abs_sum=lambda d: d.abs().sum(axis=1)) \
                     .sort_values("abs_sum", ascending=False).index
    M = values_df.loc[order, ["OCHO_COOH", "OCHO_H"]].to_numpy()

    labels = [_alpha_ij_to_pair(idx) for idx in order]

    fig_w = max(7, 0.22 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4), dpi=300)
    
    im = ax.imshow(M.T, aspect="auto", cmap="YlGnBu", norm=TwoSlopeNorm(vcenter=0.0))
    ax.set_yticks([0, 1])
    
    ax.set_yticklabels(
        ["*COOH/*OCHO", "*H/*OCHO"],
        fontsize=12,
        rotation=90,
        va='center',
        ha='center'
    )

    ax.tick_params(axis='y', which='major', pad=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_title(r"$\mathcal{S}_{\mathrm{sel}}$ (eV)", fontsize=12, pad=10, x=1.8, ha='center')

    plt.tight_layout()
    plt.show()


# ----------------------------
# Main Process Controller
# ----------------------------

def main():
    # Ingest aligned data tables
    path_oc = os.path.join(OUTPUT_DIR, f"pairs_OCHO_COOH_matched_ge_{PERC}.csv")
    path_oh = os.path.join(OUTPUT_DIR, f"pairs_OCHO_H_matched_ge_{PERC}.csv")
    df_oc = _load_csv_or_warn(path_oc, "Missing targeted OCHO--COOH matched compilation file")
    df_oh = _load_csv_or_warn(path_oh, "Missing targeted OCHO--H matched compilation file")

    # Parse structural environment features
    ocho_csro = _load_csv_or_warn(OCHO_ALPHA_IJ_CSV, "Missing baseline OCHO alpha_ij reference sheet")
    alpha_cols = _available_alpha_ij_cols(ocho_csro, pattern=ALPHA_IJ_REGEX)
    if not alpha_cols:
        print("[WARN] No active short-range order indices identified within column declarations."); return

    # Merge dataset rows across target profiles
    df_oc = _merge_alpha_ij(df_oc, ocho_csro, alpha_cols)
    df_oh = _merge_alpha_ij(df_oh, ocho_csro, alpha_cols)

    rows = []
    for a_col in alpha_cols:
        if a_col not in df_oc.columns or a_col not in df_oh.columns:
            continue
        s_oc = weighted_mean_diff_pointwise(df_oc, "delta_G_OCHO", "delta_G_COOH", a_col,
                                            beta=BETA, min_group_n=MIN_GROUP_N)
        s_oh = weighted_mean_diff_pointwise(df_oh, "delta_G_OCHO", "delta_G_H", a_col,
                                            beta=BETA, min_group_n=MIN_GROUP_N)
        rows.append(dict(alpha=a_col, OCHO_COOH=s_oc, OCHO_H=s_oh))

    values_df = pd.DataFrame(rows).set_index("alpha")
    print("[INFO] Summary view of computed metrics (Displaying top 10 profiles):")
    print(values_df.head(10))

    title = f"Pointwise linear weighting | beta={BETA}, min_group={MIN_GROUP_N}"
    plot_heat(values_df, title=title)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Joint Local Environment Vector Matcher & Free Energy Merger

This script aligns structural configurations of different adsorbates (OCHO vs. COOH/H) 
by computing the mathematical similarity between their joint chemical descriptors.

The descriptor vector is a combined representation:
    Joint Vector = [Warren-Cowley CSRO (alpha_ij) + Local Composition Deviation (delta_j)]

Matching Criteria & Workflow:
1. Adsorbates are only paired if they share the same 'Nearest_Atom' (active site element).
2. Computes pair-wise similarity using either Cosine Similarity or L1-Norm normalization.
3. Pairs exceeding the SIMILARITY_THRESHOLD are captured, and their respective 
   Gibbs free energy values (delta_G) are merged from updated Excel reference sheets.
4. Binary preference flags are generated for downstream thermodynamic trend screening:
   - flag_OCHO_vs_X = 1 if delta_G_OCHO < delta_G_X
   - flag_OCHO_vs_X = 0 if delta_G_OCHO > delta_G_X
   - flag_OCHO_vs_X = NaN if values match exactly or data points are missing.
"""

import os
import re
import numpy as np
import pandas as pd

# ----------------------------
# Global System Configurations
# ----------------------------

# Primary directory mapping for chemical environment datasets
BASE_DATA_DIR = "/path/to/your/project/data"

FOLDERS = {
    "COOH": os.path.join(BASE_DATA_DIR, "COOH"),
    "H":    os.path.join(BASE_DATA_DIR, "H"),
    "OCHO": os.path.join(BASE_DATA_DIR, "OCHO"),
}

# Source data tracking metrics from previous structural descriptor extractions
CSRO_FILES  = {k: os.path.join(v, f"{k}_site_csro.csv")       for k, v in FOLDERS.items()}  # alpha_ij values
CCSRO_FILES = {k: os.path.join(v, f"{k}_site_deviation.csv")  for k, v in FOLDERS.items()}  # delta_j values

# References for matching DFT calculated Gibbs Free Energies (delta_G)
EXCEL_FILES = {
    "COOH": os.path.join(BASE_DATA_DIR, "thermo_sheets", "updated_COOH.xlsx"),
    "H":    os.path.join(BASE_DATA_DIR, "thermo_sheets", "updated_H.xlsx"),
    "OCHO": os.path.join(BASE_DATA_DIR, "thermo_sheets", "updated_OCHO.xlsx"),
}

# Standardized output destination for aligned dataset tables
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "joint_environment_matching")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Hyperparameters & Tuning
# ----------------------------
SIMILARITY_METRIC = "cosine"       # Supported metrics: 'cosine' or 'l1_norm'
SIMILARITY_THRESHOLD = 0.90        # Minimum similarity cutoff score to certify a valid structural pair
TOP_K_PRINT_LIMIT = 10             # Number of elite matches displayed in the console logs

# Element baseline arrays for validation sequence tracking
CATALYST_ELEMENTS = ["Bi", "La", "Sn"]
DEFAULT_ALPHA_J_COLS = [f"alpha_{e}" for e in CATALYST_ELEMENTS]


# ----------------------------
# Pattern Recognition Helpers
# ----------------------------

def _get_alpha_ij_columns(df):
    """Filters dataframes to isolate pairwise short-range order string features (alpha_A_B)."""
    pattern = re.compile(r"^alpha_[A-Za-z0-9]+_[A-Za-z0-9]+$")
    return sorted([col for col in df.columns if isinstance(col, str) and pattern.match(col)])


def _get_deviation_columns(df, fallback=None):
    """Identifies and pulls local deviation structural tags (delta_A or legacy alpha_A)."""
    # Accept standard delta nomenclature or check fallback systems
    pattern_delta = re.compile(r"^delta_[A-Za-z]+$")
    pattern_legacy = re.compile(r"^alpha_[A-Za-z]+$")
    
    cols = [c for c in df.columns if isinstance(c, str) and (pattern_delta.match(c) or pattern_legacy.match(c))]
    if fallback:
        cols = [c for c in cols if c in fallback or c.replace("delta_", "alpha_") in fallback]
    return sorted(cols)


def _load_individual_adsorbate_data(adsorbate, csro_paths, ccsro_paths):
    """Loads and marries structural descriptors from decoupled matrix feature sets."""
    csro_path = csro_paths.get(adsorbate)
    ccsro_path = ccsro_paths.get(adsorbate)
    
    if not (csro_path and os.path.isfile(csro_path)):
        raise FileNotFoundError(f"Missing core SRO record array for: {csro_path}")
    if not (ccsro_path and os.path.isfile(ccsro_path)):
        raise FileNotFoundError(f"Missing composition deviation signature array for: {ccsro_path}")

    df_csro = pd.read_csv(csro_path)
    df_deviation = pd.read_csv(ccsro_path)

    # Validate structure rules are maintained
    required_headers = ["material_id", "adsorbate_type", "Nearest_Atom"]
    for current_df, source_label in [(df_csro, csro_path), (df_deviation, ccsro_path)]:
        for req_head in required_headers:
            if req_head not in current_df.columns:
                raise ValueError(f"Required structural registry identifier '{req_head}' missing in: {source_label}")

    alpha_ij_features = _get_alpha_ij_columns(df_csro)
    deviation_features = _get_deviation_columns(df_deviation, fallback=DEFAULT_ALPHA_J_COLS)
    
    if not deviation_features:
        deviation_features = _get_deviation_columns(df_deviation, fallback=None)

    keep_csro = required_headers + alpha_ij_features
    keep_dev = required_headers + deviation_features
    
    # Outer join profiles to catch complete coordinate descriptions
    merged_features = df_csro[keep_csro].merge(df_deviation[keep_dev], on=required_headers, how="outer")
    return merged_features, alpha_ij_features, deviation_features


def load_joint_vectors(csro_paths, ccsro_paths):
    """Assembles unified structural vectors combining local ordering and site deviances."""
    adsorbate_data_registry = {}
    aggregated_ij_cols = set()
    aggregated_j_cols = set()

    for ads in ["OCHO", "COOH", "H"]:
        df, ij_cols, j_cols = _load_individual_adsorbate_data(ads, csro_paths, ccsro_paths)
        adsorbate_data_registry[ads] = df
        aggregated_ij_cols.update(ij_cols)
        aggregated_j_cols.update(j_cols)

    sorted_ij_cols = sorted(aggregated_ij_cols)
    sorted_j_cols = sorted(aggregated_j_cols)
    complete_vector_schema = sorted_ij_cols + sorted_j_cols

    def _homogenize_matrix_columns(df):
        """Pads absent matrix dimensions with flat baseline zeros for alignment compliance."""
        for col in sorted_ij_cols:
            if col not in df.columns:
                df[col] = 0.0
        for col in sorted_j_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        target_schema = ["material_id", "adsorbate_type", "Nearest_Atom"] + complete_vector_schema
        standardized_copy = df[target_schema].copy()
        standardized_copy[complete_vector_schema] = standardized_copy[complete_vector_schema].fillna(0.0)
        return standardized_copy

    unified_pool = pd.concat([_homogenize_matrix_columns(adsorbate_data_registry[ads]) for ads in ["OCHO", "COOH", "H"]], ignore_index=True)
    return unified_pool, complete_vector_schema


# ----------------------------
# Vector Comparison Mechanics
# ----------------------------

def calculate_cosine_similarity(u, v, epsilon=1e-12):
    """Calculates standard directional cosine alignment metric bounds."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < epsilon and norm_v < epsilon: 
        return 1.0
    if norm_u < epsilon or norm_v < epsilon:  
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def calculate_l1_similarity(u, v, epsilon=1e-12):
    """Calculates custom scale-invariant L1 taxicab distance profile boundaries."""
    absolute_error = float(np.sum(np.abs(u - v)))
    scale_factor = float(np.sum(np.abs(u)) + np.sum(np.abs(v)) + epsilon)
    similarity_score = 1.0 - (absolute_error / scale_factor)
    return float(np.clip(similarity_score, 0.0, 1.0))


def evaluate_similarity(u, v, metric="cosine"):
    """Routing hub handling distinct mathematical framework alignments."""
    if metric == "cosine":
        return calculate_cosine_similarity(u, v)
    elif metric == "l1_norm":
        return calculate_l1_similarity(u, v)
    else:
        raise ValueError(f"Requested similarity tracking logic algorithm is not implemented: {metric}")


# ----------------------------
# Data Linking & Analysis
# ----------------------------

def parse_free_energy_profiles(adsorbate, excel_path):
    """Extracts energy rows from raw sheets and standardizes structure naming records."""
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Missing free energy source book mapping: {excel_path}")
    
    df_excel = pd.read_excel(excel_path)

    # Handle slight variations in user formatting for delta_G columns
    alt_match_1 = [c for c in df_excel.columns if str(c).startswith(f"deta_G_{adsorbate}")]
    alt_match_2 = [c for c in df_excel.columns if str(c).startswith(f"delta_G_{adsorbate}")]
    valid_energy_headers = alt_match_1 if len(alt_match_1) > 0 else alt_match_2
    
    if len(valid_energy_headers) != 1:
        raise ValueError(f"Ambiguous energy data columns tracked in {excel_path}. Evaluated options: {valid_energy_headers}")
    energy_col_header = valid_energy_headers[0]

    if adsorbate in ["COOH", "H"]:
        working_set = df_excel[["POSCAR", energy_col_header]].copy()
        working_set = working_set.rename(columns={"POSCAR": "structure_name", energy_col_header: "delta_G"})
        working_set["structure_name"] = working_set["structure_name"].astype(str).str.strip()
        working_set["material_id"] = working_set["structure_name"].apply(lambda label: f"CONTCAR-{label}")
        return dict(zip(working_set["material_id"], working_set["delta_G"]))
        
    elif adsorbate == "OCHO":
        # Account for unique dual-oxygen anchor tracking setup for bidentate format structures
        working_set = df_excel[["O1_adsorb_site", "O2_adsorb_site", energy_col_header]].copy()
        working_set = working_set.rename(columns={energy_col_header: "delta_G"})
        working_set["O1_adsorb_site"] = working_set["O1_adsorb_site"].astype(str).str.strip()
        working_set["O2_adsorb_site"] = working_set["O2_adsorb_site"].astype(str).str.strip()
        working_set["material_id"] = working_set.apply(lambda r: f"CONTCAR-{r['O1_adsorb_site']}-{r['O2_adsorb_site']}", axis=1)
        return dict(zip(working_set["material_id"], working_set["delta_G"]))
    else:
        raise ValueError(f"Unsupported molecular adsorbate configuration setup query: {adsorbate}")


def construct_cross_adsorbate_pairs(unified_df, vector_headers, adsorbate_a, adsorbate_b, metric="cosine"):
    """Performs pair-wise comparison of local profiles restricted to identical catalyst sites."""
    evaluation_records = []
    
    for active_element, group_df in unified_df.groupby("Nearest_Atom", sort=True):
        sub_df_a = group_df[group_df["adsorbate_type"] == adsorbate_a].reset_index(drop=True)
        sub_df_b = group_df[group_df["adsorbate_type"] == adsorbate_b].reset_index(drop=True)
        
        if sub_df_a.empty or sub_df_b.empty:
            continue
            
        matrix_a = sub_df_a[vector_headers].to_numpy(float)
        matrix_b = sub_df_b[vector_headers].to_numpy(float)
        
        for i in range(len(sub_df_a)):
            for j in range(len(sub_df_b)):
                similarity_value = evaluate_similarity(matrix_a[i], matrix_b[j], metric=metric)
                evaluation_records.append({
                    "Nearest_Atom": active_element,
                    "material_id_a": sub_df_a.loc[i, "material_id"],
                    "adsorbate_type_a": adsorbate_a,
                    "material_id_b": sub_df_b.loc[j, "material_id"],
                    "adsorbate_type_b": adsorbate_b,
                    "similarity": similarity_value
                })
                
    return pd.DataFrame(evaluation_records)


def attach_free_energies(pairs_df, adsorbate_a, adsorbate_b, energy_maps):
    """Maps free energy metrics onto existing paired configuration structures."""
    updated_pairs = pairs_df.copy()
    label_a = f"delta_G_{adsorbate_a}"
    label_b = f"delta_G_{adsorbate_b}"
    
    updated_pairs[label_a] = updated_pairs["material_id_a"].map(energy_maps[adsorbate_a])
    updated_pairs[label_b] = updated_pairs["material_id_b"].map(energy_maps[adsorbate_b])
    
    missing_count_a = updated_pairs[label_a].isna().sum()
    missing_count_b = updated_pairs[label_b].isna().sum()
    
    if missing_count_a or missing_count_b:
        print(f"[WARN] Incomplete profile mapping caught -> {label_a}: {missing_count_a} missing; {label_b}: {missing_count_b} missing.")
    return updated_pairs


def evaluate_thermodynamic_preferences(df, adsorbate_a, adsorbate_b):
    """
    Appends binary logic flags pointing to local energy dominance configurations:
      - If delta_G_A < delta_G_B -> 1 (System prefers configuration path A)
      - If delta_G_A > delta_G_B -> 0 (System prefers configuration path B)
      - Ties or empty readings return NaN fields.
    """
    modified_dataframe = df.copy()
    col_a = f"delta_G_{adsorbate_a}"
    col_b = f"delta_G_{adsorbate_b}"
    target_flag_header = f"flag_{adsorbate_a}_vs_{adsorbate_b}"
    
    condition_a_dominant = (modified_dataframe[col_a].notna()) & (modified_dataframe[col_b].notna()) & (modified_dataframe[col_a] < modified_dataframe[col_b])
    condition_b_dominant = (modified_dataframe[col_a].notna()) & (modified_dataframe[col_b].notna()) & (modified_dataframe[col_a] > modified_dataframe[col_b])
    
    modified_dataframe[target_flag_header] = np.nan
    modified_dataframe.loc[condition_a_dominant, target_flag_header] = 1
    modified_dataframe.loc[condition_b_dominant, target_flag_header] = 0
    
    return modified_dataframe, target_flag_header


def export_matched_pairs(all_pairs, filtered_pairs, label_tag, target_dir, cutoff_threshold):
    """Handles routine flat file generation tasks for downstream visualization pipelines."""
    rounded_percent = int(round(cutoff_threshold * 100))
    
    path_raw_all = os.path.join(target_dir, f"pairs_{label_tag}_unfiltered.csv")
    path_matched = os.path.join(target_dir, f"pairs_{label_tag}_matched_ge_{rounded_percent}.csv")
    
    all_pairs.to_csv(path_raw_all, index=False, encoding="utf-8-sig")
    filtered_pairs.to_csv(path_matched, index=False, encoding="utf-8-sig")
    
    # Calculate group totals based on reference benchmark configuration structures
    aggregated_counts = (filtered_pairs
                         .groupby(["Nearest_Atom", "material_id_b"], as_index=False)
                         .agg(matched_counterpart_count=("material_id_a", "nunique")))
                         
    path_counts = os.path.join(target_dir, f"pairs_{label_tag}_frequency_counts_ge_{rounded_percent}.csv")
    aggregated_counts.to_csv(path_counts, index=False, encoding="utf-8-sig")
    
    return path_raw_all, path_matched, path_counts


# ----------------------------
# Execution Driver Engine
# ----------------------------

def main():
    """Sequential processing orchestrator loop execution pathway."""
    unified_structural_pool, global_vector_schema = load_joint_vectors(CSRO_FILES, CCSRO_FILES)
    print(f"[INFO] Constructed Joint Vector Dimension Matrix: {len(global_vector_schema)} parameters wide.")

    # Load lookup dictionaries
    energy_lookup_tables = {
        "COOH": parse_free_energy_profiles("COOH", EXCEL_FILES["COOH"]),
        "H":    parse_free_energy_profiles("H",    EXCEL_FILES["H"]),
        "OCHO": parse_free_energy_profiles("OCHO", EXCEL_FILES["OCHO"]),
    }

    # Construct paired environments
    pairs_ocho_cooh = construct_cross_adsorbate_pairs(unified_structural_pool, global_vector_schema, "OCHO", "COOH", metric=SIMILARITY_METRIC)
    pairs_ocho_h    = construct_cross_adsorbate_pairs(unified_structural_pool, global_vector_schema, "OCHO", "H",    metric=SIMILARITY_METRIC)

    # Attach thermodynamic profiles
    pairs_oc_hydrated = attach_free_energies(pairs_ocho_cooh, "OCHO", "COOH", energy_lookup_tables)
    pairs_oh_hydrated = attach_free_energies(pairs_ocho_h,    "OCHO", "H",    energy_lookup_tables)

    # Isolate highly correlated configurations matching structural vectors
    matches_oc = pairs_oc_hydrated[pairs_oc_hydrated["similarity"] >= SIMILARITY_THRESHOLD].copy()
    matches_oh = pairs_oh_hydrated[pairs_oh_hydrated["similarity"] >= SIMILARITY_THRESHOLD].copy()

    # Append directional preference indicator tags
    matches_oc, flag_header_oc = evaluate_thermodynamic_preferences(matches_oc, "OCHO", "COOH")
    matches_oh, flag_header_oh = evaluate_thermodynamic_preferences(matches_oh, "OCHO", "H")

    print("\n>>> Logging Processing Pipeline Milestones: [OCHO <-> COOH Aligned Sets]")
    oc_raw_out, oc_match_out, oc_count_out = export_matched_pairs(
        pairs_oc_hydrated, matches_oc, label_tag="OCHO_COOH",
        target_dir=OUTPUT_DIR, cutoff_threshold=SIMILARITY_THRESHOLD
    )
    print(f"  [Output Generated] Unfiltered alignment configurations matrix exported -> {oc_raw_out}")
    print(f"  [Output Generated] Profile matches including dominance trends ({flag_header_oc}) exported -> {oc_match_out}")
    print(f"  [Output Generated] Target distribution frequencies compiled -> {oc_count_out}")

    print("\n>>> Logging Processing Pipeline Milestones: [OCHO <-> H Aligned Sets]")
    oh_raw_out, oh_match_out, oh_count_out = export_matched_pairs(
        pairs_oh_hydrated, matches_oh, label_tag="OCHO_H",
        target_dir=OUTPUT_DIR, cutoff_threshold=SIMILARITY_THRESHOLD
    )
    print(f"  [Output Generated] Unfiltered alignment configurations matrix exported -> {oh_raw_out}")
    print(f"  [Output Generated] Profile matches including dominance trends ({flag_header_oh}) exported -> {oh_match_out}")
    print(f"  [Output Generated] Target distribution frequencies compiled -> {oh_count_out}")

    def console_log_elite_matches(pairs_df, descriptor_label):
        """Displays top-tier matches to check output sanity properties directly inside standard error/out streams."""
        if pairs_df.empty:
            print(f"\n[Empty Set Encountered] '{descriptor_label}' contains zero data matrix links.")
            return
        elite_records = pairs_df.sort_values("similarity", ascending=False).head(TOP_K_PRINT_LIMIT)
        print(f"\n[Preview Verification Log] Top-{len(elite_records)} Aligned Structural Pockets for {descriptor_label}:")
        for _, record in elite_records.iterrows():
            second_adsorbate = record['adsorbate_type_b']
            print(f"  -[Site: {record['Nearest_Atom']}] {record['material_id_a']} (OCHO) <-> {record['material_id_b']} ({second_adsorbate}): "
                  f"Similarity Score = {record['similarity']:.4f} | "
                  f"delta_G_OCHO = {record.get('delta_G_OCHO', np.nan)} eV | "
                  f"delta_G_{second_adsorbate} = {record.get('delta_G_' + second_adsorbate, np.nan)} eV")

    console_log_elite_matches(pairs_oc_hydrated, "OCHO vs COOH Systems")
    console_log_elite_matches(pairs_oh_hydrated, "OCHO vs H Systems")

    print(f"\n[Process Terminated Successfully] Cleaned runtime files written out to project directory tree root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
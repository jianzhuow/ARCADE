# -*- coding: utf-8 -*-
"""
Local Warren-Cowley Chemical Short-Range Order (CSRO) & Disorder Analyzer

This script analyzes VASP CONTCAR structures to characterize the local chemical 
environment around adsorbate binding sites (COOH, H, OCHO) on trimetallic catalysts.

Key Workflow:
1. Identifies the closest catalyst surface site to the adsorbate using a 
   covalent-radius-normalized distance metric.
2. Selects a local pocket of catalyst atoms (i-atoms) by taking the union of 
   spherical cutoff regions around all target atoms in the adsorbate molecule.
3. Computes the Warren-Cowley CSRO (alpha_ij) and local compositional disorder 
   for these i-atoms within a defined coordination shell.
4. Exports results to a clean CSV file for each adsorbate type.
"""

import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure

# ----------------------------
# Global Configurations
# ----------------------------

# Dictionary holding paths to directories containing CONTCAR files for each adsorbate
# NOTE: Replace these with your actual relative or absolute paths before running.
DATA_DIRECTORIES = {
    "COOH": os.path.join(".", "data", "COOH"),
    "H":    os.path.join(".", "data", "H"),
    "OCHO": os.path.join(".", "data", "OCHO"),
}

# Specific atom type in each adsorbate used to anchor the binding site identification
ADSORBATE_ANCHORS = {
    "COOH": ["C"],
    "H":    ["H"],
    "OCHO": ["O"],
}

# Active elements present in the catalyst/nanoparticle matrix
CATALYST_ELEMENTS = ["Bi", "La", "Sn"]

# Covalent radii (in Angstroms) used for normalized distance screening
COVALENT_RADII = {
    "H":  0.31,
    "C":  0.76,
    "O":  0.66,
    "Bi": 1.48,
    "La": 2.07,
    "Sn": 1.39,
}

# Distance cutoff parameters (in Angstroms)
LOCAL_RADIUS = 4.5       # Radius to capture the local catalyst cluster (i-atoms) around the adsorbate
SRO_CUTOFF_RADIUS = 3.0  # Cutoff shell for counting neighbors (j-atoms) around each i-atom


# ----------------------------
# Core Helper Functions
# ----------------------------

def get_element_symbol(site):
    """Extracts the element symbol as a string from a pymatgen Site object."""
    return str(site.specie)


def load_structure(file_path):
    """Parses a VASP geometry file (CONTCAR/POSCAR) into a pymatgen Structure."""
    return Structure.from_file(file_path)


def find_binding_site(structure, adsorbate_symbols, catalyst_elements, radii_dict):
    """
    Finds the closest catalyst atom to the adsorbate molecule based on 
    covalent-radius-normalized distance.

    Normalized distance is defined as:
        d_norm = d_cartesian / (r_cov(adsorbate) + r_cov(catalyst))
    
    This helps accurately map the binding site even when dealing with elements
    of drastically different atomic sizes. No periodic boundary conditions (PBC) 
    are applied here (assumes a discrete cluster or surface model slab setup).
    """
    coords = np.array([site.coords for site in structure.sites])
    atom_types = [get_element_symbol(site) for site in structure.sites]

    ads_indices = [i for i, t in enumerate(atom_types) if t in adsorbate_symbols]
    cat_indices = [i for i, t in enumerate(atom_types) if t in catalyst_elements]

    if not ads_indices:
        raise ValueError("Target adsorbate atoms not found in the structure.")
    if not cat_indices:
        raise ValueError("No catalyst atoms detected in the structure.")

    min_norm_dist = float("inf")
    closest_cat_element = None
    closest_cat_idx = None
    closest_ads_idx = None

    # Exhaustive pair search between target adsorbate atoms and catalyst atoms
    for ia in ads_indices:
        type_a = atom_types[ia]
        if type_a not in radii_dict:
            raise KeyError(f"Missing covalent radius definition for adsorbate element: {type_a}")
            
        for ic in cat_indices:
            type_c = atom_types[ic]
            if type_c not in radii_dict:
                raise KeyError(f"Missing covalent radius definition for catalyst element: {type_c}")
                
            d_cartesian = np.linalg.norm(coords[ia] - coords[ic])
            norm_dist = d_cartesian / (radii_dict[type_a] + radii_dict[type_c])
            
            if norm_dist < min_norm_dist:
                min_norm_dist = norm_dist
                closest_cat_idx = ic
                closest_cat_element = type_c
                closest_ads_idx = ia

    return closest_cat_element, closest_cat_idx, closest_ads_idx, min_norm_dist


def compute_global_fractions(atom_types, catalyst_elements):
    """
    Calculates the global molar fractions (c_j) restricted solely to the 
    catalyst atom subset. Adsorbate atoms are ignored in this baseline concentration.
    """
    filtered_cat_atoms = [t for t in atom_types if t in catalyst_elements]
    if not filtered_cat_atoms:
        raise ValueError("No valid catalyst atoms found to compute baseline concentrations.")

    # Preserve order based on original CATALYST_ELEMENTS config
    present_elements = [e for e in catalyst_elements if e in set(filtered_cat_atoms)]
    counts = {e: 0 for e in present_elements}
    
    for t in filtered_cat_atoms:
        if t in counts:
            counts[t] += 1
            
    total_catalyst_atoms = float(len(filtered_cat_atoms))
    global_concentrations = {e: counts[e] / total_catalyst_atoms for e in present_elements}
    
    return present_elements, global_concentrations


def map_local_catalyst_pocket(structure, adsorbate_symbols, catalyst_elements, radius):
    """
    Gathers all catalyst atoms that fall within a defined spherical radius of 
    ANY of the target adsorbate atoms. Returns a sorted, unique set of local catalyst indices.
    """
    coords = np.array([site.coords for site in structure.sites])
    atom_types = [get_element_symbol(site) for site in structure.sites]

    ads_indices = [i for i, t in enumerate(atom_types) if t in adsorbate_symbols]
    if not ads_indices:
        raise ValueError("Target adsorbate atoms missing; cannot map local pocket.")

    pocket_indices = set()
    for ia in ads_indices:
        center_coord = coords[ia]
        distances = np.linalg.norm(coords - center_coord, axis=1)
        
        for k, dist in enumerate(distances):
            if atom_types[k] in catalyst_elements and dist <= radius:
                pocket_indices.add(k)

    return sorted(pocket_indices), ads_indices


def compute_local_sro(structure, local_i_indices, catalyst_elements, sro_cutoff):
    """
    Computes the Warren-Cowley chemical short-range order parameter (alpha_ij) 
    averaged over the selected pocket of local catalyst atoms (i-atoms).
    
    Formula: 
        alpha_ij = 1 - (p_ij / c_j)
    where p_ij is the local coordination fraction of component j around atom i, 
    and c_j is the global baseline concentration of component j.
    """
    if not local_i_indices:
        return {}, 0.0

    coords = np.array([site.coords for site in structure.sites])
    atom_types = [get_element_symbol(site) for site in structure.sites]

    # Get baseline catalyst composition
    species_list, global_conc = compute_global_fractions(atom_types, catalyst_elements)
    sro_tracking = {}

    for i_idx in local_i_indices:
        elem_i = atom_types[i_idx]
        pos_i = coords[i_idx]

        # Calculate distances from current central i-atom to all other atoms
        all_dists = np.linalg.norm(coords - pos_i, axis=1)
        
        # Identify valid neighbors within the cutoff shell (excluding itself and adsorbates)
        neighbors = [
            j for j in range(len(atom_types))
            if (j != i_idx) and (atom_types[j] in catalyst_elements) and (all_dists[j] <= sro_cutoff)
        ]
        coordination_num = len(neighbors)

        # Count occurrences of each element type in the shell
        shell_counts = {e: 0 for e in species_list}
        for j in neighbors:
            elem_j = atom_types[j]
            if elem_j in shell_counts:
                shell_counts[elem_j] += 1

        # Evaluate SRO metrics
        if coordination_num > 0:
            for elem_j in species_list:
                b_j = shell_counts.get(elem_j, 0)
                c_j = global_conc.get(elem_j, 0.0)
                
                if c_j > 0.0:
                    alpha = 1.0 - (b_j / coordination_num) / c_j
                    key = f"alpha_{elem_i}_{elem_j}"
                    sro_tracking.setdefault(key, []).append(alpha)
        else:
            # Handle isolated active atoms gracefully
            for elem_j in species_list:
                key = f"alpha_{elem_i}_{elem_j}"
                sro_tracking.setdefault(key, []).append(0.0)

    # Compute averages across the entire localized target pocket
    averaged_sro = {k: float(np.mean(v)) for k, v in sro_tracking.items()}
    
    # Structural disorder index defined as the mean absolute value of active SRO parameters
    disorder = float(np.mean([abs(val) for val in averaged_sro.values()])) if averaged_sro else 0.0
    
    return averaged_sro, disorder


# ----------------------------
# Batch Processing Driver
# ----------------------------

def process_adsorbate_batch(ads_type, folder_path, adsorb_configs, catalyst_elements, 
                            radii_dict, local_radius, sro_cutoff):
    """
    Loops through all CONTCAR files in a specific folder, extracts the local binding 
    geometry, computes the short-range order metrics, and compiles everything into a CSV.
    """
    if not os.path.exists(folder_path):
        print(f"[{ads_type}] Skipping: Directory not found at '{folder_path}'")
        return

    results = []
    registered_alpha_keys = set()

    # Screen and sort VASP CONTCAR output structures
    files = [f for f in os.listdir(folder_path) if f.startswith("CONTCAR")]
    files.sort()

    for fname in files:
        fpath = os.path.join(folder_path, fname)

        try:
            struct = load_structure(fpath)

            # Determine binding site identity
            nearest_elem, _, _, min_norm = find_binding_site(
                struct, adsorb_configs[ads_type], catalyst_elements, radii_dict
            )

            # Map the local pocket matrix
            local_i_indices, _ = map_local_catalyst_pocket(
                structure=struct,
                adsorbate_symbols=adsorb_configs[ads_type],
                catalyst_elements=catalyst_elements,
                radius=local_radius
            )

            # Compute SRO signatures
            local_sro, disorder = compute_local_sro(
                structure=struct,
                local_i_indices=local_i_indices,
                catalyst_elements=catalyst_elements,
                sro_cutoff=sro_cutoff
            )

        except Exception as e:
            print(f"[{ads_type}] Failed to process {fname}: {e}")
            continue

        # Build data record row
        row = {
            "material_id": os.path.splitext(fname)[0],
            "adsorbate_type": ads_type,
            "Nearest_Atom": nearest_elem,
            "Nearest_Normalized_Distance": min_norm,
            "disorder": disorder,
        }
        
        for k, v in local_sro.items():
            row[k] = v
            registered_alpha_keys.add(k)

        results.append(row)

    if not results:
        print(f"[{ads_type}] No valid records generated. Skipping file generation.")
        return

    # Standardize column layouts for consistency across files
    core_headers = [
        "material_id",
        "adsorbate_type",
        "Nearest_Atom",
        "Nearest_Normalized_Distance",
        "disorder",
    ]
    alpha_headers = sorted(registered_alpha_keys)
    final_column_layout = core_headers + alpha_headers

    df = pd.DataFrame(results)
    
    # Backfill missing interaction pairs with default zeros
    for col in alpha_headers:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df[final_column_layout]

    # Save output dataset
    output_csv = os.path.join(folder_path, f"{ads_type}_site_csro.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[{ads_type}] Analysis saved successfully -> {output_csv}")


def main():
    """Main execution block processing datasets sequentially."""
    for ads_type, folder in DATA_DIRECTORIES.items():
        process_adsorbate_batch(
            ads_type=ads_type,
            folder_path=folder,
            adsorb_configs=ADSORBATE_ANCHORS,
            catalyst_elements=CATALYST_ELEMENTS,
            radii_dict=COVALENT_RADII,
            local_radius=LOCAL_RADIUS,
            sro_cutoff=SRO_CUTOFF_RADIUS,
        )


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Local Compositional/Density Deviation & Structural Disorder Analyzer

This script parses VASP CONTCAR geometry files to evaluate the localized composition 
deviation (delta_j) and atomic disorder surrounding molecular adsorbates (COOH, H, OCHO) 
on trimetallic catalyst surfaces or nanoparticles.

Analytical Workflow:
1. Identifies the primary binding site using a covalent-radius-normalized distance metric.
2. Constructs a localized catalytic region (pocket) by taking the union of spheres 
   (defined by LOCAL_RADIUS) centered on all anchor atoms of the adsorbate molecule.
3. Computes the localized density deviation metric:
       delta_j = 1 - (P_j / c_j)
   where P_j is the local pocket fraction and c_j is the global baseline fraction 
   of catalyst element j.
4. Derives a local structural disorder index by averaging the absolute values of delta_j.
5. Backfills alpha_j values if downstream script compatibility is required.
"""

import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure

# ----------------------------
# Global System Configurations
# ----------------------------

# Dictionary holding paths to directories containing CONTCAR files for each adsorbate
# NOTE: Replace these with your actual relative or absolute paths before running.
DATA_DIRECTORIES = {
    "COOH": os.path.join(".", "data", "COOH"),
    "H":    os.path.join(".", "data", "H"),
    "OCHO": os.path.join(".", "data", "OCHO"),
}

# Anchor elements in each adsorbate used to map out the local surface pocket
ADSORBATE_ANCHORS = {
    "COOH": ["C"],
    "H":    ["H"],
    "OCHO": ["O"],
}

# Principal elements making up the catalyst core matrix
CATALYST_ELEMENTS = ["Bi", "La", "Sn"]

# Tabulated covalent radii (in Angstroms) for normalized proximity calculations
COVALENT_RADII = {
    "H":  0.31,
    "C":  0.76,
    "O":  0.66,
    "Bi": 1.48,
    "La": 2.07,
    "Sn": 1.39,
}

# Spherical cutoff radius (in Angstroms) around adsorbate anchors to harvest the catalyst cluster
LOCAL_RADIUS = 4.5  

# Set True to mirror delta_j values into alpha_j columns to maintain backward compatibility
KEEP_ALPHA_COMPATIBILITY = True


# ----------------------------
# Core Helper Routines
# ----------------------------

def get_element_symbol(site):
    """Returns the clean string representation of a site's element type."""
    return str(site.specie)


def load_structure(file_path):
    """Safely initializes a pymatgen Structure object from a coordinate file."""
    return Structure.from_file(file_path)


def locate_closest_catalyst_site(structure, adsorbate_symbols, catalyst_elements, radii_dict):
    """
    Screens structural pairs to find the absolute closest catalyst surface atom 
    based on a covalent-radius-normalized distance metric.

    Formula used:
        d_norm = d_cartesian / (r_cov(adsorbate) + r_cov(catalyst))
    
    This ensures proper chemical pairing across environments involving atoms 
    with highly disparate ionic/covalent sizes. Periodic boundary conditions (PBC) 
    are omitted (tailored for standalone cluster geometries or standard cluster slices).
    """
    coords = np.array([site.coords for site in structure.sites])
    atom_types = [get_element_symbol(site) for site in structure.sites]

    ads_indices = [i for i, t in enumerate(atom_types) if t in adsorbate_symbols]
    cat_indices = [i for i, t in enumerate(atom_types) if t in catalyst_elements]

    if not ads_indices:
        raise ValueError("Could not find requested adsorbate target atoms in the file structure.")
    if not cat_indices:
        raise ValueError("No substrate catalyst atoms found in the structure.")

    min_norm_dist = float("inf")
    closest_cat_element = None
    closest_cat_idx = None
    closest_ads_idx = None

    for ia in ads_indices:
        type_a = atom_types[ia]
        if type_a not in radii_dict:
            raise KeyError(f"Covalent radius for adsorbate element '{type_a}' is missing from the dictionary.")
            
        for ic in cat_indices:
            type_c = atom_types[ic]
            if type_c not in radii_dict:
                raise KeyError(f"Covalent radius for substrate element '{type_c}' is missing from the dictionary.")
                
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
    Extracts baseline molar fractions (c_j) for elements within the catalyst matrix.
    Any contribution from ambient adsorbates is completely omitted here.
    """
    active_catalyst_pool = [t for t in atom_types if t in catalyst_elements]
    if not active_catalyst_pool:
        raise ValueError("Structure contains zero catalyst elements; baseline cannot be calculated.")

    present_elements = [e for e in catalyst_elements if e in set(active_catalyst_pool)]
    counts = {e: 0 for e in present_elements}
    
    for t in active_catalyst_pool:
        if t in counts:
            counts[t] += 1
            
    total_catalyst_count = float(len(active_catalyst_pool))
    global_fractions = {e: counts[e] / total_catalyst_count for e in present_elements}
    
    return present_elements, global_fractions


def extract_local_pocket_indices(structure, adsorbate_symbols, catalyst_elements, radius):
    """
    Maps out the local neighborhood by drawing spheres around every anchor atom 
    on the adsorbate. Returns the sorted, unique collection of matching catalyst indices.
    """
    coords = np.array([site.coords for site in structure.sites])
    atom_types = [get_element_symbol(site) for site in structure.sites]

    ads_indices = [i for i, t in enumerate(atom_types) if t in adsorbate_symbols]
    if not ads_indices:
        raise ValueError("Adsorbate key tags missing; unable to parse local chemical environment.")

    pocket_indices = set()
    for ia in ads_indices:
        center_xyz = coords[ia]
        distances = np.linalg.norm(coords - center_xyz, axis=1)
        
        for k, dist in enumerate(distances):
            if atom_types[k] in catalyst_elements and dist <= radius:
                pocket_indices.add(k)

    return sorted(pocket_indices), ads_indices


def compute_compositional_deviation(structure, local_indices, catalyst_elements):
    """
    Evaluates density/compositional deviation (delta_j = 1 - P_j / c_j) over the defined pocket zone.
    The final disorder parameter represents the mean absolute value of these shifts.
    """
    if not local_indices:
        return {}, 0.0

    atom_types = [get_element_symbol(site) for site in structure.sites]
    species_list, global_fractions = compute_global_fractions(atom_types, catalyst_elements)

    # Tabulate counts inside the localized environment
    local_counts = {e: 0 for e in species_list}
    for idx in local_indices:
        elem_type = atom_types[idx]
        if elem_type in local_counts:
            local_counts[elem_type] += 1
            
    total_pocket_atoms = float(len(local_indices))
    if total_pocket_atoms == 0:
        return {}, 0.0

    pocket_fractions = {e: (local_counts[e] / total_pocket_atoms) for e in species_list}

    # Solve deviation profiles: delta_j = 1 - P_j / c_j
    delta_metrics = {}
    for e in species_list:
        c_j = global_fractions.get(e, 0.0)
        if c_j > 0.0:
            delta_metrics[e] = 1.0 - (pocket_fractions[e] / c_j)
        else:
            delta_metrics[e] = 0.0

    disorder = float(np.mean([abs(val) for val in delta_metrics.values()])) if delta_metrics else 0.0
    formatted_output = {f"delta_{e}": float(delta_metrics[e]) for e in delta_metrics.keys()}
    
    return formatted_output, disorder


# ----------------------------
# Batch Process Controller
# ----------------------------

def run_adsorbate_directory_analysis(ads_type, folder_path, adsorb_configs, catalyst_elements, 
                                     radii_dict, local_radius, keep_alpha_compat=True):
    """
    Runs systematic profiling across every matching output structure inside a given adsorbate path,
    calculates localized parameters, and formats a compiled performance report to CSV.
    """
    if not os.path.exists(folder_path):
        print(f"[{ads_type}] Target directoy skipped (Path not resolved): '{folder_path}'")
        return

    compiled_records = []
    monitored_columns = set()

    files = [f for f in os.listdir(folder_path) if f.startswith("CONTCAR")]
    files.sort()

    for fname in files:
        fpath = os.path.join(folder_path, fname)

        try:
            struct = load_structure(fpath)

            # Isolate surface adsorption origin coordinates
            nearest_elem, nearest_cat_idx, _, min_norm = locate_closest_catalyst_site(
                struct, adsorb_configs[ads_type], catalyst_elements, radii_dict
            )

            # Isolate localized catalyst framework indices
            local_indices, _ = extract_local_pocket_indices(
                structure=struct,
                adsorbate_symbols=adsorb_configs[ads_type],
                catalyst_elements=catalyst_elements,
                radius=local_radius
            )

            # Fallback action if the calculated cutoff leaves the pocket list empty
            if len(local_indices) == 0 and nearest_cat_idx is not None:
                local_indices = [nearest_cat_idx]

            # Process deviation parameters over cluster area
            local_delta, disorder = compute_compositional_deviation(
                structure=struct,
                local_indices=local_indices,
                catalyst_elements=catalyst_elements
            )

        except Exception as e:
            print(f"[{ads_type}] Parsing anomaly caught on file {fname}: {e}")
            continue

        # Standard baseline dictionary structure
        row_entry = {
            "material_id": os.path.splitext(fname)[0],
            "adsorbate_type": ads_type,
            "Nearest_Atom": nearest_elem,
            "Nearest_Normalized_Distance": min_norm,
            "disorder": disorder,
        }

        # Inject standard delta metrics
        for k, v in local_delta.items():
            row_entry[k] = v
            monitored_columns.add(k)

        # Inject mirrored alpha markers if compatibility flags are active
        if keep_alpha_compat:
            for k, v in local_delta.items():
                element_tag = k.replace("delta_", "")
                alpha_key = f"alpha_{element_tag}"
                row_entry[alpha_key] = v
                monitored_columns.add(alpha_key)

        compiled_records.append(row_entry)

    if not compiled_records:
        print(f"[{ads_type}] No output fields recorded. Terminating routine without file export.")
        return

    # Frame structure schema rules
    structural_headers = [
        "material_id",
        "adsorbate_type",
        "Nearest_Atom",
        "Nearest_Normalized_Distance",
        "disorder",
    ]
    metric_headers = sorted(monitored_columns)
    final_table_schema = structural_headers + metric_headers

    df = pd.DataFrame(compiled_records)
    
    # Fill any null measurements with neutral zero fields
    for col in metric_headers:
        if col not in df.columns:
            df[col] = 0.0
    df = df[final_table_schema]

    # Write out data package
    output_filename = os.path.join(folder_path, f"{ads_type}_site_deviation.csv")
    df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"[{ads_type}] Evaluation successfully exported -> {output_filename}")


def main():
    """Sequentially loops through global chemical dataset configurations."""
    for ads_type, directory in DATA_DIRECTORIES.items():
        run_adsorbate_directory_analysis(
            ads_type=ads_type,
            folder_path=directory,
            adsorb_configs=ADSORBATE_ANCHORS,
            catalyst_elements=CATALYST_ELEMENTS,
            radii_dict=COVALENT_RADII,
            local_radius=LOCAL_RADIUS,
            keep_alpha_compat=KEEP_ALPHA_COMPATIBILITY,
        )


if __name__ == "__main__":
    main()
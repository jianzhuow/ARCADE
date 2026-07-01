from ase.db import connect
import pandas as pd
from ase import Atoms, Atom
from ase.constraints import FixAtoms
from ase.build import fcc111, add_adsorbate
import numpy as np
import itertools as it
from copy import deepcopy
import os
import shutil
from ase.io import read, write
from ase.visualize import view
from ase.neighborlist import NeighborList
from scipy.spatial import Voronoi, ConvexHull  # Import ConvexHull
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

structure = read('./CONTCAR')
def calculate_voronoi_areas(atoms):
    positions = atoms.get_positions()
    
    # Perform Voronoi tessellation
    vor = Voronoi(positions)
    
    # Initialize area list for each atom
    areas = np.zeros(len(positions))
    
    # Iterate over the Voronoi vertices and calculate areas
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        
        # Exclude unbounded Voronoi regions
        if -1 in region:
            areas[i] = np.inf
        else:
            polygon = [vor.vertices[j] for j in region]
            # In 3D, ConvexHull.volume is the volume of the Voronoi cell.
            areas[i] = ConvexHull(polygon).volume
    
    return areas

# Keep only Bi and Sn atoms from the supplied indices.
def filter_bi_sn_atoms(atoms, indices):
    symbols = atoms.get_chemical_symbols()
    return [i for i in indices if symbols[i] in ['Bi', 'Sn']]

areas = calculate_voronoi_areas(structure)
threshold = 23 # Adjust the threshold as needed
outer_atoms_indices = [i for i, area in enumerate(areas) if area > threshold]
#bi_sn_outer_atoms_indices = filter_bi_sn_atoms(structure, outer_atoms_indices)
all_indices = set(range(len(structure)))

# Freeze atoms that are not classified as surface atoms.
inner_atoms_indices = list(all_indices - set(outer_atoms_indices))
constraint = FixAtoms(inner_atoms_indices )
structure.set_constraint(constraint)


def create_neighbor_list(structure, cutoff=2.5):
    """Create the neighbor list."""
    nl = NeighborList([cutoff / 2 * 1.1] * len(structure), bothways=True, self_interaction=False)
    nl.update(structure)
    return nl

def filter_surface_neighbors(neighbor_indices, outer_atoms_indices):
    """Keep only neighbors classified as surface atoms."""
    return np.array([idx for idx in neighbor_indices if idx in outer_atoms_indices])
    
def calculate_adsorption_direction(adsorb_atom_position, cluster_center, distance_offset=2):
    """Return an H position displaced outward from an adsorption site."""
    up_direction = adsorb_atom_position - cluster_center
    up_direction /= np.linalg.norm(up_direction)
    return adsorb_atom_position + up_direction * distance_offset

def adjust_hydrogen_position(h_position, neighbor_position, min_distance, max_distance, step_size):
    """Adjust an H position until its neighbor distance is in range."""
    while True:
        current_distance = np.linalg.norm(h_position - neighbor_position)
        if min_distance < current_distance < max_distance:
            return h_position
        direction_vector = (h_position - neighbor_position) if current_distance < min_distance else (neighbor_position - h_position)
        direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
        h_position += direction_unit_vector * step_size

def ensure_target_distance(h_position, adsorb_atom_position, target_distance):
    """Enforce a minimum H-metal distance."""
    current_distance = np.linalg.norm(h_position - adsorb_atom_position)
    if current_distance < target_distance:
        direction = h_position - adsorb_atom_position
        direction /= np.linalg.norm(direction)
        h_position += direction * (target_distance - current_distance)
    return h_position


def place_hydrogen(structure, target_position, neighbors, cluster_center,
                   distance_offset=2, min_distance=3.1, max_distance=3.6,
                   step_size=0.1, target_distance=2):
    """Return an initial H position satisfying the requested distance constraints."""
    h_position = calculate_adsorption_direction(
        target_position, cluster_center, distance_offset
    )
    
    # Adjust H relative to neighboring surface atoms.
    for neighbor_index in neighbors:
        neighbor_position = structure.get_positions()[neighbor_index]
        h_position = adjust_hydrogen_position(
            h_position, neighbor_position, min_distance, max_distance, step_size
        )
    
    # Enforce the requested H-metal distance.
    return ensure_target_distance(h_position, target_position, target_distance)


# Initialize nanoparticle geometry data.
atom_combination_index = []
positions = structure.get_positions()
cluster_center = np.mean(positions, axis=0)

nl = create_neighbor_list(structure)
adsorb_count  = 0

# Generate one H adsorption structure for each surface site.
atom_combination_index = []
positions = structure.get_positions()
cluster_center = np.mean(positions, axis=0)
nl = create_neighbor_list(structure)
adsorb_count  = 0
for adsorb_atom_index in outer_atoms_indices:
    adsorb_count += 1
    adsorb_atom_position = structure.get_positions()[adsorb_atom_index]
    target_position = positions[adsorb_atom_index]
    # Obtain neighbors and keep only surface atoms.
    neighbor_indices, offsets = nl.get_neighbors(adsorb_atom_index)
    filtered_neighbors = filter_surface_neighbors(neighbor_indices, outer_atoms_indices)
    
    # Place one hydrogen atom above the selected site.
    h_position = place_hydrogen(
        structure=structure,
        target_position=adsorb_atom_position,
        neighbors=filtered_neighbors,
        cluster_center=cluster_center,
        distance_offset=2,
        min_distance=3.1,
        max_distance=3.6,
        step_size=0.1,
        target_distance=2,
    )
    h_atom = Atoms('H', positions=[h_position])
    system = structure + h_atom

            
    adsorb_metal = structure[adsorb_atom_index].symbol
    #neighbor_metal = structure[neighbor_index].symbol
    directory_1 = os.path.join(f"{adsorb_count}_{adsorb_metal}_{adsorb_atom_index}")

    if not os.path.exists(directory_1):
        os.makedirs(directory_1)
    #directory_2 = os.path.join(directory_1,f"{neighbor_count}_{neighbor_metal}_{neighbor_index}")
    #if not os.path.exists(directory_2):
    	#os.makedirs(directory_2)

    system.write(f'{directory_1}/POSCAR')

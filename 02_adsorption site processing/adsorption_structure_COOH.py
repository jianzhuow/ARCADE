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
    """Return a point displaced outward from an adsorption site."""
    up_direction = adsorb_atom_position - cluster_center
    up_direction /= np.linalg.norm(up_direction)
    return adsorb_atom_position + up_direction * distance_offset

def adjust_oxygen_position(o_position, neighbor_position, min_distance, max_distance, step_size):
    """Adjust a trial adsorbate position until its neighbor distance is in range."""
    while True:
        current_distance = np.linalg.norm(o_position - neighbor_position)
        if min_distance < current_distance < max_distance:
            return o_position
        direction_vector = (o_position - neighbor_position) if current_distance < min_distance else (neighbor_position - o_position)
        direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
        o_position += direction_unit_vector * step_size
    return o_position

def ensure_target_distance(o_position, adsorb_atom_position, target_distance):
    """Enforce a minimum distance between the trial adsorbate and target atom."""
    distance_c_atom = np.linalg.norm(o_position - adsorb_atom_position)
    if distance_c_atom < target_distance:
        direction_c_atom_vector = o_position - adsorb_atom_position
        direction_c_atom_unit_vector = direction_c_atom_vector / np.linalg.norm(direction_c_atom_vector)
        o_position += direction_c_atom_unit_vector * (target_distance - distance_c_atom)
    return o_position
def place_carbon(structure, target_position, neighbors, cluster_center, distance_offset=2, min_distance=3.1, max_distance=3.6, step_size=0.1, target_distance=2):
    """Return the initial carbon position while satisfying neighbor-distance constraints."""
    # target_position: coordinates of the adsorption site.
    # neighbors: indices of atoms adjacent to the adsorption site.
    o_position = calculate_adsorption_direction(target_position, cluster_center, distance_offset)
    
    # Adjust the trial carbon position relative to nearby surface atoms.
    for neighbor_index in neighbors:
        neighbor_position = structure.get_positions()[neighbor_index]
        o_position = adjust_oxygen_position(o_position, neighbor_position, min_distance, max_distance, step_size)
    
    # Enforce the requested carbon-metal distance.
    o_position = ensure_target_distance(o_position, target_position, target_distance)
    
    return o_position


# Initialize nanoparticle geometry data.
atom_combination_index = []
positions = structure.get_positions()
cluster_center = np.mean(positions, axis=0)

nl = create_neighbor_list(structure)
adsorb_count  = 0

# Generate one COOH structure for each surface adsorption site.
atom_combination_index = []
positions = structure.get_positions()
cluster_center = np.mean(positions, axis=0)
nl = create_neighbor_list(structure)
adsorb_count  = 0
for adsorb_atom_index in outer_atoms_indices:
    adsorb_count += 1
    adsorb_atom_position = structure.get_positions()[adsorb_atom_index]
    target_position = positions[adsorb_atom_index]
    up_direction = target_position - cluster_center
    up_direction /= np.linalg.norm(up_direction)  # Normalize.
    # Obtain neighbors and keep only surface atoms.
    neighbor_indices, offsets = nl.get_neighbors(adsorb_atom_index)
    filtered_neighbors = filter_surface_neighbors(neighbor_indices, outer_atoms_indices)
    
    # Place the carbon atom above the selected surface atom.
    c_position = place_carbon(structure=structure,target_position=adsorb_atom_position,neighbors=filtered_neighbors,
                            cluster_center=cluster_center,distance_offset=2,min_distance=3.1,max_distance=3.6,step_size=0.1,target_distance=2)
    c_atom = Atoms('C', positions=[c_position])
    distance_offset_O = 1  # Initial C-O offset.
    
    up_c_direction = c_position - target_position  # Direction from the metal site to C.
    up_c_direction /= np.linalg.norm(up_c_direction)  # Normalize.

    # Set initial positions for the two oxygen atoms.
    o1_position = c_position + up_direction * distance_offset_O
    o2_position = c_position + up_c_direction * distance_offset_O

   # distance_O1_O2 = np.linalg.norm(o1_position - o2_position)
    mid_point_O = (o1_position + o2_position) / 2
    c_to_mid_vector = mid_point_O - c_position
    c_to_mid_unit_vector = c_to_mid_vector / np.linalg.norm(c_to_mid_vector)
    
    # Define vectors between the two oxygen atoms.
    o1_to_o2_vector = o2_position - o1_position
    o1_to_o2_unit_vector = o1_to_o2_vector / np.linalg.norm(o1_to_o2_vector)
    o2_to_o1_vector = o1_position - o2_position
    o2_to_o1_unit_vector = o2_to_o1_vector / np.linalg.norm(o2_to_o1_vector)
    
    # Use a dot product to test perpendicularity.
    dot_product = np.dot(c_to_mid_unit_vector, o1_to_o2_unit_vector)
    
    # Check whether the vectors are perpendicular.
    tolerance = 1e-6
    if abs(dot_product) > tolerance:
        # Adjust both oxygen positions when they are not perpendicular.
        perpendicular_component = np.cross(o1_to_o2_unit_vector, c_to_mid_unit_vector)
        perpendicular_component /= np.linalg.norm(perpendicular_component)
        
        # Update O1 and O2 symmetrically around their midpoint.
        adjustment_distance = np.linalg.norm(c_to_mid_vector) * dot_product
        o1_position += perpendicular_component * (adjustment_distance / 2)
        o2_position -= perpendicular_component * (adjustment_distance / 2)
    # Adjust the O-O separation.
    distance_O1_O2 = np.linalg.norm(o1_position - o2_position)
    
    if distance_O1_O2 < 2.3:
        tune_step = (2.1-distance_O1_O2)/2
        o2_position = o2_position + o1_to_o2_unit_vector * tune_step
        o1_position = o1_position + o2_to_o1_unit_vector * tune_step

    o1_to_c_vector = c_position - o1_position
    o1_to_c_vector_unit_vector = o1_to_c_vector/np.linalg.norm(o1_to_c_vector)
    h_position = o2_position + c_to_mid_unit_vector * 0.9

    o1_atom = Atoms('O', positions=[o1_position])
    o2_atom = Atoms('O', positions=[o2_position])
    h_atom = Atoms('H', positions=[h_position])
    system = structure + c_atom + o1_atom + o2_atom + h_atom  # Add COOH to the nanoparticle.
    # view(system)

            
    adsorb_metal = structure[adsorb_atom_index].symbol
    #neighbor_metal = structure[neighbor_index].symbol
    directory_1 = os.path.join(f"{adsorb_count}_{adsorb_metal}_{adsorb_atom_index}")

    if not os.path.exists(directory_1):
        os.makedirs(directory_1)
    #directory_2 = os.path.join(directory_1,f"{neighbor_count}_{neighbor_metal}_{neighbor_index}")
    #if not os.path.exists(directory_2):
    	#os.makedirs(directory_2)

    system.write(f'{directory_1}/POSCAR')

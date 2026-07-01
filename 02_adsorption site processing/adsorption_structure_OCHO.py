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
import random
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
threshold = 25 # Adjust the threshold as needed

outer_atoms_indices = [i for i, area in enumerate(areas) if area > threshold]
bi_sn_outer_atoms_indices = filter_bi_sn_atoms(structure, outer_atoms_indices)
all_indices = set(range(len(structure)))

# Freeze atoms that are not classified as surface atoms.
inner_atoms_indices = list(all_indices - set(outer_atoms_indices))
constraint = FixAtoms(inner_atoms_indices )
structure.set_constraint(constraint)


def create_neighbor_list(structure, cutoff=3):
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

def ensure_target_distance(o_position, adsorb_atom_position, target_distance):
    """Place an oxygen atom at the requested distance from a target atom."""
    distance_c_atom = np.linalg.norm(o_position - adsorb_atom_position)
    if distance_c_atom < target_distance:
        direction_c_atom_vector = o_position - adsorb_atom_position
        direction_c_atom_unit_vector = direction_c_atom_vector / np.linalg.norm(direction_c_atom_vector)
        o_position += direction_c_atom_unit_vector * (target_distance - distance_c_atom)
    elif distance_c_atom > target_distance:
        direction_c_atom_vector =  adsorb_atom_position - o_position 
        direction_c_atom_unit_vector = direction_c_atom_vector / np.linalg.norm(direction_c_atom_vector)
        o_position = o_position + direction_c_atom_unit_vector * (distance_c_atom-target_distance)
    return o_position

def add_oxygen(structure, filtered_neighbors, adsorb_atom_position,cluster_center):
    if len(filtered_neighbors) > 0:
        neighbor_positions = structure.positions[filtered_neighbors]
        neighbor_positions_mean = np.mean(neighbor_positions, axis=0)
        # Estimate the local surface normal from neighboring surface atoms.
        normal_vector = adsorb_atom_position - neighbor_positions_mean
        normal_vector /= np.linalg.norm(normal_vector)  # Normalize.
        # Ensure that the normal points away from the cluster center.
        if np.dot(normal_vector, adsorb_atom_position - cluster_center) < 0:
            normal_vector = -normal_vector
    else:
        normal_vector = np.array([0, 0, 1])  # Fallback direction.
    o_position = adsorb_atom_position + normal_vector * 2    
    return o_position,normal_vector

def adjust_oxygen_position(o_position, neighbor_position, min_distance, max_distance, step_size):
    """Adjust an oxygen position until its neighbor distance is in range."""
    while True:
        current_distance = np.linalg.norm(o_position - neighbor_position)
        if min_distance < current_distance < max_distance:
            return o_position  # The target range is satisfied.
        direction_vector = (o_position - neighbor_position) if current_distance < min_distance else (neighbor_position - o_position)
        direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
        o_position += direction_unit_vector * step_size
    return o_position
    
# def adjust_two_oxygen_position(molecure_position, opposite_adsorb_position, step_size,limit_size):
#     while True:
#         # Calculate the current distance to the neighboring atom.
#         current_distance = np.linalg.norm(molecure_position - opposite_adsorb_position)
                
#         # Check whether the distance is in the target range.
#         if current_distance < limit_size:
#             # print("The current distance is acceptable:", current_distance)
#             break
                
#         else: #current_distance > max_distance:
#             # Calculate the direction and update the oxygen position.
#             #print(current_distance)
#             direction_vector = opposite_adsorb_position - molecure_position
#             direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
#             molecure_position += direction_unit_vector * step_size
#             direction_vector_opposite = molecure_position - opposite_adsorb_position
#             direction_vector_opposite_unit = direction_vector_opposite / np.linalg.norm(direction_vector_opposite)
#             opposite_adsorb_position +=  direction_vector_opposite_unit * step_size
#     molecure_position_new = molecure_position      
#     opposite_adsorb_position_new = opposite_adsorb_position
#     return molecure_position_new, opposite_adsorb_position_new

    
def adjust_two_oxygen_position(o1_position, o2_position):
    midpoint_position = (o1_position + o2_position) / 2
        
    distance_O1_O2 = np.linalg.norm(o1_position - o2_position)
    difference = distance_O1_O2 - 2.3
    direction_vector_toward_O1 = o1_position - o2_position
    direction_vector_toward_O1 /= np.linalg.norm(direction_vector_toward_O1)  # Normalize.
        
    direction_vector_toward_O2 = o2_position - o1_position
    direction_vector_toward_O2 /= np.linalg.norm(direction_vector_toward_O2)  # Normalize.
        
    if difference > 0:
        o1_position_new = o1_position + direction_vector_toward_O2 * (difference/2 * 1)
        o2_position_new = o2_position + direction_vector_toward_O1 * (difference/2 * 1)
    elif difference < 0:
        o1_position_new = o1_position + direction_vector_toward_O1 * (difference/2 * 1)
        o2_position_new = o2_position + direction_vector_toward_O2 * (difference/2 * 1)
    elif difference == 0:
        o1_position_new = o1_position
        o2_position_new = o2_position
    return o1_position_new, o2_position_new


def adjust_carbon_positions(c_position, oxygen_position1, oxygen_position2, tolerance=0.005, step_size1=0.005, step_size2=0.005):

    while True:
        distances_O1 = np.linalg.norm(c_position - oxygen_position1)
        distances_O2 = np.linalg.norm(c_position - oxygen_position2)
        
        # Stop when the two C-O distances are nearly equal.
        if abs(distances_O1 - distances_O2) < tolerance:
            # print("The current distance is acceptable:", distances_O1)
            break
        
        # Move carbon according to the C-O distance difference.
        if distances_O1 > distances_O2:
            # Move relative to the first oxygen atom.
            direction_vector = oxygen_position1 - oxygen_position2
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            c_position += direction_unit_vector * step_size1
        else:
            # Move relative to the second oxygen atom.
            direction_vector = oxygen_position2 - oxygen_position1
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            c_position += direction_unit_vector * step_size2

    return c_position

def adjust_carbon_within_distance(c_position, oxygen_position1, oxygen_position2, midpoint_position, min_distance=1.35, max_distance=1.4, step_size_far=0.02, step_size_close=0.01):
    while True:
        distances_O1 = np.linalg.norm(c_position - oxygen_position1)
        distances_O2 = np.linalg.norm(c_position - oxygen_position2)
        
        # Check whether the C-O distance is in the target range.
        if min_distance < distances_O1 < max_distance:
            # print("The current distance is acceptable:", distances_O1)
            break
        
        # Adjust the carbon position according to its C-O distance.
        if distances_O1 > max_distance:
            # Move carbon toward the oxygen midpoint.
            direction_vector = midpoint_position - c_position
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            c_position += direction_unit_vector * step_size_far
        elif distances_O1 < min_distance:
            # Move carbon away from the oxygen midpoint.
            direction_vector = c_position - midpoint_position
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            c_position += direction_unit_vector * step_size_close
        
        # Update distances.
        distances_O1 = np.linalg.norm(c_position - oxygen_position1)
        distances_O2 = np.linalg.norm(c_position - oxygen_position2)

    return c_position
    
def adjust_o1_based_on_distance(o1_position, filtered_neighbors, positions):
    # Collect neighboring atom coordinates.
    neighbor_positions = np.array([positions[id] for id in filtered_neighbors])
    
    # Calculate distances from O1 to each neighbor.
    distances = np.linalg.norm(neighbor_positions - o1_position, axis=1)
    avg_distance = np.mean(distances)
    #print(f"Average distance to neighbors: {avg_distance:.3f}")
    
    # Quantify the spread in O1-neighbor distances.
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    distance_variation = max_distance - min_distance
    #print(f"Max distance: {max_distance:.3f}, Min distance: {min_distance:.3f}, Variation: {distance_variation:.3f}")
    #print(distances)
    return avg_distance, distance_variation

# def adjust_oxygen_position(o_position, neighbor_position, min_distance, max_distance, step_size):
#     """Adjust an oxygen position according to its neighbor distance."""
#     while True:
#         current_distance = np.linalg.norm(o_position - neighbor_position)
#         if min_distance < current_distance < max_distance:
#             return o_position
#         direction_vector = (o_position - neighbor_position) if current_distance < min_distance else (neighbor_position - o_position)
#         direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
#         o_position += direction_unit_vector * step_size
#     return o_position
    
def change_oxygen(structure, target_position, o_position, neighbors, distance_offset=2, min_distance=3.1, max_distance=3.4, step_size=0.15, target_distance=2):   
    # Adjust the oxygen position to satisfy neighbor-distance requirements.
    for neighbor_index in neighbors:
        neighbor_position = structure.get_positions()[neighbor_index]
        o_position = adjust_oxygen_position(o_position, neighbor_position, min_distance, max_distance, step_size)   
    # Restore the requested oxygen-target distance.
    o_position = ensure_target_distance(o_position, target_position, target_distance)    
    return o_position
    
    
atom_combination_index = []
positions = structure.get_positions()
cluster_center = np.mean(positions, axis=0)

nl = create_neighbor_list(structure)
adsorb_count  = 0
for adsorb_atom_index in outer_atoms_indices:
    adsorb_count += 1
    print(adsorb_atom_index)
    adsorb_atom_position = structure.get_positions()[adsorb_atom_index]
    
    # Obtain neighbors and keep only surface atoms.
    neighbor_indices, offsets = nl.get_neighbors(adsorb_atom_index)
    filtered_neighbors = filter_surface_neighbors(neighbor_indices, outer_atoms_indices)
    
    # Place the first anchoring oxygen atom (O1).
    o1_position,normal_vector1 = add_oxygen(structure, filtered_neighbors,adsorb_atom_position,cluster_center)
    for id in list(filtered_neighbors):
        distance = np.linalg.norm(positions[id] - o1_position)
        print(id)
        print(distance)
    avg_distance, distance_variation = adjust_o1_based_on_distance(o1_position, filtered_neighbors, positions)
    if distance_variation > 0.9: 
        print('yes,distance_variation is larger')
        o1_position1 = change_oxygen(structure, adsorb_atom_position, o1_position, filtered_neighbors, distance_offset=2, min_distance=avg_distance, max_distance=avg_distance+0.5, step_size=0.05, target_distance=2)
        for id in list(filtered_neighbors):
            distance = np.linalg.norm(positions[id] - o1_position)
            print(id)
            print(distance)
    else:
        o1_position1 = o1_position
    # for id in list(filtered_neighbors):
    #     distance = np.linalg.norm(positions[id] - o1_position)
    #     print(id)
    #     print(distance)
    #ef add_oxygen(structure, filtered_neighbors, adsorb_atom_position,cluster_center):
    #print("First O atom position (O1) for adsorb_atom_index", adsorb_atom_index, ":", o1_position)
    neighbor_count  = 0
    # Place the second anchoring oxygen atom (O2) on a neighboring site.
    for neighbor_index in filtered_neighbors:
        neighbor_count  += 1
        print(neighbor_index)
        if [adsorb_atom_index, neighbor_index] in atom_combination_index or [neighbor_index, adsorb_atom_index] in atom_combination_index:
            print('jump')
            continue
        else:
            neighbor_position = structure.get_positions()[neighbor_index]
            neighbor_neighbor_indices, offsets = nl.get_neighbors(neighbor_index)
            
            filtered_neighbors_neighbor = filter_surface_neighbors(neighbor_neighbor_indices, outer_atoms_indices)
            #o2_position = neighbor_position + normal_vector * 2
            o2_position,normal_vector2 = add_oxygen(structure,filtered_neighbors_neighbor,neighbor_position,cluster_center)
            for id1 in list(filtered_neighbors_neighbor):
                distance1 = np.linalg.norm(positions[id1] - o2_position)
                #print(id1)
                #print(distance1)
            avg_distance1, distance_variation1 = adjust_o1_based_on_distance(o2_position, filtered_neighbors_neighbor, positions)
            if distance_variation1 > 0.9: 
                #print('yes,distance_variation1 is larger')
                o2_position1 = change_oxygen(structure, neighbor_position, o2_position, filtered_neighbors_neighbor, distance_offset=2, min_distance=avg_distance1, max_distance=avg_distance1+0.5, step_size=0.05, target_distance=2)
                for id1 in list(filtered_neighbors_neighbor):
                    distance1 = np.linalg.norm(positions[id1] - o2_position)
                    print(id1)
                    print(distance1)
            else:
                o2_position1 = o2_position
                
            midpoint_two_metal_position = (adsorb_atom_position + neighbor_position) / 2
            
            o1_position1_new = adjust_oxygen_position(o1_position1, neighbor_position, 3.6, 3.8, 0.01)
            o2_position1_new = adjust_oxygen_position(o2_position1, adsorb_atom_position, 3.6, 3.8, 0.01)
            
            o1_position_new,o2_position_new = adjust_two_oxygen_position(o1_position1_new, o2_position1_new)

            o1_position_new11 =  ensure_target_distance(o1_position_new, adsorb_atom_position, 2)
            o2_position_new21 =  ensure_target_distance(o2_position_new, neighbor_position, 2)
           # o1_position_new,o2_position_new = adjust_two_oxygen_position(o1_position1, o2_position1, step_size=0.15,limit_size=3)
            #o1_position_new11 =  ensure_target_distance(o1_position_new, adsorb_atom_position, 2)
            #o2_position_new21 =  ensure_target_distance(o2_position_new, neighbor_position, 2)
            o2_position_new_211 = adjust_oxygen_position(o2_position_new21, o1_position_new11, min_distance=2.2, max_distance=2.3, step_size=0.05)
            #adjust_oxygen_position(o_position, neighbor_position, min_distance, max_distance, step_size):
        # # # ##################################C adsorption ######################################
            midpoint_position = (o1_position_new11 + o2_position_new21) / 2
    
            # Define the outward direction through the midpoint of the two O atoms.
            direction_from_cluster = midpoint_position - cluster_center
            direction_from_cluster /= np.linalg.norm(direction_from_cluster)  # Normalize.
            
            # Initialize carbon farther away from the cluster center.
            c_position = midpoint_position + direction_from_cluster * 1.5
            c_position_new = adjust_carbon_positions(c_position, o1_position_new11,o2_position_new21, tolerance=0.005, step_size1=0.005, step_size2=0.005)
            #distances_c_o = np.linalg.norm(c_position - o2_position_new_211)
            c_position_new1 = adjust_carbon_within_distance(c_position_new, o1_position_new11, o2_position_new21, midpoint_position, min_distance=1.35, max_distance=1.4, step_size_far=0.02, step_size_close=0.01)
            
            direction_middle_C = c_position_new1 - midpoint_position  
            direction_middle_C = direction_middle_C / np.linalg.norm(direction_middle_C)
    
            h_position = c_position_new1 + direction_middle_C * 1.09
            
            o_atom1 = Atoms('O', positions=[o1_position_new11])
            o_atom2 = Atoms('O', positions=[o2_position_new_211])
            c_atom = Atoms('C', positions=[c_position_new1])
            h_atom = Atoms('H', positions=[h_position])
            system = structure + o_atom1 + o_atom2 + c_atom + h_atom  # Add OCHO to the nanoparticle.
            atom_combination_index.append([adsorb_atom_index,neighbor_index])
            #view(system) 
            adsorb_metal = structure[adsorb_atom_index].symbol
            neighbor_metal = structure[neighbor_index].symbol
            directory_1 = os.path.join(f"{adsorb_count}_{adsorb_metal}_{adsorb_atom_index}")
                # Create directory if it doesn't exist
            if not os.path.exists(directory_1):
                os.makedirs(directory_1)
            directory_2 = os.path.join(directory_1,f"{neighbor_count}_{neighbor_metal}_{neighbor_index}")
            if not os.path.exists(directory_2):
    	        os.makedirs(directory_2)

            system.write(f'{directory_2}/POSCAR')

import numpy as np
from pymatgen.core import Structure
# 'logger' and 'argparse' are not used in this class, so they are removed for clarity.
# If your full script uses them elsewhere, you can keep them.

class lattice:
    """
    Class representing a crystal lattice structure.

    This class uses the pymatgen library for highly efficient neighbor list generation,
    and can partition neighbors into multiple shells based on cutoff distances.
    """
    def __init__(self,latt_con,species, coords, latt_vec, cutoffs):
        """
        Initializes the lattice object.

        Args:
            species (list of str): List of atomic symbols for each atom 
                                   (e.g., ['Ni', 'Cr', 'Ni', ...]).
            coords (np.ndarray): NumPy array of atomic coordinates (N_atoms, 3).
            latt_vec (np.ndarray): The 3x3 lattice matrix.
            cutoffs (list of float): A list of cutoff radii for each shell, in increasing order.
        """
        print("--- Creating Lattice Object using Pymatgen ---")
        self.latt_con  = np.array(latt_con, dtype=np.float16)
        self.coords = np.array(coords)
        self.latt_vec = np.array(latt_vec)
        self.species = species
        self.cutoffs = sorted(cutoffs)
        self.natoms = len(self.coords)

        # 1. Create a pymatgen Structure object. This is the core of the fast approach.
        #    Pymatgen handles PBC internally.
        self.structure = Structure(
            lattice=self.latt_vec,
            species=self.species,
            coords=self.coords,
            coords_are_cartesian=True
        )

        # 2. Build the neighbor list using the new, fast method.
        self.nbor_list = self.build_neighbor_list_pymatgen()
        
        print(f"Total Atoms: {self.natoms}")
        print(f"Neighbor list built for {len(self.cutoffs)} shells.")
        print("---------------------------------------------\n")

    def build_neighbor_list_pymatgen(self):
        """
        Builds a neighbor list for each atom, partitioned into shells,
        using the highly efficient pymatgen library.
        
        This method replaces the slow, brute-force O(N^2) approach.

        Returns:
            list: A nested list where list[i][j] is a sorted NumPy array of 
                  neighbor indices for atom 'i' in shell 'j'.
        """
        # Define the shell boundaries. e.g., cutoffs [2.8, 4.0] -> edges [0, 2.8, 4.0]
        # Shell 1: (0, 2.8], Shell 2: (2.8, 4.0]
        cutoff_edges = [0.0] + self.cutoffs
        max_cutoff = self.cutoffs[-1]

        print(f"Finding all neighbors within the largest cutoff radius ({max_cutoff} Å)...")
        
        # STEP 1: Perform the efficient neighbor search ONCE using pymatgen.
        # This is the O(N) step that makes the code fast. It returns all neighbors
        # for every atom within the largest cutoff radius.
        all_neighbors_raw = self.structure.get_all_neighbors(max_cutoff)
        
        # STEP 2: Post-process the results to sort neighbors into shells.
        # This part is very fast as no more distances need to be calculated.
        final_neighbor_list = []
        for i in range(self.natoms):
            # Create empty lists to hold neighbor indices for each shell
            neighbors_per_shell = [[] for _ in self.cutoffs]
            
            # site_neighbors is a list of (neighbor_object, distance, index, image_offset)
            site_neighbors = all_neighbors_raw[i]
            
            for neighbor_tuple in site_neighbors:
                distance = neighbor_tuple[1]
                neighbor_index = neighbor_tuple[2]
                
                # Find which shell this neighbor belongs to based on its distance
                for shell_idx in range(len(self.cutoffs)):
                    if cutoff_edges[shell_idx] < distance <= cutoff_edges[shell_idx + 1]:
                        neighbors_per_shell[shell_idx].append(neighbor_index)
                        break  # Found the correct shell, move to the next neighbor

            # Convert each shell's list to a sorted NumPy array, matching the original format
            sorted_neighbors_per_shell = [np.sort(np.array(lst, dtype=int)) for lst in neighbors_per_shell]
            final_neighbor_list.append(sorted_neighbors_per_shell)
            
        return final_neighbor_list

    @property
    def shells(self):
        return len(self.cutoffs)

# Example of how you would use this new class:
if __name__ == '__main__':
    # Define some example data for a simple 2-atom system in a 10x10x10 box
    example_species = ['Ni', 'Cr']
    example_coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    example_latt_vec = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    example_cutoffs = [3.0, 5.0] # Shell 1 up to 3.0 Å, Shell 2 from 3.0 to 5.0 Å

    # Create the lattice object
    my_lattice = lattice(
        species=example_species,
        coords=example_coords,
        latt_vec=example_latt_vec,
        cutoffs=example_cutoffs
    )

    # The neighbor list is now available in my_lattice.nbor_list
    # Let's inspect the neighbors of the first atom (index 0)
    print("Neighbors for Atom 0:")
    for shell_idx, nbors in enumerate(my_lattice.nbor_list[0]):
        print(f"  Shell {shell_idx + 1}: {nbors}")
        
    # Expected output for this simple example:
    # Neighbors for Atom 0:
    #   Shell 1: [1]   (Atom 1 is at dist 2.5, which is in shell 1)
    #   Shell 2: []    (No atoms are in shell 2)

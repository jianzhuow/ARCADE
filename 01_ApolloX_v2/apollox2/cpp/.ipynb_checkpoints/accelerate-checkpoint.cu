#include "accelerate.h"

#include <ctime>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <cstdint>

#include <numeric>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <fstream>
namespace accelerate {
namespace gpu {

/**
 * @brief Checks if CUDA is available on the system.
 * 
 * @return true if at least one CUDA device is available, false otherwise.
 */
bool cuda_available() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

#define THREADS_PER_BLOCK 256

/**
 * @brief Gets the current time in microseconds.
 * 
 * @return unsigned long long Current time in microseconds since epoch.
 */
unsigned long long get_microseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * @brief Calculates fitness values for multiple lattices based on Short-Range Order (SRO) coefficients.
 * 
 * @tparam Integer Integer type used for indices and counts.
 * @tparam Real Floating-point type used for calculations.
 * @param[in] num_atoms Number of atoms in each lattice.
 * @param[in] num_types Number of different atom types.
 * @param[in] num_shells Number of neighbor shells to consider.
 * @param[in] neighbor_list List of neighbor indices for each atom.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list for each atom.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Computed SRO coefficients.
 * @param[out] fitness Computed fitness values for each lattice.
 * @param[in] lattices Array of lattice configurations.
 */
template <typename Integer, typename Real>
__global__ void calculate_fitness_of_lattices(
        Integer num_atoms, Integer num_types, Integer num_shells,
        const Integer* neighbor_list, const Integer* neighbor_list_indices, 
        const Integer* species, const Real* weights, 
        const Real* target_sro, Real* coefficients, 
        Real* fitness, Integer* lattices,
        const Integer* atom_to_sro_group_map,
        std::size_t num_sro_groups)
{
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;
    const Integer num_coefficients_per_group = num_types * num_types * num_shells;
    const Integer total_coefficients = num_sro_groups * num_coefficients_per_group;
    // Each block addresses a different lattice
    const Integer* lattice = lattices + bid * num_atoms;

    extern __shared__ Real shared_mem[];//extern：告诉编译器，这个数组的大小在 kernel 调用时动态指定，不是在编译期固定;shared：声明这是共享内存，所有同一个 block 的线程可以访问。
    Real* shared_gamma = shared_mem;
    Integer* shared_group_species_counts = (Integer*)(shared_mem + total_coefficients);
    Integer* shared_group_total_atoms = (Integer*)(shared_group_species_counts + num_sro_groups * num_types);
    for (Integer ii = tid; ii < total_coefficients; ii += THREADS_PER_BLOCK) {
        shared_gamma[ii] = 0.0;
    }
    for (Integer i = tid; i < num_sro_groups * num_types; i +=THREADS_PER_BLOCK ) {
        shared_group_species_counts[i] = 0;
    }
    for (Integer i = tid; i < num_sro_groups; i += THREADS_PER_BLOCK) {
        shared_group_total_atoms[i] = 0;
    }
    __syncthreads();//让当前 block 中的所有线程在这个位置等待，直到 block 内每个线程都执行到这一行,只作用于同一个 block 的线程，不会跨 block
    for (Integer atom_idx = tid; atom_idx < num_atoms; atom_idx += THREADS_PER_BLOCK) {
        Integer group_idx = atom_to_sro_group_map[atom_idx];
        if (group_idx >= 0) {
            Integer atom_type = lattice[atom_idx];
            atomicAdd(&shared_group_species_counts[group_idx * num_types + atom_type], 1);
            atomicAdd(&shared_group_total_atoms[group_idx], 1);
        }
    }
    __syncthreads();
    // Calculate SRO coefficients for each group
    for (Integer atom_site_idx = tid; atom_site_idx < num_atoms; atom_site_idx += THREADS_PER_BLOCK) {
        Integer sro_group_idx = atom_to_sro_group_map[atom_site_idx];
        // Skip atoms that are not in any SRO group
        if (sro_group_idx < 0) {
            continue;
        }

        Integer atom_type = lattice[atom_site_idx];
        Integer num_atoms_in_group=shared_group_total_atoms[sro_group_idx];
        if(num_atoms_in_group==0)continue;
        for (Integer shell_idx = 0; shell_idx < num_shells; ++shell_idx) {
            Integer start = neighbor_list_indices[atom_site_idx * num_shells + shell_idx];
            Integer end = neighbor_list_indices[atom_site_idx * num_shells + shell_idx + 1];
            Integer neighbor_size = end - start;
            if(neighbor_size==0) continue;
            for (Integer k = 0; k < neighbor_size; ++k) {
                Integer neighbor_site_idx = neighbor_list[start + k];
                Integer neighbor_type = lattice[neighbor_site_idx];
                Integer group_offset=sro_group_idx*num_coefficients_per_group;
                // Index for this specific pair in its group's coefficient list
                Integer local_idx = atom_type * num_types * num_shells + neighbor_type * num_shells + shell_idx;

                atomicAdd(&shared_gamma[group_offset + local_idx], 1.0 / (neighbor_size * num_atoms_in_group));
            }
        }
    }
   
    __syncthreads();

    for (Integer ii = tid; ii < total_coefficients; ii += THREADS_PER_BLOCK) {
        coefficients[bid * total_coefficients + ii] = shared_gamma[ii];
    }

    if (tid == 0) {
    Real total_error = 0.0;

    for (Integer group_idx = 0; group_idx < num_sro_groups; ++group_idx) {
        Integer num_atoms_in_group=shared_group_total_atoms[group_idx];
        if(num_atoms_in_group==0) continue;
        for (Integer shell = 0; shell < num_shells; ++shell) {
            Real fitness_shell = 0.0;
            // MODIFIED: Loop through ALL possible type pairs, do not skip
            for (Integer i = 0; i < num_types; ++i) { // Type of central atom
                for (Integer j = 0; j < num_types; ++j) { // Type of neighbor atom
                    Integer species_count_i = shared_group_species_counts[group_idx * num_types + i];
                    Real conc_j = static_cast<Real>(species[j]) / num_atoms;
                    
                    Real actual_sro;
                    if(species_count_i == 0){
                        actual_sro = 0;
                    }
                    else{
                    Real conc_i = static_cast<Real>(species_count_i) / num_atoms_in_group;
                    Integer gamma_idx = group_idx * num_coefficients_per_group + i * num_types * num_shells + j * num_shells + shell;
                    actual_sro = 1.0 - (shared_gamma[gamma_idx] / conc_i) / conc_j;
                    }
    

                    // Index into the flattened target_sro array
                    Integer target_idx = group_idx * num_shells * num_types * num_types + shell * num_types * num_types + i * num_types + j;
                    Real target_sro_val = target_sro[target_idx];
                    
                    // The error is the squared difference between the actual and target SRO
                    fitness_shell += (actual_sro - target_sro_val) * (actual_sro - target_sro_val);
                    
                }
            }
            total_error += weights[shell] * fitness_shell;
        }
    }
    fitness[bid] = sqrt(total_error);
}
}
/**
 * @brief Initializes CURAND states for random number generation.
 * 
 * @param[out] rng_states Array of CURAND states to initialize.
 * @param[in] seed Random seed value.
 */
__global__ void init_curand(curandState* rng_states, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &rng_states[id]);
}

/**
 * @brief Generates initial lattice configurations based on species distribution.
 * 
 * @tparam Integer Integer type used for indices and counts.
 * @param[in] num_atoms Total number of atoms per lattice.
 * @param[in] num_types Number of different atom types.
 * @param[in] species Array containing count of each atom type.
 * @param[out] lattices Output array for generated lattices.
 */
template <typename Integer>
__global__ void generate_normal_lattices(const Integer num_atoms, const Integer num_types, const Integer* species, Integer* lattices) //__global__:在cpu调用函数,在gpu生效
{
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;
    
    // Pointer to the start of the lattice for the current block
    Integer* lattice = lattices + bid * num_atoms;

    for (Integer ii = tid; ii < num_atoms; ii += THREADS_PER_BLOCK) {
        Integer sum = 0;
        for (Integer type = 0; type < num_types; ++type) {
            sum += species[type];
            if (ii < sum) {
                lattice[ii] = type;  
                break;
            }
        }
    }
}

/**
 * @brief Locates the index of the best fitness value in a block.
 * 
 * @tparam Integer Integer type used for indices.
 * @tparam Real Floating-point type used for fitness values.
 * @param[in] fitness Array of fitness values.
 * @param[in,out] indices Array of indices.
 * @return int Index of the best fitness value.
 */
template <typename Integer, typename Real>
__device__ __forceinline__ int locate_best_fitness(Real* fitness, Integer* indices) {
    const Integer tid = threadIdx.x;
    const Integer num_threads = blockDim.x;
    // Shared memory for indices
    indices[tid] = tid; // Initialize indices
    __syncthreads();
    // Reduction to find the minimum value and its index
    for (int ii = num_threads / 2; ii > 0; ii /= 2) {
        if (tid < ii) {
            if (fitness[tid] > fitness[tid + ii]) {
                fitness[tid] = fitness[tid + ii];
                indices[tid] = indices[tid + ii];
            }
        }
        __syncthreads();
    }
    // Return the index of the minimum value
    return indices[0]; // Index of the minimum value in shared memory
}

/**
 * @brief Calculates SRO coefficients for a given lattice configuration.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] num_atoms Number of atoms in the lattice.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[out] coefficients Output array for computed coefficients.
 * @param[in] lattice Current lattice configuration.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ void calculate_coefficients(
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* atom_to_sro_group_map,
    const Integer num_sro_groups,
    Real* coefficients,
    Integer* shared_group_species_counts,
    Integer* shared_group_total_atoms,
    LOW_Integer* lattice)
{   
    __syncthreads();
    const Integer tid = threadIdx.x;
    const Integer num_threads = blockDim.x;
    // Define the number of coefficients
    const Integer NUM_COEFF_PER_GROUP = NUM_TYPES * NUM_TYPES * NUM_SHELLS;
    for (Integer i = tid; i < num_sro_groups * NUM_TYPES; i += num_threads) {
        shared_group_species_counts[i] = 0;
    }
    for (Integer i = tid; i < num_sro_groups; i += num_threads) {
        shared_group_total_atoms[i] = 0;
    }
    __syncthreads();
    for (Integer atom_idx = tid; atom_idx < num_atoms; atom_idx += num_threads) {
        Integer group_idx = atom_to_sro_group_map[atom_idx];
        if (group_idx >= 0) {
            Integer atom_type = lattice[atom_idx];
            atomicAdd(&shared_group_species_counts[group_idx * NUM_TYPES + atom_type], 1);
            atomicAdd(&shared_group_total_atoms[group_idx], 1);
        }
    }
    __syncthreads();
    // Loop over each atom
    for (Integer ii = threadIdx.x; ii < num_atoms; ii += blockDim.x) {
        Integer sro_group_idx = atom_to_sro_group_map[ii];
        if (sro_group_idx < 0) {
            continue; // Skip atoms not in any SRO group
        }
        LOW_Integer atom = lattice[ii];
        Integer num_atoms_in_group=shared_group_total_atoms[sro_group_idx];
        if(num_atoms_in_group==0) continue;
        // Loop over each shell
        for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
            Integer neighbor_size = neighbor_list_indices[ii * NUM_SHELLS + shell + 1] 
                                  - neighbor_list_indices[ii * NUM_SHELLS + shell];
            if (neighbor_size == 0) continue;
            for (Integer kk = 0; kk < neighbor_size; ++kk) {                
                Integer idx = neighbor_list[neighbor_list_indices[ii * NUM_SHELLS + shell] + kk];
                LOW_Integer neighbor = lattice[idx];
                Integer group_offset = sro_group_idx * NUM_COEFF_PER_GROUP;
                Integer local_idx = atom * NUM_TYPES * NUM_SHELLS + neighbor * NUM_SHELLS + shell;
                // Compute the index for accessing the arrays
                atomicAdd(&coefficients[group_offset + local_idx], 1.0 / (neighbor_size * num_atoms_in_group));
            }
        }
    }
    __syncthreads();
}

/**
 * @brief Calculates fitness for a proposed lattice configuration with two atoms swapped.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] atom1 Index of first atom to swap.
 * @param[in] atom2 Index of second atom to swap.
 * @param[in] num_atoms Total number of atoms.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Temporary array for coefficient calculations.
 * @param[in,out] lattice Current lattice configuration.
 * @return Real Computed fitness value for the proposed configuration.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS,int N_SRO_GROUPS_CE>
__device__ __forceinline__ Real calculate_fitness_incremental(
    const Integer& atom1, const Integer& atom2,
    const Integer& num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* atom_to_sro_group_map,
    const Integer* initial_group_species_counts,
    const Integer* initial_group_total_atoms,
    const Integer num_sro_groups,
    const Real* target_sro, Real* coefficients,
    LOW_Integer* lattice)
{
    const Integer NUM_COEFF_PER_GROUP = NUM_TYPES * NUM_TYPES * NUM_SHELLS;
    Integer local_group_species_counts[N_SRO_GROUPS_CE * NUM_TYPES]; // Note: Requires NUM_SRO_GROUPS_CE as compile-time const
    Integer local_group_total_atoms[N_SRO_GROUPS_CE];
    for(int i = 0; i < num_sro_groups * NUM_TYPES; ++i) {
        local_group_species_counts[i] = initial_group_species_counts[i];
    }
    for(int i = 0; i < num_sro_groups; ++i) {
        local_group_total_atoms[i] = initial_group_total_atoms[i];
    }
    LOW_Integer type1_orig = lattice[atom1];
    LOW_Integer type2_orig = lattice[atom2];
    Integer group_idx1 = atom_to_sro_group_map[atom1];
    Integer group_idx2 = atom_to_sro_group_map[atom2];
    if (group_idx1 >= 0) {
        local_group_species_counts[group_idx1 * NUM_TYPES + type1_orig]--;
        local_group_species_counts[group_idx1 * NUM_TYPES + type2_orig]++;
    }
    if (group_idx2 >= 0) {
        local_group_species_counts[group_idx2 * NUM_TYPES + type2_orig]--;
        local_group_species_counts[group_idx2 * NUM_TYPES + type1_orig]++;
    }
    // ====================================================================
    // Part 1: Incrementally update gamma coefficients for the swap
    // ====================================================================

    for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
        // --- Block 1: Handle neighbors of the first swapped atom (atom1) ---
        LOW_Integer type1_orig = lattice[atom1];
        LOW_Integer type1_new = lattice[atom2];
        Integer neighbor_size1 = neighbor_list_indices[atom1 * NUM_SHELLS + shell + 1] - neighbor_list_indices[atom1 * NUM_SHELLS + shell];

        for (Integer kk = 0; kk < neighbor_size1; ++kk) {
            Integer neighbor_idx = neighbor_list[neighbor_list_indices[atom1 * NUM_SHELLS + shell] + kk];
            LOW_Integer neighbor_type_orig = lattice[neighbor_idx];
            
            LOW_Integer neighbor_type_new = neighbor_type_orig;
            if (neighbor_idx == atom2) { // If the neighbor is the other swapped atom
                neighbor_type_new = type1_orig; // Its new type is the original type of atom1
            }

            // --- Subtract the OLD bond's contribution (symmetrically) ---
            // A.1: From atom1's perspective
            Integer group_idx_1 = atom_to_sro_group_map[atom1];
            Integer num_atoms_in_group_1 = (group_idx1 >= 0) ? initial_group_total_atoms[group_idx1] : 0;
            if (group_idx_1 >= 0 && neighbor_size1 > 0 && num_atoms_in_group_1 > 0) {
                    Integer offset = group_idx_1 * NUM_COEFF_PER_GROUP;
                    Integer idx = type1_orig * NUM_TYPES * NUM_SHELLS + neighbor_type_orig * NUM_SHELLS + shell;
                    coefficients[offset + idx] -= 1.0 / (neighbor_size1 * num_atoms_in_group_1);
            }
            // A.2: From the neighbor's perspective
            Integer group_idx_n = atom_to_sro_group_map[neighbor_idx];
            Integer num_atoms_in_group_n = (group_idx_n >= 0) ? initial_group_total_atoms[group_idx_n] : 0;
            Integer neighbor_size_n = neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell + 1] - neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell];
            if (group_idx_n >= 0&& neighbor_size_n > 0 && num_atoms_in_group_n > 0) {
     
                Integer offset = group_idx_n * NUM_COEFF_PER_GROUP;
                Integer idx = neighbor_type_orig * NUM_TYPES * NUM_SHELLS + type1_orig * NUM_SHELLS + shell;
                coefficients[offset + idx] -= 1.0 / (neighbor_size_n * num_atoms_in_group_n);

            }

            // --- Add the NEW bond's contribution (symmetrically) ---
            // B.1: From atom1's new perspective
            if (group_idx_1 >= 0&& neighbor_size1 > 0 && num_atoms_in_group_1 > 0) {
                    Integer offset = group_idx_1 * NUM_COEFF_PER_GROUP;
                    Integer idx = type1_new * NUM_TYPES * NUM_SHELLS + neighbor_type_new * NUM_SHELLS + shell;
                    coefficients[offset + idx] += 1.0 / (neighbor_size1 * num_atoms_in_group_1);
            }
            // B.2: From the neighbor's new perspective
            if (group_idx_n >= 0) {
                Integer neighbor_size_n = neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell + 1] - neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell];
                Integer offset = group_idx_n * NUM_COEFF_PER_GROUP;
                Integer idx = neighbor_type_new * NUM_TYPES * NUM_SHELLS + type1_new * NUM_SHELLS + shell;
                coefficients[offset + idx] += 1.0 / (neighbor_size_n * num_atoms_in_group_n);
            }
        }

        // --- Block 2: Handle neighbors of the second swapped atom (atom2) ---
        LOW_Integer type2_orig = lattice[atom2];
        LOW_Integer type2_new = lattice[atom1];
        Integer neighbor_size2 = neighbor_list_indices[atom2 * NUM_SHELLS + shell + 1] - neighbor_list_indices[atom2 * NUM_SHELLS + shell];

        for (Integer kk = 0; kk < neighbor_size2; ++kk) {
            Integer neighbor_idx = neighbor_list[neighbor_list_indices[atom2 * NUM_SHELLS + shell] + kk];

            // CRITICAL: Skip the neighbor if it is atom1, as that bond has already been fully processed in Block 1.
            if (neighbor_idx == atom1) continue;

            LOW_Integer neighbor_type_orig = lattice[neighbor_idx];
            
            // The neighbor's type does not change in this block because we've already excluded atom1.
            LOW_Integer neighbor_type_new = neighbor_type_orig;
            Integer num_atoms_in_group2 = (group_idx2 >= 0) ? initial_group_total_atoms[group_idx2] : 0;
            // --- Subtract the OLD bond's contribution (symmetrically) ---
            // A.1: From atom2's perspective
            Integer group_idx_2 = atom_to_sro_group_map[atom2];
            if (group_idx_2 >= 0&&neighbor_size2>0&&num_atoms_in_group2>0) {
                Integer offset = group_idx_2 * NUM_COEFF_PER_GROUP;
                Integer idx = type2_orig * NUM_TYPES * NUM_SHELLS + neighbor_type_orig * NUM_SHELLS + shell;
                coefficients[offset + idx] -= 1.0 / (neighbor_size2 * num_atoms_in_group2);
            }
            // A.2: From the neighbor's perspective
            Integer group_idx_n = atom_to_sro_group_map[neighbor_idx];
            Integer num_atoms_in_group_n = (group_idx_n >= 0) ? initial_group_total_atoms[group_idx_n] : 0;
            Integer neighbor_size_n = neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell + 1] - neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell];
            if (group_idx_n >= 0&& neighbor_size_n > 0 && num_atoms_in_group_n > 0) {
             
                Integer offset = group_idx_n * NUM_COEFF_PER_GROUP;
                Integer idx = neighbor_type_orig * NUM_TYPES * NUM_SHELLS + type2_orig * NUM_SHELLS + shell;
                coefficients[offset + idx] -= 1.0 / (neighbor_size_n * num_atoms_in_group_n);

            }

            // --- Add the NEW bond's contribution (symmetrically) ---
            // B.1: From atom2's new perspective
            if (group_idx_2 >= 0&&neighbor_size2>0&&num_atoms_in_group2>0) {
                    Integer offset = group_idx_2 * NUM_COEFF_PER_GROUP;
                    Integer idx = type2_new * NUM_TYPES * NUM_SHELLS + neighbor_type_new * NUM_SHELLS + shell;
                    coefficients[offset + idx] += 1.0 / (neighbor_size2 * num_atoms_in_group2);
            }
            // B.2: From the neighbor's new perspective
            if (group_idx_n >= 0&& neighbor_size_n > 0 && num_atoms_in_group_n > 0) {
                Integer neighbor_size_n = neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell + 1] - neighbor_list_indices[neighbor_idx * NUM_SHELLS + shell];
                    Integer offset = group_idx_n * NUM_COEFF_PER_GROUP;
                    Integer idx = neighbor_type_new * NUM_TYPES * NUM_SHELLS + type2_new * NUM_SHELLS + shell;
                    coefficients[offset + idx] += 1.0 / (neighbor_size_n * num_atoms_in_group_n);
            }
        }
    }

    // ====================================================================
    // Part 2: Recalculate total fitness from the updated coefficients
    // This part is identical to the logic in calculate_fitness_of_lattices
    // ====================================================================
    Real total_error = 0.0;
    for (Integer group_idx = 0; group_idx < num_sro_groups; ++group_idx) {
        Integer num_atoms_in_group = local_group_total_atoms[group_idx];
        if (num_atoms_in_group == 0) continue;
        for (Integer shell = 0; shell < NUM_SHELLS; ++shell) {
            Real fitness_shell = 0.0;
            for (Integer i = 0; i < NUM_TYPES; ++i) {
                for (Integer j = 0; j < NUM_TYPES; ++j) {
                    Integer species_count_i = local_group_species_counts[group_idx * NUM_TYPES + i];
                    Real actual_sro;
                    if (species_count_i == 0) {
                        actual_sro = 0; 
                    } else {
                        Real conc_i = static_cast<Real>(species_count_i) / num_atoms_in_group;
                        Real conc_j = static_cast<Real>(species[j]) / num_atoms;
                    Integer gamma_idx = group_idx * NUM_COEFF_PER_GROUP + i * NUM_TYPES * NUM_SHELLS + j * NUM_SHELLS + shell;
                    actual_sro = 1.0 - (coefficients[gamma_idx] / conc_i) / conc_j;
                    }
                    Integer target_idx = group_idx * NUM_SHELLS * NUM_TYPES * NUM_TYPES + shell * NUM_TYPES * NUM_TYPES + i * NUM_TYPES + j;
                    Real target_sro_val = target_sro[target_idx];
                    fitness_shell += (actual_sro - target_sro_val) * (actual_sro - target_sro_val);
                }
            }
            total_error += weights[shell] * fitness_shell;
        }
    }
    return sqrt(total_error);
}


/**
 * @brief Performs parallel Monte Carlo optimization of lattice configurations.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @tparam LOW_Integer Lower precision integer type for lattice values.
 * @tparam LOW_Real Lower precision floating-point type for intermediate calculations.
 * @tparam NUM_TYPES Number of atom types (compile-time constant).
 * @tparam NUM_SHELLS Number of neighbor shells (compile-time constant).
 * @param[in] threshold Acceptance threshold for Monte Carlo moves.
 * @param[in] search_depth Number of Monte Carlo steps to perform.
 * @param[in] seed Random seed value.
 * @param[in] num_atoms Number of atoms per lattice.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] fitness Array for computed fitness values.
 * @param[in,out] lattices Array of lattice configurations.
 */
template <typename Integer, typename Real,
          typename LOW_Integer, typename LOW_Real,
          int NUM_TYPES, int NUM_SHELLS,int NUM_TOTAL_COEFF,int N_SRO_GROUPS_CE>
__global__ void parallel_monte_carlo(
    const Real threshold, const Integer search_depth, const unsigned long long random_seed,
    const Integer num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,const Integer* flat_groups, const Integer* group_offsets, const Integer* group_sizes, std::size_t num_swap_groups,
        const Integer* free_indices, const Integer num_free_indices,
    const Integer* atom_to_sro_group_map,
    std::size_t num_sro_groups,const Real* target_sro,
    Real* fitness, Integer* lattices)
{
    extern __shared__ Real shared[];
    const Integer padded_num_atoms = (num_atoms + 3) & ~3;
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;

    const Integer num_blocks = gridDim.x;
    const Integer num_threads = blockDim.x;
    const Integer NUM_COEFF_PER_GROUP = NUM_SHELLS * NUM_TYPES * NUM_TYPES;
    curandState local_state;
    curand_init(random_seed, tid, 0, &local_state);

    Integer*     shared_depth     = reinterpret_cast<Integer*>(shared);
    Integer*     shared_indices   = shared_depth   + num_threads;
    LOW_Integer* shared_lattice   = reinterpret_cast<LOW_Integer*>(shared_indices + num_threads); // LOW_Integer type for shared lattice
    Real* shared_fitness          = reinterpret_cast<Real*>(shared_lattice + padded_num_atoms); // Adjust for LOW_Integer
    Real* shared_coefficients     = shared_fitness + num_threads;
    Integer* shared_group_species_counts = (Integer*)(shared_coefficients + NUM_TOTAL_COEFF);
    Integer* shared_group_total_atoms = (Integer*)(shared_group_species_counts + num_sro_groups * NUM_TYPES);
    shared_fitness[tid] = 100;

    // Init fitness
    if (tid == 0) {
        shared_depth[0] = 0;
    }
    Real reg_fitness = fitness[bid];

    // Load lattice from global memory (int) to shared memory (LOW_Integer)
    for (Integer ii = tid; ii < num_atoms; ii += num_threads) {
        shared_lattice[ii] = static_cast<LOW_Integer>(lattices[bid * num_atoms + ii]);  // Cast int to LOW_Integer
    }
    for (Integer ii = num_atoms + tid; ii < padded_num_atoms; ii += num_threads) {
        shared_lattice[ii] = 255;
    }
    for (Integer ii = tid; ii < NUM_TOTAL_COEFF; ii += num_threads) {
        shared_coefficients[ii] = 0.0;
    }

    calculate_coefficients<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS>(
        num_atoms, species, weights, neighbor_list, neighbor_list_indices,atom_to_sro_group_map, num_sro_groups, shared_coefficients,shared_group_species_counts,shared_group_total_atoms,shared_lattice);

    Integer reg_depth = shared_depth[0];
    if (num_free_indices < 2 && num_swap_groups == 0){ 
    return;}
    while (reg_depth < search_depth) {
        Real local_coefficients[NUM_TOTAL_COEFF];
        for (Integer ii = 0; ii < NUM_TOTAL_COEFF; ii++) {
            local_coefficients[ii] = shared_coefficients[ii];
        }

        // ============ MODIFICATION: Restricted atom selection logic ============
        Integer atom1, atom2;
        do {
            // Decide whether to swap within a restricted group or the free group
            // Add 1 to num_swap_groups only if the free group is swappable
            Integer total_swappable_entities = num_swap_groups + (num_free_indices > 1 ? 1 : 0);
            if (total_swappable_entities == 0) {
                // This case should be caught by the early exit, but as a safeguard:
                atom1 = 0; atom2 = 0; // force loop to continue if it somehow gets here
                continue;
            }

            Integer choice = curand(&local_state) % total_swappable_entities;

            if (choice < num_swap_groups) { // Pick from a restricted swap group
                Integer group_size = group_sizes[choice];
                if (group_size < 2) {
                    atom1 = atom2 = 0; // Invalid choice, try again
                    continue;
                }
                Integer offset = group_offsets[choice];
                Integer idx1 = curand(&local_state) % group_size;
                Integer idx2 = curand(&local_state) % group_size;
                atom1 = flat_groups[offset + idx1];
                atom2 = flat_groups[offset + idx2];
            }
            else { // Pick from the free indices group
                Integer idx1 = curand(&local_state) % num_free_indices;
                Integer idx2 = curand(&local_state) % num_free_indices;
                atom1 = free_indices[idx1];
                atom2 = free_indices[idx2];
            }
        } while (atom1 == atom2 || shared_lattice[atom1] == shared_lattice[atom2]);
        
        shared_fitness[tid] = calculate_fitness_incremental<Integer, Real, LOW_Integer, LOW_Real, NUM_TYPES, NUM_SHELLS,N_SRO_GROUPS_CE>(
            atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices, atom_to_sro_group_map,shared_group_species_counts,shared_group_total_atoms, num_sro_groups,target_sro, local_coefficients, shared_lattice);

        Integer id = locate_best_fitness(shared_fitness, shared_indices);

        if ((shared_fitness[id] < reg_fitness)) {
            reg_fitness = shared_fitness[id];
            if (tid == id) {
                LOW_Integer swap = shared_lattice[atom1];
                shared_lattice[atom1] = shared_lattice[atom2];
                shared_lattice[atom2] = swap;
                for (Integer ii = 0; ii < NUM_TOTAL_COEFF; ii++) {
                    shared_coefficients[ii] = local_coefficients[ii];
                }
            }
            if (tid == 0) {
                shared_depth[0] = 0;
            }
        }
       
        else {
      if (tid == 0) {
        shared_depth[0] += 1;
       }

        }
        __syncthreads();
        reg_depth = shared_depth[0];
    }
    __syncthreads();

    if (tid == 0) {
        fitness[bid] = reg_fitness;
    }
    // Write the final lattice from shared memory (LOW_Integer) back to global memory (int)
    for (Integer ii = tid; ii < num_atoms; ii += num_threads) {
        lattices[bid * num_atoms + ii] = static_cast<Integer>(shared_lattice[ii]);  // Cast LOW_Integer back to int
    }
}



/**
 * @brief Sorts lattices based on their fitness values.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for fitness values.
 * @param[in,out] lattices Array of lattice configurations.
 * @param[in,out] fitness Array of fitness values.
 * @param[in] num_lattices Number of lattices to sort.
 * @param[in] num_atoms Number of atoms per lattice.
 */
template <typename Integer, typename Real>
void sort_lattices_by_fitness(Integer* lattices, Real* fitness, size_t num_lattices, size_t num_atoms) {
    Integer* indices = new Integer[num_lattices];
    std::iota(indices, indices + num_lattices, 0);
    
    Real* host_fitness = new Real[num_lattices * 2];
    cudaMemcpy(host_fitness, fitness, num_lattices * sizeof(Real), cudaMemcpyDeviceToHost);
    for (size_t id = 0; id < num_lattices; ++id) {
        host_fitness[num_lattices + id] = host_fitness[id];
    }
    
    std::sort(indices, indices + num_lattices, [&host_fitness](Integer a, Integer b) {
        return host_fitness[a] < host_fitness[b];
    });
    
    Integer* sorted_lattices = nullptr;
    cudaMalloc(&sorted_lattices, num_lattices * num_atoms * sizeof(Integer));
    
    for (size_t id = 0; id < num_lattices; ++id) {
        Integer sorted_index = indices[id];
        host_fitness[id] = host_fitness[num_lattices + sorted_index];
        cudaMemcpy(sorted_lattices + id * num_atoms, lattices + sorted_index * num_atoms, num_atoms * sizeof(Integer), cudaMemcpyDeviceToDevice);
    }
    
    cudaMemcpy(fitness, host_fitness, num_lattices * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(lattices, sorted_lattices, num_lattices * num_atoms * sizeof(Integer), cudaMemcpyDeviceToDevice);

    delete[] indices;
    delete[] host_fitness;
    cudaFree(sorted_lattices);
}

/**
 * @brief Generates random lattice configurations using parallel Monte Carlo.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for calculations.
 * @param[in] num_lattices Number of lattices to generate.
 * @param[in] num_types Number of atom types.
 * @param[in] num_atoms Number of atoms per lattice.
 * @param[in] num_shells Number of neighbor shells.
 * @param[in] species Array containing count of each atom type.
 * @param[in] weights Weight factors for each shell.
 * @param[in] neighbor_list List of neighbor indices.
 * @param[in] neighbor_list_indices Starting indices in neighbor_list.
 * @param[in] target_sro Target SRO values to compare against.
 * @param[out] coefficients Array for computed coefficients.
 * @param[out] fitness Array for computed fitness values.
 * @param[out] lattices Output array for generated lattices.
 */
template <typename Integer>
__global__ void constrained_norm_kernel(
    Integer* lattices,
    const Integer* d_flat_groups,
    const Integer* d_group_offsets,
    const Integer* d_group_sizes,
    std::size_t num_groups,
    const Integer* d_free_indices,
    std::size_t num_free_indices,
    Integer num_atoms,
    unsigned long long seed)
{
    // Each block is responsible for one lattice
    const Integer bid = blockIdx.x;
    if (bid == 0) {
        return;
    }
    const Integer tid = threadIdx.x;
    const Integer block_size = blockDim.x;
    // Pointer to the start of the current lattice in global memory
    Integer* current_lattice = lattices + bid * num_atoms;

    // Shared memory buffer for shuffling. Its size is passed at launch.
    extern __shared__ Integer shared_buffer[];

    // Initialize random state for each thread.
    // Unique seed for each thread across all blocks.
    curandState state;
    curand_init(seed, bid * block_size + tid, 0, &state);

    // --- Step 1: Shuffle the "free" indices ---
    if (num_free_indices > 0) {
        // a) Parallel Load (Gather): Copy atom types from global to shared memory
        for (Integer i = tid; i < num_free_indices; i += block_size) {
            shared_buffer[i] = current_lattice[d_free_indices[i]];
        }
        __syncthreads();

        // b) Sequential Shuffle in Shared Memory (done by one thread)
        if (tid == 0) {
            for (Integer i = num_free_indices - 1; i > 0; i--) {
                // Generate a random index 'j' in the range [0, i]
                Integer j = curand(&state) % (i + 1);
                // Swap elements
                Integer temp = shared_buffer[i];
                shared_buffer[i] = shared_buffer[j];
                shared_buffer[j] = temp;
            }
        }
        __syncthreads();

        // c) Parallel Store (Scatter): Copy shuffled types from shared back to global memory
        for (Integer i = tid; i < num_free_indices; i += block_size) {
            current_lattice[d_free_indices[i]] = shared_buffer[i];
        }
    }

    // --- Step 2: Shuffle within each group ---
    for (Integer g = 0; g < num_groups; ++g) {
        Integer group_size = d_group_sizes[g];
        if (group_size <= 1) continue; // No need to shuffle a group of 0 or 1

        __syncthreads(); // Ensure scatter from previous step is complete

        Integer group_offset = d_group_offsets[g];

        // a) Parallel Load (Gather) for the current group
        for (Integer i = tid; i < group_size; i += block_size) {
            Integer global_idx = d_flat_groups[group_offset + i];
            shared_buffer[i] = current_lattice[global_idx];
        }
        __syncthreads();

        // b) Sequential Shuffle in Shared Memory
        if (tid == 0) {
            for (Integer i = group_size - 1; i > 0; i--) {
                Integer j = curand(&state) % (i + 1);
                Integer temp = shared_buffer[i];
                shared_buffer[i] = shared_buffer[j];
                shared_buffer[j] = temp;
            }
        }
        __syncthreads();

        // c) Parallel Store (Scatter) for the current group
        for (Integer i = tid; i < group_size; i += block_size) {
            Integer global_idx = d_flat_groups[group_offset + i];
            current_lattice[global_idx] = shared_buffer[i];
        }
    }
}
template <typename Integer>
__global__ void constrained_shuffle_kernel(
    Integer* lattices,
    const Integer* d_flat_groups,
    const Integer* d_group_offsets,
    const Integer* d_group_sizes,
    std::size_t num_groups,
    const Integer* d_free_indices,
    std::size_t num_free_indices,
    Integer num_atoms,
    unsigned long long seed)
{
    // Each block is responsible for one lattice
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;
    const Integer block_size = blockDim.x;
    // Pointer to the start of the current lattice in global memory
    Integer* current_lattice = lattices + bid * num_atoms;

    // Shared memory buffer for shuffling. Its size is passed at launch.
    extern __shared__ Integer shared_buffer[];

    // Initialize random state for each thread.
    // Unique seed for each thread across all blocks.
    curandState state;
    curand_init(seed, bid * block_size + tid, 0, &state);

    // --- Step 1: Shuffle the "free" indices ---
    if (num_free_indices > 0) {
        // a) Parallel Load (Gather): Copy atom types from global to shared memory
        for (Integer i = tid; i < num_free_indices; i += block_size) {
            shared_buffer[i] = current_lattice[d_free_indices[i]];
        }
        __syncthreads();

        // b) Sequential Shuffle in Shared Memory (done by one thread)
        if (tid == 0) {
            for (Integer i = num_free_indices - 1; i > 0; i--) {
                // Generate a random index 'j' in the range [0, i]
                Integer j = curand(&state) % (i + 1);
                // Swap elements
                Integer temp = shared_buffer[i];
                shared_buffer[i] = shared_buffer[j];
                shared_buffer[j] = temp;
            }
        }
        __syncthreads();

        // c) Parallel Store (Scatter): Copy shuffled types from shared back to global memory
        for (Integer i = tid; i < num_free_indices; i += block_size) {
            current_lattice[d_free_indices[i]] = shared_buffer[i];
        }
    }

    // --- Step 2: Shuffle within each group ---
    for (Integer g = 0; g < num_groups; ++g) {
        Integer group_size = d_group_sizes[g];
        if (group_size <= 1) continue; // No need to shuffle a group of 0 or 1

        __syncthreads(); // Ensure scatter from previous step is complete

        Integer group_offset = d_group_offsets[g];

        // a) Parallel Load (Gather) for the current group
        for (Integer i = tid; i < group_size; i += block_size) {
            Integer global_idx = d_flat_groups[group_offset + i];
            shared_buffer[i] = current_lattice[global_idx];
        }
        __syncthreads();

        // b) Sequential Shuffle in Shared Memory
        if (tid == 0) {
            for (Integer i = group_size - 1; i > 0; i--) {
                Integer j = curand(&state) % (i + 1);
                Integer temp = shared_buffer[i];
                shared_buffer[i] = shared_buffer[j];
                shared_buffer[j] = temp;
            }
        }
        __syncthreads();

        // c) Parallel Store (Scatter) for the current group
        for (Integer i = tid; i < group_size; i += block_size) {
            Integer global_idx = d_flat_groups[group_offset + i];
            current_lattice[global_idx] = shared_buffer[i];
        }
    }
}
template <typename Integer, typename Real>
void generate_random_lattices(
    Integer num_lattices,
    Integer num_atoms, Integer num_types, Integer num_shells,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro, Real* coefficients,
    Real* fitness, Integer* lattices,
    // Arguments for constrained shuffling on the GPU
    const Integer* d_flat_groups, const Integer* d_group_offsets,
    const Integer* d_group_sizes, std::size_t num_groups,
    const Integer* d_free_indices, std::size_t num_free_indices,
    size_t shared_mem_size,const Integer* d_flat_sro_groups, const Integer* d_sro_group_offsets, const Integer* d_sro_group_sizes, std::size_t num_sro_groups,
    const Integer* d_atom_to_sro_group_map)
{
    // Step 1: Create the initial, ordered lattices on the GPU
    generate_normal_lattices << <num_lattices, THREADS_PER_BLOCK >> > (
        num_atoms, num_types, species, lattices);

    // Step 2: Launch the single, high-performance kernel to shuffle all lattices
    unsigned long long seed = get_microseconds();
    constrained_shuffle_kernel << <num_lattices, THREADS_PER_BLOCK, shared_mem_size >> > (
        lattices, d_flat_groups, d_group_offsets, d_group_sizes, num_groups,
        d_free_indices, num_free_indices, num_atoms, seed);

    // Check for any kernel launch errors (good practice)
    cudaDeviceSynchronize();

    // Step 3: Calculate fitness for the newly shuffled lattices
    const Integer num_coefficients_per_group = num_types * num_types * num_shells;
    size_t fitness_shared_mem = num_sro_groups * num_coefficients_per_group * sizeof(Real);
    calculate_fitness_of_lattices<<<num_lattices, THREADS_PER_BLOCK, fitness_shared_mem>>>(
        num_atoms, num_types, num_shells,
        neighbor_list, neighbor_list_indices,
        species, weights,
        target_sro, coefficients,
        fitness, lattices,
        // NEW: Pass SRO group info to the kernel
        d_atom_to_sro_group_map,
        num_sro_groups);
}
template <typename Integer, typename Real>
void generate_norm_lattices(
    Integer num_lattices,
    Integer num_atoms, Integer num_types, Integer num_shells,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Real* target_sro, Real* coefficients,
    Real* fitness, Integer* lattices,
    // Arguments for constrained shuffling on the GPU
    const Integer* d_flat_groups, const Integer* d_group_offsets,
    const Integer* d_group_sizes, std::size_t num_groups,
    const Integer* d_free_indices, std::size_t num_free_indices,
    size_t shared_mem_size,const Integer* d_flat_sro_groups, const Integer* d_sro_group_offsets, const Integer* d_sro_group_sizes, std::size_t num_sro_groups,
    const Integer* d_atom_to_sro_group_map)
{
    // Step 1: Create the initial, ordered lattices on the GPU
    generate_normal_lattices << <num_lattices, THREADS_PER_BLOCK >> > (
        num_atoms, num_types, species, lattices);
    unsigned long long seed = get_microseconds();
    constrained_norm_kernel << <num_lattices, THREADS_PER_BLOCK, shared_mem_size >> > (
        lattices, d_flat_groups, d_group_offsets, d_group_sizes, num_groups,
        d_free_indices, num_free_indices, num_atoms, seed);

    // Check for any kernel launch errors (good practice)
    cudaDeviceSynchronize();
    // Step 2: Calculate fitness for the lattices
    const Integer num_coefficients_per_group = num_types * num_types * num_shells;
    size_t fitness_shared_mem = num_sro_groups * num_coefficients_per_group * sizeof(Real);
    calculate_fitness_of_lattices<<<num_lattices, THREADS_PER_BLOCK, fitness_shared_mem>>>(
        num_atoms, num_types, num_shells,
        neighbor_list, neighbor_list_indices,
        species, weights,
        target_sro, coefficients,
        fitness, lattices,
        // NEW: Pass SRO group info to the kernel
        d_atom_to_sro_group_map,
        num_sro_groups);
}
/**
 * @brief Selects best lattice configurations from current and new populations.
 * 
 * @tparam Integer Integer type for indices.
 * @tparam Real Floating-point type for fitness values.
 * @param[in] num_lattices Number of lattices to select.
 * @param[in] num_atoms Number of atoms per lattice.
 * @param[in,out] lattices Current lattice configurations.
 * @param[in] new_lattices New lattice configurations.
 * @param[in,out] fitness Current fitness values.
 * @param[in] new_fitness New fitness values.
 */
#define LAUNCH_PMC_KERNEL(NT, NS, N_SRO_GROUPS_CE) \
    do { \
        /* 基于编译时常量计算出最终的 TOTAL_COEFF */ \
        const int NUM_COEFF_PER_GROUP = (NT) * (NT) * (NS); \
        const int TOTAL_COEFF = NUM_COEFF_PER_GROUP * (N_SRO_GROUPS_CE); \
        \
        /* 使用完整的模板参数获取和设置核函数属性 */ \
        cudaFuncGetAttributes(&attr, parallel_monte_carlo<Integer, Real, LOW_Integer, Real, NT, NS, TOTAL_COEFF, N_SRO_GROUPS_CE>); \
        cudaFuncSetAttribute(parallel_monte_carlo<Integer, Real, LOW_Integer, Real, NT, NS, TOTAL_COEFF, N_SRO_GROUPS_CE>, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             prop.sharedMemPerBlockOptin); \
        \
        /* 启动对应的核函数模板实例 */ \
        /* 注意：最后的 d_temp_coefficients_workspace 参数已被移除 */ \
        parallel_monte_carlo<Integer, Real, LOW_Integer, Real, NT, NS, TOTAL_COEFF, N_SRO_GROUPS_CE><<<num_lattices, num_tasks, shared_usage>>>( \
            threshold, search_depth, seed, num_atoms, species, weights, neighbor_list, neighbor_list_indices, \
            d_flat_groups, d_group_offsets, d_group_sizes, num_swap_groups, d_free_indices, num_free_indices, \
            d_atom_to_sro_group_map,num_sro_groups, \
            target_sro, fitness, lattices); \
    } while (0)
template <typename Integer, typename Real>
void local_parallel_monte_carlo(
    const Integer& num_lattices, const Integer& num_atoms, 
    const Integer& num_types, const Integer& num_shells, const Integer& num_tasks,
    const Integer& search_depth, const Real& threshold,    
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* species, 
    const std::vector<Integer>& host_species,
    const Real* weights,
    const Real* target_sro,const Integer* d_flat_groups, const Integer* d_group_offsets, const Integer* d_group_sizes, const std::size_t& num_swap_groups,
    const Integer* d_free_indices, const std::size_t& num_free_indices,
    const Integer* d_flat_sro_groups, const Integer* d_sro_group_offsets, const Integer* d_sro_group_sizes, const std::size_t& num_sro_groups,
    const Integer* d_atom_to_sro_group_map,
    Real* fitness, Integer* lattices)
{
    using LOW_Integer = uint8_t;
    unsigned long long seed = get_microseconds();
    const int num_coefficients_per_group = num_types * num_types * num_shells;
    size_t shared_usage = 2 * num_tasks * sizeof(Integer) 
        + num_atoms * sizeof(LOW_Integer)
        + num_tasks * sizeof(Real)
        + num_sro_groups*num_coefficients_per_group * sizeof(Real)+num_sro_groups*num_types*sizeof(Integer)+num_sro_groups*sizeof(Integer);
    // Get device properties to check maximum shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    // Check if shared memory requirement exceeds maximum available
    if (shared_usage > prop.sharedMemPerBlockOptin) 
    {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cerr << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - ApolloX2 - INFO - " 
                  << "Error: Required shared memory (" << shared_usage 
                  << " bytes) exceeds maximum available shared memory (" 
                  << prop.sharedMemPerBlockOptin << " bytes)" << std::endl;
        exit(1);
    }
    shared_usage = prop.sharedMemPerBlockOptin;

    // Set maximum shared memory for the kernel
    cudaFuncAttributes attr;
    if (num_types == 3 && num_shells == 3) {
        // 内层分支：基于 num_sro_groups 的范围
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(3, 3, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(3, 3, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(3, 3, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(3, 3, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(3, 3, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(3, 3, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(3, 3, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(3, 3, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(3, 3, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(3, 3, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    } 
    else if (num_types == 4 && num_shells == 3) {
        // 内层分支：基于 num_sro_groups 的范围
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(4, 3, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(4, 3, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(4, 3, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(4, 3, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(4, 3, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(4, 3, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(4, 3, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(4, 3, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(4, 3, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(4, 3, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    } 
    else if (num_types == 5 && num_shells == 3) {
        // 内层分支：基于 num_sro_groups 的范围
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(5, 3, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(5, 3, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(5, 3, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(5, 3, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(5, 3, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(5, 3, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(5, 3, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(5, 3, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(5, 3, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(5, 3, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
        
    } 
    else if (num_types == 6 && num_shells == 3) {
        // 内层分支：基于 num_sro_groups 的范围
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(6, 3, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(6, 3, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(6, 3, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(6, 3, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(6, 3, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(6, 3, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(6, 3, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(6, 3, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(6, 3, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(6, 3, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    }
    else if (num_types == 3 && num_shells == 2) {
        // 内层分支：基于 num_sro_groups 的范围
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(3, 2, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(3, 2, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(3, 2, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(3, 2, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(3, 2, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(3, 2, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(3, 2, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(3, 2, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(3, 2, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(3, 2, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    } 
    else if (num_types == 4 && num_shells == 2) {
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(4, 2, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(4, 2, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(4, 2, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(4, 2, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(4, 2, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(4, 2, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(4, 2, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(4, 2, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(4, 2, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(4, 2, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
        
    } 
    else if (num_types == 5 && num_shells == 2) {
       if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(5, 2, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(5, 2, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(5, 2, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(5, 2, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(5, 2, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(5, 2, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(5, 2, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(5, 2, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(5, 2, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(5, 2, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    }
    else if (num_types == 6 && num_shells == 2) {
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(6, 2, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(6, 2, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(6, 2, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(6, 2, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(6, 2, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(6, 2, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(6, 2, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(6, 2, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(6, 2, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(6, 2, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    }
    else if (num_types == 3 && num_shells == 1) {
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(3,1, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(3, 1, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(3, 1, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(3, 1, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(3, 1, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(3, 1, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(3, 1, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(3, 1, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(3, 1, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(3, 1, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    }
     else if (num_types == 4 && num_shells == 1) {
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(4,1, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(4, 1, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(4, 1, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(4, 1, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(4, 1, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(4, 1, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(4, 1, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(4, 1, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(4, 1, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(4, 1, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    }
     else if (num_types == 5 && num_shells == 1) {
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(5, 1, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(5, 1, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(5, 1, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(5, 1, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(5, 1, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(5, 1, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(5, 1, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(5, 1, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(5, 1, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(5, 1, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    } 
    else if (num_types == 6 && num_shells == 1) {
        if (num_sro_groups >= 1 && num_sro_groups <= 5) {
            LAUNCH_PMC_KERNEL(6,1, 5);
        } else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(6, 1, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(6, 1, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(6, 1, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(6, 1, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(6, 1, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(6, 1, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(6, 1, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(6, 1, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(6, 1, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    }
    else if (num_types == 10 && num_shells == 1) {
        if (num_sro_groups == 1) {
            LAUNCH_PMC_KERNEL(10, 1, 1);
        }else if (num_sro_groups >= 2 && num_sro_groups <= 5){
            LAUNCH_PMC_KERNEL(10, 1, 5);
        }else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(10, 1, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(10, 1, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(10, 1, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(10, 1, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(10, 1, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(10, 1, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(10, 1, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(10, 1, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(10, 1, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    } 
    else if (num_types == 14 && num_shells == 1) {
        if (num_sro_groups == 1) {
            LAUNCH_PMC_KERNEL(14, 1, 1);
        }else if (num_sro_groups >= 2 && num_sro_groups <= 5){
            LAUNCH_PMC_KERNEL(14, 1, 5);
        }else if (num_sro_groups > 5 && num_sro_groups <= 10) {
            LAUNCH_PMC_KERNEL(14, 1, 10);
        } else if (num_sro_groups > 10 && num_sro_groups <= 15) {
            LAUNCH_PMC_KERNEL(14, 1, 15);
        } else if (num_sro_groups > 15 && num_sro_groups <= 20) {
            LAUNCH_PMC_KERNEL(14, 1, 20);
        } else if (num_sro_groups > 20 && num_sro_groups <= 25) {
            LAUNCH_PMC_KERNEL(14, 1, 25);
        } else if (num_sro_groups > 25 && num_sro_groups <= 30) {
            LAUNCH_PMC_KERNEL(14, 1, 30);
        } else if (num_sro_groups > 30 && num_sro_groups <= 35) {
            LAUNCH_PMC_KERNEL(14, 1, 35);
        } else if (num_sro_groups > 35 && num_sro_groups <= 40) {
            LAUNCH_PMC_KERNEL(14, 1, 40);
        } else if (num_sro_groups > 40 && num_sro_groups <= 45) {
            LAUNCH_PMC_KERNEL(14, 1, 45);
        } else if (num_sro_groups > 45 && num_sro_groups <= 50) {
            LAUNCH_PMC_KERNEL(14, 1, 50);
        } else {
            // 处理不支持的组数
            std::cerr << "错误: 不支持的 num_sro_groups (" << num_sro_groups << ")" << std::endl;
            exit(1);
        }
    } 
    else {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cerr <<  std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - ApolloX2 - INFO - " 
                  << "Error: Parallel Monte Carlo optimization "
                  << "GPU implementation is not available for the given number of types " << num_types 
                  << " and shells " << num_shells << std::endl;
        exit(1);
    }
}
#undef LAUNCH_PMC_KERNEL
template <typename Integer, typename Real>
void calculate_best_lattices(Integer num_lattices, Integer num_atoms, Integer* lattices, Integer* new_lattices, Real* fitness, Real* new_fitness) {
    sort_lattices_by_fitness(lattices, fitness, 2 * num_lattices, num_atoms);
}

/**
 * @brief Main optimization function for finding optimal lattice configurations.
 * 
 * @param[in] num_iters Number of optimization iterations.
 * @param[in] num_lattices Number of lattices in population.
 * @param[in] host_species Array containing count of each atom type.
 * @param[in] host_weights Weight factors for each shell.
 * @param[in] host_nbor Neighbor list information.
 * @param[in] host_target_sro Target SRO values to achieve.
 * @param[in] host_lattices Initial lattice configurations.
 * @param[out] best_lattices Output array for best found configurations.
 * @param[out] best_fitness Output array for best fitness values.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> 
run_local_parallel_hcs_cuda(
        const int num_lattices,
        const int num_iters,
        const int k,
        const int num_tasks,
        const int search_depth,
        const double threshold,
        const std::vector<std::vector<std::vector<int>>>& neighbor_list,
        const std::vector<int>& host_species,
        const std::vector<double>& host_weights,
        const std::vector<std::vector<std::vector<double>>>& host_target_sro,const std::vector<std::vector<int>>& host_swap_groups,const std::vector<std::vector<int>>& host_sro_groups)        
{ std::cout << "DEBUG C++: Received " << host_swap_groups.size() << " swap groups." << std::endl;
    for (size_t i = 0; i < host_swap_groups.size(); ++i) {
        std::cout << "  Group " << i << " (size " << host_swap_groups[i].size() << "): [ ";
        for (int member : host_swap_groups[i]) {
            std::cout << member << " ";
        }
        std::cout << "]" << std::endl;
    }
    int idx = 0;
    const int num_shells = host_weights.size();
    double sum = 0.0;
    for (double w : host_weights) {
        sum += w;
    }
    const int num_types  = host_species.size();
    const int num_atoms  = std::accumulate(host_species.begin(), host_species.end(), 0);
    const int num_coefficients = num_types * num_types * num_shells;
    std::vector<int> host_flat_nbor;
    std::vector<int> host_flat_nbor_idx;
    // Flatten neighbor_list for easier copying to device
    std::vector<int> flat_groups;//展平的host_swap_groups
    std::vector<int> group_offsets;//存着每个group起点的指标
    std::vector<int> group_sizes;//存着每个group的大小
    size_t max_group_size = 0;
    int offset = 0;
    for (const auto& group : host_swap_groups) {
        group_offsets.push_back(offset);
        group_sizes.push_back(group.size());
        if (group.size() > max_group_size) {
            max_group_size = group.size();
        }
        flat_groups.insert(flat_groups.end(), group.begin(), group.end());
        offset += group.size();
    }
    std::vector<bool> is_restricted(num_atoms, false);
    for (int idx : flat_groups) {
        is_restricted[idx] = true;
    }
    std::vector<int> host_free_indices;
    for (int i = 0; i < num_atoms; ++i) {
        if (!is_restricted[i]) {
            host_free_indices.push_back(i);
        }
    }
     // NEW: --- Flatten `host_sro_groups` (Replicating logic from swap_groups) ---
    std::cout << "DEBUG C++: Received " << host_sro_groups.size() << " SRO groups." << std::endl;
    std::vector<int> flat_sro_groups;
    std::vector<int> sro_group_offsets;
    std::vector<int> sro_group_sizes;
    int sro_offset = 0;
    int count=0;
    for (const auto& group : host_sro_groups) {
        sro_group_offsets.push_back(sro_offset);
        sro_group_sizes.push_back(group.size());
        flat_sro_groups.insert(flat_sro_groups.end(), group.begin(), group.end());
        sro_offset += group.size();
        count+=1;
    }
    // NEW: --- Prepare SRO group data for GPU ---
    int* d_flat_sro_groups;
    int* d_sro_group_offsets;
    int* d_sro_group_sizes;
    cudaMalloc(&d_flat_sro_groups, flat_sro_groups.size() * sizeof(int));
    cudaMalloc(&d_sro_group_offsets, sro_group_offsets.size() * sizeof(int));
    cudaMalloc(&d_sro_group_sizes, sro_group_sizes.size() * sizeof(int));
    cudaMemcpy(d_flat_sro_groups, flat_sro_groups.data(), flat_sro_groups.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sro_group_offsets, sro_group_offsets.data(), sro_group_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sro_group_sizes, sro_group_sizes.data(), sro_group_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);
    int* d_flat_groups;
    int* d_group_offsets;
    int* d_group_sizes;
    int* d_free_indices;

    cudaMalloc(&d_flat_groups, flat_groups.size() * sizeof(int));
    cudaMalloc(&d_group_offsets, group_offsets.size() * sizeof(int));
    cudaMalloc(&d_group_sizes, group_sizes.size() * sizeof(int));
    cudaMalloc(&d_free_indices, host_free_indices.size() * sizeof(int));

    cudaMemcpy(d_flat_groups, flat_groups.data(), flat_groups.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_offsets, group_offsets.data(), group_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, group_sizes.data(), group_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_free_indices, host_free_indices.data(), host_free_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    size_t shuffle_buffer_size = std::max(max_group_size, host_free_indices.size());//取出分组的最大值
    size_t shared_mem_size = shuffle_buffer_size * sizeof(int);

    for (const auto& shell : neighbor_list) {
        int count = 0;
        for (const auto& neighbor_list : shell) {
            host_flat_nbor.insert(host_flat_nbor.end(), neighbor_list.begin(), neighbor_list.end());
            host_flat_nbor_idx.push_back(idx);
            idx += neighbor_list.size();
            count++;
            if (count == num_shells) {
                break;
            }
        }
    }
    host_flat_nbor_idx.push_back(idx);
    // Flatten target SRO for device
   std::vector<float> host_flat_target_sro;
    for (const auto& group_sro : host_target_sro) {
        for (const auto& shell_sro : group_sro) {
            // This assumes shell_sro is already flat (vector<double>)
            for (double val : shell_sro) {
                host_flat_target_sro.push_back(static_cast<float>(val));
            }
        }
    }
    // NEW: Create a map from atom index to its SRO group index
    std::vector<int> host_atom_to_sro_group_map(num_atoms, -1); // -1 means not in any SRO group
    for (size_t group_idx = 0; group_idx < host_sro_groups.size(); ++group_idx) {
        for (int atom_idx : host_sro_groups[group_idx]) {
            // Add a check to prevent an atom from being in multiple SRO groups
            if (atom_idx >= 0 && atom_idx < num_atoms) {
                host_atom_to_sro_group_map[atom_idx] = group_idx;
            }
        }
    }
    // NEW: Allocate and copy this new information to the GPU
    int* d_atom_to_sro_group_map;
    cudaMalloc(&d_atom_to_sro_group_map, host_atom_to_sro_group_map.size() * sizeof(int));
    cudaMemcpy(d_atom_to_sro_group_map, host_atom_to_sro_group_map.data(), host_atom_to_sro_group_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    int *species = nullptr, *flat_nbor = nullptr, *flat_nbor_idx = nullptr, *lattices = nullptr, *new_lattices = nullptr;
    float *weights = nullptr, *fitness = nullptr, *new_fitness = nullptr, *target_sro = nullptr, *coefficients = nullptr;
    cudaMalloc(&species, host_species.size() * sizeof(int));
    cudaMemcpy( species, host_species.data(), host_species.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<float> host_float_weights(host_weights.size(), 0.0);
    for (size_t ii = 0; ii < host_weights.size(); ii++) {
        host_float_weights[ii] = static_cast<float>(host_weights[ii]);
    }
    cudaMalloc(&weights, host_weights.size() * sizeof(float));
    cudaMemcpy( weights, host_float_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory on the GPU for the fitness array
    // The size is 2 * num_lattices * sizeof(float)
    cudaMalloc(&fitness,  2 * num_lattices * sizeof(float));
    // Allocate memory on the GPU for the lattices array
    // The size is 2 * num_lattices * num_atoms * sizeof(int)
    cudaMalloc(&lattices, 2 * num_lattices * num_atoms * sizeof(int));
    // Set new_fitness to point to the second half of the fitness array
    new_fitness  = fitness  + num_lattices;
    // Set new_lattices to point to the second half of the lattices array
    new_lattices = lattices + num_lattices * num_atoms;

    cudaMalloc(&flat_nbor, host_flat_nbor.size() * sizeof(int));
    cudaMemcpy( flat_nbor, host_flat_nbor.data(), host_flat_nbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&flat_nbor_idx, host_flat_nbor_idx.size() * sizeof(int));
    cudaMemcpy( flat_nbor_idx, host_flat_nbor_idx.data(), host_flat_nbor_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    const int total_coefficients = count * num_coefficients;
    cudaMalloc(&coefficients, num_lattices *total_coefficients* sizeof(float));

    // Allocate device memory for target SRO
    cudaMalloc(&target_sro, host_flat_target_sro.size() * sizeof(float));
    cudaMemcpy(target_sro, host_flat_target_sro.data(), host_flat_target_sro.size() * sizeof(float), cudaMemcpyHostToDevice);
    std::ofstream evolution_file("evolution_data.txt");
    if (!evolution_file.is_open()) {
        std::cerr << "Error: Failed to open evolution_data.txt for writing." << std::endl;
    }
    std::unordered_set<std::string> seen_lattices;
    generate_norm_lattices(
        num_lattices, num_atoms, num_types, num_shells, species, weights,
        flat_nbor, flat_nbor_idx, target_sro, coefficients, fitness, lattices,
        d_flat_groups, d_group_offsets, d_group_sizes, host_swap_groups.size(),
        d_free_indices, host_free_indices.size(), shared_mem_size,
        d_flat_sro_groups, d_sro_group_offsets, d_sro_group_sizes, host_sro_groups.size(),
        d_atom_to_sro_group_map
    );
    std::vector<float> host_fitness(num_lattices, 0);
    //cudaMemcpy(host_fitness.data(), fitness, sizeof(float)*num_lattices, cudaMemcpyDeviceToHost);
    // Global Search Loop
    for (int ii = 0; ii < num_iters; ii++) {
        // Perpuatation: Generate new lattices randomly
        generate_random_lattices(
            num_lattices, num_atoms, num_types, num_shells, species, weights,
            flat_nbor, flat_nbor_idx, target_sro, coefficients, new_fitness, new_lattices,
            d_flat_groups, d_group_offsets, d_group_sizes, host_swap_groups.size(),
            d_free_indices, host_free_indices.size(), shared_mem_size,
            d_flat_sro_groups, d_sro_group_offsets, d_sro_group_sizes, host_sro_groups.size(),
            d_atom_to_sro_group_map
        );
        cudaDeviceSynchronize();

    // 2. Create a vector on the CPU (host) to hold the results
    std::vector<float> host_new_fitness(num_lattices);

    // 3. Copy the fitness values from the GPU (device) to the CPU (host)
    cudaMemcpy(
        host_new_fitness.data(),      // Destination: host memory
        new_fitness,                  // Source: device memory
        num_lattices * sizeof(float), // Total bytes to copy
        cudaMemcpyDeviceToHost        // Direction of copy
    );

    // 4. Loop through the host vector and print each value
    std::cout << "--- Printing new_fitness values after generation ---" << std::endl;
    for (int i = 0; i < num_lattices; ++i) {
        std::cout << "Lattice " << i << " new fitness: " << host_new_fitness[i] << std::endl;
    }
    std::cout << "----------------------------------------------------" << std::endl;
        // Local Search  I: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(
            /* inputs  */ num_lattices, num_atoms, num_types, num_shells, num_tasks, search_depth, (float)threshold, flat_nbor, flat_nbor_idx, species, host_species, weights, target_sro,
            /* swap info */ d_flat_groups,d_group_offsets,d_group_sizes,host_swap_groups.size(),d_free_indices,host_free_indices.size(),
            /* sro info */ d_flat_sro_groups, d_sro_group_offsets, d_sro_group_sizes, host_sro_groups.size(), d_atom_to_sro_group_map,
            /* outputs */ new_fitness, new_lattices);
        // Ranking: Calculate the best lattices and update the fitness values
        calculate_best_lattices(num_lattices, num_atoms, lattices, new_lattices, fitness, new_fitness);
        // Local Search II: Perform local parallel Monte Carlo optimization
        local_parallel_monte_carlo(
            /* inputs  */ num_lattices, num_atoms, num_types, num_shells, num_tasks, search_depth, (float)threshold, flat_nbor, flat_nbor_idx, species, host_species, weights, target_sro,
            /* swap info */ d_flat_groups,d_group_offsets,d_group_sizes,host_swap_groups.size(),d_free_indices,host_free_indices.size(),
            /* sro info */ d_flat_sro_groups, d_sro_group_offsets, d_sro_group_sizes, host_sro_groups.size(), d_atom_to_sro_group_map,
            /* outputs */ fitness, lattices);
        if (ii >= k) {
            
            // Create temporary host vectors to hold the data for this iteration
            std::vector<int> current_host_lattices(num_lattices * num_atoms);
            std::vector<float> current_host_fitness(num_lattices);

            // Copy the current population's data from GPU to host
            cudaMemcpy(current_host_lattices.data(), lattices, num_lattices * num_atoms * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(current_host_fitness.data(), fitness, num_lattices * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Write the data to the file
            evolution_file << "Iteration " << ii << "\n";
            for (int lat_idx = 0; lat_idx < num_lattices; ++lat_idx) {
                std::string lattice_signature = "";
                for (int atom_idx = 0; atom_idx < num_atoms; ++atom_idx) {
            lattice_signature += std::to_string(current_host_lattices[lat_idx * num_atoms + atom_idx]) + " ";
        }
            if (seen_lattices.count(lattice_signature)) {
            continue; // 直接跳到 for 循环的下一次迭代
        }
        seen_lattices.insert(lattice_signature);
                evolution_file << "Lattice " << lat_idx << " Fitness: " << current_host_fitness[lat_idx]/num_types/sqrt(sum*host_sro_groups.size()) << "\n";
                evolution_file << "Data: " << lattice_signature << "\n";
            
        }
        evolution_file << "---\n"; // Separator for clarity
        }
        cudaMemcpy(host_fitness.data(), fitness, sizeof(float), cudaMemcpyDeviceToHost);
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cout << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << " - ApolloX2 - INFO - " << "Iter " << ii << " with best fitness:      " << host_fitness[0] << std::endl;
    }
    evolution_file.close();
    int * temp_lattices = new int[num_lattices * num_atoms];
    std::vector<std::vector<int>> final_lattices(num_lattices, std::vector<int>(num_atoms, 0));
    std::vector<float> temp_fitness(num_lattices, 0.0);
    std::vector<double> final_fitness(num_lattices, 0.0);
    cudaMemcpy(temp_lattices, lattices, num_lattices * num_atoms * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_fitness.data(), fitness, num_lattices * sizeof(float), cudaMemcpyDeviceToHost);
    for (int ii = 0; ii < num_lattices; ii++) {
        for (int jj = 0; jj < num_atoms; jj++) {
            final_lattices[ii][jj] = temp_lattices[ii * num_atoms + jj];
        }
        final_fitness[ii] = temp_fitness[ii];
    }
    delete [] temp_lattices;

    cudaFree(species);
    cudaFree(weights);

    cudaFree(fitness);
    cudaFree(lattices);

    cudaFree(flat_nbor);
    cudaFree(flat_nbor_idx);

    cudaFree(target_sro);
    cudaFree(coefficients);
    cudaFree(d_flat_groups);
    cudaFree(d_group_offsets);
    cudaFree(d_group_sizes);
    cudaFree(d_free_indices);
    cudaFree(d_flat_sro_groups);
    cudaFree(d_sro_group_offsets);
    cudaFree(d_sro_group_sizes);
    cudaFree(d_atom_to_sro_group_map);
    return std::make_tuple(final_lattices, final_fitness);
}
    
} // namespace cpu
} // namespace accelerate
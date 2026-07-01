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
#include <sstream>
#include <string>
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
namespace accelerate {
namespace gpu2 {
using LatticeType=uint8_t;
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
struct LatticeEntry {
    float fitness;               // 一个浮点数，表示适应度(常见于优化/搜索算法)
    std::vector<LatticeType> data;       // 一个整型向量，用来存储与晶格相关的数据(比如原子排列、索引等)

    // 一个比较运算符，用于排序
    bool operator<(const LatticeEntry& other) const {
        return fitness < other.fitness;
    }
};

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
        Real* fitness, LatticeType* lattices,
        const Integer* atom_to_sro_group_map,
        std::size_t num_sro_groups)
{
    const Integer bid = blockIdx.x;
    const Integer tid = threadIdx.x;
    const Integer num_coefficients_per_group = num_types * num_types * num_shells;
    const Integer total_coefficients = num_sro_groups * num_coefficients_per_group;
    // Each block addresses a different lattice
    const LatticeType* lattice = lattices + (long long)bid * num_atoms;

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
            LatticeType atom_type = lattice[atom_idx];
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

        LatticeType atom_type = lattice[atom_site_idx];
        Integer num_atoms_in_group=shared_group_total_atoms[sro_group_idx];
        if (num_atoms_in_group==0) continue;
        for (Integer shell_idx = 0; shell_idx < num_shells; ++shell_idx) {
            Integer start = neighbor_list_indices[atom_site_idx * num_shells + shell_idx];
            Integer end = neighbor_list_indices[atom_site_idx * num_shells + shell_idx + 1];
            Integer neighbor_size = end - start;

            if (neighbor_size == 0) continue;

            for (Integer k = 0; k < neighbor_size; ++k) {
                Integer neighbor_site_idx = neighbor_list[start + k];
                Integer neighbor_type = lattice[neighbor_site_idx];
                Integer group_offset = sro_group_idx * num_coefficients_per_group;
                Integer local_idx = atom_type * num_types * num_shells + neighbor_type * num_shells + shell_idx;

                // The normalization for P_ij must be divided by N_i, the number of central atoms of that type in the group.
                atomicAdd(&shared_gamma[group_offset + local_idx], 1.0f / (neighbor_size * num_atoms_in_group));
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
        Integer num_atoms_in_group= shared_group_total_atoms[group_idx];
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
 * @brief Cooperatively calculates SRO coefficients for a block's initial lattice.
 *
 * All threads in a block call this function to work together on populating a
 * single coefficients array stored in shared memory.
 */
template <typename Integer, typename Real, typename LatticeType,
          int NUM_TYPES, int NUM_SHELLS>
__device__ __forceinline__ void calculate_shared_coefficients(
    const Integer& num_atoms,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* atom_to_sro_group_map,
    const Integer num_sro_groups,
    Real* shared_coefficients, // Writes to a shared memory array
    Integer* shared_group_species_counts,
    Integer* shared_group_total_atoms,
    const LatticeType* initial_lattice_for_block) // Reads from the single initial lattice
{
    const Integer tid = threadIdx.x;
    const Integer num_threads = blockDim.x;
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
            LatticeType atom_type = initial_lattice_for_block[atom_idx];
            atomicAdd(&shared_group_species_counts[group_idx * NUM_TYPES + atom_type], 1);
            atomicAdd(&shared_group_total_atoms[group_idx], 1);
        }
    }
    __syncthreads();
    // Parallel loop over all atoms in the lattice
    for (Integer atom_site_idx = tid; atom_site_idx < num_atoms; atom_site_idx += num_threads) {
        Integer sro_group_idx = atom_to_sro_group_map[atom_site_idx];
        if (sro_group_idx < 0) continue;

        LatticeType atom_type = initial_lattice_for_block[atom_site_idx];
        Integer num_atoms_in_group = shared_group_total_atoms[sro_group_idx];
        if (num_atoms_in_group == 0) continue;


        for (Integer shell_idx = 0; shell_idx < NUM_SHELLS; ++shell_idx) {
            Integer start = neighbor_list_indices[atom_site_idx * NUM_SHELLS + shell_idx];
            Integer end = neighbor_list_indices[atom_site_idx * NUM_SHELLS + shell_idx + 1];
            Integer neighbor_size = end - start;
            if (neighbor_size == 0) continue;

            for (Integer k = 0; k < neighbor_size; ++k) {
                Integer neighbor_site_idx = neighbor_list[start + k];
                LatticeType neighbor_type = initial_lattice_for_block[neighbor_site_idx];
                
                Integer group_offset = sro_group_idx * NUM_COEFF_PER_GROUP;
                Integer local_idx = (Integer)atom_type * NUM_TYPES * NUM_SHELLS + (Integer)neighbor_type * NUM_SHELLS + shell_idx;
                
                // Use atomicAdd because all threads write to the same shared array
                atomicAdd(&shared_coefficients[group_offset + local_idx], 1.0f / (neighbor_size * num_atoms_in_group));
            }
        }
    }
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
          typename LatticeType, int NUM_TYPES, int NUM_SHELLS,int N_SRO_GROUPS_CE>
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
    LatticeType* lattice)
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
    LatticeType type1_orig = lattice[atom1];
    LatticeType type2_orig = lattice[atom2];
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
        LatticeType type1_orig = lattice[atom1];
        LatticeType type1_new = lattice[atom2];
        Integer neighbor_size1 = neighbor_list_indices[atom1 * NUM_SHELLS + shell + 1] - neighbor_list_indices[atom1 * NUM_SHELLS + shell];

        for (Integer kk = 0; kk < neighbor_size1; ++kk) {
            Integer neighbor_idx = neighbor_list[neighbor_list_indices[atom1 * NUM_SHELLS + shell] + kk];
            LatticeType neighbor_type_orig = lattice[neighbor_idx];
            
            LatticeType neighbor_type_new = neighbor_type_orig;
            if (neighbor_idx == atom2) { // If the neighbor is the other swapped atom
                neighbor_type_new = type1_orig; // Its new type is the original type of atom1
            }

            // --- Subtract the OLD bond's contribution (symmetrically) ---
            // A.1: From atom1's perspective
            Integer group_idx_1 = atom_to_sro_group_map[atom1];
            Integer num_atoms_in_group1 = (group_idx1 >= 0) ? initial_group_total_atoms[group_idx1] : 0;
            if (group_idx_1 >= 0&& neighbor_size1 > 0 && num_atoms_in_group1 > 0) {
                    Integer offset = group_idx_1 * NUM_COEFF_PER_GROUP;
                    Integer idx = type1_orig * NUM_TYPES * NUM_SHELLS + neighbor_type_orig * NUM_SHELLS + shell;
                    coefficients[offset + idx] -= 1.0 / (neighbor_size1 * num_atoms_in_group1);
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
            if (group_idx_1 >= 0&& neighbor_size1 > 0 && num_atoms_in_group1 > 0) {
                    Integer offset = group_idx_1 * NUM_COEFF_PER_GROUP;
                    Integer idx = type1_new * NUM_TYPES * NUM_SHELLS + neighbor_type_new * NUM_SHELLS + shell;
                    coefficients[offset + idx] += 1.0 / (neighbor_size1 * num_atoms_in_group1);

            }
            // B.2: From the neighbor's new perspective
            if (group_idx_n >= 0&& neighbor_size_n > 0 && num_atoms_in_group_n > 0) {
                Integer offset = group_idx_n * NUM_COEFF_PER_GROUP;
                Integer idx = neighbor_type_new * NUM_TYPES * NUM_SHELLS + type1_new * NUM_SHELLS + shell;
                coefficients[offset + idx] += 1.0 / (neighbor_size_n * num_atoms_in_group_n);
            }
        }

        // --- Block 2: Handle neighbors of the second swapped atom (atom2) ---
        LatticeType type2_orig = lattice[atom2];
        LatticeType type2_new = lattice[atom1];
        Integer neighbor_size2 = neighbor_list_indices[atom2 * NUM_SHELLS + shell + 1] - neighbor_list_indices[atom2 * NUM_SHELLS + shell];

        for (Integer kk = 0; kk < neighbor_size2; ++kk) {
            Integer neighbor_idx = neighbor_list[neighbor_list_indices[atom2 * NUM_SHELLS + shell] + kk];

            // CRITICAL: Skip the neighbor if it is atom1, as that bond has already been fully processed in Block 1.
            if (neighbor_idx == atom1) continue;

            LatticeType neighbor_type_orig = lattice[neighbor_idx];
            
            // The neighbor's type does not change in this block because we've already excluded atom1.
            LatticeType neighbor_type_new = neighbor_type_orig;
            Integer num_atoms_in_group2 = (group_idx2 >= 0) ? initial_group_total_atoms[group_idx2] : 0;
            // --- Subtract the OLD bond's contribution (symmetrically) ---
            // A.1: From atom2's perspective
            Integer group_idx_2 = atom_to_sro_group_map[atom2];
            if (group_idx_2 >= 0&& neighbor_size2 > 0 && num_atoms_in_group2 > 0) {
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
            if (group_idx_2 >= 0&& neighbor_size2 > 0 && num_atoms_in_group2 > 0) {
                    Integer offset = group_idx_2 * NUM_COEFF_PER_GROUP;
                    Integer idx = type2_new * NUM_TYPES * NUM_SHELLS + neighbor_type_new * NUM_SHELLS + shell;
                    coefficients[offset + idx] += 1.0 / (neighbor_size2 * num_atoms_in_group2);
            }
            // B.2: From the neighbor's new perspective
            if (group_idx_n >= 0&& neighbor_size_n > 0 && num_atoms_in_group_n > 0) {
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
template <typename Integer, typename Real, typename LatticeType,
          int NUM_TYPES, int NUM_SHELLS, int NUM_TOTAL_COEFF,int N_SRO_GROUPS_CE>
__global__ void independent_monte_carlo(
    // Input parameters
    const Real sum,const Integer num_types,
    const Real fitness_threshold, const Integer search_depth, const unsigned long long random_seed,
    const Integer num_atoms,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* flat_groups, const Integer* group_offsets, const Integer* group_sizes, std::size_t num_swap_groups,
    const Integer* free_indices, const Integer num_free_indices,
    const Integer* atom_to_sro_group_map,
    std::size_t num_sro_groups, const Real* target_sro,
    const LatticeType* initial_lattices, // ⭐ CHANGE: Accepts array of uint8_t lattices
    const Real* initial_fitnesses,      // Array of fitnesses for each starting lattice
    const Real temperature,

    // Workspace and Output Buffers
    LatticeType* thread_workspaces,      // Workspace is uint8_t
    Real* result_fitness_buffer,
    LatticeType* result_lattices_buffer, // ⭐ CHANGE: Output buffer is also uint8_t
    Integer* result_count
)
{   extern __shared__ Real shared_mem[];
    Real* shared_coefficients = shared_mem;
    Integer* shared_group_species_counts = (Integer*)(shared_mem + NUM_TOTAL_COEFF);
    Integer* shared_group_total_atoms = (Integer*)(shared_group_species_counts + num_sro_groups * NUM_TYPES);
    const Integer block_id = blockIdx.x;
    const Integer tid = threadIdx.x;
    const Integer num_threads=blockDim.x;
    const Integer global_tid = block_id * num_threads + tid;
    for (int i = tid; i < NUM_TOTAL_COEFF; i += num_threads) {
        shared_coefficients[i] = 0.0f;
    }
    __syncthreads();
    const LatticeType* single_initial_lattice = initial_lattices + (long long)block_id * num_atoms;
    calculate_shared_coefficients<Integer, Real, LatticeType, NUM_TYPES, NUM_SHELLS>(
        num_atoms,
        neighbor_list, neighbor_list_indices,
        atom_to_sro_group_map, 
        num_sro_groups,
        shared_coefficients,
        shared_group_species_counts, // Pass pointers to the count arrays
        shared_group_total_atoms,
        single_initial_lattice
    );
    __syncthreads();
    curandState local_state;
    curand_init(random_seed, global_tid, 0, &local_state);

    Real local_coefficients[NUM_TOTAL_COEFF];
    LatticeType* local_lattice = thread_workspaces + (long long)global_tid * num_atoms;
    for (int i = 0; i < num_atoms; ++i) {
        local_lattice[i] = single_initial_lattice[i];
    }
    for (int i = 0; i < NUM_TOTAL_COEFF; ++i) {
        local_coefficients[i] = shared_coefficients[i];
    }
    Real reg_fitness = initial_fitnesses[block_id];
    Integer unsuccessful_attempts = 0;
    while (unsuccessful_attempts < search_depth) {
        Real temp_coefficients[NUM_TOTAL_COEFF];
        for (Integer ii = 0; ii < NUM_TOTAL_COEFF; ii++) {
            temp_coefficients[ii] = local_coefficients[ii];
        }

        // ⭐ COMPLETED: Atom selection logic, now operating on the private local_lattice.
        Integer atom1, atom2;
        do {
            Integer total_swappable_entities = num_swap_groups + (num_free_indices > 1 ? 1 : 0);
            if (total_swappable_entities == 0) continue;

            Integer choice = curand(&local_state) % total_swappable_entities;

            if (choice < num_swap_groups) { // Pick from a restricted swap group
                Integer group_size = group_sizes[choice];
                if (group_size < 2) continue;
                Integer offset = group_offsets[choice];
                Integer idx1 = curand(&local_state) % group_size;
                Integer idx2 = curand(&local_state) % group_size;
                atom1 = flat_groups[offset + idx1];
                atom2 = flat_groups[offset + idx2];
            } else { // Pick from the free indices group
                Integer idx1 = curand(&local_state) % num_free_indices;
                Integer idx2 = curand(&local_state) % num_free_indices;
                atom1 = free_indices[idx1];
                atom2 = free_indices[idx2];
            }
        } while (atom1 == atom2 || local_lattice[atom1] == local_lattice[atom2]);

        // Calculate the fitness of the proposed swap incrementally.
        // NOTE: calculate_fitness_incremental must also be updated to work with LatticeType.
        Real new_fitness = calculate_fitness_incremental<Integer, Real, LatticeType,NUM_TYPES, NUM_SHELLS,N_SRO_GROUPS_CE>(
            atom1, atom2, num_atoms, species, weights, neighbor_list, neighbor_list_indices,
            atom_to_sro_group_map, shared_group_species_counts,shared_group_total_atoms,
            num_sro_groups, target_sro, temp_coefficients, local_lattice);

        // ⭐ COMPLETED: Metropolis acceptance logic for this thread.
        if (new_fitness < reg_fitness) {
            reg_fitness = new_fitness;
            // Apply the swap to the thread's private lattice
            LatticeType swap = local_lattice[atom1];
            local_lattice[atom1] = local_lattice[atom2];
            local_lattice[atom2] = swap;
            // Commit the updated coefficients
            for (Integer ii = 0; ii < NUM_TOTAL_COEFF; ii++) {
                local_coefficients[ii] = temp_coefficients[ii];
            }
            unsuccessful_attempts = 0;
        } else if (temperature > 1e-9f) { // Use 'f' suffix for float constants
            Real delta_fitness = new_fitness - reg_fitness;
            Real acceptance_prob = expf(-delta_fitness / temperature);
            if (curand_uniform(&local_state) < acceptance_prob) {
                reg_fitness = new_fitness;
            // Apply the swap to the thread's private lattice
            LatticeType swap = local_lattice[atom1];
            local_lattice[atom1] = local_lattice[atom2];
            local_lattice[atom2] = swap;
            // Commit the updated coefficients
            for (Integer ii = 0; ii < NUM_TOTAL_COEFF; ii++) {
                local_coefficients[ii] = temp_coefficients[ii];
            }
            }
            unsuccessful_attempts++;
        }

        else {
            unsuccessful_attempts++;
        }
    }
    reg_fitness=reg_fitness/num_types/sqrt(sum*num_sro_groups);
    // --- Step 4: If the result is good, write it to the global buffer ---
    if (reg_fitness < fitness_threshold) {
        Integer write_index = atomicAdd(result_count, 1);
        // Safety check to prevent buffer overflow if too many results are found.
        // The max number of results is the total number of threads launched in the grid.
        if (write_index < (blockDim.x * gridDim.x)) {
            result_fitness_buffer[write_index] = reg_fitness;
            LatticeType* result_lattice_start = result_lattices_buffer + (long long)write_index * num_atoms;
            for (Integer i = 0; i < num_atoms; ++i) {
                result_lattice_start[i] = local_lattice[i];
            }
        }
    }
}

/**
 * @brief Reads lattice configurations from a file, sorts them by fitness, and returns them.
 */
/**
 * @brief Reads only the lattice configurations from a file. Ignores fitness values.
 *
 * @param filename The name of the file to read from.
 * @param num_atoms The expected number of atoms per lattice.
 * @return A vector containing all valid lattice configurations found in the file.
 */
std::vector<std::vector<LatticeType>> read_lattices_from_file(const std::string& filename, int num_atoms) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "FATAL ERROR: Could not open initial configuration file: " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<LatticeType>> all_lattices;
    std::string line;

    while (std::getline(infile, line)) {
        // Only look for "Data:" lines, ignore everything else.
        if (line.find("Data:") != std::string::npos) {
            std::stringstream ss(line.substr(line.find(":") + 1));
            std::vector<LatticeType> current_data;
            current_data.reserve(num_atoms);
            int atom_type_int;
            while (ss >> atom_type_int) {
                current_data.push_back(static_cast<LatticeType>(atom_type_int));
            }

            if (current_data.size() == num_atoms) {
                all_lattices.push_back(current_data);
            }
        }
    }
    infile.close();
    std::cout << "Found " << all_lattices.size() << " valid lattice configurations in file." << std::endl;
    return all_lattices;
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
        /* Calculate the total number of coefficients based on compile-time constants. */ \
        const int NUM_COEFF_PER_GROUP = (NT) * (NT) * (NS); \
        const int TOTAL_COEFF = NUM_COEFF_PER_GROUP * (N_SRO_GROUPS_CE); \
        \
        /* Get a unique seed for this launch. */ \
        unsigned long long seed = get_microseconds(); \
        \
        /* Launch the corresponding kernel template instance. */ \
        independent_monte_carlo<Integer, Real, LatticeType, NT, NS, TOTAL_COEFF, N_SRO_GROUPS_CE><<<num_blocks, threads_per_block,shared_mem_size>>>( \
            sum,num_types,fitness_threshold, search_depth, seed, num_atoms, \
            species, weights, neighbor_list, neighbor_list_indices, \
            d_flat_groups, d_group_offsets, d_group_sizes, num_swap_groups, \
            d_free_indices, num_free_indices, d_atom_to_sro_group_map, \
           num_sro_groups, target_sro, \
            d_initial_lattices, d_initial_fitnesses, temperature, \
            d_thread_workspaces, d_result_fitness_buffer, d_result_lattices_buffer, d_result_count); \
    } while (0)
template <typename Integer, typename Real>
void launch_independent_mc(
    // Launch configuration
    double sum,const Integer& num_blocks, const Integer& threads_per_block,
    // Kernel parameters
    const Real& fitness_threshold, const Integer& search_depth,
    const Integer& num_atoms, const Integer& num_types, const Integer& num_shells,
    const Integer* species, const Real* weights,
    const Integer* neighbor_list, const Integer* neighbor_list_indices,
    const Integer* d_flat_groups, const Integer* d_group_offsets, const Integer* d_group_sizes, const std::size_t& num_swap_groups,
    const Integer* d_free_indices, const std::size_t& num_free_indices,
    const Integer* d_atom_to_sro_group_map,
    const std::size_t& num_sro_groups, const Real* target_sro,
    const LatticeType* d_initial_lattices, const Real* d_initial_fitnesses,
    const Real& temperature,
    // Workspace and output buffers
    LatticeType* d_thread_workspaces,
    Real* d_result_fitness_buffer,
    LatticeType* d_result_lattices_buffer,
    Integer* d_result_count)
{   const int num_coefficients_per_group = num_types * num_types * num_shells;
    const int total_coefficients_per_lattice = num_sro_groups * num_coefficients_per_group;
    size_t gamma_mem_size = (size_t)total_coefficients_per_lattice * sizeof(Real);
    size_t counts_mem_size = (size_t)num_sro_groups * num_types * sizeof(Integer);
    size_t totals_mem_size = (size_t)num_sro_groups * sizeof(Integer);
    size_t shared_mem_size = gamma_mem_size + counts_mem_size + totals_mem_size;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
    std::cerr << "错误: 请求的共享内存大小超过设备限制" << std::endl;
    exit(1);
}
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
void save_results_to_file(
    const std::string& filename,
    int num_results,
    int num_atoms,
    const std::vector<float>& fitnesses,
    // ⭐ CHANGE: Accepts LatticeType vector
    const std::vector<LatticeType>& lattices,
    bool append,std::unordered_set<std::string>& seen_lattices)
{
    std::ofstream outfile;
    if (append) {
        outfile.open(filename, std::ios_base::app);
    } else {
        outfile.open(filename);
    }
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }
    int new_unique_count = 0;
    for (int i = 0; i < num_results; ++i) {
        std::stringstream ss;
        for (int j = 0; j < num_atoms; ++j) {
            // ⭐ CHANGE: Cast uint8_t to int for printing to avoid it being treated as a character.
            ss << static_cast<int>(lattices[(long long)i * num_atoms + j]) << (j == num_atoms - 1 ? "" : " ");
        }
        std::string signature = ss.str();
        if (seen_lattices.find(signature) == seen_lattices.end()) {
            outfile << "Lattice " << (seen_lattices.size()) << " Fitness: " << std::fixed << std::setprecision(7) << fitnesses[i] << "\n";
            outfile << "Data: " << signature << "\n";
            seen_lattices.insert(signature);
            new_unique_count++;
        }
    }
    outfile.close();
    std::cout << "Saved " << new_unique_count << " unique configurations to " << filename << std::endl;
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

void run_mc_cuda(
        const int num_lattices_to_read,
        const int g_generations,  
        const int num_tasks,//threads_per_block
        const int search_depth,
        const double fitness_threshold,
        const double initial_temp,
        const double cooling_rate,
        const int annealing_steps,
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
    int *species = nullptr, *flat_nbor = nullptr, *flat_nbor_idx = nullptr;
    float *weights = nullptr,*target_sro = nullptr;
    cudaMalloc(&species, host_species.size() * sizeof(int));
    cudaMemcpy( species, host_species.data(), host_species.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<float> host_float_weights(host_weights.size(), 0.0);
    for (size_t ii = 0; ii < host_weights.size(); ii++) {
        host_float_weights[ii] = static_cast<float>(host_weights[ii]);
    }
    cudaMalloc(&weights, host_weights.size() * sizeof(float));
    cudaMemcpy( weights, host_float_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&flat_nbor, host_flat_nbor.size() * sizeof(int));
    cudaMemcpy( flat_nbor, host_flat_nbor.data(), host_flat_nbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&flat_nbor_idx, host_flat_nbor_idx.size() * sizeof(int));
    cudaMemcpy( flat_nbor_idx, host_flat_nbor_idx.data(), host_flat_nbor_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate device memory for target SRO
    cudaMalloc(&target_sro, host_flat_target_sro.size() * sizeof(float));
    cudaMemcpy(target_sro, host_flat_target_sro.data(), host_flat_target_sro.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<std::vector<LatticeType>> h_lattices_from_file = read_lattices_from_file("evolution_data.txt", num_atoms);
    if (h_lattices_from_file.empty()) {
        std::cerr << "Error: No valid lattices found in the seed file. Exiting." << std::endl;
        cudaFree(d_sro_group_offsets);
        cudaFree(d_sro_group_sizes);
        cudaFree(d_flat_groups);
        cudaFree(d_group_offsets);
        cudaFree(d_group_sizes);
        cudaFree(d_free_indices);
        cudaFree(d_atom_to_sro_group_map);
        cudaFree(species);
        exit(1);
    }
        // b) Prepare host and device buffers for recalculation.
    const int num_found_lattices = h_lattices_from_file.size();
    std::vector<LatticeType> h_lattices_flat;
    h_lattices_flat.reserve((long long)num_found_lattices * num_atoms);
    for(const auto& lat : h_lattices_from_file) {
        h_lattices_flat.insert(h_lattices_flat.end(), lat.begin(), lat.end());
    }

    //LatticeType* d_temp_lattices;
    //float* d_temp_fitnesses;
    //float* d_temp_coefficients;
    //cudaMalloc(&d_temp_lattices, (long long)num_found_lattices * num_atoms * sizeof(LatticeType));
    //cudaMalloc(&d_temp_fitnesses, num_found_lattices * sizeof(float));
    const int num_coefficients_per_group = num_types * num_types * num_shells;
    const int total_coefficients_per_lattice = host_sro_groups.size() * num_coefficients_per_group;
    //cudaMalloc(&d_temp_coefficients, (long long)num_found_lattices * total_coefficients_per_lattice * sizeof(float));

    // c) Copy the lattices to the GPU.
    //cudaMemcpy(d_temp_lattices, h_lattices_flat.data(), (long long)num_found_lattices * num_atoms * sizeof(LatticeType), cudaMemcpyHostToDevice);
    const int max_batch_size =4096;
    // d) Launch the fitness calculation kernel.
    std::cout << "Recalculating fitness for " << num_found_lattices << " initial configurations..." << std::endl;
    LatticeType* d_batch_lattices;
    float* d_batch_fitnesses;
    float* d_batch_coefficients;

    cudaMalloc(&d_batch_lattices, (size_t)max_batch_size * num_atoms * sizeof(LatticeType));
    cudaMalloc(&d_batch_fitnesses, max_batch_size * sizeof(float));
    cudaMalloc(&d_batch_coefficients, (size_t)max_batch_size * total_coefficients_per_lattice * sizeof(float));
    std::vector<float> h_recalculated_fitnesses(num_found_lattices);

    // 2. Loop through all lattices in batches.
    for (int offset = 0; offset < num_found_lattices; offset += max_batch_size) {
        // Determine the size of the current batch (this handles the last, possibly smaller, batch).
        const int current_batch_size = std::min(max_batch_size, num_found_lattices - offset);
        std::cout << "  -> Processing batch starting at index " << offset 
                  << " (size: " << current_batch_size << ")" << std::endl;

        // Prepare and flatten the lattices for THIS BATCH ONLY.
        std::vector<LatticeType> h_batch_lattices_flat;
        h_batch_lattices_flat.reserve((size_t)current_batch_size * num_atoms);
        for (int i = 0; i < current_batch_size; ++i) {
            // The source index is offset + i
            h_batch_lattices_flat.insert(h_batch_lattices_flat.end(), 
                                         h_lattices_from_file[offset + i].begin(), 
                                         h_lattices_from_file[offset + i].end());
        }

        // 3. Copy the current batch's data to the GPU.
        cudaMemcpy(d_batch_lattices, h_batch_lattices_flat.data(), (size_t)current_batch_size * num_atoms * sizeof(LatticeType), cudaMemcpyHostToDevice);

        // 4. Launch the kernel for the current batch.
        size_t gamma_mem_size = (size_t)host_sro_groups.size() * num_types * num_types * num_shells * sizeof(float);
        size_t counts_mem_size = (size_t)host_sro_groups.size() * num_types * sizeof(int);
        size_t totals_mem_size = (size_t)host_sro_groups.size() * sizeof(int);
        size_t fitness_shared_mem = gamma_mem_size + counts_mem_size + totals_mem_size;

        calculate_fitness_of_lattices<<<current_batch_size, THREADS_PER_BLOCK, fitness_shared_mem>>>(
            num_atoms, num_types, num_shells,
            flat_nbor, flat_nbor_idx,
            species, weights, target_sro, 
            d_batch_coefficients,    // Use batch-sized coefficient buffer
            d_batch_fitnesses,       // Use batch-sized fitness buffer
            d_batch_lattices,        // Use batch-sized lattice buffer
            d_atom_to_sro_group_map, 
            host_sro_groups.size());
        // Wait for the kernel to finish THIS BATCH before proceeding.
        cudaDeviceSynchronize();

        // 5. Copy the batch's results from the GPU back to the correct slice of the host vector.
        cudaMemcpy(h_recalculated_fitnesses.data() + offset, // <-- Use pointer arithmetic to write to the correct spot
                   d_batch_fitnesses, 
                   current_batch_size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
    }

    // 6. Free the temporary batch buffers on the GPU now that we're done.
    cudaFree(d_batch_lattices);
    cudaFree(d_batch_fitnesses);
    cudaFree(d_batch_coefficients);
    
    std::cout << "--- Printing all values from h_recalculated_fitnesses ---" << std::endl;
    for (size_t i = 0; i < h_recalculated_fitnesses.size(); ++i) {
        std::cout << "Index " << i << ": " << h_recalculated_fitnesses[i] << std::endl;
    }
    std::cout << "---------------------------------------------------------" << std::endl;
    
    // f) Combine lattices and their new fitnesses, then sort.
    std::vector<LatticeEntry> all_seed_lattices;
    all_seed_lattices.reserve(num_found_lattices);
    for (int i = 0; i < num_found_lattices; ++i) {
        all_seed_lattices.push_back({h_recalculated_fitnesses[i], h_lattices_from_file[i]});
    }
    std::sort(all_seed_lattices.begin(), all_seed_lattices.end());
    std::cout << "Initial configurations have been read, recalculated, and sorted." << std::endl;
    std::cout << "--- Sorted Fitnesses of Initial Lattices ---" << std::endl;
    for (const auto& entry : all_seed_lattices) {
        std::cout << "Fitness: " << entry.fitness << std::endl;
    }
    std::cout << "--------------------------------------------" << std::endl;
    // Free the temporary buffers used for recalculation
    //cudaFree(d_temp_lattices);
    //cudaFree(d_temp_fitnesses);
    //cudaFree(d_temp_coefficients);
    const int max_blocks_per_gen = num_lattices_to_read;
    const int threads_per_block = num_tasks;
    const int max_total_threads = max_blocks_per_gen * threads_per_block;
    LatticeType* d_thread_workspaces;
    float* d_result_fitness_buffer;
    LatticeType* d_result_lattices_buffer;
    int* d_result_count;
    LatticeType* d_initial_lattices;
    float* d_initial_fitnesses;
    cudaMalloc(&d_thread_workspaces, (long long)max_total_threads * num_atoms * sizeof(LatticeType));
    cudaMalloc(&d_result_fitness_buffer, max_total_threads * sizeof(float));
    cudaMalloc(&d_result_lattices_buffer, (long long)max_total_threads * num_atoms * sizeof(LatticeType));
    cudaMalloc(&d_result_count, sizeof(int));
    cudaMalloc(&d_initial_lattices, (long long)max_blocks_per_gen * num_atoms * sizeof(LatticeType));
    cudaMalloc(&d_initial_fitnesses, max_blocks_per_gen * sizeof(float));
    std::unordered_set<std::string> globally_seen_lattices;
    bool first_write_to_file = true;
    for (int gen = 0; gen < g_generations; ++gen) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "--- Starting Generation " << gen + 1 << " / " << g_generations << " ---" << std::endl;
        std::cout << "============================================================" << std::endl;

        // a Determine the batch of lattices for this generation
        int start_index = gen * num_lattices_to_read;
        if (start_index >= all_seed_lattices.size()) {
            std::cout << "All available seed lattices have been processed. Stopping early." << std::endl;
            break; // Stop if we've run out of seeds
        }
        int end_index = std::min(start_index + num_lattices_to_read, (int)all_seed_lattices.size());
        int num_blocks_for_this_gen = end_index - start_index;

        if (num_blocks_for_this_gen <= 0) continue;

        std::cout << "Processing lattices from index " << start_index << " to " << end_index - 1 << "." << std::endl;
        std::vector<LatticeType> h_initial_lattices_flat;
        std::vector<float> h_initial_fitnesses;
        h_initial_lattices_flat.reserve((long long)num_blocks_for_this_gen * num_atoms);
        h_initial_fitnesses.reserve(num_blocks_for_this_gen);
        for (int i = start_index; i < end_index; ++i) {
            h_initial_lattices_flat.insert(h_initial_lattices_flat.end(),
                                           all_seed_lattices[i].data.begin(),
                                           all_seed_lattices[i].data.end());
            h_initial_fitnesses.push_back(all_seed_lattices[i].fitness);
        }
        std::cout << "--- Printing Initial Fitnesses for Generation ---" << std::endl;
        for (size_t i = 0; i < h_initial_fitnesses.size(); ++i) {
            std::cout << "Lattice " << start_index + i << " -> Fitness: " << h_initial_fitnesses[i] << std::endl;
        }
        std::cout << "-----------------------------------------------" << std::endl;
        cudaMemcpy(d_initial_lattices, h_initial_lattices_flat.data(), (long long)num_blocks_for_this_gen * num_atoms * sizeof(LatticeType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_initial_fitnesses, h_initial_fitnesses.data(), num_blocks_for_this_gen * sizeof(float), cudaMemcpyHostToDevice);
        double current_temp = initial_temp;
        for (int step = 0; step < annealing_steps; ++step) {
            std::cout << "  Annealing Step " << step + 1 << "/" << annealing_steps
                      << ", Temperature: " << std::fixed << std::setprecision(5) << current_temp << std::endl;

            cudaMemset(d_result_count, 0, sizeof(int));
            launch_independent_mc(sum,num_blocks_for_this_gen, threads_per_block,(float)fitness_threshold,search_depth,num_atoms,num_types, num_shells,species,weights,flat_nbor,flat_nbor_idx,d_flat_groups, d_group_offsets, d_group_sizes, host_swap_groups.size(),d_free_indices,host_free_indices.size(),d_atom_to_sro_group_map,host_sro_groups.size(), target_sro,d_initial_lattices, d_initial_fitnesses, (float)current_temp,d_thread_workspaces, d_result_fitness_buffer, d_result_lattices_buffer,d_result_count
);
            CUDA_CHECK(cudaGetLastError()); // 检查核函数启动是否成功提交
            CUDA_CHECK(cudaDeviceSynchronize()); // 等待核函数完成并检查运行时错误
            cudaDeviceSynchronize();
            int host_result_count = 0;
            cudaMemcpy(&host_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_result_count > 0) {
                std::vector<float> h_result_fitnesses(host_result_count);
                std::vector<LatticeType> h_result_lattices((long long)host_result_count * num_atoms);
                cudaMemcpy(h_result_fitnesses.data(), d_result_fitness_buffer, host_result_count * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_result_lattices.data(), d_result_lattices_buffer, (long long)host_result_count * num_atoms * sizeof(LatticeType), cudaMemcpyDeviceToHost);
                save_results_to_file("mc_independent_results.txt", host_result_count, num_atoms, h_result_fitnesses, h_result_lattices, !first_write_to_file,globally_seen_lattices);
                first_write_to_file = false;
            } else {
                 std::cout << "    -> No new configurations found below threshold at this temperature." << std::endl;
            }
            current_temp *= cooling_rate;
            }
        }
    cudaFree(species);
    cudaFree(weights);
    cudaFree(flat_nbor);
    cudaFree(flat_nbor_idx);
    cudaFree(target_sro);
    cudaFree(d_flat_groups);
    cudaFree(d_group_offsets);
    cudaFree(d_group_sizes);
    cudaFree(d_free_indices);
    cudaFree(d_flat_sro_groups);
    cudaFree(d_sro_group_offsets);
    cudaFree(d_sro_group_sizes);
    cudaFree(d_atom_to_sro_group_map);
    cudaFree(d_thread_workspaces);
    cudaFree(d_result_fitness_buffer);
    cudaFree(d_result_lattices_buffer);
    cudaFree(d_result_count);
    cudaFree(d_initial_lattices);
    cudaFree(d_initial_fitnesses);
}
    
} // namespace cpu
} // namespace accelerate
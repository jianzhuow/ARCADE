#ifndef ACCELERATE_H
#define ACCELERATE_H

#include <vector>
#include <iomanip>

namespace accelerate {
namespace cpu {
/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm on CPU.
 *
 * @param num_lattices Number of networks or lattices to generate
 * @param step Maximum number of iterations
 * @param task Task identifier for Monte Carlo method
 * @param depth Depth parameter for Monte Carlo method
 * @param threshold Threshold value for fitness score
 * @param neighbor_list Neighborhood relationships (3D vector)
 * @param species Elements or nodes in lattices
 * @param weight Weights associated with elements
 * @return std::tuple<std::vector<std::vector<int>>, std::vector<double>> 
 *         Optimized lattices and their fitness scores
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> run_local_parallel_hcs(
    const int num_lattices,
    const int step,
    const int task,
    const int depth,
    const double threshold,
    const std::vector<std::vector<std::vector<int>>>& neighbor_list,
    const std::vector<int>& species,
    const std::vector<double>& weight,
    const std::vector<std::vector<std::vector<double>>>& target_sro,const std::vector<std::vector<int>>& host_swap_groups);
} // namespace cpu

#ifdef USE_CUDA
namespace gpu {
/**
 * @brief Executes a local parallel heuristic search (HCS) algorithm on GPU.
 *
 * @param num_solutions Number of networks or solutions to generate
 * @param step Maximum number of iterations
 * @param task Task identifier for Monte Carlo method
 * @param depth Depth parameter for Monte Carlo method
 * @param threshold Threshold value for fitness score
 * @param neighbor_list Neighborhood relationships (3D vector)
 * @param species Elements or nodes in solutions
 * @param weight Weights associated with elements
 * @return std::tuple<std::vector<std::vector<int>>, std::vector<double>>
 *         Optimized solutions and their fitness scores
 */
std::tuple<std::vector<std::vector<int>>, std::vector<double>> run_local_parallel_hcs_cuda(
    const int num_solutions,
    const int step,
    const int k,
    const int task,
    const int depth,
    const double threshold,
    const std::vector<std::vector<std::vector<int>>>& neighbor_list,
    const std::vector<int>& species,
    const std::vector<double>& weight,
    const std::vector<std::vector<std::vector<double>>>& target_sro,const std::vector<std::vector<int>>& host_swap_groups,const std::vector<std::vector<int>>& host_sro_groups);
    bool cuda_available();
}
/**
 * @brief Executes a massively parallel Monte Carlo simulation using an independent agent model.
 *
 * This function processes batches of seed lattices over several generations. For each seed,
 * it launches thousands of independent Monte Carlo threads that perform a simulated annealing
 * search. Results with a fitness below the threshold are saved directly to a file.
 *
 * @param num_lattices_to_read The number of seed lattices to process per generation (batch size).
 * @param g_generations The total number of generations (batches) to run.
 * @param num_tasks The number of parallel threads to launch per seed lattice (threads per block).
 * @param search_depth The search patience; number of unsuccessful attempts before a thread stops.
 * @param fitness_threshold The fitness score below which a configuration is saved.
 * @param initial_temp The starting temperature for the simulated annealing process.
 * @param cooling_rate The multiplicative rate at which the temperature decreases (e.g., 0.99).
 * @param annealing_steps The number of temperature steps in the annealing schedule.
 * @param neighbor_list Neighborhood data for all atoms.
 * @param host_species A vector containing the count of each atom type.
 * @param host_weights A vector of weights for each neighbor shell.
 * @param host_target_sro The target SRO parameters for the simulation.
 * @param host_swap_groups A list of atom groups where swapping is constrained to occur within a group.
 * @param host_sro_groups A list of atom groups for which SRO is calculated independently.
 */
namespace gpu2{
void run_mc_cuda(
        const int num_lattices_to_read,
        const int g_generations,
        const int num_tasks,
        const int search_depth,
        const double fitness_threshold,
        const double initial_temp,
        const double cooling_rate,
        const int annealing_steps,
        const std::vector<std::vector<std::vector<int>>>& neighbor_list,
        const std::vector<int>& host_species,
        const std::vector<double>& host_weights,
        const std::vector<std::vector<std::vector<double>>>& host_target_sro,
        const std::vector<std::vector<int>>& host_swap_groups,
        const std::vector<std::vector<int>>& host_sro_groups);

/**
 * @brief Check if CUDA is available at runtime
 * @return bool True if CUDA is available
 */
bool cuda_available();
} // namespace gpu
#endif // USE_CUDA

} // namespace accelerate

#endif // ACCELERATE_H
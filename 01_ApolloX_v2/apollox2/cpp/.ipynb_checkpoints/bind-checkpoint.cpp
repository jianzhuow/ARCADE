#include <pybind11/stl.h>     // for binding STL containers
#include <pybind11/pybind11.h>

#include "accelerate.h"     // Include the header file where `calc_fits` is declared

PYBIND11_MODULE(accelerate, m) {
    m.def("run_local_parallel_hcs", &accelerate::cpu::run_local_parallel_hcs, 
        pybind11::arg("nnet"),
        pybind11::arg("step"),
        pybind11::arg("task"),
        pybind11::arg("depth"),
        pybind11::arg("thr"),
        pybind11::arg("nbor"),
        pybind11::arg("element"),
        pybind11::arg("weight"),
        pybind11::arg("target_sro"),
        pybind11::arg("swap_groups"),
        R"pbdoc(
            Run the local parallel hybrid cuckoo search algorithm.

            Parameters
            ----------
            nnet : int
                The number of nests to generate.
            step : int
                The total number of atoms in each nest.
            task : int
                A vector specifying the maximum number of each atom type allowed.
            depth : int
                The total number of atoms to be distributed in each nest.
            thr : float
                The threshold for fitness comparison.
            nbor : list of list of int
                A list of lists representing the neighbors of each atom for each shell.
            element : list of int
                A list indicating the number of each type of atom.
            weight : list of float
                A list of weights used to calculate the weighted error.
            target_sro : list of float
                A list of target SRO values for each shell.

            Returns
            -------
            tuple
                A tuple containing the lattice structures and their fitness values.
        )pbdoc"
    );

#ifdef USE_CUDA
    m.def("run_local_parallel_hcs_cuda", &accelerate::gpu::run_local_parallel_hcs_cuda, 
        pybind11::arg("nnet"),
        pybind11::arg("step"),
        pybind11::arg("k"),
        pybind11::arg("task"),
        pybind11::arg("depth"),
        pybind11::arg("thr"),
        pybind11::arg("nbor"),
        pybind11::arg("element"),
        pybind11::arg("weight"),
        pybind11::arg("target_sro"),
        pybind11::arg("swap_groups"),
        pybind11::arg("sro_groups"),
        R"pbdoc(
            Run the local parallel hybrid cuckoo search algorithm.

            Parameters
            ----------
            nnet : int
                The number of nests to generate.
            step : int
                The total number of atoms in each nest.
            task : int
                A vector specifying the maximum number of each atom type allowed.
            depth : int
                The total number of atoms to be distributed in each nest.
            thr : float
                The threshold for fitness comparison.
            nbor : list of list of int
                A list of lists representing the neighbors of each atom for each shell.
            element : list of int
                A list indicating the number of each type of atom.
            weight : list of float
                A list of weights used to calculate the weighted error.
            target_sro : list of float
                A list of target SRO values for each shell.

            Returns
            -------
            tuple
                A tuple containing the lattice structures and their fitness values.
        )pbdoc"
    );
    m.def("run_mc_cuda", &accelerate::gpu::run_mc_cuda,
        "Runs the advanced, independent agent Monte Carlo simulation on the GPU with annealing.",
        pybind11::arg("num_lattices_to_read"),
        pybind11::arg("g_generations"),
        pybind11::arg("num_tasks"),
        pybind11::arg("search_depth"),
        pybind11::arg("fitness_threshold"),
        pybind11::arg("initial_temp"),
        pybind11::arg("cooling_rate"),
        pybind11::arg("annealing_steps"),
        pybind11::arg("neighbor_list"),
        pybind11::arg("host_species"),
        pybind11::arg("host_weights"),
        pybind11::arg("host_target_sro"),
        pybind11::arg("host_swap_groups"),
        pybind11::arg("host_sro_groups")
    );

    m.def("cuda_available", &accelerate::gpu::cuda_available, 
        R"pbdoc(
            Run the local parallel hybrid cuckoo search algorithm.

            Parameters
            ----------
            nnet : int
                The number of nests to generate.
            step : int
                The total number of atoms in each nest.
            task : int
                A vector specifying the maximum number of each atom type allowed.
            depth : int
                The total number of atoms to be distributed in each nest.
            thr : float
                The threshold for fitness comparison.
            nbor : list of list of int
                A list of lists representing the neighbors of each atom for each shell.
            element : list of int
                A list indicating the number of each type of atom.
            weight : list of float
                A list of weights used to calculate the weighted error.
            comm : MPI_Comm
                The MPI communicator.

            Returns
            -------
            tuple
                A tuple containing the lattice structures and their fitness values.
        )pbdoc"
    );
#endif // USE_CUDA
}

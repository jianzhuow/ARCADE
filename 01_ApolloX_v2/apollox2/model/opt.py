from apollox2.utils.logger import logger
from apollox2.version import __version__  # Import version information
from apollox2.io.output import write_structure
try:
    from apollox2.cpp import accelerate as acc
except ImportError:
    logger.warning("Failed to import accelerate module: No module named 'apollox2.cpp.accelerate'\nThis might happen if the C++ extension was not properly built.")
    acc = None

def file_final_results(nest, latt, ntyp, elem, file,element_type,output_format='vasp/poscar'):
    """Write the final results to the specified output format.
    
    @param nest list List of atom types
    @param latt Lattice Lattice object containing vectors and coordinates
    @param ntyp int Number of atom types
    @param elem list List of elements 
    @param file str Output file name
    @param output_format str Desired output format (default: 'vasp/poscar')
                           Supported formats: vasp/poscar, lammps/lmp
    @return None
    """
    write_structure(nest, latt, ntyp, elem, file,element_type, output_format)

class opt_model:
    def __init__(self, latt, conf, comm):
        """Initialize the optimization model.
        
        @param latt Lattice Lattice object containing structure information
        @param conf Config Configuration object containing simulation parameters
        @param comm Comm Communication object for parallel processing
        @return None
        """
        self.latt = latt
        self.comm = comm
        self.conf = conf  # Store conf for access to output format

        self.thr  = 0.001
        self.ntyp = conf.type
        self.nnet = conf.solutions
        self.nbor = latt.nbor_list
        self.step = conf.total_iter
        self.task = conf.parallel_task
        
        self.depth   = conf.converge_depth
        self.device  = conf.device
        self.element = conf.element
        self.target_sro = conf.target_sro
        self.element_type= conf.element_type
        self.swap_group=conf.swap_group
        self.sro_group=conf.sro_group
        self.k=conf.cut_iter
        if (len(conf.weight) > latt.shells):
            logger.error("The number of weight is larger than the maximum number(shell) of neighbors.")
            exit()
        self.weight = conf.weight
        self.mc_num_lattice_per_batch=conf.mc_num_lattice_per_batch
        self.mc_num_tasks=conf.mc_num_tasks
        self.mc_search_depth=conf.mc_search_depth
        self.mc_batch_num=conf.mc_batch_num
        self.mc_fitness_threshold=conf.mc_fitness_threshold
        self.mc_initial_temperature=conf.mc_initial_temperature
        self.mc_cooling_rate=conf.mc_cooling_rate
        self.mc_annealing_steps=conf.mc_annealing_steps
    def run_optimization(self):
        """Run the Local Parallel HCS optimization algorithm.
        
        Executes the High-throughput Computing Screening (HCS) optimization
        algorithm in parallel to find the optimal configuration for the high entropy alloy.
        
        @return tuple (list, list) Returns optimized lattice structures and their fitness values
        """
        logger.info("Running Local Parallel HCS optimization...")
        
        if (self.device == "gpu"):
            if (self.comm.Get_size() > 1):
                logger.error("Cannot run on multiple GPUs. Please run on a single GPU.")
                exit()
            if acc is not None and acc.cuda_available() == True:
                # Run the Local Parallel HCS algorithm
                print("start!")
                latts, fitss = acc.run_local_parallel_hcs_cuda(
                    self.nnet, self.step,self.k,self.task, self.depth, self.thr, self.nbor, self.element, self.weight, self.target_sro,self.swap_group,self.sro_group)
            else:
                logger.error("CUDA is not available on this machine or accelerate module is not available. Please ensure the package is properly installed and CUDA support is available.")
                exit()
        else:
            logger.error("Invalid device type. Please specify 'gpu'.")
            exit()

        # Write the final lattice structure to a file
        if (self.comm.Get_rank() == 0):
            output_format = getattr(self.conf, 'output_format', 'vasp/poscar')
            output_name = getattr(self.conf, 'output_name', 'output')
            
            # Determine output filename based on format
            if output_format == 'lammps/lmp':
                output_file = f"{output_name}.lmp"
            else:  # vasp/poscar
                output_file = f"{output_name}.vasp"

            file_final_results(latts[0], self.latt, self.ntyp, self.element, output_file,self.element_type,output_format=output_format)
            logger.info(f"Final lattice structure saved to file {output_file}")
        
        # Return the optimized latts and fitness values
        return latts, fitss
    def run_mc(self):
        """Run the Local Parallel HCS optimization algorithm.
        
        Executes the High-throughput Computing Screening (HCS) optimization
        algorithm in parallel to find the optimal configuration for the high entropy alloy.
        
        @return tuple (list, list) Returns optimized lattice structures and their fitness values
        """
        logger.info("Running MC")
        
    
        if (self.device == "gpu"):
            if (self.comm.Get_size() > 1):
                logger.error("GPU mode supports only a single MPI process.")
                exit()
            if acc is not None and acc.cuda_available() == True:
                print("start!")
                acc.run_mc_cuda(self.mc_num_lattice_per_batch,self.mc_batch_num,self.mc_num_tasks,self.mc_search_depth,self.mc_fitness_threshold,self.mc_initial_temperature,self.mc_cooling_rate,self.mc_annealing_steps,self.nbor,self.element,self.weight, self.target_sro,self.swap_group,self.sro_group)
            else:
                logger.error("CUDA is not available on this machine or accelerate module is not available. Please ensure the package is properly installed and CUDA support is available.")
                exit()
        else:
            logger.error("Invalid device type.")
            exit()

        
        # Return the optimized latts and fitness values
        return
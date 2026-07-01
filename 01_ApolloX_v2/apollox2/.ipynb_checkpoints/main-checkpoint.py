import os
import time

from apollox2.io import input_config
from apollox2.io.params import parse_args

from apollox2.comm import comm

from apollox2.utils import logger
from apollox2.utils.analyze import analyze_structure, analyze_result
from apollox2.model import opt_model
from apollox2.lattice import lattice

def main():
    """Main function to run the PyHEA lattice simulation."""
    # Parse command line arguments
    args = parse_args()
    
    # Print welcome message
    logger.info("ApolloX2: A high-performance implementation for constructing fixed-site lattice models.")

    # Handle different commands       
    if args.command == 'global':
        logger.info(f"Running ApolloX2 with the configuration file: {args.config_file}")
        # Use the parsed argument
        config = input_config(args.config_file)
        logger.info(f"Configuration loaded successfully.")
        logger.info(f"{'Element types:':<30} {config.element}")
        logger.info(f"{'Sro shell weights:':<30} {config.weight}")
        logger.info(f"{'Number of solutions:':<30} {config.solutions}")
        logger.info(f"{'Number of shells:':<30} {config.max_shell_num}")
        logger.info(f"{'Total iterations:':<30} {config.total_iter}")
        logger.info(f"{'Convergence depth:':<30} {config.converge_depth}")
        logger.info(f"{'Parallel Monte Carlo tasks:':<30} {config.parallel_task}")
        logger.info(f"{'Running with processes:':<30} {comm.Get_size()}")
        # logger.info(f"{'Target SRO:'} {config.target_sro.tolist()}")
        # logger.info(f"{'Lattice structure:'} {config.structure}\n\n")

        # Initialize and run the simulation
        lattice_instance = lattice(
            config.latt_const,
            config.species,
            config.position,
            config.latt_vectors,
            config.cutoff)
        
        start = time.time()
        model = opt_model(lattice_instance, config, comm)
        solutions, fitness = model.run_optimization()
        logger.info(f"Total time taken: {time.time() - start} seconds.\n\n")

        #Analyze SRO parameters and compare with target values
        logger.info("Post-processing: Analyzing SRO parameters...")
        result_sro, mae, rmse = analyze_result(
            f'{config.output_name}.{config.output_format}',
            config.target_sro,
            config.element_type,
            config.cutoff
        )
    elif args.command == 'detail':
        config = input_config(args.config_file)
        lattice_instance = lattice(
            config.latt_const,
            config.species,
            config.position,
            config.latt_vectors,
            config.cutoff)
        start = time.time()
        model = opt_model(lattice_instance, config, comm)
        model.run_mc()
        logger.info(f"Total time taken: {time.time() - start} seconds.\n\n")

    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()

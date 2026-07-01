"""
Command line argument parsing for ApolloX2.
"""

import argparse
from apollox2.version import __version__

def create_parser():
    """Create and return the argument parser for ApolloX2.
    
    @return argparse.ArgumentParser The configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="ApolloX2: A high-performance implementation for constructing fixed-site lattice models."
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add the main simulation command
    sim_parser = subparsers.add_parser('global', help='Step1: global search')
    sim_parser.add_argument('config_file', type=str, help='Path to the configuration YAML file.')
    
    # Add the analyze command
    analyze_parser = subparsers.add_parser('detail', help='Step2: detail search')
    analyze_parser.add_argument('config_file', type=str, help='Path to the configuration YAML file.')
    
    
    # Add version option to main parser
    parser.add_argument('--version', action='version', version=f'ApolloX {__version__}')
    
    return parser

def parse_args():
    """Parse command line arguments.
    
    @return argparse.Namespace The parsed command line arguments
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        exit(1)
        
    return args

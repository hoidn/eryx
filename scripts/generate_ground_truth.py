#!/usr/bin/env python3
"""
Generate ground truth data for testing PyTorch implementation.

This script instruments NumPy functions to capture inputs and outputs,
which can then be used to validate the PyTorch implementation.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Callable, Optional

# Add parent directory to path to import eryx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.autotest.debug import Debug
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

def instrument_function(module_name: str, function_name: str) -> Callable:
    """
    Instrument a function to capture inputs and outputs.
    
    Args:
        module_name: Name of the module containing the function
        function_name: Name of the function to instrument
        
    Returns:
        Instrumented function
    """
    logger.info(f"Instrumenting {module_name}.{function_name}")
    
    # Import the module and get the function
    module = __import__(module_name, fromlist=[function_name])
    func = getattr(module, function_name)
    
    # Instrument the function
    debug = Debug().decorate
    instrumented_func = debug(func)
    
    # Replace the original function with the instrumented one
    setattr(module, function_name, instrumented_func)
    
    return instrumented_func

def instrument_map_utils():
    """
    Instrument functions in eryx.map_utils.
    """
    functions = [
        "generate_grid",
        "get_symmetry_equivalents",
        "get_ravel_indices",
        "compute_resolution",
        "get_resolution_mask",
        "get_dq_map",
        "get_centered_sampling",
        "resize_map"
    ]
    
    for func_name in functions:
        instrument_function("eryx.map_utils", func_name)

def instrument_scatter():
    """
    Instrument functions in eryx.scatter.
    """
    functions = [
        "compute_form_factors",
        "structure_factors_batch",
        "structure_factors"
    ]
    
    for func_name in functions:
        instrument_function("eryx.scatter", func_name)

def instrument_models():
    """
    Instrument methods in eryx.models.
    """
    # Import the module
    import eryx.models
    
    # Instrument OnePhonon methods
    debug = Debug().decorate
    eryx.models.OnePhonon.apply_disorder = debug(eryx.models.OnePhonon.apply_disorder)
    eryx.models.OnePhonon.compute_hessian = debug(eryx.models.OnePhonon.compute_hessian)
    eryx.models.OnePhonon.compute_covariance_matrix = debug(eryx.models.OnePhonon.compute_covariance_matrix)
    
    # Instrument other model classes as needed
    eryx.models.RigidBodyTranslations.apply_disorder = debug(eryx.models.RigidBodyTranslations.apply_disorder)
    eryx.models.LiquidLikeMotions.apply_disorder = debug(eryx.models.LiquidLikeMotions.apply_disorder)

def generate_onephonon_data(pdb_path: str, output_dir: str):
    """
    Generate ground truth data for OnePhonon model.
    
    Args:
        pdb_path: Path to PDB file
        output_dir: Directory to save ground truth data
    """
    logger.info(f"Generating OnePhonon data for {pdb_path}")
    
    # Import the model
    from eryx.models import OnePhonon
    
    # Create model instance
    model = OnePhonon(
        pdb_path,
        [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    
    # Apply disorder to generate data
    Id = model.apply_disorder(use_data_adp=True)
    
    # Save the result
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "onephonon_intensity.npy"), Id)
    
    logger.info(f"Saved OnePhonon data to {output_dir}")

def generate_structure_factor_data(output_dir: str):
    """
    Generate ground truth data for structure factor calculations.
    
    Args:
        output_dir: Directory to save ground truth data
    """
    logger.info("Generating structure factor data")
    
    # Import necessary functions
    import numpy as np
    from eryx.scatter import compute_form_factors, structure_factors
    
    # Create test data
    n_points = 100
    n_atoms = 10
    
    # Generate random q-vectors
    q_grid = np.random.rand(n_points, 3) * 10
    
    # Generate random atomic positions
    xyz = np.random.rand(n_atoms, 3) * 20
    
    # Generate random form factor coefficients
    ff_a = np.random.rand(n_atoms, 4)
    ff_b = np.random.rand(n_atoms, 4)
    ff_c = np.random.rand(n_atoms)
    
    # Generate random displacement parameters
    U = np.random.rand(n_atoms) * 0.1
    
    # Compute form factors
    ff = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    
    # Compute structure factors
    sf = structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U)
    
    # Save the inputs and outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "q_grid.npy"), q_grid)
    np.save(os.path.join(output_dir, "xyz.npy"), xyz)
    np.save(os.path.join(output_dir, "ff_a.npy"), ff_a)
    np.save(os.path.join(output_dir, "ff_b.npy"), ff_b)
    np.save(os.path.join(output_dir, "ff_c.npy"), ff_c)
    np.save(os.path.join(output_dir, "U.npy"), U)
    np.save(os.path.join(output_dir, "form_factors.npy"), ff)
    np.save(os.path.join(output_dir, "structure_factors.npy"), sf)
    
    logger.info(f"Saved structure factor data to {output_dir}")

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Generate ground truth data for testing")
    parser.add_argument("--pdb-path", default="tests/pdbs/5zck_p1.pdb", help="Path to PDB file")
    parser.add_argument("--output-dir", default="tests/ground_truth", help="Directory to save ground truth data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Instrument functions
    logger.info("Instrumenting functions")
    instrument_map_utils()
    instrument_scatter()
    instrument_models()
    
    # Generate ground truth data
    generate_onephonon_data(args.pdb_path, args.output_dir)
    generate_structure_factor_data(args.output_dir)
    
    logger.info("Ground truth data generation complete")

if __name__ == "__main__":
    main()

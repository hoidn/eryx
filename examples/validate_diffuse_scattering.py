#!/usr/bin/env python3
"""
Minimal example script for validating diffuse scattering data consistency.

This script demonstrates how to safely load diffuse scattering data
from different simulation modes and create validation visualizations.
"""

import os
import sys
import logging

# Add parent directory to path for importing eryx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization module
from eryx.visualization import load_simulation_data, visualize_basic_scatter

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Validate diffuse scattering data consistency')
    parser.add_argument('--input-dir', default=".", help='Directory containing simulation outputs')
    parser.add_argument('--output-dir', default="validation_output", help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Log the sampling parameter issue warning
    logger.warning("IMPORTANT: Be aware of sampling parameter interpretation issues!")
    logger.warning("Different simulation methods may use different interpretations:")
    logger.warning("1. Direct: h_dim = int(hsampling[2])")
    logger.warning("2. Formula: hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)")
    logger.warning("This can lead to mismatched dimensions between q-vectors and intensities")
    
    try:
        # Load data with consistency checks
        logger.info(f"Loading simulation data from {args.input_dir}")
        data_dict = load_simulation_data(args.input_dir)
        
        # Create basic validation visualizations
        logger.info(f"Creating validation visualizations in {args.output_dir}")
        visualize_basic_scatter(data_dict, args.output_dir)
        
        logger.info("Validation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

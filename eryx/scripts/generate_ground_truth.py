#!/usr/bin/env python3
"""
Ground truth data generation script for PyTorch port.

This script runs the NumPy implementation with different parameter sets
to generate ground truth data for testing the PyTorch implementation.
"""

import os
import sys
import logging
import time
import numpy as np

# Set DEBUG_MODE environment variable if not already set
if os.environ.get("DEBUG_MODE") != "1":
    os.environ["DEBUG_MODE"] = "1"
    print("Set DEBUG_MODE=1 environment variable")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("ground_truth_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import after setting DEBUG_MODE
import sys
import os
# Add the project root to the path so we can import run_debug.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from run_debug import run_np
from eryx.autotest_config import config

def main():
    """
    Main function to generate ground truth data.
    """
    start_time = time.time()
    logging.info("Starting ground truth data generation")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Ensure log directory exists
    log_dir = config.getLogFilePrefix()
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Log files will be saved to: {log_dir}")
    
    # Run with different parameter sets
    variants = ["small", "medium", "default"]
    
    for variant in variants:
        variant_start = time.time()
        logging.info(f"Running with {variant} parameters")
        try:
            run_np(variant)
            logging.info(f"Completed {variant} run in {time.time() - variant_start:.2f} seconds")
        except Exception as e:
            logging.error(f"Error in {variant} run: {e}")
    
    # Verify log files were created
    log_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(log_files)} log files")
    
    total_time = time.time() - start_time
    logging.info(f"Ground truth data generation completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

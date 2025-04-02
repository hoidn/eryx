#!/usr/bin/env python3
"""
Debug script to compare grid-based and explicit q-vector modes using a minimal dataset.

This script:
1. Creates a minimal set of q-vectors
2. Runs both modes with identical input vectors
3. Compares results at each step of calculation
4. Helps identify where the calculation paths diverge
"""

import os
import sys
import logging
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def setup_minimal_test():
    """
    Create a minimal test case with just a few q-vectors.
    """
    from eryx.models_torch import OnePhonon
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    device = torch.device('cpu')
    
    # Create a tiny grid for better debugging (3x3x3 = 27 points)
    h_dim, k_dim, l_dim = 3, 3, 3
    
    # First create a grid-based model with minimal grid
    logger.info("Creating grid-based model with tiny grid...")
    grid_model = OnePhonon(
        pdb_path=pdb_path,
        hsampling=[-1, 1, h_dim],
        ksampling=[-1, 1, k_dim],
        lsampling=[-1, 1, l_dim],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    # Extract q-vectors directly from hkl grid
    from eryx.pdb import AtomicModel
    model = AtomicModel(pdb_path, expand_p1=True)
    
    # Create q-grid directly using map_utils
    from eryx.map_utils import generate_grid
    logger.info(f"Generating q-vectors directly from hkl grid")
    hsampling = [-1, 1, h_dim]
    ksampling = [-1, 1, k_dim]
    lsampling = [-1, 1, l_dim]
    
    hkl_grid, map_shape = generate_grid(model.A_inv, 
                                      hsampling,
                                      ksampling,
                                      lsampling,
                                      return_hkl=True)
    
    # Convert to tensor
    hkl_grid_tensor = torch.tensor(hkl_grid, dtype=torch.float32, device=device)
    
    # Compute q-grid directly: q_grid = 2π * A_inv^T * hkl_grid^T
    A_inv_tensor = torch.tensor(model.A_inv, dtype=torch.float32, device=device)
    q_vectors = 2 * torch.pi * torch.matmul(A_inv_tensor.T, hkl_grid_tensor.T).T
    
    logger.info(f"Extracted {q_vectors.shape[0]} q-vectors directly from hkl grid")
    
    # Create model with explicit q-vectors - use exact same parameters
    logger.info("Creating model with explicit q-vectors...")
    gnm_cutoff = 4.0
    gamma_intra = 1.0
    gamma_inter = 1.0
    
    explicit_model = OnePhonon(
        pdb_path=pdb_path,
        q_vectors=q_vectors,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=gnm_cutoff,
        gamma_intra=gamma_intra,
        gamma_inter=gamma_inter,
        device=device
    )
    
    return grid_model, explicit_model

def compare_models(grid_model, explicit_model):
    """
    Compare key attributes and calculation results between models.
    """
    logger.info("Comparing models...")
    
    # Compare q-grids
    grid_q = grid_model.q_grid.detach().cpu().numpy()
    explicit_q = explicit_model.q_grid.detach().cpu().numpy()
    
    q_diff = np.abs(grid_q - explicit_q)
    logger.info(f"q-grid max difference: {np.max(q_diff):.8e}")
    
    # Compare k-vectors
    grid_k = grid_model.kvec.detach().cpu().numpy()
    explicit_k = explicit_model.kvec.detach().cpu().numpy()
    
    k_diff = np.abs(grid_k - explicit_k)
    logger.info(f"k-vectors max difference: {np.max(k_diff):.8e}")
    
    # Compare eigenvalues (Winv)
    grid_winv = grid_model.Winv.detach().cpu().numpy()
    explicit_winv = explicit_model.Winv.detach().cpu().numpy()
    
    # Compare shapes
    logger.info(f"Winv shapes - Grid: {grid_winv.shape}, Explicit: {explicit_winv.shape}")
    
    # Compare eigenvalues for first few points
    logger.info("Comparing eigenvalues for first 3 points:")
    for i in range(min(3, len(grid_winv))):
        # Get non-NaN values for comparison
        grid_pt = grid_winv[i]
        explicit_pt = explicit_winv[i]
        
        # Filter NaN values
        grid_valid = ~np.isnan(grid_pt)
        explicit_valid = ~np.isnan(explicit_pt)
        
        # Check if indices of NaN values match
        nan_match = np.array_equal(grid_valid, explicit_valid)
        logger.info(f"  Point {i} - NaN patterns match: {nan_match}")
        
        # Compare values where both are valid
        common_valid = grid_valid & explicit_valid
        if np.any(common_valid):
            grid_values = grid_pt[common_valid]
            explicit_values = explicit_pt[common_valid]
            
            # Calculate statistics
            abs_diff = np.abs(grid_values - explicit_values)
            rel_diff = np.mean(abs_diff / np.maximum(np.abs(grid_values), 1e-10)) * 100
            
            logger.info(f"  Point {i} - Mean relative difference: {rel_diff:.4f}%")
            logger.info(f"  First few values - Grid: {grid_values[:5]}")
            logger.info(f"  First few values - Explicit: {explicit_values[:5]}")
        else:
            logger.warning(f"  Point {i} - No common valid values!")

def compare_intensities(grid_model, explicit_model):
    """
    Compare final intensity values from both models.
    """
    logger.info("Comparing final intensity calculations...")
    
    # Calculate intensities
    grid_intensity = grid_model.apply_disorder(use_data_adp=True)
    explicit_intensity = explicit_model.apply_disorder(use_data_adp=True)
    
    # Convert to numpy
    grid_intensity_np = grid_intensity.detach().cpu().numpy()
    explicit_intensity_np = explicit_intensity.detach().cpu().numpy()
    
    # Create masks for valid values
    grid_valid = ~np.isnan(grid_intensity_np)
    explicit_valid = ~np.isnan(explicit_intensity_np)
    
    # Check valid positions
    common_valid = grid_valid & explicit_valid
    logger.info(f"Common valid points: {np.sum(common_valid)}/{len(grid_intensity_np)}")
    
    # Compare values where both are valid
    if np.any(common_valid):
        grid_values = grid_intensity_np[common_valid]
        explicit_values = explicit_intensity_np[common_valid]
        
        # Calculate statistics
        abs_diff = np.abs(grid_values - explicit_values)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        rel_diff = np.mean(abs_diff / np.maximum(np.abs(grid_values), 1e-10)) * 100
        
        # Compute correlation
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(grid_values, explicit_values)
        
        logger.info(f"Intensity statistics:")
        logger.info(f"  Max absolute difference: {max_diff:.6e}")
        logger.info(f"  Mean absolute difference: {mean_diff:.6e}")
        logger.info(f"  Mean relative difference: {rel_diff:.4f}%")
        logger.info(f"  Correlation: {correlation:.6f}")
        
        # Print sample values
        n_samples = min(5, np.sum(common_valid))
        sample_indices = np.where(common_valid)[0][:n_samples]
        
        logger.info("Sample intensity values:")
        for i, idx in enumerate(sample_indices):
            logger.info(f"  Point {i} (index {idx}):")
            logger.info(f"    Grid: {grid_intensity_np.flat[idx]:.6e}")
            logger.info(f"    Explicit: {explicit_intensity_np.flat[idx]:.6e}")
            if grid_intensity_np.flat[idx] != 0:
                ratio = explicit_intensity_np.flat[idx] / grid_intensity_np.flat[idx]
                logger.info(f"    Ratio: {ratio:.6f}")
    else:
        logger.warning("No common valid points to compare!")

def main():
    """Run the debug script."""
    logger.info("Starting minimal test to debug q-vector inconsistency...")
    
    # Setup minimal test
    grid_model, explicit_model = setup_minimal_test()
    
    # Compare models
    compare_models(grid_model, explicit_model)
    
    # Compare intensities
    compare_intensities(grid_model, explicit_model)
    
    logger.info("Debug test complete.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug regression between original and optimized implementations."""

import torch
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

def compare_implementations():
    """Compare key values at each stage."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Small grid to debug
    h_samp = (-0.5, 0.5, 2)  # 3 points, avoids integers
    k_samp = (-0.5, 0.5, 2)  # 3 points
    l_samp = (-0.5, 0.5, 2)  # 3 points
    
    print("Creating models...")
    
    # Original
    model_orig = OnePhonon(
        pdb_path,
        h_samp, k_samp, l_samp,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    # Optimized
    model_opt = OnePhononOptimized(
        pdb_path,
        h_samp, k_samp, l_samp,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    print("\n=== INITIALIZATION CHECK ===")
    print(f"Original atoms: {model_orig.asu_xyz.shape if hasattr(model_orig, 'asu_xyz') else 'N/A'}")
    print(f"Optimized atoms: {model_opt.asu_xyz.shape if hasattr(model_opt, 'asu_xyz') else 'N/A'}")
    
    # Check q_grid
    print("\n=== Q-GRID CHECK ===")
    print(f"Original q_grid shape: {model_orig.q_grid.shape}")
    print(f"Optimized q_grid shape: {model_opt.q_grid.shape}")
    q_diff = torch.abs(model_orig.q_grid - model_opt.q_grid).max().item()
    print(f"Max q_grid difference: {q_diff}")
    
    # Compute phonons
    print("\n=== COMPUTING PHONONS ===")
    model_orig.compute_gnm_phonons()
    model_opt.compute_gnm_phonons()
    
    # Check phonon matrices
    print(f"Original V shape: {model_orig.V.shape if model_orig.V is not None else 'None'}")
    print(f"Optimized V shape: {model_opt.V.shape if model_opt.V is not None else 'None'}")
    
    if model_orig.V is not None and model_opt.V is not None:
        v_diff = torch.abs(model_orig.V - model_opt.V).max().item()
        print(f"Max V difference: {v_diff}")
    
    # Compute covariance
    print("\n=== COMPUTING COVARIANCE ===")
    model_orig.compute_covariance_matrix()
    model_opt.compute_covariance_matrix()
    
    # Check ADP values
    print("\n=== ADP CHECK ===")
    adp_orig = model_orig.calc_adp().detach().cpu()
    adp_opt = model_opt.calc_adp().detach().cpu()
    print(f"Original ADP shape: {adp_orig.shape}")
    print(f"Optimized ADP shape: {adp_opt.shape}")
    print(f"Original ADP range: [{adp_orig.min():.2e}, {adp_orig.max():.2e}]")
    print(f"Optimized ADP range: [{adp_opt.min():.2e}, {adp_opt.max():.2e}]")
    adp_diff = torch.abs(adp_orig - adp_opt).max().item()
    print(f"Max ADP difference: {adp_diff}")
    
    # Apply disorder
    print("\n=== APPLYING DISORDER ===")
    intensity_orig = model_orig.apply_disorder(use_data_adp=True)
    intensity_opt = model_opt.apply_disorder(use_data_adp=True)
    
    print(f"Original intensity shape: {intensity_orig.shape}")
    print(f"Optimized intensity shape: {intensity_opt.shape}")
    
    # Compare valid points
    valid_orig = intensity_orig[~torch.isnan(intensity_orig)]
    valid_opt = intensity_opt[~torch.isnan(intensity_opt)]
    
    print(f"Original valid points: {valid_orig.numel()}/{intensity_orig.numel()}")
    print(f"Optimized valid points: {valid_opt.numel()}/{intensity_opt.numel()}")
    
    if valid_orig.numel() > 0:
        print(f"Original range: [{valid_orig.min().item():.2e}, {valid_orig.max().item():.2e}]")
    
    if valid_opt.numel() > 0:
        print(f"Optimized range: [{valid_opt.min().item():.2e}, {valid_opt.max().item():.2e}]")
    
    # Sample comparison
    print("\n=== SAMPLE VALUES (first 10) ===")
    print(f"Original: {intensity_orig[:10].cpu().numpy()}")
    print(f"Optimized: {intensity_opt[:10].cpu().numpy()}")
    
    # Check for specific issues
    print("\n=== POTENTIAL ISSUES ===")
    if valid_opt.numel() > 0 and valid_orig.numel() > 0:
        scale_factor = valid_opt.mean() / valid_orig.mean()
        print(f"Scale factor (opt/orig): {scale_factor:.2e}")
        
        if scale_factor > 1000:
            print("⚠️ CRITICAL: Optimized values are >1000x larger!")
            print("   Possible causes:")
            print("   - Missing normalization")
            print("   - Incorrect matrix operations")
            print("   - Wrong units/scaling")

if __name__ == "__main__":
    compare_implementations()
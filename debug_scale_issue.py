#!/usr/bin/env python3
"""Debug scale difference between original and optimized implementations."""

import torch
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

def compare_phonon_matrices():
    """Compare V and Winv between implementations."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Small test grid
    h_samp = (-0.5, 0.5, 2)  # 3 points, avoids integers
    k_samp = (-0.5, 0.5, 2)
    l_samp = (-0.5, 0.5, 2)
    
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
    
    print("\n=== COMPUTING PHONONS ===")
    model_orig.compute_gnm_phonons()
    model_orig.compute_covariance_matrix()
    
    model_opt.compute_gnm_phonons()
    model_opt.compute_covariance_matrix()
    
    print("\n=== V MATRIX COMPARISON ===")
    print(f"Original V shape: {model_orig.V.shape if model_orig.V is not None else 'None'}")
    print(f"Optimized V shape: {model_opt.V.shape if model_opt.V is not None else 'None'}")
    
    if model_orig.V is not None and model_opt.V is not None:
        # Compare norms
        v_norm_orig = torch.abs(model_orig.V).mean().item()
        v_norm_opt = torch.abs(model_opt.V).mean().item()
        print(f"Original V mean abs: {v_norm_orig:.2e}")
        print(f"Optimized V mean abs: {v_norm_opt:.2e}")
        print(f"Ratio (opt/orig): {v_norm_opt/v_norm_orig:.2e}")
    
    print("\n=== Winv COMPARISON ===")
    print(f"Original Winv shape: {model_orig.Winv.shape if model_orig.Winv is not None else 'None'}")
    print(f"Optimized Winv shape: {model_opt.Winv.shape if model_opt.Winv is not None else 'None'}")
    
    if model_orig.Winv is not None and model_opt.Winv is not None:
        # Compare values (handle complex numbers)
        winv_orig_abs = torch.abs(model_orig.Winv)
        winv_opt_abs = torch.abs(model_opt.Winv)
        
        # Filter out NaN
        valid_orig_mask = ~torch.isnan(winv_orig_abs)
        valid_opt_mask = ~torch.isnan(winv_opt_abs)
        
        winv_orig_valid = winv_orig_abs[valid_orig_mask]
        winv_opt_valid = winv_opt_abs[valid_opt_mask]
        
        if winv_orig_valid.numel() > 0:
            print(f"Original |Winv| range: [{winv_orig_valid.min().item():.2e}, {winv_orig_valid.max().item():.2e}]")
            print(f"Original |Winv| mean: {winv_orig_valid.mean().item():.2e}")
        
        if winv_opt_valid.numel() > 0:
            print(f"Optimized |Winv| range: [{winv_opt_valid.min().item():.2e}, {winv_opt_valid.max().item():.2e}]")
            print(f"Optimized |Winv| mean: {winv_opt_valid.mean().item():.2e}")
            
            if winv_orig_valid.numel() > 0:
                scale_factor = winv_opt_valid.mean().item() / winv_orig_valid.mean().item()
                print(f"Scale factor (opt/orig): {scale_factor:.2e}")
    
    # Test apply_disorder
    print("\n=== APPLY_DISORDER TEST ===")
    intensity_orig = model_orig.apply_disorder(use_data_adp=True)
    intensity_opt = model_opt.apply_disorder(use_data_adp=True)
    
    valid_orig = intensity_orig[~torch.isnan(intensity_orig)]
    valid_opt = intensity_opt[~torch.isnan(intensity_opt)]
    
    if valid_orig.numel() > 0:
        print(f"Original intensity mean: {valid_orig.mean().item():.2e}")
    if valid_opt.numel() > 0:
        print(f"Optimized intensity mean: {valid_opt.mean().item():.2e}")
        
    if valid_orig.numel() > 0 and valid_opt.numel() > 0:
        intensity_scale = valid_opt.mean().item() / valid_orig.mean().item()
        print(f"Intensity scale factor (opt/orig): {intensity_scale:.2e}")
        
        if intensity_scale > 100:
            print("\n⚠️ CRITICAL: Optimized values are >100x larger!")
            print("Possible causes:")
            print("  - Missing normalization in eigendecomposition")
            print("  - Incorrect matrix transformations")
            print("  - Wrong scaling in apply_disorder")

if __name__ == "__main__":
    compare_phonon_matrices()
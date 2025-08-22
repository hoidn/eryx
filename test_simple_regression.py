#!/usr/bin/env python3
"""Simple test to identify regression source."""

import torch
import numpy as np

def test_apply_disorder():
    """Test apply_disorder directly."""
    from eryx.models_torch import OnePhonon
    from eryx.models_torch_optimized import OnePhononOptimized
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Very small test grid  
    h_samp = (-0.5, 0.5, 1)  # 2 points
    k_samp = (-0.5, 0.5, 1)  # 2 points
    l_samp = (-0.5, 0.5, 1)  # 2 points
    
    print("Creating original model...")
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
    
    print("Computing original intensity...")
    model_orig.compute_gnm_phonons()
    model_orig.compute_covariance_matrix()
    intensity_orig = model_orig.apply_disorder(use_data_adp=True)
    
    print(f"Original intensity: shape={intensity_orig.shape}")
    valid_orig = intensity_orig[~torch.isnan(intensity_orig)]
    if valid_orig.numel() > 0:
        print(f"  Range: [{valid_orig.min().item():.2e}, {valid_orig.max().item():.2e}]")
        print(f"  Mean: {valid_orig.mean().item():.2e}")
    
    # Now test optimized - but use original implementation for now
    print("\nUsing original implementation again as control...")
    model_ctrl = OnePhonon(
        pdb_path,
        h_samp, k_samp, l_samp,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    model_ctrl.compute_gnm_phonons()
    model_ctrl.compute_covariance_matrix()
    intensity_ctrl = model_ctrl.apply_disorder(use_data_adp=True)
    
    print(f"Control intensity: shape={intensity_ctrl.shape}")
    valid_ctrl = intensity_ctrl[~torch.isnan(intensity_ctrl)]
    if valid_ctrl.numel() > 0:
        print(f"  Range: [{valid_ctrl.min().item():.2e}, {valid_ctrl.max().item():.2e}]")
        print(f"  Mean: {valid_ctrl.mean().item():.2e}")
    
    # Compare
    print("\nComparison:")
    if torch.allclose(intensity_orig, intensity_ctrl, rtol=1e-5, atol=1e-8, equal_nan=True):
        print("✅ Control matches original perfectly")
    else:
        print("❌ Control differs from original (unexpected!)")

if __name__ == "__main__":
    test_apply_disorder()
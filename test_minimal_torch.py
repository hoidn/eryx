#!/usr/bin/env python3
"""Minimal test to check if PyTorch implementation works at all."""

import torch
from eryx.models_torch import OnePhonon

print("Testing minimal PyTorch implementation...")
device = torch.device('cpu')  # Use CPU to avoid CUDA issues
pdb_path = "tests/pdbs/5zck_p1.pdb"

# Extremely small grid
print("Creating model with 2x2x2 grid...")
model = OnePhonon(
    pdb_path,
    [-0.25, 0.25, 1],  # 2 points
    [-0.25, 0.25, 1],  # 2 points  
    [-0.25, 0.25, 1],  # 2 points
    expand_p1=True,  # Need to expand to P1 for proper initialization
    res_limit=5.0,    # High resolution limit to reduce points
    gnm_cutoff=4.0,
    gamma_intra=1.0,
    gamma_inter=1.0,
    device=device
)

print(f"Model created. q_grid shape: {model.q_grid.shape}")

# Try to apply disorder without computing phonons first
# (they should be computed in __init__)
print("Applying disorder...")
intensity = model.apply_disorder(use_data_adp=False)  # Don't use data ADP

print(f"Success! Intensity shape: {intensity.shape}")
valid = intensity[~torch.isnan(intensity)]
if valid.numel() > 0:
    print(f"Valid points: {valid.numel()}/{intensity.numel()}")
    print(f"Range: [{valid.min().item():.2e}, {valid.max().item():.2e}]")
else:
    print("All values are NaN")
#!/usr/bin/env python3
"""Create a version with uniform opacity instead of intensity-based opacity."""

from optimized_masking_final import create_final_optimized_masking
import numpy as np
import os

# Load the high-res data
if os.path.exists('torch_diffuse_intensity.npy'):
    print("Loading torch_diffuse_intensity.npy...")
    data = np.load('torch_diffuse_intensity.npy')
    volume_data = data.reshape(81, 81, 81)
else:
    print("Creating test data...")
    x = np.linspace(-2, 2, 41)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    volume_data = np.exp(-(X**2 + Y**2 + Z**2))

print(f"Creating uniform opacity version...")
print(f"Volume shape: {volume_data.shape}")

# Create with uniform opacity
output_file = create_final_optimized_masking(
    volume_data,
    output_file="uniform_opacity_masking.html",
    alpha_coef=15.0,  # Ignored when uniform_opacity=True
    color_range_percentile=90,
    uniform_opacity=True  # This enables uniform opacity
)

print(f"\n✅ Created: uniform_opacity_masking.html")
print("This version uses uniform opacity across all intensity values")
print("All voxels have the same transparency regardless of their intensity")
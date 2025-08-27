#!/usr/bin/env python3
"""
Example: Create a slicing animation of diffuse scattering data.

This example shows how to use the visualization module to create
animated GIFs of 3D intensity data being progressively sliced.
"""

import numpy as np
from eryx.visualization import create_slicing_animation

# Example 1: Simple animation along diagonal plane
print("Creating diagonal slicing animation...")
create_slicing_animation(
    "torch_grid_results.npz",           # Data file from run_torch.py
    normal=[1, 1, 1],                    # Diagonal cutting plane
    n_frames=30,                         # Number of frames
    output_path="diagonal_slice.gif",   # Output file
    nan_strategy='interpolate',         # Handle NaN at lattice points
    fps=10,                              # Frames per second
    subsample=4,                         # Subsample for performance
    colormap='viridis',                  # Color scheme
    view_angles=(30, 45),                # Camera angles
    figsize=(10, 8),                     # Figure size
    dpi=100                              # Resolution
)
print("Saved: diagonal_slice.gif")

# Example 2: Animation along crystallographic axes
for axis, normal in [('h', [1, 0, 0]), ('k', [0, 1, 0]), ('l', [0, 0, 1])]:
    print(f"Creating {axis}-axis slicing animation...")
    create_slicing_animation(
        "torch_grid_results.npz",
        normal=normal,
        n_frames=25,
        output_path=f"{axis}_axis_slice.gif",
        nan_strategy='mask',
        fps=8,
        subsample=6,
        show_plane=True,
        show_volume=True,
        colormap='coolwarm'
    )
    print(f"Saved: {axis}_axis_slice.gif")

# Example 3: High-quality animation with custom parameters
print("Creating high-quality animation...")
create_slicing_animation(
    "torch_grid_results.npz",
    normal=[1, 2, 0.5],                 # Custom normal vector
    n_frames=50,                         # More frames for smoother animation
    output_path="hq_slice.gif",
    nan_strategy='interpolate',
    fps=15,
    subsample=2,                         # Less subsampling for higher detail
    figsize=(12, 10),
    dpi=150,                              # Higher resolution
    colormap='plasma',
    volume_alpha=0.4,                    # More transparent volume
    plane_alpha=0.2,                     # Subtle plane
    slice_alpha=0.95,                    # Prominent slice
    voxel_size=0.5,                      # Smaller voxels
    title_format='Diffuse Intensity - Frame {frame}/{total}'
)
print("Saved: hq_slice.gif")

print("\nAll animations created successfully!")
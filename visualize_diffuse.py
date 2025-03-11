#!/usr/bin/env python3
"""
Visualize the diffuse intensity output from run_debug.py.

This script loads the "np_diffuse_intensity.npy" file and displays a 2D slice
(using matplotlib). If the output is 3D (e.g. shape (dim_h, dim_k, dim_l)),
you can choose to display a central slice along one dimension.

Usage:
    python visualize_diffuse.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Path to the output diffuse intensity file (produced by run_debug.py)
    npy_file = "torch_diffuse_intensity.npy"
    
    if not os.path.exists(npy_file):
        print(f"File not found: {npy_file}")
        return

    # Load the diffuse intensity data
    intensity = np.load(npy_file)
    
    # Check the dimensions of the output
    print(f"Diffuse intensity shape: {intensity.shape}")
    
    # If intensity is 1D, we assume it should be reshaped.
    # For example, if the log indicated map_shape = (25, 103, 175)
    if intensity.ndim == 1:
        # Change the shape here if needed based on your actual grid dimensions.
        map_shape = (25, 103, 175)
        intensity = intensity.reshape(map_shape)
        print(f"Reshaped diffuse intensity to: {intensity.shape}")
    
    # If it's 3D, we can take a central slice for visualization.
    if intensity.ndim == 3:
        # Take the central slice along the first dimension
        slice_index = intensity.shape[0] // 2
        intensity_slice = intensity[slice_index, :, :]
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity_slice, cmap='viridis', origin='lower')
        plt.title(f"Diffuse Intensity (Slice {slice_index} of {intensity.shape[0]})")
        plt.xlabel("Axis 2")
        plt.ylabel("Axis 3")
        plt.colorbar(label="Intensity")
        plt.show()
    else:
        # If not 3D, try to plot it directly
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity, cmap='viridis')
        plt.title("Diffuse Intensity")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Intensity")
        plt.show()
    return intensity

if __name__ == '__main__':
    main()


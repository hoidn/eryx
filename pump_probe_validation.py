#!/usr/bin/env python
"""Pump-probe spectroscopy validation for the Eryx phonon modeling package.

This module simulates pump-probe experiments on protein crystals by comparing
thermal equilibrium phonon populations with non-equilibrium pumped states.
It demonstrates how selective excitation of phonon modes affects diffuse
scattering patterns in X-ray crystallography.

Key Features:
- Single model instance for efficient computation
- Direct Winv tensor manipulation for accurate physics
- Support for both delta function and Gaussian pump profiles
- Visual comparison of equilibrium vs pumped states
- Weighted population histograms showing actual phonon occupancies

Physics Background:
- Winv tensor represents 1/ω² scaled by thermal occupation factors
- Pumping increases specific mode populations above thermal equilibrium
- Changes in phonon populations modify the diffuse scattering intensity
- The difference maps reveal pump-induced structural dynamics

Usage:
    python pump_probe_validation.py [options]
    python pump_probe_validation.py --list-examples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Ensure the eryx library is in the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from eryx.models_torch import OnePhonon
except ImportError as e:
    print(f"ERROR: Could not import OnePhonon. Make sure eryx is in your PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)

def run_high_res_validation(pump_magnitude=3.0, pump_energy_percentile=50.0, slice_idx=None, 
                           pump_type='delta', pump_width=0.5):
    """
    Compares default thermal model against model with pumped phonon mode using
    direct in-memory manipulation of the Winv tensor.
    
    This implementation avoids duplicate model instantiation and directly modifies
    the phonon populations for accurate pump-probe simulation.
    
    Parameters
    ----------
    pump_magnitude : float
        Multiplier for pump intensity relative to thermal maximum (default: 3.0)
    pump_energy_percentile : float
        Percentile of frequency distribution to pump (default: 50.0)
    slice_idx : int, optional
        Index of the slice to plot along the k-axis (default: shape[1]//4)
    pump_type : str
        Type of pump profile: 'delta' for single mode, 'gaussian' for finite bandwidth (default: 'delta')
    pump_width : float
        Width of Gaussian pump in THz (sigma parameter) (default: 0.5)
    """
    print("--- Starting Pumped Mode PDOS Visual Validation ---")

    # Setup Model - using single instance for efficiency
    pdb_path = "tests/pdbs/6o2h_clean.pdb"
    if not os.path.exists(pdb_path):
        print(f"ERROR: Test PDB file not found at '{pdb_path}'")
        return

    hsampling_high_res = [-4, 4, 4]
    ksampling_high_res = [-4, 4, 4]
    lsampling_high_res = [-4, 4, 4]

    print(f"Initializing model (sampling rate: {hsampling_high_res[2]})...")
    
    # Single model instance - no duplicate instantiation
    model = OnePhonon(
        pdb_path=pdb_path,
        hsampling=hsampling_high_res,
        ksampling=ksampling_high_res,
        lsampling=lsampling_high_res,
        device=torch.device('cpu'),
        gamma_intra=1.5,
        gamma_inter=0.7
    )
    print("Model initialized.")

    # Compute phonons first to populate Winv
    print("Computing phonon modes...")
    _ = model.apply_disorder(use_data_adp=False)  # This populates model.Winv
    print("Phonon computation complete.")
    
    # Store original Winv tensor
    original_winv = model.Winv.clone()
    
    # --- 1. Get ALL frequencies from the model, including NaNs ---
    # Extract frequencies from Winv tensor: ω = sqrt(1/Winv)
    # NaN values indicate singular modes (zero frequency)
    winv_tensor = model.Winv.real.detach().cpu()
    omega_squared = torch.where(torch.isnan(winv_tensor) | (winv_tensor <= 1e-12), 
                                torch.tensor(float('nan')), 1.0 / winv_tensor)
    omega_rad = torch.sqrt(omega_squared)
    # Apply same empirical conversion as get_frequencies_thz()
    raw_freqs_hz = (omega_rad / (2 * np.pi))
    all_freqs_thz = (raw_freqs_hz * 100).flatten().numpy()  # Empirical factor for THz
    
    # --- 2. Identify the valid frequencies and their original indices ---
    # Filter out singular modes for frequency-based operations
    valid_mask = ~np.isnan(all_freqs_thz)
    valid_freqs = all_freqs_thz[valid_mask]
    original_indices = np.arange(all_freqs_thz.size)
    valid_original_indices = original_indices[valid_mask]
    
    # --- 3. Find the target frequency and its index IN THE VALID LIST ---
    print(f"Identifying mode at {pump_energy_percentile}th percentile for pumping...")
    pump_frequency_thz = np.percentile(valid_freqs, pump_energy_percentile)
    closest_idx_in_valid_list = np.argmin(np.abs(valid_freqs - pump_frequency_thz))
    
    # --- 4. Use the valid-list-index to find the TRUE index in the original tensor ---
    pump_index_flat = valid_original_indices[closest_idx_in_valid_list]
    actual_pump_freq_thz = all_freqs_thz[pump_index_flat]
    
    print(f"Target pump frequency (percentile): {pump_frequency_thz:.2f} THz")
    print(f"Actual pump frequency (closest mode): {actual_pump_freq_thz:.2f} THz")
    print(f"Mode index in flattened tensor: {pump_index_flat}")
    
    # Create modified Winv for pumped state
    pumped_winv = original_winv.clone()
    
    # Calculate pump intensity based on magnitude and max Winv value
    # Filter out any NaN/inf values when finding the max
    # Pump intensity scales with the thermal population maximum
    winv_real_abs = torch.abs(original_winv.real)
    finite_mask = torch.isfinite(winv_real_abs)
    if finite_mask.any():
        pump_intensity = torch.max(winv_real_abs[finite_mask]) * pump_magnitude
    else:
        # Fallback if all values are non-finite
        pump_intensity = torch.tensor(1.0) * pump_magnitude
    
    # Apply pump based on profile type
    pumped_winv_flat = pumped_winv.view(-1)
    
    if pump_type == 'delta':
        # Delta function pump - single mode excitation
        # Models ideal monochromatic pump (e.g., single-frequency laser)
        pumped_winv_flat[pump_index_flat] = pumped_winv_flat[pump_index_flat] + pump_intensity
        print(f"Applied delta pump: intensity {pump_intensity:.2e} to mode {pump_index_flat}")
        
    elif pump_type == 'gaussian':
        # Gaussian pump - affects multiple modes with finite bandwidth
        # Models realistic pump pulse with spectral width
        # Width parameter controls selectivity vs mode coverage tradeoff
        print(f"Applying Gaussian pump: center={pump_frequency_thz:.2f} THz, width={pump_width:.2f} THz")
        
        # Apply Gaussian weighting to all valid modes
        modes_affected = 0
        total_added = 0.0
        
        for i in range(len(all_freqs_thz)):
            if not np.isnan(all_freqs_thz[i]):
                # Calculate frequency distance from pump center
                freq_diff = all_freqs_thz[i] - pump_frequency_thz
                
                # Gaussian weight: exp(-0.5 * (Δf/σ)²)
                # Standard Gaussian profile centered at pump frequency
                weight = np.exp(-0.5 * (freq_diff / pump_width)**2)
                
                # Apply weighted pump intensity (with cutoff for efficiency)
                if weight > 0.01:  # Only affect modes with >1% contribution
                    pump_contribution = pump_intensity * weight
                    pumped_winv_flat[i] = pumped_winv_flat[i] + pump_contribution
                    modes_affected += 1
                    total_added += pump_contribution.item() if torch.is_tensor(pump_contribution) else pump_contribution
        
        print(f"Gaussian pump affected {modes_affected} modes, total intensity added: {total_added:.2e}")
    
    else:
        raise ValueError(f"Unknown pump_type: {pump_type}. Use 'delta' or 'gaussian'")
    
    pumped_winv = pumped_winv_flat.view(original_winv.shape)
    
    # Create histogram data showing actual phonon populations (not just density of states)
    # The Winv values represent the actual mode populations that determine scattering
    # This is a key improvement: using Winv weights gives true occupancies
    
    # Extract thermal populations from original Winv
    winv_values_thermal = original_winv.real.flatten().detach().cpu().numpy()
    # Extract pumped populations from modified Winv
    winv_values_pumped = pumped_winv.real.flatten().detach().cpu().numpy()
    
    # Create histograms weighted by actual populations
    # We need to ensure weights match the valid frequencies
    # Higher Winv = higher thermal population for that mode
    bins = 200
    thermal_hist_values, bin_edges = np.histogram(
        valid_freqs,  # Only valid frequencies
        weights=winv_values_thermal[valid_mask],  # Corresponding Winv values
        bins=bins, 
        density=False  # Don't normalize - keep absolute population values
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create pumped histogram with modified populations
    pumped_pop_values, _ = np.histogram(
        valid_freqs,  # Same frequencies
        weights=winv_values_pumped[valid_mask],  # Modified Winv values
        bins=bins,
        density=False  # Keep absolute values to see pump effect
    )

    # Run Simulations for Both Scenarios using the same model instance

    # Scenario A: Default Thermal Model (with original Winv)
    print("\nRunning Scenario A: Default Thermal Model...")
    model.Winv = original_winv
    intensity_default = model.apply_disorder(use_data_adp=False)
    intensity_default_np = intensity_default.detach().cpu().numpy().reshape(model.map_shape)
    print("Scenario A complete.")
    
    # Save thermal state (same format as run_torch.py)
    q_vectors_thermal = model.q_grid.detach().cpu().numpy()
    intensity_thermal_flat = intensity_default.detach().cpu().numpy()
    
    np.savez_compressed("thermal_grid_results.npz",
                        q_vectors=q_vectors_thermal,
                        intensity=intensity_thermal_flat,
                        map_shape=model.map_shape)
    
    np.save("thermal_diffuse_intensity.npy", intensity_thermal_flat)
    torch.save(intensity_default, "thermal_diffuse_intensity.pt")
    print(f"Saved thermal results: thermal_grid_results.npz ({intensity_thermal_flat.shape})")
    print(f"Thermal intensity range: [{np.min(intensity_thermal_flat):.2e}, {np.max(intensity_thermal_flat):.2e}]")

    # Scenario B: Pumped Model (with modified Winv)
    print("\nRunning Scenario B: Pumped Model...")
    model.Winv = pumped_winv
    intensity_pumped = model.apply_disorder(use_data_adp=False)
    intensity_pumped_np = intensity_pumped.detach().cpu().numpy().reshape(model.map_shape)
    print("Scenario B complete.")
    
    # Restore original state
    model.Winv = original_winv
    
    # Save pumped state (same format as run_torch.py)
    q_vectors_pumped = model.q_grid.detach().cpu().numpy()  # Same q-grid as thermal
    intensity_pumped_flat = intensity_pumped.detach().cpu().numpy()
    
    np.savez_compressed("pumped_grid_results.npz",
                        q_vectors=q_vectors_pumped,
                        intensity=intensity_pumped_flat,
                        map_shape=model.map_shape)
    
    np.save("pumped_diffuse_intensity.npy", intensity_pumped_flat)
    torch.save(intensity_pumped, "pumped_diffuse_intensity.pt")
    print(f"Saved pumped results: pumped_grid_results.npz ({intensity_pumped_flat.shape})")
    print(f"Pumped intensity range: [{np.min(intensity_pumped_flat):.2e}, {np.max(intensity_pumped_flat):.2e}]")
    
    # Calculate and save difference (pump-probe signal)
    intensity_difference = intensity_pumped_flat - intensity_thermal_flat
    
    np.savez_compressed("difference_grid_results.npz",
                        q_vectors=q_vectors_thermal,  # Same q-grid
                        intensity=intensity_difference,
                        map_shape=model.map_shape)
    
    np.save("difference_diffuse_intensity.npy", intensity_difference)
    print(f"Saved difference map: difference_grid_results.npz ({intensity_difference.shape})")
    print(f"Difference range: [{np.min(intensity_difference):.2e}, {np.max(intensity_difference):.2e}]")

    # Create the 2x2 Visualization
    print("\nGenerating 2x2 comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle("Default Thermal vs. Pumped Phonon Population", fontsize=18)

    # Plot A: Default Thermal Population Model
    ax = axes[0, 0]
    ax.plot(bin_centers, thermal_hist_values, color='royalblue', lw=2)
    ax.fill_between(bin_centers, thermal_hist_values, color='royalblue', alpha=0.2)
    ax.set_title("A) Thermal Phonon Population (Equilibrium)", fontsize=14)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon Population (arb. units)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=0)
    # FIX: Prevent matplotlib offset notation artifacts
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.ticklabel_format(style='plain', axis='x')

    # Plot B: "Pumped" Thermal Population Model
    ax = axes[0, 1]
    ax.plot(bin_centers, pumped_pop_values, color='orangered', lw=2)
    ax.fill_between(bin_centers, pumped_pop_values, color='orangered', alpha=0.2)
    ax.axvline(actual_pump_freq_thz, color='red', linestyle='--', lw=2, label=f'Pumped Mode ({actual_pump_freq_thz:.2f} THz)')
    ax.set_title("B) Pumped Phonon Population (Non-Equilibrium)", fontsize=14)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon Population (arb. units)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=0)
    # FIX: Prevent matplotlib offset notation artifacts
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.ticklabel_format(style='plain', axis='x')
    ax.legend()

    # Helper function for plotting intensity slices with log scaling
    def plot_slice(ax, intensity_map, title, hkl_grid):
        """Plot 2D slice of 3D diffuse intensity with Miller index labels.
        
        Args:
            ax: Matplotlib axis to plot on
            intensity_map: 3D array of diffuse intensities
            title: Plot title string
            hkl_grid: 4D array of Miller indices (h,k,l) at each grid point
        """
        nonlocal slice_idx
        if slice_idx is None:
            slice_idx = intensity_map.shape[1] // 4
        # Validate slice_idx is within bounds
        slice_idx = max(0, min(slice_idx, intensity_map.shape[1] - 1))
        slice_data = intensity_map[:, slice_idx, :]
        
        # Calculate Miller Index extents for labels
        h_coords = hkl_grid[:, slice_idx, 0, 0]  # h-values for the slice
        l_coords = hkl_grid[0, slice_idx, :, 2]  # l-values for the slice
        
        h_min, h_max = h_coords.min(), h_coords.max()
        l_min, l_max = l_coords.min(), l_coords.max()
        
        # The extent for imshow is [left, right, bottom, top]
        plot_extent = [l_min, l_max, h_min, h_max]
        
        # Apply log scaling for better visualization
        # Add small epsilon to avoid log(0)
        # Log scale reveals weak features in diffuse scattering
        epsilon = 1e-10
        log_slice_data = np.log10(slice_data + epsilon)
        
        # Set color scale based on log data
        valid_log_data = log_slice_data[np.isfinite(log_slice_data)]
        if valid_log_data.size > 0:
            vmax = np.percentile(valid_log_data, 99.5)
            vmin = np.percentile(valid_log_data, 1)  # Use 1st percentile instead of minimum
        else:
            vmin, vmax = 0, 1

        # FIX: Replace NaN values with vmin to prevent blank regions in imshow
        # This ensures proper visualization even with singular modes
        log_slice_data_fixed = log_slice_data.copy()
        log_slice_data_fixed[~np.isfinite(log_slice_data_fixed)] = vmin

        im = ax.imshow(log_slice_data_fixed, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, 
                       interpolation='bilinear', extent=plot_extent, aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Miller Index (l)")
        ax.set_ylabel("Miller Index (h)")
        fig.colorbar(im, ax=ax, label="log₁₀(Diffuse Intensity)")

    # Plot C: Diffuse Intensity from Default Model
    hkl_grid_np = model.hkl_grid.detach().cpu().numpy().reshape(model.map_shape + (3,))
    plot_slice(axes[1, 0], intensity_default_np, "C) Diffuse Intensity (Equilibrium)", hkl_grid_np)

    # Plot D: Diffuse Intensity from Pumped Model
    plot_slice(axes[1, 1], intensity_pumped_np, "D) Diffuse Intensity (Pumped Mode)", hkl_grid_np)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Generate output filename with parameters
    output_filename = f"pump_probe_validation_mag{pump_magnitude:.1f}_pct{pump_energy_percentile:.0f}.png"
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'.")
    plt.show()

def main():
    """Parse command-line arguments and run validation."""
    parser = argparse.ArgumentParser(description='PDOS pumped mode validation')
    parser.add_argument('--pump-magnitude', '-m', type=float, default=3.0,
                        help='Pump intensity multiplier relative to thermal maximum (default: 3.0)')
    parser.add_argument('--pump-energy-percentile', '-p', type=float, default=50.0,
                        help='Percentile of frequency distribution to pump (default: 50.0)')
    parser.add_argument('--slice-idx', '-s', type=int, default=None,
                        help='Index of the slice to plot along the k-axis (default: shape[1]//4)')
    parser.add_argument('--pump-type', '-t', type=str, choices=['delta', 'gaussian'], default='delta',
                        help='Type of pump profile: delta (single mode) or gaussian (finite bandwidth) (default: delta)')
    parser.add_argument('--pump-width', '-w', type=float, default=0.5,
                        help='Width of Gaussian pump in THz (sigma parameter) (default: 0.5)')
    parser.add_argument('--list-examples', action='store_true',
                        help='Show example usage and exit')
    
    args = parser.parse_args()
    
    if args.list_examples:
        print("Example usage:")
        print("  python pump_probe_validation.py                        # Default: 3x delta pump at 50 percentile")
        print("  python pump_probe_validation.py -m 5.0 -p 90          # 5x delta pump at 90th percentile")
        print("  python pump_probe_validation.py -t gaussian -w 0.5    # Gaussian pump with 0.5 THz width")
        print("  python pump_probe_validation.py -t gaussian -w 1.0 -m 10  # Strong Gaussian pump, 1 THz width")
        print("  python pump_probe_validation.py -m 1.5 -p 50 -s 0     # 1.5x pump at median frequency, first slice")
        print("  python pump_probe_validation.py -m 10.0 -p 99 -s 3    # 10x pump at highest frequencies, slice 3")
        sys.exit(0)
    
    # Validate arguments
    if args.pump_magnitude <= 0:
        print("ERROR: Pump magnitude must be positive")
        sys.exit(1)
    if not (0 <= args.pump_energy_percentile <= 100):
        print("ERROR: Pump energy percentile must be between 0 and 100")
        sys.exit(1)
    if args.pump_width <= 0:
        print("ERROR: Pump width must be positive")
        sys.exit(1)
    if args.slice_idx is not None and args.slice_idx < 0:
        print("ERROR: Slice index must be non-negative")
        sys.exit(1)
    
    print(f"Running with pump magnitude: {args.pump_magnitude}x")
    print(f"Running with pump energy percentile: {args.pump_energy_percentile}%")
    print(f"Running with pump type: {args.pump_type}")
    if args.pump_type == 'gaussian':
        print(f"Running with Gaussian width: {args.pump_width} THz")
    if args.slice_idx is not None:
        print(f"Running with slice index: {args.slice_idx}")
    else:
        print("Running with slice index: auto (shape[1]//4)")
    
    run_high_res_validation(args.pump_magnitude, args.pump_energy_percentile, args.slice_idx,
                          args.pump_type, args.pump_width)

if __name__ == "__main__":
    main()
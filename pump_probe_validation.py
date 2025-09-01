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
    from pump_probe_utils import PumpProbeSimulator
except ImportError as e:
    print(f"ERROR: Could not import required modules. Make sure eryx and pump_probe_utils are in your PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)

def run_high_res_validation(pump_magnitude=3.0, pump_energy_percentile=50.0, slice_idx=None):
    """
    Compares default thermal model against model with directly pumped phonon modes.
    
    Uses direct Winv modification to avoid PDOS binning artifacts and provide accurate
    pump-probe simulation results without intensity inflation.
    
    Parameters
    ----------
    pump_magnitude : float
        Pump magnitude for direct Winv modification (default: 3.0)
    pump_energy_percentile : float
        Percentile of frequency distribution to pump (default: 50.0)
    slice_idx : int, optional
        Index of the slice to plot along the k-axis (default: shape[1]//4)
    """
    print("--- Starting Direct Winv Modification Pump-Probe Validation ---")

    # Setup Model
    pdb_path = "tests/pdbs/6o2h_clean.pdb"
    #pdb_path = "tests/pdbs/1896374.pdb"
    if not os.path.exists(pdb_path):
        print(f"ERROR: Test PDB file not found at '{pdb_path}'")
        return

    hsampling_high_res = [-4, 4, 4]
    ksampling_high_res = [-4, 4, 4]
    lsampling_high_res = [-4, 4, 4]

    print(f"Initializing base model (sampling rate: {hsampling_high_res[2]})...")
    
    base_model = OnePhonon(
        pdb_path=pdb_path,
        hsampling=hsampling_high_res,
        ksampling=ksampling_high_res,
        lsampling=lsampling_high_res,
        device=torch.device('cpu'),
        gamma_intra=1.5,
        gamma_inter=0.7
    )
    print("Base model initialized.")

    # Calculate pump frequency from model's frequency distribution
    simulator = PumpProbeSimulator(base_model)
    frequencies = simulator._calculate_frequencies()
    freq_valid = frequencies[torch.isfinite(frequencies)].detach().cpu().numpy()
    pump_frequency_thz = float(np.percentile(freq_valid, pump_energy_percentile))
    
    print(f"Pump frequency target: {pump_frequency_thz:.2f} THz ({pump_energy_percentile}th percentile)")
    print(f"Frequency range: {np.min(freq_valid):.2f} - {np.max(freq_valid):.2f} THz")

    # Run Simulations for Both Scenarios

    # Scenario A: Thermal equilibrium (unmodified Winv)
    print("\nRunning Scenario A: Thermal Equilibrium Model...")
    intensity_thermal = base_model.apply_disorder(use_data_adp=False)
    intensity_thermal_np = intensity_thermal.detach().cpu().numpy().reshape(base_model.map_shape)
    print("Scenario A complete.")
    
    # Save thermal state (same format as run_torch.py)
    q_vectors_thermal = base_model.q_grid.detach().cpu().numpy()
    intensity_thermal_flat = intensity_thermal.detach().cpu().numpy()
    
    np.savez_compressed("thermal_grid_results.npz",
                        q_vectors=q_vectors_thermal,
                        intensity=intensity_thermal_flat,
                        map_shape=base_model.map_shape)
    
    np.save("thermal_diffuse_intensity.npy", intensity_thermal_flat)
    torch.save(intensity_thermal, "thermal_diffuse_intensity.pt")
    print(f"Saved thermal results: thermal_grid_results.npz ({intensity_thermal_flat.shape})")
    print(f"Thermal intensity range: [{np.min(intensity_thermal_flat):.2e}, {np.max(intensity_thermal_flat):.2e}]")

    # Scenario B: Direct Winv modification (pumped state)
    print("\nRunning Scenario B: Direct Winv Pump Application...")
    # Instead of deepcopy, create new model and apply pump to the original model
    # We'll save the original Winv and restore it later
    
    # Apply pump directly to base_model (we'll restore it later)
    simulator_for_pump = PumpProbeSimulator(base_model)
    
    # Apply Gaussian pump with 5% bandwidth around target frequency
    pump_bandwidth = 0.05 * pump_frequency_thz  # 5% of center frequency
    pump_info = simulator_for_pump.apply_gaussian_pump(
        center_freq=pump_frequency_thz,
        width=pump_bandwidth,
        magnitude=pump_magnitude
    )
    
    print(f"Applied pump: {pump_info['n_affected_modes']} modes affected")
    print(f"Pump details: {pump_info['center_freq']:.2f} THz ± {pump_bandwidth:.2f} THz")
    print(f"Peak scale factor: {pump_info['peak_scale_factor']:.3f}")
    
    # Calculate pumped intensity with modified Winv
    intensity_pumped = base_model.apply_disorder(use_data_adp=False)
    intensity_pumped_np = intensity_pumped.detach().cpu().numpy().reshape(base_model.map_shape)
    
    # Reset the model back to thermal state for consistency
    simulator_for_pump.reset()
    print("Scenario B complete.")
    
    # Save pumped state (same format as run_torch.py)
    q_vectors_pumped = base_model.q_grid.detach().cpu().numpy()  # Same q-grid as thermal
    intensity_pumped_flat = intensity_pumped.detach().cpu().numpy()
    
    np.savez_compressed("pumped_grid_results.npz",
                        q_vectors=q_vectors_pumped,
                        intensity=intensity_pumped_flat,
                        map_shape=base_model.map_shape)
    
    np.save("pumped_diffuse_intensity.npy", intensity_pumped_flat)
    torch.save(intensity_pumped, "pumped_diffuse_intensity.pt")
    print(f"Saved pumped results: pumped_grid_results.npz ({intensity_pumped_flat.shape})")
    print(f"Pumped intensity range: [{np.min(intensity_pumped_flat):.2e}, {np.max(intensity_pumped_flat):.2e}]")
    
    # Calculate and save difference (pump-probe signal)
    intensity_difference = intensity_pumped_flat - intensity_thermal_flat
    
    np.savez_compressed("difference_grid_results.npz",
                        q_vectors=q_vectors_thermal,  # Same q-grid
                        intensity=intensity_difference,
                        map_shape=base_model.map_shape)
    
    np.save("difference_diffuse_intensity.npy", intensity_difference)
    print(f"Saved difference map: difference_grid_results.npz ({intensity_difference.shape})")
    print(f"Difference range: [{np.min(intensity_difference):.2e}, {np.max(intensity_difference):.2e}]")
    
    # Calculate pump effect statistics
    relative_change = np.abs(intensity_difference) / np.abs(intensity_thermal_flat)
    mean_relative_change = np.nanmean(relative_change) * 100
    max_relative_change = np.nanmax(relative_change) * 100
    print(f"Mean relative change: {mean_relative_change:.3f}%")
    print(f"Max relative change: {max_relative_change:.3f}%")

    # Create the 2x2 Visualization
    print("\nGenerating 2x2 comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle("Direct Winv Modification Pump-Probe Results", fontsize=18)

    # Plot A: Frequency Distribution of Phonon Modes
    ax = axes[0, 0]
    bins = 100
    counts, bin_edges = np.histogram(freq_valid, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, counts, color='royalblue', lw=2)
    ax.fill_between(bin_centers, counts, color='royalblue', alpha=0.2)
    ax.axvline(pump_frequency_thz, color='red', linestyle='--', lw=2, 
               label=f'Pump Target ({pump_frequency_thz:.2f} THz)')
    ax.axvspan(pump_frequency_thz - pump_bandwidth, pump_frequency_thz + pump_bandwidth,
               alpha=0.2, color='red', label=f'Pump Bandwidth (±{pump_bandwidth:.2f} THz)')
    ax.set_title("A) Phonon Frequency Distribution", fontsize=14)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Number of Modes")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot B: Pump Effect Statistics
    ax = axes[0, 1]
    # Show intensity change statistics
    intensity_ratio = intensity_pumped_flat / intensity_thermal_flat
    valid_ratio = intensity_ratio[np.isfinite(intensity_ratio)]
    
    if len(valid_ratio) > 0:
        bins_ratio = np.linspace(0.5, 2.0, 50)
        counts_ratio, _ = np.histogram(valid_ratio, bins=bins_ratio, density=True)
        bin_centers_ratio = (bins_ratio[:-1] + bins_ratio[1:]) / 2
        ax.plot(bin_centers_ratio, counts_ratio, color='orange', lw=2)
        ax.fill_between(bin_centers_ratio, counts_ratio, color='orange', alpha=0.3)
        ax.axvline(1.0, color='black', linestyle='-', lw=1, alpha=0.5, label='No Change')
        ax.axvline(np.median(valid_ratio), color='red', linestyle='--', lw=2,
                   label=f'Median Ratio: {np.median(valid_ratio):.3f}')
    
    ax.set_title("B) Intensity Change Distribution", fontsize=14)
    ax.set_xlabel("Pumped/Thermal Intensity Ratio")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Helper function for plotting intensity slices with log scaling
    def plot_slice(ax, intensity_map, title, hkl_grid):
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
        log_slice_data_fixed = log_slice_data.copy()
        log_slice_data_fixed[~np.isfinite(log_slice_data_fixed)] = vmin

        im = ax.imshow(log_slice_data_fixed, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, 
                       interpolation='bilinear', extent=plot_extent, aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Miller Index (l)")
        ax.set_ylabel("Miller Index (h)")
        fig.colorbar(im, ax=ax, label="log₁₀(Diffuse Intensity)")

    # Plot C: Diffuse Intensity from Thermal Model
    hkl_grid_np = base_model.hkl_grid.detach().cpu().numpy().reshape(base_model.map_shape + (3,))
    plot_slice(axes[1, 0], intensity_thermal_np, "C) Diffuse Intensity (Thermal Equilibrium)", hkl_grid_np)

    # Plot D: Diffuse Intensity from Pumped Model  
    plot_slice(axes[1, 1], intensity_pumped_np, "D) Diffuse Intensity (Direct Winv Pump)", hkl_grid_np)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Generate output filename with parameters
    output_filename = f"pump_probe_validation_mag{pump_magnitude:.1f}_pct{pump_energy_percentile:.0f}.png"
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'.")
    
    # Summary of all generated files and results
    print("\n=== Direct Winv Modification Results Summary ===")
    print(f"Pump Parameters:")
    print(f"  - Target frequency: {pump_frequency_thz:.2f} THz ({pump_energy_percentile}th percentile)")
    print(f"  - Pump magnitude: {pump_magnitude}")
    print(f"  - Pump bandwidth: ±{pump_bandwidth:.2f} THz") 
    print(f"  - Affected modes: {pump_info['n_affected_modes']}")
    print(f"Intensity Changes:")
    print(f"  - Mean relative change: {mean_relative_change:.3f}%")
    print(f"  - Max relative change: {max_relative_change:.3f}%")
    print("Generated Files:")
    print("NPZ files (compatible with run_torch.py format):")
    print("  - thermal_grid_results.npz    : Thermal equilibrium state")
    print("  - pumped_grid_results.npz     : Direct Winv pumped state")  
    print("  - difference_grid_results.npz : Pump-probe difference signal")
    print("NPY files (NumPy arrays):")
    print("  - thermal_diffuse_intensity.npy")
    print("  - pumped_diffuse_intensity.npy")
    print("  - difference_diffuse_intensity.npy")
    print("PyTorch tensors:")
    print("  - thermal_diffuse_intensity.pt")
    print("  - pumped_diffuse_intensity.pt")
    print("Visualization:")
    print(f"  - {output_filename}")
    print("==================================================\n")
    
    plt.show()

def main():
    """Parse command-line arguments and run validation."""
    parser = argparse.ArgumentParser(description='Direct Winv modification pump-probe validation')
    parser.add_argument('--pump-magnitude', '-m', type=float, default=3.0,
                        help='Pump magnitude for direct Winv modification (default: 3.0)')
    parser.add_argument('--pump-energy-percentile', '-p', type=float, default=50.0,
                        help='Percentile of frequency distribution to pump (default: 50.0)')
    parser.add_argument('--slice-idx', '-s', type=int, default=None,
                        help='Index of the slice to plot along the k-axis (default: shape[1]//4)')
    parser.add_argument('--list-examples', action='store_true',
                        help='Show example usage and exit')
    
    args = parser.parse_args()
    
    if args.list_examples:
        print("Example usage (Direct Winv modification):")
        print("  python pump_probe_validation.py                        # Default: 3.0 magnitude pump at 50th percentile")
        print("  python pump_probe_validation.py -m 0.5 -p 90          # 0.5 magnitude pump at 90th percentile")
        print("  python pump_probe_validation.py -m 0.0000001 -p 50    # Near-zero pump (test for inflation)")
        print("  python pump_probe_validation.py -m 2.0 -p 25 -s 0     # 2.0 magnitude pump at low frequencies")
        sys.exit(0)
    
    # Validate arguments
    if args.pump_magnitude <= 0:
        print("ERROR: Pump magnitude must be positive")
        sys.exit(1)
    if not (0 <= args.pump_energy_percentile <= 100):
        print("ERROR: Pump energy percentile must be between 0 and 100")
        sys.exit(1)
    if args.slice_idx is not None and args.slice_idx < 0:
        print("ERROR: Slice index must be non-negative")
        sys.exit(1)
    
    print(f"Running with pump magnitude: {args.pump_magnitude}x")
    print(f"Running with pump energy percentile: {args.pump_energy_percentile}%")
    if args.slice_idx is not None:
        print(f"Running with slice index: {args.slice_idx}")
    else:
        print("Running with slice index: auto (shape[1]//4)")
    
    run_high_res_validation(args.pump_magnitude, args.pump_energy_percentile, args.slice_idx)

if __name__ == "__main__":
    main()

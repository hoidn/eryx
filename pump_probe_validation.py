import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tempfile
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

def run_high_res_validation(pump_magnitude=3.0, pump_energy_percentile=50.0, slice_idx=None):
    """
    Compares default thermal model against model with pumped phonon mode.
    
    Parameters
    ----------
    pump_magnitude : float
        Multiplier for pump intensity relative to thermal maximum (default: 3.0)
    pump_energy_percentile : float
        Percentile of frequency distribution to pump (default: 50.0)
    slice_idx : int, optional
        Index of the slice to plot along the k-axis (default: shape[1]//4)
    """
    print("--- Starting Pumped Mode PDOS Visual Validation ---")

    # Setup Model and Generate its own PDOS
    #pdb_path = "tests/pdbs/6o2h_clean.pdb"
    pdb_path = "tests/pdbs/1896374.pdb"
    if not os.path.exists(pdb_path):
        print(f"ERROR: Test PDB file not found at '{pdb_path}'")
        return

#    hsampling_high_res = [-2, 2, 4]
#    ksampling_high_res = [-2, 2, 4]
#    lsampling_high_res = [-2, 2, 4]

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

    # Generate the unweighted PDOS using the new method
    pdos_data = base_model.generate_pdos(bins=200, density=False)
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.dat') as tmp_f:
        np.savetxt(tmp_f, pdos_data, fmt="%.15e")  # Use high precision
        temp_pdos_path = tmp_f.name
    print(f"Model's own PDOS saved to temporary file: {temp_pdos_path}")

    # Create the Two Phonon Population Models

    # Extract raw frequencies (ω) and thermal populations (Winv ~ 1/ω²)
    winv_tensor = base_model.Winv.real.detach().cpu()
    omega_squared = torch.where(torch.isnan(winv_tensor) | (winv_tensor <= 1e-12), torch.tensor(float('nan')), 1.0 / winv_tensor)
    omega_rad = torch.sqrt(omega_squared)
    
    # Convert angular frequency to Hz then THz
    # omega_rad is in rad/s, f = omega/(2π) gives Hz
    # The model's internal units require multiplication by 100 to get actual THz
    raw_freqs_hz = (omega_rad / (2 * np.pi)).flatten().numpy()
    raw_freqs_hz = raw_freqs_hz[~np.isnan(raw_freqs_hz)]
    raw_freqs_thz = raw_freqs_hz * 100  # Convert to actual THz
    
    thermal_population_weights = winv_tensor.flatten().numpy()
    thermal_population_weights = thermal_population_weights[~np.isnan(thermal_population_weights)]

    # Model A: Create the Default Thermal Population histogram
    bins = 200
    thermal_hist_values, bin_edges = np.histogram(raw_freqs_thz, bins=bins, weights=thermal_population_weights, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Model B: Create the "Pumped" Thermal Population
    print(f"Creating pumped thermal PDOS by exciting mode at {pump_energy_percentile}th percentile...")
    pumped_pop_values = thermal_hist_values.copy()
    pump_frequency_thz = np.percentile(raw_freqs_thz, pump_energy_percentile)
    pump_bin_index = np.argmin(np.abs(bin_centers - pump_frequency_thz))
    
    actual_pump_freq_thz = bin_centers[pump_bin_index]
    
    pump_intensity = np.max(thermal_hist_values) * pump_magnitude
    pumped_pop_values[pump_bin_index] += pump_intensity
    
    print(f"Target pump frequency (percentile): {pump_frequency_thz:.2f} THz")
    print(f"Actual pump frequency (bin center): {actual_pump_freq_thz:.2f} THz")
    print(f"Added pump of intensity {pump_intensity:.2e} to thermal population at bin {pump_bin_index}")

    # Save the "pumped" PDOS to a temporary file for the simulation
    # Convert frequencies back to model's internal units
    bin_centers_model_units = bin_centers / 100  # Convert THz back to model units
    pumped_pdos_data = np.vstack((bin_centers_model_units, pumped_pop_values)).T
    with open(temp_pdos_path, 'w') as f: # Overwrite the temp file
        np.savetxt(f, pumped_pdos_data, fmt="%.15e")  # Use high precision

    # Run Simulations for Both Scenarios

    # Scenario A: Get intensity from the already-run base model (Default Thermal)
    print("\nRunning Scenario A: Default Thermal Model...")
    intensity_default = base_model.apply_disorder(use_data_adp=False)
    intensity_default_np = intensity_default.detach().cpu().numpy().reshape(base_model.map_shape)
    print("Scenario A complete.")

    # Scenario B: Run with the "pumped" PDOS in 'direct' mode
    print("\nRunning Scenario B: 'Pumped' PDOS in 'direct' mode...")
    model_pumped = OnePhonon(
        pdb_path=pdb_path,
        hsampling=hsampling_high_res, ksampling=ksampling_high_res, lsampling=lsampling_high_res,
        device=torch.device('cpu'),
        gamma_intra=1.5, gamma_inter=0.7,
        pdos_path=temp_pdos_path,
        pdos_mode='direct'
    )
    intensity_pumped = model_pumped.apply_disorder(use_data_adp=False)
    intensity_pumped_np = intensity_pumped.detach().cpu().numpy().reshape(base_model.map_shape)
    print("Scenario B complete.")

    # Clean up the temporary file
    os.remove(temp_pdos_path)

    # Create the 2x2 Visualization
    print("\nGenerating 2x2 comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle("Default Thermal vs. Pumped Phonon Population", fontsize=18)

    # Plot A: Default Thermal Population Model
    ax = axes[0, 0]
    ax.plot(bin_centers, thermal_hist_values, color='royalblue', lw=2)
    ax.fill_between(bin_centers, thermal_hist_values, color='royalblue', alpha=0.2)
    ax.set_title("A) Effective Thermal Population (Equilibrium)", fontsize=14)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Weighted Density (Population)")
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
    ax.set_title("B) Pumped Thermal Population (Non-Equilibrium)", fontsize=14)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Population (Arbitrary Units)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=0)
    # FIX: Prevent matplotlib offset notation artifacts
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.ticklabel_format(style='plain', axis='x')
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

    # Plot C: Diffuse Intensity from Default Model
    hkl_grid_np = base_model.hkl_grid.detach().cpu().numpy().reshape(base_model.map_shape + (3,))
    plot_slice(axes[1, 0], intensity_default_np, "C) Diffuse Intensity (Equilibrium)", hkl_grid_np)

    # Plot D: Diffuse Intensity from "Pumped" Model
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
                        help='Percentile of frequency distribution to pump (default: 95.0)')
    parser.add_argument('--slice-idx', '-s', type=int, default=None,
                        help='Index of the slice to plot along the k-axis (default: shape[1]//4)')
    parser.add_argument('--list-examples', action='store_true',
                        help='Show example usage and exit')
    
    args = parser.parse_args()
    
    if args.list_examples:
        print("Example usage:")
        print("  python pump_probe_validation.py                        # Default: 3x pump at 50 percentile, auto slice")
        print("  python pump_probe_validation.py -m 5.0 -p 90          # 5x pump at 90th percentile")
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

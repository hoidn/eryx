#!/usr/bin/env python3
import os
#os.environ["DEBUG_MODE"] = "1"
import logging
import numpy as np
from eryx.models import OnePhonon

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    filename="debug_output.log",
    filemode="w"
)
# Also set up console logging here if desired:
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logging.getLogger("").addHandler(console)


def setup_logging():
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="debug_output.log",
        filemode="w"
    )
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_np():
    # Use a small grid for testing; adjust parameters as necessary.
    logging.info("Starting NP branch computation")
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    onephonon_np = OnePhonon(
        pdb_path,
        [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    Id_np = onephonon_np.apply_disorder(use_data_adp=True)
    logging.debug(f"NP: hkl_grid shape = {onephonon_np.hkl_grid.shape}")
    logging.debug("NP: hkl_grid coordinate ranges:")
    logging.debug(f"  Dimension 0: min = {onephonon_np.hkl_grid[:,0].min()}, max = {onephonon_np.hkl_grid[:,0].max()}")
    logging.debug(f"  Dimension 1: min = {onephonon_np.hkl_grid[:,1].min()}, max = {onephonon_np.hkl_grid[:,1].max()}")
    logging.debug(f"  Dimension 2: min = {onephonon_np.hkl_grid[:,2].min()}, max = {onephonon_np.hkl_grid[:,2].max()}")
    logging.debug(f"NP: q_grid range: min = {onephonon_np.q_grid.min()}, max = {onephonon_np.q_grid.max()}")
    logging.info("NP branch diffuse intensity stats: min=%s, max=%s", np.nanmin(Id_np), np.nanmax(Id_np))
    # Save for later comparison
    np.save("np_diffuse_intensity.npy", Id_np)

def run_torch():
    """Run PyTorch version of the diffuse scattering simulation."""
    try:
        import torch
        from eryx.models_torch import OnePhonon
        
        # Get the device (use CUDA if available)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        logging.info(f"Starting PyTorch branch computation on {device}")
        
        # Use the same parameters as in run_np
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        
        print(f"DEBUG run_torch: Creating OnePhonon with parameters:")
        print(f"  pdb_path = {pdb_path}")
        print(f"  hsampling = {[-4, 4, 3]}")
        print(f"  ksampling = {[-17, 17, 3]}")
        print(f"  lsampling = {[-29, 29, 3]}")
        print(f"  q_vectors = None (should use grid-based sampling)")
        print(f"DEBUG: In run_torch, OnePhonon class has q_vectors_input attribute: {'q_vectors_input' in dir(OnePhonon)}")
        
        onephonon_torch = OnePhonon(
            pdb_path=pdb_path,
            hsampling=[-4, 4, 3],
            ksampling=[-17, 17, 3],
            lsampling=[-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Apply disorder
        Id_torch = onephonon_torch.apply_disorder(use_data_adp=True)
        
        # Log debug information
        logging.debug(f"PyTorch: hkl_grid shape = {onephonon_torch.hkl_grid.shape}")
        logging.debug("PyTorch: hkl_grid coordinate ranges:")
        logging.debug(f"  Dimension 0: min = {onephonon_torch.hkl_grid[:,0].min().item()}, max = {onephonon_torch.hkl_grid[:,0].max().item()}")
        logging.debug(f"  Dimension 1: min = {onephonon_torch.hkl_grid[:,1].min().item()}, max = {onephonon_torch.hkl_grid[:,1].max().item()}")
        logging.debug(f"  Dimension 2: min = {onephonon_torch.hkl_grid[:,2].min().item()}, max = {onephonon_torch.hkl_grid[:,2].max().item()}")
        logging.debug(f"PyTorch: q_grid range: min = {onephonon_torch.q_grid.min().item()}, max = {onephonon_torch.q_grid.max().item()}")
        
        # Save q-vectors for comparison
        np.save("torch_grid_q_vectors.npy", onephonon_torch.q_grid.detach().cpu().numpy())
        
        # Save intensity results for comparison
        torch.save(Id_torch, "torch_diffuse_intensity.pt")
        np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())
        
        return Id_torch
        
    except RuntimeError as e:
        # Handle CUDA errors by falling back to CPU
        if 'CUDA' in str(e):
            logging.error(f"CUDA error: {e}. Attempting to run on CPU instead.")
            # Modify the environment to force CPU usage and retry
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return run_torch()  # Recursive call will use CPU now
        else:
            logging.error(f"Error in PyTorch computation: {e}")
            raise
    except Exception as e:
        logging.error(f"Unexpected error in PyTorch computation: {e}")
        raise

def run_torch_with_explicit_q():
    """Run PyTorch version of the diffuse scattering simulation with explicit q-vectors."""
    # Add clear warning about sampling parameters interpretation
    logging.warning("IMPORTANT: When using explicit q-vectors, they must match what would be generated by map_utils.generate_grid()")
    logging.warning("Grid size formula: int(sampling[2] * (sampling[1] - sampling[0]) + 1)")
    logging.warning("For example, with [-1, 1, 3], this gives 3*(1-(-1))+1 = 7 steps per dimension")
    
    try:
        import torch
        from eryx.models_torch import OnePhonon
        
        # Get the device (use CPU for consistency with other runs)
        device = torch.device('cpu')
        logging.info(f"Starting PyTorch branch with explicit q-vectors on {device}")
        
        # Use the same sampling parameters as in run_np and run_torch
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        hsampling = [-4, 4, 3]
        ksampling = [-17, 17, 3]
        lsampling = [-29, 29, 3]
        
        # Extract q-vectors directly without phonon computation
        q_vectors = extract_q_vectors(
            pdb_path=pdb_path,
            hsampling=hsampling,
            ksampling=ksampling,
            lsampling=lsampling,
            device=device
        )
        
        # Save exact q-vectors for comparison
        np.save("torch_explicit_q_vectors.npy", q_vectors.detach().cpu().numpy())
        
        # Use the EXACT same parameters as the grid-based mode
        gnm_cutoff = 4.0
        gamma_intra = 1.0
        gamma_inter = 1.0
        
        # Create OnePhonon instance with explicit q-vectors
        onephonon_torch = OnePhonon(
            pdb_path=pdb_path,
            q_vectors=q_vectors,  # Pass extracted q-vectors
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=gnm_cutoff,
            gamma_intra=gamma_intra,
            gamma_inter=gamma_inter,
            device=device
        )
        
        # Apply disorder
        Id_torch = onephonon_torch.apply_disorder(use_data_adp=True)
        
        # Log debug information
        logging.debug(f"PyTorch with explicit q: q_vectors shape = {q_vectors.shape}")
        logging.debug(f"PyTorch with explicit q: q_grid shape = {onephonon_torch.q_grid.shape}")
        logging.debug(f"PyTorch with explicit q: q_grid range: min = {onephonon_torch.q_grid.min().item()}, max = {onephonon_torch.q_grid.max().item()}")
        # Handle NaN values properly for PyTorch versions that don't have torch.nanmin/nanmax
        valid_values = Id_torch[~torch.isnan(Id_torch)]
        min_val = valid_values.min().item() if valid_values.numel() > 0 else float('nan')
        max_val = valid_values.max().item() if valid_values.numel() > 0 else float('nan')
        logging.info(f"PyTorch explicit-q branch diffuse intensity stats: min={min_val}, max={max_val}")
        
        # Save for later comparison
        torch.save(Id_torch, "torch_explicit_q_diffuse_intensity.pt")
        # Also save as NumPy array for easier comparison
        np.save("torch_explicit_q_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())
        
        # Also save q-vectors and mask for visualization
        from eryx.visualization import save_q_vectors
        save_q_vectors(
            q_vectors=onephonon_torch.q_grid,
            mask=onephonon_torch.res_mask,
            output_dir=".",
            q_vectors_file="explicit_q_vectors.npy",
            mask_file="explicit_resolution_mask.npy"
        )
        
        return Id_torch
        
    except RuntimeError as e:
        # Handle CUDA errors by falling back to CPU
        if 'CUDA' in str(e):
            logging.error(f"CUDA error with explicit q-vectors: {e}. Attempting to run on CPU instead.")
            # Modify the environment to force CPU usage and retry
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return run_torch_with_explicit_q()  # Recursive call will use CPU now
        else:
            logging.error(f"Error in PyTorch computation with explicit q-vectors: {e}")
            raise

def extract_q_vectors(pdb_path, hsampling, ksampling, lsampling, device=None):
    """
    Extract q-vectors directly using the same approach as OnePhonon's grid-based implementation.
    This ensures exact consistency between grid-based and explicit q-vector modes.
    
    Args:
        pdb_path: Path to the PDB file
        hsampling, ksampling, lsampling: Sampling parameters
        device: PyTorch device (default: CPU)
        
    Returns:
        q_vectors: PyTorch tensor of q-vectors
    """
    # Import here to avoid circular imports
    import torch
    import numpy as np
    import logging
    
    if device is None:
        device = torch.device('cpu')
    
    # Load the model to get cell parameters
    from eryx.pdb import AtomicModel
    model = AtomicModel(pdb_path, expand_p1=True)
    
    # Use the EXACT same approach as OnePhonon's grid-based implementation
    # h_dim = int(self.hsampling[2]) in OnePhonon, not the formula from map_utils
    h_dim = int(hsampling[2])
    k_dim = int(ksampling[2])
    l_dim = int(lsampling[2])
    total_points = h_dim * k_dim * l_dim
    
    logging.info(f"Creating q-vectors grid with dimensions {h_dim}x{k_dim}x{l_dim} = {total_points} points")
    
    # Create linspace for each dimension
    h_grid = np.linspace(hsampling[0], hsampling[1], h_dim)
    k_grid = np.linspace(ksampling[0], ksampling[1], k_dim)
    l_grid = np.linspace(lsampling[0], lsampling[1], l_dim)
    
    # Create meshgrid - using indexing='ij' to match NumPy's default behavior
    h_mesh, k_mesh, l_mesh = np.meshgrid(h_grid, k_grid, l_grid, indexing='ij')
    
    # Reshape and stack
    hkl_grid = np.stack([h_mesh.flatten(), k_mesh.flatten(), l_mesh.flatten()], axis=1)
    
    # Convert to tensor
    hkl_grid_tensor = torch.tensor(hkl_grid, dtype=torch.float32, device=device)
    
    # Compute q-grid directly: q_grid = 2π * A_inv^T * hkl_grid^T
    A_inv_tensor = torch.tensor(model.A_inv, dtype=torch.float32, device=device)
    q_vectors = 2 * torch.pi * torch.matmul(A_inv_tensor.T, hkl_grid_tensor.T).T
    
    # Log information about the extracted q-vectors
    logging.info(f"Extracted {q_vectors.shape[0]} q-vectors matching grid-based dimensions")
    logging.info(f"q-vector range: [{q_vectors.min().item():.2f}, {q_vectors.max().item():.2f}]")
    
    return q_vectors
def validate_q_vector_consistency():
    """Validate that all three simulation modes use consistent q-vectors and produce comparable results."""
    try:
        import torch
        import numpy as np
        from scipy.stats import pearsonr
        
        logging.info("Validating q-vector consistency across simulation modes...")
        
        # Check if all required files exist
        required_files = [
            "np_diffuse_intensity.npy",
            "torch_diffuse_intensity.npy",
            "torch_explicit_q_diffuse_intensity.npy"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            logging.error(f"Missing required files for validation: {missing_files}")
            logging.error("Please run all simulation modes first with --run-mode all")
            return False
        
        # Load results
        np_result = np.load("np_diffuse_intensity.npy")
        torch_grid_result = np.load("torch_diffuse_intensity.npy")
        torch_explicit_result = np.load("torch_explicit_q_diffuse_intensity.npy")
        
        # Check q-vector consistency by loading the raw tensors
        q_vector_files = {
            "grid": "q_vectors.npy",  # From save_simulation_outputs in run_torch
            "explicit": "torch_explicit_q_vectors.npy"  # From run_torch_with_explicit_q
        }
        
        if all(os.path.exists(f) for f in q_vector_files.values()):
            grid_q_vectors = np.load(q_vector_files["grid"])
            explicit_q_vectors = np.load(q_vector_files["explicit"])
            
            # Check if shapes match
            logging.info(f"Grid q-vectors shape: {grid_q_vectors.shape}, Explicit q-vectors shape: {explicit_q_vectors.shape}")
            
            # Compute absolute difference between vectors
            if grid_q_vectors.shape == explicit_q_vectors.shape:
                q_vector_diff = np.abs(grid_q_vectors - explicit_q_vectors)
                max_diff = np.max(q_vector_diff)
                mean_diff = np.mean(q_vector_diff)
                logging.info(f"Q-vector max difference: {max_diff:.8e}, mean difference: {mean_diff:.8e}")
                
                # Check if close enough (should be nearly identical)
                q_vectors_match = max_diff < 1e-5
                logging.info(f"Q-vectors are {'identical' if q_vectors_match else 'DIFFERENT'}")
                
                # Print a few examples
                logging.info("Sample q-vectors comparison (first 3 points):")
                for i in range(min(3, len(grid_q_vectors))):
                    logging.info(f"  Point {i}:")
                    logging.info(f"    Grid: {grid_q_vectors[i]}")
                    logging.info(f"    Explicit: {explicit_q_vectors[i]}")
                    logging.info(f"    Diff: {q_vector_diff[i]}")
            else:
                logging.error("Q-vector shapes do not match, cannot compare directly")
                q_vectors_match = False
        else:
            logging.warning("Q-vector files not found, cannot validate q-vector consistency")
            q_vectors_match = None
        
        # Create masks for non-NaN values in all arrays
        valid_mask_np = ~np.isnan(np_result)
        valid_mask_torch_grid = ~np.isnan(torch_grid_result)
        valid_mask_torch_explicit = ~np.isnan(torch_explicit_result)
        
        # Get common valid mask
        common_valid_mask = valid_mask_np & valid_mask_torch_grid & valid_mask_torch_explicit
        valid_count = np.sum(common_valid_mask)
        
        logging.info(f"Number of common valid points across all modes: {valid_count}")
        
        if valid_count == 0:
            logging.warning("No common valid points found across all modes!")
            return False
        
        # Extract valid values
        np_values = np_result[common_valid_mask]
        torch_grid_values = torch_grid_result[common_valid_mask]
        torch_explicit_values = torch_explicit_result[common_valid_mask]
        
        # Print some example values for direct comparison
        logging.info("Sample intensity values (first 5 common valid points):")
        sample_indices = np.where(common_valid_mask)[0][:5]
        for idx, i in enumerate(sample_indices):
            logging.info(f"  Point {idx} (index {i}):")
            logging.info(f"    NumPy: {np_result.flat[i]:.4f}")
            logging.info(f"    PyTorch Grid: {torch_grid_result.flat[i]:.4f}")
            logging.info(f"    PyTorch Explicit: {torch_explicit_result.flat[i]:.4f}")
            if torch_grid_result.flat[i] != 0:
                ratio = torch_explicit_result.flat[i] / torch_grid_result.flat[i]
                logging.info(f"    Explicit/Grid Ratio: {ratio:.4f}")
        
        # Compute correlation between PyTorch grid and explicit modes
        corr_grid_explicit, _ = pearsonr(torch_grid_values, torch_explicit_values)
        
        # Compute relative difference
        rel_diff = np.mean(np.abs(torch_grid_values - torch_explicit_values) / 
                          np.maximum(np.abs(torch_grid_values), 1e-10)) * 100
        
        logging.info(f"Correlation between PyTorch grid and explicit modes: {corr_grid_explicit:.6f}")
        logging.info(f"Mean relative difference: {rel_diff:.4f}%")
        
        # Check if correlation is close to 1.0 (perfect correlation)
        is_consistent = corr_grid_explicit > 0.99 and rel_diff < 1.0
        
        if is_consistent:
            logging.info("VALIDATION PASSED: q-vector consistency confirmed across simulation modes")
        else:
            logging.warning("VALIDATION FAILED: q-vector consistency issues detected")
            
            # Additional diagnostic: Distribution comparison
            logging.info("Statistical comparison:")
            for name, values in [("NumPy", np_values), ("PyTorch Grid", torch_grid_values), 
                                ("PyTorch Explicit", torch_explicit_values)]:
                logging.info(f"  {name}: min={np.min(values):.4f}, max={np.max(values):.4f}, "
                           f"mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        return is_consistent
        
    except Exception as e:
        logging.error(f"Error validating q-vector consistency: {e}")
        return False

if __name__ == "__main__":
    setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser(description='Run diffuse scattering simulations')
    parser.add_argument('--run-mode', choices=['all', 'np', 'torch', 'torch-explicit-q'], default='all',
                       help='Specify which implementation to run (default: all)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate q-vector consistency across modes')
    args = parser.parse_args()
    
    # Run the specified implementation(s)
    if args.run_mode in ['all', 'np']:
        run_np()
    
    if args.run_mode in ['all', 'torch']:
        run_torch()
    
    if args.run_mode in ['all', 'torch-explicit-q']:
        run_torch_with_explicit_q()
    
    # Run validation if requested
    if args.validate:
        # If not running all modes but validation is requested, check if files exist
        if args.run_mode != 'all':
            logging.warning("Validation requested but not all modes were run.")
            logging.warning("Validation will proceed with existing output files.")
        
        validate_q_vector_consistency()
    
    logging.info("Completed debug run. Please check debug_output.log and the generated .npy files")

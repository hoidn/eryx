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
        onephonon_torch = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
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
        
        # Save for later comparison
        torch.save(Id_torch, "torch_diffuse_intensity.pt")
        # Also save as NumPy array for easier comparison
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
        
        # Extract q-vectors from grid-based approach
        q_vectors = extract_q_vectors(
            pdb_path=pdb_path,
            hsampling=hsampling,
            ksampling=ksampling,
            lsampling=lsampling,
            device=device
        )
        
        # Create OnePhonon instance with explicit q-vectors
        onephonon_torch = OnePhonon(
            pdb_path,
            q_vectors=q_vectors,  # Pass extracted q-vectors
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
        logging.debug(f"PyTorch with explicit q: q_vectors shape = {q_vectors.shape}")
        logging.debug(f"PyTorch with explicit q: q_grid shape = {onephonon_torch.q_grid.shape}")
        logging.debug(f"PyTorch with explicit q: q_grid range: min = {onephonon_torch.q_grid.min().item()}, max = {onephonon_torch.q_grid.max().item()}")
        
        # Save for later comparison
        torch.save(Id_torch, "torch_explicit_q_diffuse_intensity.pt")
        # Also save as NumPy array for easier comparison
        np.save("torch_explicit_q_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())
        
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
    Extract q-vectors from a grid-based model without computing phonons.
    
    Args:
        pdb_path: Path to the PDB file
        hsampling, ksampling, lsampling: Sampling parameters
        device: PyTorch device (default: CPU)
        
    Returns:
        q_vectors: PyTorch tensor of q-vectors
    """
    # Import here to avoid circular imports
    import torch
    from eryx.models_torch import OnePhonon
    
    if device is None:
        device = torch.device('cpu')
    
    # Create OnePhonon model with params to skip phonon computation
    model = OnePhonon(
        pdb_path=pdb_path,
        hsampling=hsampling,
        ksampling=ksampling,
        lsampling=lsampling,
        expand_p1=True,
        # Skip phonon computation by using these params
        model="gnm",  # Use GNM as it's faster
        res_limit=0.0,
        gnm_cutoff=0.0,  # Set to zero to minimize computation
        gamma_intra=0.0,
        gamma_inter=0.0,
        device=device
    )
    
    # Get q_grid and make a copy to ensure it's detached
    q_vectors = model.q_grid.clone().detach()
    
    # Log information about the extracted q-vectors
    logging.info(f"Extracted {q_vectors.shape[0]} q-vectors from grid-based model")
    logging.info(f"q-vector range: [{q_vectors.min().item():.2f}, {q_vectors.max().item():.2f}]")
    
    return q_vectors
def validate_q_vector_consistency():
    """Validate that all three simulation modes use consistent q-vectors and produce comparable results."""
    try:
        import torch
        import numpy as np
        from scipy.stats import pearsonr
        
        logging.info("Validating q-vector consistency across simulation modes...")
        
        # Load results
        np_result = np.load("np_diffuse_intensity.npy")
        torch_grid_result = np.load("torch_diffuse_intensity.npy")
        torch_explicit_result = np.load("torch_explicit_q_diffuse_intensity.npy")
        
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
    if args.validate and args.run_mode == 'all':
        validate_q_vector_consistency()
    
    logging.info("Completed debug run. Please check debug_output.log and the generated .npy files")

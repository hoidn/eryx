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
    """Run PyTorch version with explicit q-vectors extracted from the grid-based approach."""
    try:
        import torch
        from eryx.models_torch import OnePhonon
        
        # Get the device
        device = torch.device('cpu')
        logging.info(f"Starting PyTorch branch with explicit q-vectors on {device}")
        
        # First, run the grid-based approach to extract q-vectors
        logging.info("Running grid-based approach first to extract q-vectors")
        grid_onephonon = OnePhonon(
            "tests/pdbs/5zck_p1.pdb",
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Extract q-vectors from the grid-based approach
        q_vectors = grid_onephonon.q_grid.clone().detach()
        logging.info(f"Extracted {q_vectors.shape[0]} q-vectors from grid-based approach")
        
        # Save q-vectors for visualization
        np.save("grid_q_vectors.npy", q_vectors.cpu().numpy())
        
        # Now run with explicit q-vectors
        logging.info("Running with explicit q-vectors")
        explicit_onephonon = OnePhonon(
            "tests/pdbs/5zck_p1.pdb",
            q_vectors=q_vectors,
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Apply disorder using the explicit q-vectors
        Id_explicit = explicit_onephonon.apply_disorder(use_data_adp=True)
        
        # Log debug information
        logging.debug(f"PyTorch with explicit q: q_vectors shape = {q_vectors.shape}")
        logging.debug(f"PyTorch with explicit q: q_grid shape = {explicit_onephonon.q_grid.shape}")
        logging.debug(f"PyTorch with explicit q: q_grid min = {explicit_onephonon.q_grid.min().item()}, max = {explicit_onephonon.q_grid.max().item()}")
        
        # Save results
        torch.save(Id_explicit, "torch_explicit_q_diffuse_intensity.pt")
        np.save("torch_explicit_q_diffuse_intensity.npy", Id_explicit.detach().cpu().numpy())
        
        # Run grid-based approach for comparison
        Id_grid = grid_onephonon.apply_disorder(use_data_adp=True)
        
        # Compare results
        grid_np = Id_grid.detach().cpu().numpy()
        explicit_np = Id_explicit.detach().cpu().numpy()
        
        # Calculate differences
        abs_diff = np.abs(grid_np - explicit_np)
        rel_diff = abs_diff / (np.abs(grid_np) + 1e-10)  # Avoid division by zero
        
        # Log comparison statistics
        logging.info("Comparison between grid-based and explicit q-vector approaches:")
        logging.info(f"  Mean absolute difference: {np.nanmean(abs_diff)}")
        logging.info(f"  Max absolute difference: {np.nanmax(abs_diff)}")
        logging.info(f"  Mean relative difference: {np.nanmean(rel_diff) * 100:.6f}%")
        logging.info(f"  Identical outputs: {np.allclose(grid_np, explicit_np, equal_nan=True)}")
        
        return Id_explicit
        
    except RuntimeError as e:
        # Handle CUDA errors by falling back to CPU
        if 'CUDA' in str(e):
            logging.error(f"CUDA error with explicit q-vectors: {e}. Attempting to run on CPU instead.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return run_torch_with_explicit_q()
        else:
            logging.error(f"Error in PyTorch computation with explicit q-vectors: {e}")
            raise

if __name__ == "__main__":
    setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser(description='Run diffuse scattering simulations')
    parser.add_argument('--run-mode', choices=['all', 'np', 'torch', 'torch-explicit-q'], default='all',
                       help='Specify which implementation to run (default: all)')
    args = parser.parse_args()
    
    # Run the specified implementation(s)
    if args.run_mode in ['all', 'np']:
        run_np()
    
    if args.run_mode in ['all', 'torch']:
        run_torch()
    
    if args.run_mode in ['all', 'torch-explicit-q']:
        run_torch_with_explicit_q()
    
    logging.info("Completed debug run. Please check debug_output.log and the generated .npy files")

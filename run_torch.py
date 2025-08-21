#!/usr/bin/env python3
import os
import logging
import numpy as np
import torch
import argparse

def setup_logging():
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="torch_output.log",
        filemode="w"
    )
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_torch(device=None):
    """Run PyTorch version of the diffuse scattering simulation."""
    try:
        from eryx.models_torch import OnePhonon
        
        # Get the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
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
        
        # --- Save q-vectors and intensity to NPZ ---
        q_vectors_torch = onephonon_torch.q_grid.detach().cpu().numpy()
        intensity_torch_np = Id_torch.detach().cpu().numpy()

        output_filename = "torch_grid_results.npz"
        try:
            np.savez_compressed(output_filename,
                                q_vectors=q_vectors_torch,
                                intensity=intensity_torch_np,
                                map_shape=onephonon_torch.map_shape)  # Save map_shape too
            logging.info(f"PyTorch Grid: Saved q-vectors ({q_vectors_torch.shape}) and intensity ({intensity_torch_np.shape}) to {output_filename}")
            logging.info("PyTorch Grid diffuse intensity stats: min=%s, max=%s", 
                         np.nanmin(intensity_torch_np), np.nanmax(intensity_torch_np))
        except Exception as e:
            logging.error(f"PyTorch Grid: Failed to save results to {output_filename}: {e}")
        
        # Also save original files for backward compatibility
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

def run_torch_arb_q(device=None):
    """Run PyTorch version in arbitrary q-vector mode."""
    try:
        from eryx.models_torch import OnePhonon
        
        # Get the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        logging.info(f"Starting PyTorch arbitrary q-vector mode computation on {device}")
        
        # First, run grid mode to extract q vectors
        logging.info("Creating grid model to extract q-vectors...")
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        grid_model = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        
        # Extract q-vectors and map shape
        q_vectors = grid_model.q_grid.clone().detach()
        map_shape = grid_model.map_shape
        hkl_grid = grid_model.hkl_grid.clone().detach()
        logging.info(f"Extracted q-vectors shape: {q_vectors.shape}")
        logging.info(f"Map shape: {map_shape}")
        
        # Save memory by deleting the grid model
        del grid_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # Now create the arbitrary q-vector model
        logging.info("Creating arbitrary q-vector model...")
        # Pass sampling parameters explicitly as they are required for ADP calculation in GNM model
        hsampling_params = [-4, 4, 3]
        ksampling_params = [-17, 17, 3]
        lsampling_params = [-29, 29, 3]
        arb_q_model = OnePhonon(
            pdb_path,
            q_vectors=q_vectors,
            hsampling=hsampling_params, # Pass sampling params
            ksampling=ksampling_params,
            lsampling=lsampling_params,
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        logging.info("Arbitrary q-vector model initialized (phonons/ADPs calculated automatically).")

        # Verify phonon tensors exist and contain non-zero values
        if hasattr(arb_q_model, 'V') and hasattr(arb_q_model, 'Winv') and arb_q_model.V is not None and arb_q_model.Winv is not None:
            v_nonzero = torch.count_nonzero(torch.abs(arb_q_model.V)).item()
            winv_nonzero = torch.count_nonzero(~torch.isnan(arb_q_model.Winv)).item() if arb_q_model.Winv is not None else 0
            
            logging.info(f"Phonon eigenvectors (V) shape: {arb_q_model.V.shape}")
            logging.info(f"Phonon eigenvalues (Winv) shape: {arb_q_model.Winv.shape}")
            logging.info(f"Non-zero elements - V: {v_nonzero}, Winv: {winv_nonzero}")
        else:
            logging.warning("Phonon tensors (V and/or Winv) are missing.")
        
        # Apply disorder to get intensity
        logging.info("Computing diffuse intensity...")
        Id_arb_q = arb_q_model.apply_disorder(use_data_adp=True)
        
        # Log stats and save results
        valid_intensity = Id_arb_q[~torch.isnan(Id_arb_q)]
        if valid_intensity.numel() > 0:
            min_intensity = valid_intensity.min().item()
            max_intensity = valid_intensity.max().item()
            mean_intensity = valid_intensity.mean().item()
            logging.info(f"Arb-Q diffuse intensity stats: min={min_intensity}, max={max_intensity}, mean={mean_intensity}")
        else:
            logging.warning("All intensity values are NaN!")
        
        # --- Save q-vectors and intensity to NPZ ---
        q_vectors_arbq_np = q_vectors.detach().cpu().numpy()
        intensity_arbq_np = Id_arb_q.detach().cpu().numpy()

        output_filename = "torch_arbq_results.npz"
        try:
            np.savez_compressed(output_filename,
                                q_vectors=q_vectors_arbq_np,
                                intensity=intensity_arbq_np,
                                map_shape=map_shape)  # Save the map_shape derived from grid
            logging.info(f"Arb-Q: Saved q-vectors ({q_vectors_arbq_np.shape}) and intensity ({intensity_arbq_np.shape}) to {output_filename}")
            
            # Log stats from flat data
            valid_intensity_np = intensity_arbq_np[~np.isnan(intensity_arbq_np)]
            if valid_intensity_np.size > 0:
                min_intensity = np.min(valid_intensity_np)
                max_intensity = np.max(valid_intensity_np)
                mean_intensity = np.mean(valid_intensity_np)
                logging.info(f"Arb-Q flat intensity stats: min={min_intensity:.4f}, max={max_intensity:.4f}, mean={mean_intensity:.4f}")
            else:
                logging.warning("Arb-Q: All flat intensity values are NaN!")
        except Exception as e:
            logging.error(f"Arb-Q: Failed to save results to {output_filename}: {e}")
        
        # Also save original files for backward compatibility
        torch.save(Id_arb_q, "arb_q_diffuse_intensity.pt")
        np.save("arb_q_diffuse_intensity.npy", Id_arb_q.detach().cpu().numpy())
        
        return Id_arb_q, map_shape
        
    except Exception as e:
        logging.error(f"Error in arbitrary q-vector mode: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyTorch diffuse scattering simulation")
    parser.add_argument("--arb-q", action="store_true", 
                       help="Run arbitrary q-vector mode instead of grid mode")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
    args = parser.parse_args()
    
    setup_logging()
    
    # Handle device selection
    if args.device == "auto":
        device = None  # Let functions auto-detect
    else:
        device = args.device
    
    if args.arb_q:
        logging.info("Running PyTorch arbitrary q-vector mode...")
        arb_q_result, map_shape = run_torch_arb_q(device=device)
        logging.info("Completed PyTorch arbitrary q-vector run")
    else:
        logging.info("Running PyTorch grid mode...")
        grid_result = run_torch(device=device)
        logging.info("Completed PyTorch grid run")
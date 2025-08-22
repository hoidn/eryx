#!/usr/bin/env python3
"""Run the optimized PyTorch implementation with same config as run_torch.py"""

import os
import logging
import numpy as np
import torch
from eryx.models_torch_optimized import OnePhononOptimized

def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="torch_optimized_output.log",
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_torch_optimized(device=None):
    """Run optimized PyTorch version."""
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        logging.info(f"Starting Optimized PyTorch computation on {device}")
        
        # Use exact same parameters as run_torch.py
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        onephonon_opt = OnePhononOptimized(
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
        Id_opt = onephonon_opt.apply_disorder(use_data_adp=True)
        
        # Log debug information
        logging.debug(f"Optimized: hkl_grid shape = {onephonon_opt.hkl_grid.shape}")
        logging.debug("Optimized: hkl_grid coordinate ranges:")
        logging.debug(f"  Dimension 0: min = {onephonon_opt.hkl_grid[:,0].min().item()}, max = {onephonon_opt.hkl_grid[:,0].max().item()}")
        logging.debug(f"  Dimension 1: min = {onephonon_opt.hkl_grid[:,1].min().item()}, max = {onephonon_opt.hkl_grid[:,1].max().item()}")
        logging.debug(f"  Dimension 2: min = {onephonon_opt.hkl_grid[:,2].min().item()}, max = {onephonon_opt.hkl_grid[:,2].max().item()}")
        logging.debug(f"Optimized: q_grid range: min = {onephonon_opt.q_grid.min().item()}, max = {onephonon_opt.q_grid.max().item()}")
        
        # Save results
        q_vectors_opt = onephonon_opt.q_grid.detach().cpu().numpy()
        intensity_opt_np = Id_opt.detach().cpu().numpy()

        output_filename = "torch_optimized_results.npz"
        try:
            np.savez_compressed(output_filename,
                                q_vectors=q_vectors_opt,
                                intensity=intensity_opt_np,
                                map_shape=onephonon_opt.map_shape)
            logging.info(f"Optimized: Saved q-vectors ({q_vectors_opt.shape}) and intensity ({intensity_opt_np.shape}) to {output_filename}")
            logging.info("Optimized diffuse intensity stats: min=%s, max=%s", 
                         np.nanmin(intensity_opt_np), np.nanmax(intensity_opt_np))
        except Exception as e:
            logging.error(f"Optimized: Failed to save results to {output_filename}: {e}")
        
        return Id_opt
        
    except Exception as e:
        logging.error(f"Error in Optimized computation: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    logging.info("Running Optimized PyTorch grid mode...")
    result = run_torch_optimized()
    logging.info("Completed Optimized PyTorch run")

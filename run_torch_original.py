#!/usr/bin/env python3
"""Run the ORIGINAL PyTorch implementation for comparison."""

import os
import logging
import numpy as np
import torch
import time

def setup_logging(filename):
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename=filename,
        filemode="w"
    )
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_original():
    """Run original PyTorch implementation."""
    from eryx.models_torch import OnePhonon
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Starting ORIGINAL PyTorch implementation on {device}")
    
    # Use the same parameters as run_torch.py
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Use smaller grid for faster testing but still meaningful
    # Original: [-4, 4, 3], [-17, 17, 3], [-29, 29, 3] = 450,625 points (too large!)
    # Use oversampling=2 to avoid BZ dimension bug with oversampling=1
    # [-2, 2, 2], [-8, 8, 2], [-14, 14, 2] = 9×33×57 = 16,929 points
    hsampling = [-2, 2, 2]
    ksampling = [-8, 8, 2]
    lsampling = [-14, 14, 2]
    
    h_pts = (hsampling[1] - hsampling[0]) * hsampling[2] + 1
    k_pts = (ksampling[1] - ksampling[0]) * ksampling[2] + 1
    l_pts = (lsampling[1] - lsampling[0]) * lsampling[2] + 1
    total_pts = h_pts * k_pts * l_pts
    
    logging.info(f"Grid size: {h_pts:.0f}×{k_pts:.0f}×{l_pts:.0f} = {total_pts:,.0f} points")
    
    # Time the entire calculation
    start_time = time.perf_counter()
    
    # Initialize model
    t_init_start = time.perf_counter()
    model = OnePhonon(
        pdb_path,
        hsampling, ksampling, lsampling,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    t_init = time.perf_counter() - t_init_start
    logging.info(f"Initialization took {t_init:.2f}s")
    
    # Compute phonons
    t_phonon_start = time.perf_counter()
    model.compute_gnm_phonons()
    t_phonon = time.perf_counter() - t_phonon_start
    logging.info(f"Phonon computation took {t_phonon:.2f}s")
    
    # Compute covariance
    t_covar_start = time.perf_counter()
    model.compute_covariance_matrix()
    t_covar = time.perf_counter() - t_covar_start
    logging.info(f"Covariance computation took {t_covar:.2f}s")
    
    # Apply disorder
    t_disorder_start = time.perf_counter()
    intensity = model.apply_disorder(use_data_adp=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_disorder = time.perf_counter() - t_disorder_start
    logging.info(f"Apply disorder took {t_disorder:.2f}s")
    
    total_time = time.perf_counter() - start_time
    logging.info(f"TOTAL TIME: {total_time:.2f}s")
    
    # Save results
    intensity_np = intensity.detach().cpu().numpy()
    output_filename = "original_results.npz"
    np.savez_compressed(
        output_filename,
        intensity=intensity_np,
        map_shape=model.map_shape,
        q_vectors=model.q_grid.detach().cpu().numpy(),
        hkl_grid=model.hkl_grid.detach().cpu().numpy(),
        timing={
            't_init': t_init,
            't_phonon': t_phonon,
            't_covar': t_covar,
            't_disorder': t_disorder,
            't_total': total_time
        }
    )
    
    # Stats
    valid_intensity = intensity_np[~np.isnan(intensity_np)]
    if valid_intensity.size > 0:
        logging.info(f"Intensity stats: min={valid_intensity.min():.4e}, max={valid_intensity.max():.4e}, mean={valid_intensity.mean():.4e}")
        logging.info(f"Valid points: {valid_intensity.size}/{intensity_np.size}")
    else:
        logging.warning("All intensity values are NaN!")
    
    logging.info(f"Results saved to {output_filename}")
    
    # Memory stats
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
        logging.info(f"Peak GPU memory: {peak_memory:.2f} GB")
    
    return intensity

if __name__ == "__main__":
    setup_logging("original_run.log")
    logging.info("="*60)
    logging.info("RUNNING ORIGINAL IMPLEMENTATION")
    logging.info("="*60)
    result = run_original()
    logging.info("="*60)
    logging.info("ORIGINAL RUN COMPLETE")
    logging.info("="*60)
#!/usr/bin/env python3
import os
import logging
import numpy as np
from eryx.models import OnePhonon

def setup_logging():
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="numpy_output.log",
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
    
    # --- Save q-vectors and intensity to NPZ ---
    q_vectors_np = onephonon_np.q_grid
    intensity_np = Id_np

    output_filename = "np_results.npz"
    try:
        np.savez_compressed(output_filename,
                            q_vectors=q_vectors_np,
                            intensity=intensity_np,
                            map_shape=onephonon_np.map_shape)  # Save map_shape too
        logging.info(f"NP: Saved q-vectors ({q_vectors_np.shape}) and intensity ({intensity_np.shape}) to {output_filename}")
        logging.info("NP branch diffuse intensity stats: min=%s, max=%s", np.nanmin(intensity_np), np.nanmax(intensity_np))
    except Exception as e:
        logging.error(f"NP: Failed to save results to {output_filename}: {e}")
    
    # Also save the original .npy file for backward compatibility
    np.save("np_diffuse_intensity.npy", Id_np)

    return Id_np

if __name__ == "__main__":
    setup_logging()
    result = run_np()
    logging.info("Completed NumPy run")
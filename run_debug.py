#!/usr/bin/env python3
import logging

import numpy as np
from eryx.onephonon_torch import OnePhononTorch
from eryx.models import OnePhonon

def setup_logging():
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="debug_output.log",
        filemode="w"
    n_processes=1
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
    logging.info("Starting Torch branch computation")
    import torch
    pdb_path = "tests/pdbs/5zck.pdb"
    device = torch.device("cpu")
    onephonon_torch = OnePhononTorch(
        pdb_path,
        [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    # onephonon_torch.hkl_grid is the NP hkl_grid used internally.
    logging.debug(f"Torch: hkl_grid shape = {onephonon_torch.hkl_grid.shape}")
    logging.debug("Torch: hkl_grid coordinate ranges:")
    logging.debug(f"  Dimension 0: min = {onephonon_torch.hkl_grid[:,0].min()}, max = {onephonon_torch.hkl_grid[:,0].max()}")
    logging.debug(f"  Dimension 1: min = {onephonon_torch.hkl_grid[:,1].min()}, max = {onephonon_torch.hkl_grid[:,1].max()}")
    logging.debug(f"  Dimension 2: min = {onephonon_torch.hkl_grid[:,2].min()}, max = {onephonon_torch.hkl_grid[:,2].max()}")
    logging.debug(f"Torch: q_grid shape (torch) = {onephonon_torch.q_grid.shape}")
    logging.debug(f"Torch: q_grid range: min = {onephonon_torch.q_grid.cpu().min().item()}, max = {onephonon_torch.q_grid.cpu().max().item()}")
    Id_torch = onephonon_torch.apply_disorder()
    logging.info("Torch branch diffuse intensity stats: min=%s, max=%s", Id_torch.min().item(), Id_torch.max().item())
    np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())
    # Save for later comparison
    np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())

def main():
    setup_logging()
    run_np()
    run_torch()
    logging.info("Completed debug run. Please check debug_output.log, np_diffuse_intensity.npy and torch_diffuse_intensity.npy")

if __name__ == "__main__":
    main()
    # After setting up and importing everything, call the run routines.
    run_np()
    run_torch()
    logging.info("Completed debug run. Please check debug_output.log, np_diffuse_intensity.npy and torch_diffuse_intensity.npy")

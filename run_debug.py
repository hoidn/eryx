#!/usr/bin/env python3
import logging
import numpy as np
from eryx.models import OnePhonon   # NP version
from eryx.onephonon_torch import OnePhononTorch

def setup_logging():
    import logging
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
    sampling = [-4, 4, 3]
    onephonon_np = OnePhonon(
        pdb_path,
        sampling, sampling, sampling,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    Id_np = onephonon_np.apply_disorder(use_data_adp=True)
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
    Id_torch = onephonon_torch.apply_disorder()
    logging.info("Torch branch diffuse intensity stats: min=%s, max=%s", Id_torch.min().item(), Id_torch.max().item())
    # Save for later comparison
    np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())

if __name__ == "__main__":
    setup_logging()
    setup_logging()
    run_np()
    run_torch()
    logging.info("Completed debug run. Please check debug_output.log, np_diffuse_intensity.npy and torch_diffuse_intensity.npy")

#!/usr/bin/env python3
import os
os.environ["DEBUG_MODE"] = "1"
import logging
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
    # TODO parallel torch implementation that computes the same thing as run_np()
    pass

if __name__ == "__main__":
    setup_logging()
    import numpy as np
    from eryx.models import OnePhonon

    # After setting up and importing everything, call the run routines.
    run_np()
    run_torch()
    logging.info("Completed debug run. Please check debug_output.log, np_diffuse_intensity.npy and torch_diffuse_intensity.npy")

#!/usr/bin/env python3
"""Debug where the code is hanging."""

import torch
import sys
import time

# Add progress tracking
def progress(msg):
    print(f"[{time.time():.2f}] {msg}")
    sys.stdout.flush()

progress("Starting test...")

from eryx.models_torch import OnePhonon

progress("Imports complete")

device = torch.device('cpu')
pdb_path = "tests/pdbs/5zck_p1.pdb"

progress("Creating model...")

# Monkey patch to add debug output
original_compute_gnm = OnePhonon.compute_gnm_phonons

def debug_compute_gnm(self):
    progress("  compute_gnm_phonons called")
    try:
        result = original_compute_gnm(self)
        progress("  compute_gnm_phonons completed")
        return result
    except Exception as e:
        progress(f"  compute_gnm_phonons failed: {e}")
        raise

OnePhonon.compute_gnm_phonons = debug_compute_gnm

# Also patch _setup_phonons
original_setup = OnePhonon._setup_phonons

def debug_setup(self, *args, **kwargs):
    progress("  _setup_phonons called")
    try:
        result = original_setup(self, *args, **kwargs)
        progress("  _setup_phonons completed")
        return result
    except Exception as e:
        progress(f"  _setup_phonons failed: {e}")
        raise

OnePhonon._setup_phonons = debug_setup

try:
    model = OnePhonon(
        pdb_path,
        [-0.25, 0.25, 1],  # 2 points
        [-0.25, 0.25, 1],  # 2 points  
        [-0.25, 0.25, 1],  # 2 points
        expand_p1=True,
        res_limit=5.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    progress("Model created successfully")
except Exception as e:
    progress(f"Model creation failed: {e}")
    import traceback
    traceback.print_exc()
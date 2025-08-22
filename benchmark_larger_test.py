#!/usr/bin/env python3
"""Benchmark with larger problem sizes matching the original reports."""

import torch
import time
from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

def benchmark_full_pipeline():
    """Run the full pipeline as in run_torch.py."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Use the actual run_torch.py parameters
    h_samp = [-4, 4, 3]  # 25 points
    k_samp = [-17, 17, 3]  # 103 points  
    l_samp = [-29, 29, 3]  # 175 points
    # Total: 450,625 points
    
    print("Testing with run_torch.py parameters:")
    print(f"  Grid: 25×103×175 = 450,625 points")
    print(f"  Device: {device}")
    print()
    
    # Test original
    print("Original implementation:")
    start = time.perf_counter()
    try:
        model_orig = OnePhonon(
            pdb_path,
            h_samp, k_samp, l_samp,
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device
        )
        intensity_orig = model_orig.apply_disorder(use_data_adp=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_orig = time.perf_counter() - start
        print(f"  Total time: {time_orig:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
        time_orig = float('inf')
    
    # Clear GPU memory
    if device.type == 'cuda':
        del model_orig, intensity_orig
        torch.cuda.empty_cache()
    
    # Test optimized
    print("\nOptimized implementation:")
    start = time.perf_counter()
    try:
        model_opt = OnePhononOptimized(
            pdb_path,
            h_samp, k_samp, l_samp,
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            gamma_intra=1.0,
            gamma_inter=1.0,
            device=device,
            max_memory_gb=8.0  # Limit memory usage
        )
        intensity_opt = model_opt.apply_disorder(use_data_adp=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        time_opt = time.perf_counter() - start
        print(f"  Total time: {time_opt:.2f}s")
    except Exception as e:
        print(f"  Failed: {e}")
        time_opt = float('inf')
    
    if time_orig != float('inf') and time_opt != float('inf'):
        speedup = time_orig / time_opt
        print(f"\n**Speedup: {speedup:.2f}x**")
        print(f"Time saved: {time_orig - time_opt:.1f}s")

if __name__ == "__main__":
    print("=" * 60)
    print("FULL SCALE BENCHMARK")
    print("=" * 60)
    benchmark_full_pipeline()
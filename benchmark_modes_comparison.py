#!/usr/bin/env python3
"""Compare runtime for arbitrary q-vector and grid modes."""

import torch
import time
import numpy as np
from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

def benchmark_mode(mode='grid', use_optimized=False):
    """Benchmark a specific mode."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Parameters for testing - use run_torch.py config
    if mode == 'grid':
        # Grid mode: same as run_torch.py
        h_samp = [-4, 4, 3]  # 25 points
        k_samp = [-17, 17, 3]  # 103 points
        l_samp = [-29, 29, 3]  # 175 points
        kwargs = {
            'hsampling': h_samp,
            'ksampling': k_samp,
            'lsampling': l_samp,
        }
        n_points = 25 * 103 * 175  # 450,625 points
    else:  # arbitrary
        # First create a grid model to get q-vectors
        grid_model = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            res_limit=0.0,
            gnm_cutoff=4.0,
            device=device
        )
        # Use same number of q-vectors as grid mode for fair comparison
        q_vectors = grid_model.q_grid.clone().detach()  # Use all q-vectors
        del grid_model
        
        kwargs = {
            'q_vectors': q_vectors,
            'hsampling': [-4, 4, 3],  # Still needed for ADP
            'ksampling': [-17, 17, 3],
            'lsampling': [-29, 29, 3],
        }
        n_points = q_vectors.shape[0]  # Same as grid
    
    # Choose implementation
    ModelClass = OnePhononOptimized if use_optimized else OnePhonon
    
    # Create model and time it
    start = time.perf_counter()
    
    model = ModelClass(
        pdb_path,
        **kwargs,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    # Compute phonons
    model.compute_gnm_phonons()
    model.compute_covariance_matrix()
    
    # Apply disorder
    intensity = model.apply_disorder(use_data_adp=True)
    
    # Ensure computation is complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start
    
    # Count valid points
    valid_points = torch.sum(~torch.isnan(intensity)).item()
    
    return total_time, valid_points, n_points

def main():
    print("=" * 70)
    print("PyTorch Mode Performance Comparison")
    print("=" * 70)
    print()
    
    results = {}
    
    # Test each combination
    for mode in ['grid', 'arbitrary']:
        print(f"\n{mode.upper()} MODE:")
        print("-" * 40)
        
        # Original implementation
        print(f"Running original implementation...")
        time_orig, valid_orig, n_pts = benchmark_mode(mode, use_optimized=False)
        print(f"  Time: {time_orig:.3f}s")
        print(f"  Valid points: {valid_orig}/{n_pts}")
        
        # Optimized implementation
        print(f"Running optimized implementation...")
        time_opt, valid_opt, n_pts = benchmark_mode(mode, use_optimized=True)
        print(f"  Time: {time_opt:.3f}s")
        print(f"  Valid points: {valid_opt}/{n_pts}")
        
        # Calculate speedup
        speedup = time_orig / time_opt
        print(f"\n  **Speedup: {speedup:.2f}x**")
        print(f"  Time saved: {time_orig - time_opt:.3f}s ({100*(1-1/speedup):.1f}% reduction)")
        
        results[mode] = {
            'original': time_orig,
            'optimized': time_opt,
            'speedup': speedup
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Mode':<15} {'Original':<12} {'Optimized':<12} {'Speedup':<10}")
    print("-" * 50)
    for mode in ['grid', 'arbitrary']:
        r = results[mode]
        print(f"{mode:<15} {r['original']:<12.3f} {r['optimized']:<12.3f} {r['speedup']:<10.2f}x")

if __name__ == "__main__":
    main()
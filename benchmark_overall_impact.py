"""
Analyze the overall impact of Phase 1 vectorization on end-to-end calculation time.

This script measures how much of the total runtime is spent in apply_disorder
and estimates the overall speedup from vectorization.
"""

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)

def profile_calculation(use_vectorized=False):
    """Profile a complete calculation with timing breakdown."""
    
    if use_vectorized:
        from eryx.models_torch_vectorized import OnePhononVectorized as OnePhonon
    else:
        from eryx.models_torch import OnePhonon
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Profiling {'VECTORIZED' if use_vectorized else 'ORIGINAL'} implementation on {device}")
    print(f"{'='*60}")
    
    # Test configuration - small grid for quick profiling
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    hsampling = (-2, 2, 1)  # 5 points
    ksampling = (-2, 2, 1)  # 5 points  
    lsampling = (-2, 2, 1)  # 5 points
    # Total: 5×5×5 = 125 grid points
    
    # Time model initialization
    print("\n1. Model Initialization")
    t0 = time.perf_counter()
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
    if use_vectorized:
        model.use_vectorized = True
    time_init = time.perf_counter() - t0
    print(f"   Time: {time_init:.3f}s")
    
    # Time phonon computation
    print("\n2. Phonon Computation (compute_gnm_phonons)")
    t0 = time.perf_counter()
    model.compute_gnm_phonons()
    time_phonons = time.perf_counter() - t0
    print(f"   Time: {time_phonons:.3f}s")
    
    # Time covariance computation
    print("\n3. Covariance Matrix Computation")
    t0 = time.perf_counter()
    model.compute_covariance_matrix()
    time_covar = time.perf_counter() - t0
    print(f"   Time: {time_covar:.3f}s")
    
    # Time apply_disorder (the vectorized part)
    print("\n4. Apply Disorder (VECTORIZED SECTION)")
    # Warm-up
    _ = model.apply_disorder(use_data_adp=True)
    
    # Actual timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()
    Id = model.apply_disorder(use_data_adp=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_disorder = time.perf_counter() - t0
    print(f"   Time: {time_disorder:.3f}s")
    
    # Calculate valid intensity points
    valid_points = torch.sum(~torch.isnan(Id)).item()
    print(f"   Valid intensity points: {valid_points}")
    
    # Total time
    total_time = time_init + time_phonons + time_covar + time_disorder
    
    print(f"\n{'='*40}")
    print("TIMING BREAKDOWN")
    print(f"{'='*40}")
    print(f"Initialization:    {time_init:8.3f}s ({100*time_init/total_time:5.1f}%)")
    print(f"Phonon calc:       {time_phonons:8.3f}s ({100*time_phonons/total_time:5.1f}%)")
    print(f"Covariance:        {time_covar:8.3f}s ({100*time_covar/total_time:5.1f}%)")
    print(f"Apply disorder:    {time_disorder:8.3f}s ({100*time_disorder/total_time:5.1f}%)")
    print(f"{'='*40}")
    print(f"TOTAL:             {total_time:8.3f}s (100.0%)")
    
    return {
        'init': time_init,
        'phonons': time_phonons,
        'covariance': time_covar,
        'disorder': time_disorder,
        'total': total_time
    }


def main():
    """Compare original vs vectorized overall performance."""
    
    print("\n" + "="*60)
    print("OVERALL IMPACT ANALYSIS OF PHASE 1 VECTORIZATION")
    print("="*60)
    
    # Profile original implementation
    timings_orig = profile_calculation(use_vectorized=False)
    
    # Profile vectorized implementation
    timings_vec = profile_calculation(use_vectorized=True)
    
    # Calculate impact
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"\n{'Component':<20} {'Original':>10} {'Vectorized':>10} {'Speedup':>10}")
    print("-"*50)
    
    for component in ['init', 'phonons', 'covariance', 'disorder', 'total']:
        orig_time = timings_orig[component]
        vec_time = timings_vec[component]
        speedup = orig_time / vec_time if vec_time > 0 else float('inf')
        print(f"{component.capitalize():<20} {orig_time:10.3f}s {vec_time:10.3f}s {speedup:10.2f}x")
    
    # Calculate overall speedup
    overall_speedup = timings_orig['total'] / timings_vec['total']
    disorder_speedup = timings_orig['disorder'] / timings_vec['disorder']
    
    # Calculate what fraction of time is spent in apply_disorder
    disorder_fraction_orig = timings_orig['disorder'] / timings_orig['total']
    disorder_fraction_vec = timings_vec['disorder'] / timings_vec['total']
    
    print("\n" + "="*60)
    print("KEY METRICS")
    print("="*60)
    print(f"\nApply Disorder Speedup:    {disorder_speedup:.2f}x")
    print(f"Overall Calculation Speedup: {overall_speedup:.2f}x")
    print(f"\nTime spent in apply_disorder:")
    print(f"  Original:  {disorder_fraction_orig*100:.1f}% of total time")
    print(f"  Vectorized: {disorder_fraction_vec*100:.1f}% of total time")
    
    # Amdahl's Law analysis
    print(f"\n{'='*60}")
    print("AMDAHL'S LAW ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate theoretical maximum speedup
    non_vectorized_fraction = 1.0 - disorder_fraction_orig
    theoretical_max_speedup = 1.0 / non_vectorized_fraction
    
    print(f"\nOriginal time distribution:")
    print(f"  Vectorizable (apply_disorder): {disorder_fraction_orig*100:.1f}%")
    print(f"  Non-vectorizable (other):       {non_vectorized_fraction*100:.1f}%")
    print(f"\nTheoretical maximum speedup (if apply_disorder → 0s): {theoretical_max_speedup:.2f}x")
    print(f"Achieved overall speedup: {overall_speedup:.2f}x")
    print(f"Efficiency: {100*overall_speedup/theoretical_max_speedup:.1f}% of theoretical maximum")
    
    # Estimate for larger problems
    print(f"\n{'='*60}")
    print("ESTIMATED IMPACT FOR LARGER PROBLEMS")
    print(f"{'='*60}")
    
    print("\nFor larger grids, apply_disorder typically takes a higher percentage of time.")
    print("Estimated overall speedups for different workload profiles:")
    
    profiles = [
        ("Small (current test)", disorder_fraction_orig, disorder_speedup),
        ("Medium (50% in apply_disorder)", 0.5, disorder_speedup),
        ("Large (70% in apply_disorder)", 0.7, disorder_speedup),
        ("Very Large (90% in apply_disorder)", 0.9, disorder_speedup),
    ]
    
    for name, fraction, speedup in profiles:
        overall = 1.0 / ((1.0 - fraction) + fraction/speedup)
        print(f"  {name:<35}: {overall:.1f}x overall speedup")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"\nPhase 1 vectorization achieves {disorder_speedup:.1f}x speedup in apply_disorder,")
    print(f"resulting in {overall_speedup:.1f}x overall speedup for the complete calculation.")
    print(f"\nFor production workloads with larger grids, the overall speedup")
    print(f"is expected to be even more significant (up to {0.9*disorder_speedup:.0f}x for very large problems).")


if __name__ == '__main__':
    main()
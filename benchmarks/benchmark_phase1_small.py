"""
Small and fast benchmark suite for Phase 1 vectorization performance.

This script compares the performance of original vs vectorized implementations
with smaller problem sizes for quick testing.
"""

import torch
import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os

from eryx.models_torch import OnePhonon
from eryx.models_torch_vectorized import OnePhononVectorized


def benchmark_arbitrary_mode_small(pdb_path: str, n_points_list: List[int], 
                            device: torch.device) -> pd.DataFrame:
    """Benchmark arbitrary q-vector mode with different numbers of points."""
    results = []
    
    for n_points in n_points_list:
        print(f"\nBenchmarking arbitrary mode with {n_points} q-vectors...")
        
        # Generate random q-vectors
        q_vectors = torch.rand(n_points, 3, device=device, dtype=torch.float64) * 2.0
        
        # Original implementation
        print("  Running original implementation...")
        # Need sampling parameters even for arbitrary q-vector mode (for ADP calculation)
        hsampling = (-1, 1, 1)  # Very small grid for ADP calculation: 3x3x3
        ksampling = (-1, 1, 1)
        lsampling = (-1, 1, 1)
        model_orig = OnePhonon(pdb_path, q_vectors=q_vectors.clone(), 
                              hsampling=hsampling, ksampling=ksampling, lsampling=lsampling,
                              device=device)
        model_orig.compute_gnm_phonons()
        
        # Warm-up
        _ = model_orig.apply_disorder()
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(3):
            _ = model_orig.apply_disorder()
            torch.cuda.synchronize() if device.type == 'cuda' else None
        time_orig = (time.perf_counter() - t0) / 3
        
        # Vectorized implementation
        print("  Running vectorized implementation...")
        model_vec = OnePhononVectorized(pdb_path, q_vectors=q_vectors.clone(), 
                                       hsampling=hsampling, ksampling=ksampling, lsampling=lsampling,
                                       device=device, use_vectorized=True)
        model_vec.compute_gnm_phonons()
        
        # Warm-up
        _ = model_vec.apply_disorder()
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(3):
            _ = model_vec.apply_disorder()
            torch.cuda.synchronize() if device.type == 'cuda' else None
        time_vec = (time.perf_counter() - t0) / 3
        
        # Calculate speedup
        speedup = time_orig / time_vec if time_vec > 0 else float('inf')
        
        results.append({
            'n_points': n_points,
            'time_original': time_orig,
            'time_vectorized': time_vec,
            'speedup': speedup
        })
        
        print(f"  Original: {time_orig:.4f}s, Vectorized: {time_vec:.4f}s, Speedup: {speedup:.2f}x")
    
    return pd.DataFrame(results)


def benchmark_grid_mode_small(pdb_path: str, grid_specs: List[Tuple], 
                       device: torch.device) -> pd.DataFrame:
    """Benchmark grid mode with different grid sizes."""
    results = []
    
    for h_range, k_range, l_range, oversampling in grid_specs:
        # Calculate actual grid size
        h_points = (h_range[1] - h_range[0]) * oversampling + 1
        k_points = (k_range[1] - k_range[0]) * oversampling + 1
        l_points = (l_range[1] - l_range[0]) * oversampling + 1
        total_points = h_points * k_points * l_points
        
        print(f"\nBenchmarking grid mode: {h_points}×{k_points}×{l_points} = {total_points} points")
        
        hsampling = (h_range[0], h_range[1], oversampling)
        ksampling = (k_range[0], k_range[1], oversampling)
        lsampling = (l_range[0], l_range[1], oversampling)
        
        # Skip very large grids
        if total_points > 500:
            print(f"  Skipping - too large for small benchmark ({total_points} points)")
            continue
            
        try:
            # Original implementation
            print("  Running original implementation...")
            model_orig = OnePhonon(pdb_path, hsampling=hsampling, ksampling=ksampling,
                                 lsampling=lsampling, device=device)
            model_orig.compute_gnm_phonons()
            
            # Warm-up
            _ = model_orig.apply_disorder()
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            for _ in range(3):
                _ = model_orig.apply_disorder()
                torch.cuda.synchronize() if device.type == 'cuda' else None
            time_orig = (time.perf_counter() - t0) / 3
            
            # Vectorized implementation
            print("  Running vectorized implementation...")
            model_vec = OnePhononVectorized(pdb_path, hsampling=hsampling, ksampling=ksampling,
                                          lsampling=lsampling, device=device, use_vectorized=True)
            model_vec.compute_gnm_phonons()
            
            # Warm-up
            _ = model_vec.apply_disorder()
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            for _ in range(3):
                _ = model_vec.apply_disorder()
                torch.cuda.synchronize() if device.type == 'cuda' else None
            time_vec = (time.perf_counter() - t0) / 3
            
            # Calculate speedup
            speedup = time_orig / time_vec if time_vec > 0 else float('inf')
            
            results.append({
                'grid_size': f"{h_points}×{k_points}×{l_points}",
                'total_points': total_points,
                'time_original': time_orig,
                'time_vectorized': time_vec,
                'speedup': speedup
            })
            
            print(f"  Original: {time_orig:.4f}s, Vectorized: {time_vec:.4f}s, Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'grid_size': f"{h_points}×{k_points}×{l_points}",
                'total_points': total_points,
                'time_original': None,
                'time_vectorized': None,
                'speedup': None,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def main():
    """Run small Phase 1 benchmarks."""
    print("="*60)
    print("Phase 1 Vectorization Performance Benchmark (Small/Fast)")
    print("="*60)
    
    # Configuration
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Arbitrary mode benchmarks (smaller sizes)
    print("\n" + "-"*40)
    print("ARBITRARY Q-VECTOR MODE BENCHMARKS")
    print("-"*40)
    n_points_list = [5, 10, 20, 50]  # Reduced from original [5, 10, 20, 50, 100, 200]
    df_arbitrary = benchmark_arbitrary_mode_small(pdb_path, n_points_list, device)
    
    # Grid mode benchmarks (smaller grids)
    print("\n" + "-"*40)
    print("GRID MODE BENCHMARKS")
    print("-"*40)
    grid_specs = [
        # (h_range, k_range, l_range, oversampling)
        ((-1, 1), (-1, 1), (-1, 1), 1),  # 3×3×3 = 27 points
        ((-2, 2), (-2, 2), (-2, 2), 1),  # 5×5×5 = 125 points
        ((-1, 1), (-1, 1), (-1, 1), 2),  # 5×5×5 = 125 points
        ((-3, 3), (-3, 3), (-3, 3), 1),  # 7×7×7 = 343 points (last one within limit)
    ]
    df_grid = benchmark_grid_mode_small(pdb_path, grid_specs, device)
    
    # Summary and results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print("\nARBITRARY MODE RESULTS:")
    print(df_arbitrary.round(4))
    
    print("\nGRID MODE RESULTS:")
    print(df_grid.round(4))
    
    # Calculate average speedups
    arbitrary_speedups = df_arbitrary['speedup'].dropna()
    grid_speedups = df_grid['speedup'].dropna()
    
    if len(arbitrary_speedups) > 0:
        print(f"\nArbitrary mode average speedup: {arbitrary_speedups.mean():.2f}x")
        print(f"Arbitrary mode max speedup: {arbitrary_speedups.max():.2f}x")
    
    if len(grid_speedups) > 0:
        print(f"Grid mode average speedup: {grid_speedups.mean():.2f}x")
        print(f"Grid mode max speedup: {grid_speedups.max():.2f}x")
    
    # Save results
    os.makedirs("benchmarks/results", exist_ok=True)
    df_arbitrary.to_csv("benchmarks/results/phase1_arbitrary_small.csv", index=False)
    df_grid.to_csv("benchmarks/results/phase1_grid_small.csv", index=False)
    
    print(f"\nResults saved to benchmarks/results/")
    
    # Performance assessment
    all_speedups = pd.concat([arbitrary_speedups, grid_speedups])
    if len(all_speedups) > 0:
        avg_speedup = all_speedups.mean()
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        if avg_speedup > 10:
            print("  ✅ TARGET MET: >10x average speedup achieved!")
        elif avg_speedup > 5:
            print("  🟡 GOOD: >5x average speedup achieved")
        elif avg_speedup > 2:
            print("  🟠 MODERATE: >2x average speedup achieved")
        else:
            print("  ❌ LOW: <2x average speedup - needs improvement")


if __name__ == "__main__":
    main()
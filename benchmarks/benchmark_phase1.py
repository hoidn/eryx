"""
Benchmark suite for Phase 1 vectorization performance.

This script compares the performance of original vs vectorized implementations
across different problem sizes and modes.
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


def benchmark_arbitrary_mode(pdb_path: str, n_points_list: List[int], 
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
        hsampling = (-2, 2, 1)  # Small grid for ADP calculation
        ksampling = (-2, 2, 1)
        lsampling = (-2, 2, 1)
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
        
        speedup = time_orig / time_vec
        print(f"  Original: {time_orig:.4f}s, Vectorized: {time_vec:.4f}s, Speedup: {speedup:.2f}x")
        
        results.append({
            'mode': 'arbitrary',
            'n_points': n_points,
            'time_original': time_orig,
            'time_vectorized': time_vec,
            'speedup': speedup
        })
    
    return pd.DataFrame(results)


def benchmark_grid_mode(pdb_path: str, grid_specs: List[Tuple], 
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
        if total_points > 10000:
            print(f"  Skipping - too large for benchmark ({total_points} points)")
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
            _ = model_orig.apply_disorder()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            time_orig = time.perf_counter() - t0
            
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
            _ = model_vec.apply_disorder()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            time_vec = time.perf_counter() - t0
            
            speedup = time_orig / time_vec
            print(f"  Original: {time_orig:.4f}s, Vectorized: {time_vec:.4f}s, Speedup: {speedup:.2f}x")
            
            results.append({
                'mode': 'grid',
                'grid_dims': f"{h_points}×{k_points}×{l_points}",
                'n_points': total_points,
                'time_original': time_orig,
                'time_vectorized': time_vec,
                'speedup': speedup
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_results(df_arbitrary: pd.DataFrame, df_grid: pd.DataFrame, output_dir: str):
    """Create performance plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Arbitrary mode - Time vs Points
    if not df_arbitrary.empty:
        ax = axes[0, 0]
        ax.plot(df_arbitrary['n_points'], df_arbitrary['time_original'], 
                'o-', label='Original', linewidth=2)
        ax.plot(df_arbitrary['n_points'], df_arbitrary['time_vectorized'], 
                's-', label='Vectorized', linewidth=2)
        ax.set_xlabel('Number of Q-vectors')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Arbitrary Mode Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Arbitrary mode - Speedup
    if not df_arbitrary.empty:
        ax = axes[0, 1]
        ax.plot(df_arbitrary['n_points'], df_arbitrary['speedup'], 
                'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Q-vectors')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Arbitrary Mode Speedup')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Add speedup annotations
        for idx, row in df_arbitrary.iterrows():
            ax.annotate(f"{row['speedup']:.1f}x", 
                       (row['n_points'], row['speedup']),
                       textcoords="offset points", xytext=(0,5), ha='center')
    
    # Grid mode - Time vs Points
    if not df_grid.empty:
        ax = axes[1, 0]
        ax.plot(df_grid['n_points'], df_grid['time_original'], 
                'o-', label='Original', linewidth=2)
        ax.plot(df_grid['n_points'], df_grid['time_vectorized'], 
                's-', label='Vectorized', linewidth=2)
        ax.set_xlabel('Total Grid Points')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Grid Mode Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Grid mode - Speedup
    if not df_grid.empty:
        ax = axes[1, 1]
        ax.bar(range(len(df_grid)), df_grid['speedup'], color='green', alpha=0.7)
        ax.set_xticks(range(len(df_grid)))
        ax.set_xticklabels(df_grid['grid_dims'], rotation=45, ha='right')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Grid Mode Speedup')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Add speedup annotations
        for idx, (i, row) in enumerate(df_grid.iterrows()):
            ax.text(idx, row['speedup'] + 0.5, f"{row['speedup']:.1f}x", 
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_performance.png'), dpi=150)
    plt.show()
    print(f"\nPlot saved to {output_dir}/phase1_performance.png")


def main():
    """Run comprehensive Phase 1 benchmarks."""
    print("="*60)
    print("Phase 1 Vectorization Performance Benchmark")
    print("="*60)
    
    # Configuration
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Arbitrary mode benchmarks
    print("\n" + "-"*40)
    print("ARBITRARY Q-VECTOR MODE BENCHMARKS")
    print("-"*40)
    n_points_list = [5, 10, 20, 50, 100, 200]
    df_arbitrary = benchmark_arbitrary_mode(pdb_path, n_points_list, device)
    
    # Grid mode benchmarks
    print("\n" + "-"*40)
    print("GRID MODE BENCHMARKS")
    print("-"*40)
    grid_specs = [
        # (h_range, k_range, l_range, oversampling)
        ((-1, 1), (-1, 1), (-1, 1), 1),  # 3×3×3 = 27 points
        ((-2, 2), (-2, 2), (-2, 2), 1),  # 5×5×5 = 125 points
        ((-3, 3), (-3, 3), (-3, 3), 1),  # 7×7×7 = 343 points
        ((-2, 2), (-2, 2), (-2, 2), 2),  # 9×9×9 = 729 points
        ((-3, 3), (-3, 3), (-3, 3), 2),  # 13×13×13 = 2197 points
    ]
    df_grid = benchmark_grid_mode(pdb_path, grid_specs, device)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if not df_arbitrary.empty:
        print("\nArbitrary Mode:")
        print(f"  Average speedup: {df_arbitrary['speedup'].mean():.2f}x")
        print(f"  Min speedup: {df_arbitrary['speedup'].min():.2f}x")
        print(f"  Max speedup: {df_arbitrary['speedup'].max():.2f}x")
    
    if not df_grid.empty:
        print("\nGrid Mode:")
        print(f"  Average speedup: {df_grid['speedup'].mean():.2f}x")
        print(f"  Min speedup: {df_grid['speedup'].min():.2f}x")
        print(f"  Max speedup: {df_grid['speedup'].max():.2f}x")
    
    # Save results
    output_dir = "benchmarks/results"
    os.makedirs(output_dir, exist_ok=True)
    
    if not df_arbitrary.empty:
        df_arbitrary.to_csv(os.path.join(output_dir, 'phase1_arbitrary_mode.csv'), index=False)
    if not df_grid.empty:
        df_grid.to_csv(os.path.join(output_dir, 'phase1_grid_mode.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Create plots
    plot_results(df_arbitrary, df_grid, output_dir)
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
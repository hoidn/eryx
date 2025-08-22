"""
Final benchmark for Phase 4: Complete performance analysis.

This script measures the overall performance improvement achieved by
all four phases of vectorization combined.
"""

import torch
import numpy as np
import time
import gc
import logging
import pandas as pd
import matplotlib.pyplot as plt

from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

logging.basicConfig(level=logging.INFO)


def benchmark_complete_calculation(pdb_path: str, sampling_params: tuple, 
                                  device: torch.device, implementation: str) -> dict:
    """Benchmark a complete calculation from initialization to final intensity."""
    
    # Clear memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    results = {}
    
    # Time full pipeline
    t_total_start = time.perf_counter()
    
    # Initialization
    t_init_start = time.perf_counter()
    if implementation == 'original':
        model = OnePhonon(
            pdb_path,
            sampling_params[0], sampling_params[1], sampling_params[2],
            device=device
        )
    else:  # optimized
        model = OnePhononOptimized(
            pdb_path,
            sampling_params[0], sampling_params[1], sampling_params[2],
            device=device,
            enable_profiling=False
        )
    t_init = time.perf_counter() - t_init_start
    
    # Phonon computation
    t_phonon_start = time.perf_counter()
    model.compute_gnm_phonons()
    t_phonon = time.perf_counter() - t_phonon_start
    
    # Covariance computation
    t_covar_start = time.perf_counter()
    model.compute_covariance_matrix()
    t_covar = time.perf_counter() - t_covar_start
    
    # Apply disorder
    t_disorder_start = time.perf_counter()
    intensity = model.apply_disorder(use_data_adp=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_disorder = time.perf_counter() - t_disorder_start
    
    t_total = time.perf_counter() - t_total_start
    
    # Collect results
    results['implementation'] = implementation
    results['t_init'] = t_init
    results['t_phonon'] = t_phonon
    results['t_covar'] = t_covar
    results['t_disorder'] = t_disorder
    results['t_total'] = t_total
    results['valid_points'] = torch.sum(~torch.isnan(intensity)).item()
    
    # Memory stats if available
    if device.type == 'cuda':
        results['peak_memory_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.reset_peak_memory_stats(device)
    else:
        results['peak_memory_gb'] = 0
    
    return results


def main():
    """Run comprehensive Phase 4 benchmarks."""
    print("="*60)
    print("PHASE 4 FINAL PERFORMANCE BENCHMARK")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Test configurations
    test_configs = [
        ("Small", (-1, 1, 1), (-1, 1, 1), (-1, 1, 1)),  # 3×3×3 = 27 points
        ("Medium", (-2, 2, 1), (-2, 2, 1), (-2, 2, 1)),  # 5×5×5 = 125 points
        # ("Large", (-3, 3, 1), (-3, 3, 1), (-3, 3, 1)),  # 7×7×7 = 343 points
    ]
    
    results_list = []
    
    for config_name, h_samp, k_samp, l_samp in test_configs:
        h_pts = (h_samp[1] - h_samp[0]) * h_samp[2] + 1
        k_pts = (k_samp[1] - k_samp[0]) * k_samp[2] + 1
        l_pts = (l_samp[1] - l_samp[0]) * l_samp[2] + 1
        total_pts = h_pts * k_pts * l_pts
        
        print(f"\n{'='*40}")
        print(f"Configuration: {config_name}")
        print(f"Grid: {h_pts}×{k_pts}×{l_pts} = {total_pts} points")
        print(f"{'='*40}")
        
        # Benchmark original
        print("\nRunning ORIGINAL implementation...")
        try:
            orig_results = benchmark_complete_calculation(
                pdb_path, (h_samp, k_samp, l_samp), device, 'original'
            )
            orig_results['config'] = config_name
            orig_results['total_points'] = total_pts
            results_list.append(orig_results)
            
            print(f"  Init: {orig_results['t_init']:.3f}s")
            print(f"  Phonon: {orig_results['t_phonon']:.3f}s")
            print(f"  Covar: {orig_results['t_covar']:.3f}s")
            print(f"  Disorder: {orig_results['t_disorder']:.3f}s")
            print(f"  TOTAL: {orig_results['t_total']:.3f}s")
            print(f"  Memory: {orig_results['peak_memory_gb']:.2f} GB")
        except Exception as e:
            print(f"  Failed: {e}")
            orig_results = None
        
        # Benchmark optimized
        print("\nRunning OPTIMIZED implementation (Phase 1-4)...")
        try:
            opt_results = benchmark_complete_calculation(
                pdb_path, (h_samp, k_samp, l_samp), device, 'optimized'
            )
            opt_results['config'] = config_name
            opt_results['total_points'] = total_pts
            results_list.append(opt_results)
            
            print(f"  Init: {opt_results['t_init']:.3f}s")
            print(f"  Phonon: {opt_results['t_phonon']:.3f}s")
            print(f"  Covar: {opt_results['t_covar']:.3f}s")
            print(f"  Disorder: {opt_results['t_disorder']:.3f}s")
            print(f"  TOTAL: {opt_results['t_total']:.3f}s")
            print(f"  Memory: {opt_results['peak_memory_gb']:.2f} GB")
        except Exception as e:
            print(f"  Failed: {e}")
            opt_results = None
        
        # Calculate speedup
        if orig_results and opt_results:
            speedup = orig_results['t_total'] / opt_results['t_total']
            print(f"\n🎯 SPEEDUP: {speedup:.2f}x")
            
            # Component speedups
            print("\nComponent Speedups:")
            for component in ['init', 'phonon', 'covar', 'disorder']:
                t_orig = orig_results[f't_{component}']
                t_opt = opt_results[f't_{component}']
                if t_opt > 0:
                    comp_speedup = t_orig / t_opt
                    print(f"  {component:10s}: {comp_speedup:.2f}x "
                          f"({t_orig:.3f}s → {t_opt:.3f}s)")
    
    # Create summary DataFrame
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Calculate average speedups
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        orig_data = config_data[config_data['implementation'] == 'original']
        opt_data = config_data[config_data['implementation'] == 'optimized']
        
        if not orig_data.empty and not opt_data.empty:
            orig_time = orig_data['t_total'].values[0]
            opt_time = opt_data['t_total'].values[0]
            speedup = orig_time / opt_time
            
            print(f"\n{config} Configuration:")
            print(f"  Original: {orig_time:.3f}s")
            print(f"  Optimized: {opt_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory reduction: {(1 - opt_data['peak_memory_gb'].values[0] / orig_data['peak_memory_gb'].values[0]) * 100:.1f}%")
    
    # Save results
    df.to_csv('benchmarks/phase4_final_results.csv', index=False)
    print(f"\nResults saved to benchmarks/phase4_final_results.csv")
    
    # Create visualization
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time comparison
        ax = axes[0]
        configs = df['config'].unique()
        x = np.arange(len(configs))
        width = 0.35
        
        orig_times = [df[(df['config'] == c) & (df['implementation'] == 'original')]['t_total'].values[0]
                     for c in configs if not df[(df['config'] == c) & (df['implementation'] == 'original')].empty]
        opt_times = [df[(df['config'] == c) & (df['implementation'] == 'optimized')]['t_total'].values[0]
                    for c in configs if not df[(df['config'] == c) & (df['implementation'] == 'optimized')].empty]
        
        if orig_times and opt_times:
            ax.bar(x - width/2, orig_times, width, label='Original', color='red', alpha=0.7)
            ax.bar(x + width/2, opt_times, width, label='Optimized', color='green', alpha=0.7)
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(configs)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Speedup
        ax = axes[1]
        speedups = [orig_times[i] / opt_times[i] for i in range(len(orig_times))]
        ax.bar(configs, speedups, color='blue', alpha=0.7)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Overall Speedup (Phase 1-4)')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(speedups):
            ax.text(i, v, f'{v:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('benchmarks/phase4_performance.png', dpi=150)
        print("Plot saved to benchmarks/phase4_performance.png")
        
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    print("\n" + "="*60)
    print("PHASE 4 BENCHMARK COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
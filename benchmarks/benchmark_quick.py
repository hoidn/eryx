"""
Quick comprehensive benchmark for Phase 1 vectorization performance.
Tests multiple sizes without the initialization overhead repeatedly.
"""

import torch
import numpy as np
import time
import os

from eryx.models_torch import OnePhonon
from eryx.models_torch_vectorized import OnePhononVectorized


def test_arbitrary_mode(pdb_path: str, device: torch.device):
    """Test arbitrary q-vector mode performance."""
    print("\n" + "="*50)
    print("ARBITRARY Q-VECTOR MODE PERFORMANCE")
    print("="*50)
    
    # Test different sizes
    test_sizes = [5, 10, 20, 50]
    results = []
    
    for n_points in test_sizes:
        print(f"\nTesting {n_points} q-vectors...")
        
        # Generate random q-vectors
        q_vectors = torch.rand(n_points, 3, device=device, dtype=torch.float64) * 2.0
        
        # Small grid for ADP calculation
        hsampling = (-1, 1, 1)  # 3x3x3 = 27 points
        ksampling = (-1, 1, 1)
        lsampling = (-1, 1, 1)
        
        try:
            # Setup models (do this once per size)
            print("  Setting up models...")
            t_setup = time.perf_counter()
            
            model_orig = OnePhonon(pdb_path, q_vectors=q_vectors.clone(), 
                                  hsampling=hsampling, ksampling=ksampling, lsampling=lsampling,
                                  device=device)
            model_orig.compute_gnm_phonons()
            
            model_vec = OnePhononVectorized(pdb_path, q_vectors=q_vectors.clone(), 
                                           hsampling=hsampling, ksampling=ksampling, lsampling=lsampling,
                                           device=device, use_vectorized=True)
            model_vec.compute_gnm_phonons()
            
            setup_time = time.perf_counter() - t_setup
            print(f"    Setup time: {setup_time:.2f}s")
            
            # Warm up
            _ = model_orig.apply_disorder()
            _ = model_vec.apply_disorder()
            
            # Benchmark original (multiple runs for accuracy)
            times_orig = []
            for _ in range(3):
                torch.cuda.synchronize() if device.type == 'cuda' else None
                t0 = time.perf_counter()
                result_orig = model_orig.apply_disorder()
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times_orig.append(time.perf_counter() - t0)
            time_orig = np.mean(times_orig)
            
            # Benchmark vectorized (multiple runs for accuracy)
            times_vec = []
            for _ in range(3):
                torch.cuda.synchronize() if device.type == 'cuda' else None
                t0 = time.perf_counter()
                result_vec = model_vec.apply_disorder()
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times_vec.append(time.perf_counter() - t0)
            time_vec = np.mean(times_vec)
            
            # Calculate speedup
            speedup = time_orig / time_vec if time_vec > 0 else float('inf')
            
            # Check accuracy
            valid_orig = ~torch.isnan(result_orig)
            valid_vec = ~torch.isnan(result_vec)
            if torch.equal(valid_orig, valid_vec) and valid_orig.sum() > 0:
                valid_mask = valid_orig
                diff = torch.abs(result_orig[valid_mask] - result_vec[valid_mask])
                rel_diff = diff / torch.abs(result_orig[valid_mask])
                max_rel_diff = rel_diff.max().item()
                accuracy = "✅" if max_rel_diff < 1e-10 else "⚠️"
            else:
                max_rel_diff = float('nan')
                accuracy = "❌"
            
            results.append({
                'n_points': n_points,
                'time_original': time_orig,
                'time_vectorized': time_vec,
                'speedup': speedup,
                'max_rel_diff': max_rel_diff,
                'accuracy': accuracy
            })
            
            print(f"    Original: {time_orig:.4f}s ± {np.std(times_orig):.4f}s")
            print(f"    Vectorized: {time_vec:.4f}s ± {np.std(times_vec):.4f}s")
            print(f"    Speedup: {speedup:.1f}x")
            print(f"    Accuracy: {accuracy} (max rel diff: {max_rel_diff:.2e})")
            
        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                'n_points': n_points,
                'time_original': None,
                'time_vectorized': None,
                'speedup': None,
                'max_rel_diff': None,
                'accuracy': '❌',
                'error': str(e)
            })
    
    return results


def test_grid_mode(pdb_path: str, device: torch.device):
    """Test grid mode performance."""
    print("\n" + "="*50)
    print("GRID MODE PERFORMANCE") 
    print("="*50)
    
    # Test different grid sizes
    grid_specs = [
        ((-1, 1), (-1, 1), (-1, 1), 1),  # 3×3×3 = 27 points
        ((-2, 2), (-2, 2), (-2, 2), 1),  # 5×5×5 = 125 points  
        ((-1, 1), (-1, 1), (-1, 1), 2),  # 5×5×5 = 125 points (different way)
        ((-3, 3), (-3, 3), (-3, 3), 1),  # 7×7×7 = 343 points
    ]
    
    results = []
    
    for h_range, k_range, l_range, oversampling in grid_specs:
        # Calculate actual grid size  
        h_points = (h_range[1] - h_range[0]) * oversampling + 1
        k_points = (k_range[1] - k_range[0]) * oversampling + 1
        l_points = (l_range[1] - l_range[0]) * oversampling + 1
        total_points = h_points * k_points * l_points
        
        print(f"\nTesting grid {h_points}×{k_points}×{l_points} = {total_points} points...")
        
        hsampling = (h_range[0], h_range[1], oversampling)
        ksampling = (k_range[0], k_range[1], oversampling)
        lsampling = (l_range[0], l_range[1], oversampling)
        
        try:
            # Setup models
            print("  Setting up models...")
            t_setup = time.perf_counter()
            
            model_orig = OnePhonon(pdb_path, hsampling=hsampling, ksampling=ksampling,
                                 lsampling=lsampling, device=device)
            model_orig.compute_gnm_phonons()
            
            model_vec = OnePhononVectorized(pdb_path, hsampling=hsampling, ksampling=ksampling,
                                          lsampling=lsampling, device=device, use_vectorized=True)
            model_vec.compute_gnm_phonons()
            
            setup_time = time.perf_counter() - t_setup
            print(f"    Setup time: {setup_time:.2f}s")
            
            # Warm up
            _ = model_orig.apply_disorder()
            _ = model_vec.apply_disorder()
            
            # Benchmark original
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            result_orig = model_orig.apply_disorder()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            time_orig = time.perf_counter() - t0
            
            # Benchmark vectorized
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            result_vec = model_vec.apply_disorder()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            time_vec = time.perf_counter() - t0
            
            # Calculate speedup
            speedup = time_orig / time_vec if time_vec > 0 else float('inf')
            
            # Check accuracy
            valid_orig = ~torch.isnan(result_orig)
            valid_vec = ~torch.isnan(result_vec)
            if torch.equal(valid_orig, valid_vec) and valid_orig.sum() > 0:
                valid_mask = valid_orig
                diff = torch.abs(result_orig[valid_mask] - result_vec[valid_mask])
                rel_diff = diff / torch.abs(result_orig[valid_mask])
                max_rel_diff = rel_diff.max().item()
                accuracy = "✅" if max_rel_diff < 1e-10 else "⚠️"
            else:
                max_rel_diff = float('nan')
                accuracy = "❌"
            
            results.append({
                'grid_size': f"{h_points}×{k_points}×{l_points}",
                'total_points': total_points,
                'time_original': time_orig,
                'time_vectorized': time_vec,
                'speedup': speedup,
                'max_rel_diff': max_rel_diff,
                'accuracy': accuracy
            })
            
            print(f"    Original: {time_orig:.4f}s")
            print(f"    Vectorized: {time_vec:.4f}s")
            print(f"    Speedup: {speedup:.1f}x")
            print(f"    Accuracy: {accuracy} (max rel diff: {max_rel_diff:.2e})")
            
        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                'grid_size': f"{h_points}×{k_points}×{l_points}",
                'total_points': total_points,
                'time_original': None,
                'time_vectorized': None,
                'speedup': None,
                'max_rel_diff': None,
                'accuracy': '❌',
                'error': str(e)
            })
    
    return results


def main():
    """Run quick comprehensive benchmark."""
    print("="*60)
    print("Phase 1 Vectorization Quick Benchmark")
    print("="*60)
    
    # Configuration
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run tests
    arbitrary_results = test_arbitrary_mode(pdb_path, device)
    grid_results = test_grid_mode(pdb_path, device)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print("\nARBITRARY MODE:")
    print("Points | Original  | Vectorized | Speedup | Accuracy")
    print("-------|-----------|------------|---------|----------")
    for result in arbitrary_results:
        if result.get('error'):
            print(f"{result['n_points']:6d} | ERROR: {result['error'][:30]}...")
        else:
            print(f"{result['n_points']:6d} | {result['time_original']:8.4f}s | {result['time_vectorized']:9.4f}s | {result['speedup']:6.1f}x | {result['accuracy']}")
    
    print("\nGRID MODE:")
    print("Grid Size    | Points | Original  | Vectorized | Speedup | Accuracy")
    print("-------------|--------|-----------|------------|---------|----------")
    for result in grid_results:
        if result.get('error'):
            print(f"{result['grid_size']:12s} | {result['total_points']:6d} | ERROR: {result['error'][:20]}...")
        else:
            print(f"{result['grid_size']:12s} | {result['total_points']:6d} | {result['time_original']:8.4f}s | {result['time_vectorized']:9.4f}s | {result['speedup']:6.1f}x | {result['accuracy']}")
    
    # Calculate overall performance
    all_speedups = []
    for result in arbitrary_results + grid_results:
        if result.get('speedup') and not np.isnan(result['speedup']) and np.isfinite(result['speedup']):
            all_speedups.append(result['speedup'])
    
    if all_speedups:
        avg_speedup = np.mean(all_speedups)
        max_speedup = np.max(all_speedups)
        min_speedup = np.min(all_speedups)
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Average speedup: {avg_speedup:.1f}x")
        print(f"  Maximum speedup: {max_speedup:.1f}x")
        print(f"  Minimum speedup: {min_speedup:.1f}x")
        
        # Assessment
        if avg_speedup > 10:
            print("  🎉 OUTSTANDING: >10x average speedup achieved!")
        elif avg_speedup > 5:
            print("  ✅ EXCELLENT: >5x average speedup achieved")
        elif avg_speedup > 2:
            print("  ✅ GOOD: >2x average speedup achieved")
        else:
            print("  ⚠️ MODERATE: <2x average speedup")
    else:
        print("\nNo valid speedup measurements obtained.")


if __name__ == "__main__":
    main()
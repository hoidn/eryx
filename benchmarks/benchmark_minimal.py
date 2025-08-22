"""
Minimal benchmark to test just the core vectorization performance.
This focuses on the apply_disorder method performance without setup overhead.
"""

import torch
import numpy as np
import time
import os

from eryx.models_torch import OnePhonon
from eryx.models_torch_vectorized import OnePhononVectorized


def main():
    """Run minimal benchmark focused on core performance."""
    print("="*50)
    print("Minimal Vectorization Benchmark")
    print("="*50)
    
    # Configuration
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test case: small arbitrary q-vectors
    print("\nTesting 10 arbitrary q-vectors...")
    n_points = 10
    q_vectors = torch.rand(n_points, 3, device=device, dtype=torch.float64) * 2.0
    
    # Small grid for ADP calculation
    hsampling = (-1, 1, 1)  # 3x3x3 = 27 points
    ksampling = (-1, 1, 1)
    lsampling = (-1, 1, 1)
    
    try:
        # Setup original model
        print("Setting up original model...")
        t0 = time.perf_counter()
        model_orig = OnePhonon(pdb_path, q_vectors=q_vectors.clone(), 
                              hsampling=hsampling, ksampling=ksampling, lsampling=lsampling,
                              device=device)
        model_orig.compute_gnm_phonons()
        setup_time_orig = time.perf_counter() - t0
        print(f"  Setup time: {setup_time_orig:.3f}s")
        
        # Test original implementation
        print("Testing original implementation...")
        t0 = time.perf_counter()
        result_orig = model_orig.apply_disorder()
        time_orig = time.perf_counter() - t0
        print(f"  Computation time: {time_orig:.3f}s")
        print(f"  Result shape: {result_orig.shape}")
        valid_values_orig = result_orig[~torch.isnan(result_orig)]
        if valid_values_orig.numel() > 0:
            print(f"  Result range: [{valid_values_orig.min():.2e}, {valid_values_orig.max():.2e}]")
        else:
            print("  Result range: [all NaN]")
        
    except Exception as e:
        print(f"Original implementation failed: {e}")
        return
    
    try:
        # Setup vectorized model
        print("\nSetting up vectorized model...")
        t0 = time.perf_counter()
        model_vec = OnePhononVectorized(pdb_path, q_vectors=q_vectors.clone(), 
                                       hsampling=hsampling, ksampling=ksampling, lsampling=lsampling,
                                       device=device, use_vectorized=True)
        model_vec.compute_gnm_phonons()
        setup_time_vec = time.perf_counter() - t0
        print(f"  Setup time: {setup_time_vec:.3f}s")
        
        # Test vectorized implementation
        print("Testing vectorized implementation...")
        t0 = time.perf_counter()
        result_vec = model_vec.apply_disorder()
        time_vec = time.perf_counter() - t0
        print(f"  Computation time: {time_vec:.3f}s")
        print(f"  Result shape: {result_vec.shape}")
        valid_values_vec = result_vec[~torch.isnan(result_vec)]
        if valid_values_vec.numel() > 0:
            print(f"  Result range: [{valid_values_vec.min():.2e}, {valid_values_vec.max():.2e}]")
        else:
            print("  Result range: [all NaN]")
        
        # Compare results
        print("\nComparison:")
        if time_vec > 0:
            speedup = time_orig / time_vec
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print("  Speedup: ∞x (vectorized time ~0)")
            
        # Accuracy check
        valid_orig = ~torch.isnan(result_orig)
        valid_vec = ~torch.isnan(result_vec)
        if torch.equal(valid_orig, valid_vec):
            valid_mask = valid_orig
            diff = torch.abs(result_orig[valid_mask] - result_vec[valid_mask])
            rel_diff = diff / torch.abs(result_orig[valid_mask])
            max_abs_diff = diff.max().item()
            max_rel_diff = rel_diff.max().item()
            print(f"  Max absolute difference: {max_abs_diff:.2e}")
            print(f"  Max relative difference: {max_rel_diff:.2e}")
            
            if max_rel_diff < 1e-10:
                print("  ✅ Results match within numerical precision")
            elif max_rel_diff < 1e-6:
                print("  ✅ Results match within acceptable tolerance")
            else:
                print("  ⚠️ Results differ significantly")
        else:
            print("  ❌ NaN patterns differ between implementations")
        
        print(f"\nSUMMARY:")
        print(f"  Original: {time_orig:.4f}s")
        print(f"  Vectorized: {time_vec:.4f}s") 
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 5:
            print("  🎉 Excellent speedup!")
        elif speedup > 2:
            print("  ✅ Good speedup")
        elif speedup > 1:
            print("  🟡 Modest speedup")
        else:
            print("  ❌ No speedup achieved")
            
    except Exception as e:
        print(f"Vectorized implementation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
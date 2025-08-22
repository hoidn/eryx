"""
Test suite for Phase 4: Integration and Optimization.

This module tests the final integrated implementation with all optimizations
and memory management features.
"""

import unittest
import torch
import numpy as np
import time
import logging
import gc

from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

logging.basicConfig(level=logging.INFO)


class TestPhase4Integration(unittest.TestCase):
    """Test suite for Phase 4 integrated optimizations."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.rtol = 1e-10
        self.atol = 1e-12
        
    def test_memory_management(self):
        """Test intelligent memory management features."""
        print("\n" + "="*60)
        print("Testing Memory Management")
        print("="*60)
        
        # Create model with memory limit
        max_memory_gb = 2.0 if self.device.type == 'cuda' else 0.5
        
        model = OnePhononOptimized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=torch.randn(100, 3, device=self.device, dtype=torch.float64),
            device=self.device,
            max_memory_gb=max_memory_gb,
            enable_profiling=True
        )
        
        print(f"Model initialized with {max_memory_gb}GB memory limit")
        print(f"Batch size: {model.batch_size}")
        print(f"Eigendecomp batch: {model.eigendecomp_batch}")
        
        # Run computation
        model.compute_gnm_phonons()
        intensity = model.apply_disorder()
        
        # Check memory stats
        stats = model.get_memory_stats()
        print("\nMemory Statistics:")
        for key, value in stats.items():
            if 'memory' in key or 'allocated' in key:
                print(f"  {key}: {value / 1e9:.3f} GB")
            else:
                print(f"  {key}: {value}")
        
        # Verify results are valid
        valid_points = torch.sum(~torch.isnan(intensity)).item()
        print(f"\nValid intensity points: {valid_points}")
        
        self.assertGreater(valid_points, 0, "No valid intensity points computed")
        print("✓ Memory management verified!")
    
    def test_batch_processing(self):
        """Test that batch processing produces correct results."""
        print("\n" + "="*60)
        print("Testing Batch Processing Correctness")
        print("="*60)
        
        # Create test q-vectors
        q_vectors = torch.randn(50, 3, device=self.device, dtype=torch.float64) * 0.5
        
        # Model with small batches
        print("Creating model with small batch size...")
        model_small_batch = OnePhononOptimized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=q_vectors.clone(),
            device=self.device,
            max_memory_gb=0.1  # Force small batches
        )
        model_small_batch.batch_size = 5  # Override to very small
        
        # Model with large batches
        print("Creating model with large batch size...")
        model_large_batch = OnePhononOptimized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=q_vectors.clone(),
            device=self.device,
            max_memory_gb=10.0
        )
        model_large_batch.batch_size = 100  # Override to large
        
        # Compute for both
        print("\nComputing with small batches...")
        model_small_batch.compute_gnm_phonons()
        intensity_small = model_small_batch.apply_disorder()
        
        print("Computing with large batches...")
        model_large_batch.compute_gnm_phonons()
        intensity_large = model_large_batch.apply_disorder()
        
        # Compare results
        nan_mask = torch.isnan(intensity_small)
        valid_mask = ~nan_mask
        
        if torch.any(valid_mask):
            valid_small = intensity_small[valid_mask]
            valid_large = intensity_large[valid_mask]
            
            abs_diff = torch.abs(valid_small - valid_large)
            max_diff = torch.max(abs_diff).item()
            
            print(f"\nMax difference between batch sizes: {max_diff:.2e}")
            
            self.assertTrue(
                torch.allclose(valid_small, valid_large, rtol=self.rtol, atol=self.atol),
                f"Batch results differ: max_diff={max_diff:.2e}"
            )
            
            print("✓ Batch processing produces consistent results!")
    
    def test_performance_comparison(self):
        """Compare performance of optimized vs original implementation."""
        print("\n" + "="*60)
        print("Testing Overall Performance Improvement")
        print("="*60)
        
        # Use small problem for quick test
        hsampling = (-1, 1, 1)  # 3 points
        ksampling = (-1, 1, 1)
        lsampling = (-1, 1, 1)
        
        print(f"Grid: 3×3×3 = 27 points")
        
        # Original implementation
        print("\nOriginal implementation:")
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        gc.collect()
        
        t0 = time.perf_counter()
        model_orig = OnePhonon(
            self.pdb_path,
            hsampling, ksampling, lsampling,
            device=self.device
        )
        model_orig.compute_gnm_phonons()
        intensity_orig = model_orig.apply_disorder()
        time_orig = time.perf_counter() - t0
        print(f"  Total time: {time_orig:.3f}s")
        
        # Optimized implementation
        print("\nOptimized implementation (Phase 4):")
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        gc.collect()
        
        t0 = time.perf_counter()
        model_opt = OnePhononOptimized(
            self.pdb_path,
            hsampling, ksampling, lsampling,
            device=self.device,
            enable_profiling=False
        )
        model_opt.compute_gnm_phonons()
        intensity_opt = model_opt.apply_disorder()
        time_opt = time.perf_counter() - t0
        print(f"  Total time: {time_opt:.3f}s")
        
        # Calculate speedup
        speedup = time_orig / time_opt if time_opt > 0 else 0
        print(f"\nOverall speedup: {speedup:.2f}x")
        
        # Verify results match
        nan_mask = torch.isnan(intensity_orig)
        valid_mask = ~nan_mask
        
        if torch.any(valid_mask):
            valid_orig = intensity_orig[valid_mask]
            valid_opt = intensity_opt[valid_mask]
            
            max_diff = torch.max(torch.abs(valid_orig - valid_opt)).item()
            print(f"Max difference: {max_diff:.2e}")
            
            self.assertTrue(
                torch.allclose(valid_orig, valid_opt, rtol=self.rtol, atol=self.atol),
                f"Results differ: max_diff={max_diff:.2e}"
            )
        
        print("✓ Performance comparison complete!")
        
        # We expect at least some speedup
        self.assertGreater(speedup, 0.9, "Optimized version should not be slower")
    
    def test_contiguous_memory(self):
        """Test that tensors are kept contiguous for optimal performance."""
        print("\n" + "="*60)
        print("Testing Memory Layout Optimization")
        print("="*60)
        
        model = OnePhononOptimized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=torch.randn(10, 3, device=self.device, dtype=torch.float64),
            device=self.device
        )
        
        # Check key tensors are contiguous after operations
        model.compute_gnm_phonons()
        
        tensors_to_check = {
            'gamma_tensor': model.gamma_tensor if hasattr(model, 'gamma_tensor') else None,
            'V': model.V,
            'Winv': model.Winv,
        }
        
        print("Checking tensor memory layout:")
        for name, tensor in tensors_to_check.items():
            if tensor is not None:
                is_contiguous = tensor.is_contiguous()
                print(f"  {name}: {'✓ contiguous' if is_contiguous else '✗ not contiguous'}")
                
                if not is_contiguous:
                    # This is not necessarily an error, but worth noting
                    print(f"    Shape: {tensor.shape}, Stride: {tensor.stride()}")
        
        intensity = model.apply_disorder()
        self.assertTrue(intensity.is_contiguous(), "Output intensity should be contiguous")
        
        print("\n✓ Memory layout check complete!")
    
    def test_inference_optimization(self):
        """Test inference mode optimization."""
        print("\n" + "="*60)
        print("Testing Inference Mode Optimization")
        print("="*60)
        
        model = OnePhononOptimized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=torch.randn(20, 3, device=self.device, dtype=torch.float64),
            device=self.device
        )
        
        # Check that gradients are tracked initially
        model.compute_gnm_phonons()
        
        initial_grad_enabled = []
        for attr_name in ['gamma_tensor', 'V', 'Winv']:
            if hasattr(model, attr_name):
                tensor = getattr(model, attr_name)
                if isinstance(tensor, torch.Tensor):
                    initial_grad_enabled.append(tensor.requires_grad)
        
        print(f"Initial gradient tracking: {any(initial_grad_enabled)}")
        
        # Optimize for inference
        model.optimize_for_inference()
        
        # Check gradients are disabled
        final_grad_enabled = []
        for attr_name in ['gamma_tensor', 'V', 'Winv']:
            if hasattr(model, attr_name):
                tensor = getattr(model, attr_name)
                if isinstance(tensor, torch.Tensor):
                    final_grad_enabled.append(tensor.requires_grad)
        
        print(f"After optimization: gradients={'enabled' if any(final_grad_enabled) else 'disabled'}")
        
        # Run inference
        with torch.no_grad():
            intensity = model.apply_disorder()
        
        valid_points = torch.sum(~torch.isnan(intensity)).item()
        print(f"Valid points computed: {valid_points}")
        
        self.assertFalse(any(final_grad_enabled), "Gradients should be disabled in inference mode")
        self.assertGreater(valid_points, 0, "Should compute valid results in inference mode")
        
        print("✓ Inference optimization verified!")


if __name__ == '__main__':
    unittest.main(verbosity=2)
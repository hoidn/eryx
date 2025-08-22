#!/usr/bin/env python3
"""
Test suite for verifying numerical equivalence between original and vectorized implementations.

This module ensures that vectorization optimizations preserve:
- Numerical accuracy (within tolerance)
- Gradient flow
- API compatibility
- Edge case handling
"""

import unittest
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
from pathlib import Path


class VectorizationEquivalenceTest(unittest.TestCase):
    """Test numerical equivalence between original and vectorized implementations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.pdb_path = "tests/pdbs/5zck_p1.pdb"
        
        # Tolerance settings (strict for regression prevention)
        cls.rtol = 1e-12
        cls.atol = 1e-14
        
        # Test grid configurations
        cls.test_grids = {
            'tiny': ([-1, 1, 2], [-1, 1, 2], [-1, 1, 2]),
            'small': ([-1, 1, 3], [-1, 1, 3], [-1, 1, 3]),
        }
    
    def setUp(self):
        """Reset state before each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def assert_tensors_equal(self, 
                            tensor1: torch.Tensor, 
                            tensor2: torch.Tensor,
                            name: str = "tensors",
                            rtol: Optional[float] = None,
                            atol: Optional[float] = None):
        """
        Assert two tensors are numerically equal within tolerance.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor  
            name: Name for error messages
            rtol: Relative tolerance (uses class default if None)
            atol: Absolute tolerance (uses class default if None)
        """
        rtol = rtol or self.rtol
        atol = atol or self.atol
        
        # Check shapes match
        self.assertEqual(tensor1.shape, tensor2.shape, 
                        f"{name} shapes don't match: {tensor1.shape} vs {tensor2.shape}")
        
        # Check dtypes are compatible
        if tensor1.dtype != tensor2.dtype:
            # Allow float32/float64 comparison
            if not (tensor1.dtype in [torch.float32, torch.float64] and 
                   tensor2.dtype in [torch.float32, torch.float64]):
                self.fail(f"{name} dtypes incompatible: {tensor1.dtype} vs {tensor2.dtype}")
        
        # Move to CPU for comparison
        t1_cpu = tensor1.detach().cpu()
        t2_cpu = tensor2.detach().cpu()
        
        # Handle NaN values
        nan_mask1 = torch.isnan(t1_cpu)
        nan_mask2 = torch.isnan(t2_cpu)
        
        # Check NaN positions match
        if not torch.equal(nan_mask1, nan_mask2):
            self.fail(f"{name} NaN positions don't match")
        
        # Compare non-NaN values
        valid_mask = ~nan_mask1
        if valid_mask.any():
            valid1 = t1_cpu[valid_mask]
            valid2 = t2_cpu[valid_mask]
            
            # Use torch.allclose for numerical comparison
            if not torch.allclose(valid1, valid2, rtol=rtol, atol=atol):
                # Calculate actual differences for debugging
                diff = torch.abs(valid1 - valid2)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                rel_diff = torch.max(diff / (torch.abs(valid1) + 1e-10)).item()
                
                self.fail(f"{name} values not equal within tolerance.\n"
                         f"Max absolute diff: {max_diff:.2e} (tolerance: {atol:.2e})\n"
                         f"Mean absolute diff: {mean_diff:.2e}\n"
                         f"Max relative diff: {rel_diff:.2e} (tolerance: {rtol:.2e})")
    
    def test_placeholder_for_vectorized_implementation(self):
        """Placeholder test for when vectorized implementation is ready."""
        # This will be replaced with actual comparison tests
        # For now, just verify the test infrastructure works
        
        t1 = torch.randn(10, 10, dtype=torch.float64, device=self.device)
        t2 = t1.clone()
        
        self.assert_tensors_equal(t1, t2, "identical tensors")
        
        # Test with small perturbation
        t3 = t1 + 1e-13
        self.assert_tensors_equal(t1, t3, "nearly identical tensors")
        
        # Test that large differences are caught
        t4 = t1 + 1e-6
        with self.assertRaises(AssertionError):
            self.assert_tensors_equal(t1, t4, "different tensors", rtol=1e-12)
    
    def verify_gradient_flow(self,
                            model: Any,
                            param_names: list,
                            loss_fn: Optional[callable] = None) -> Dict[str, bool]:
        """
        Verify gradient flow through specified parameters.
        
        Args:
            model: Model to test
            param_names: List of parameter names to check
            loss_fn: Optional custom loss function
            
        Returns:
            Dictionary mapping parameter names to gradient existence
        """
        # Create test input
        if hasattr(model, 'q_grid'):
            # Grid mode
            output = model.apply_disorder(use_data_adp=True)
        else:
            # Arbitrary q mode
            output = model.apply_disorder(use_data_adp=True)
        
        # Default loss is sum of output
        if loss_fn is None:
            loss = output.sum()
        else:
            loss = loss_fn(output)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        gradient_status = {}
        for param_name in param_names:
            if hasattr(model, param_name):
                param = getattr(model, param_name)
                if hasattr(param, 'grad') and param.grad is not None:
                    has_grad = torch.any(param.grad != 0).item()
                    gradient_status[param_name] = has_grad
                else:
                    gradient_status[param_name] = False
            else:
                gradient_status[param_name] = None
        
        return gradient_status
    
    def compare_implementations(self,
                               original_fn: callable,
                               vectorized_fn: callable,
                               args: tuple = (),
                               kwargs: dict = None,
                               name: str = "operation") -> Tuple[bool, Dict]:
        """
        Compare original and vectorized implementations.
        
        Args:
            original_fn: Original implementation function
            vectorized_fn: Vectorized implementation function
            args: Arguments to pass to both functions
            kwargs: Keyword arguments to pass to both functions
            name: Name for reporting
            
        Returns:
            Tuple of (equivalent, statistics_dict)
        """
        kwargs = kwargs or {}
        
        # Run original implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original_output = original_fn(*args, **kwargs)
        
        # Run vectorized implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectorized_output = vectorized_fn(*args, **kwargs)
        
        # Compare outputs
        try:
            self.assert_tensors_equal(original_output, vectorized_output, name)
            equivalent = True
        except AssertionError as e:
            print(f"Equivalence check failed for {name}: {e}")
            equivalent = False
        
        # Calculate statistics
        stats = {
            'name': name,
            'equivalent': equivalent,
            'original_shape': list(original_output.shape),
            'vectorized_shape': list(vectorized_output.shape),
        }
        
        if equivalent:
            diff = torch.abs(original_output - vectorized_output)
            stats.update({
                'max_diff': torch.max(diff).item(),
                'mean_diff': torch.mean(diff).item(),
                'std_diff': torch.std(diff).item(),
            })
        
        return equivalent, stats
    
    # Placeholder methods for future vectorized implementation tests
    
    def test_grid_mode_equivalence(self):
        """Test grid mode produces identical results."""
        self.skipTest("Waiting for vectorized implementation")
        
        # Will test:
        # 1. Create model with original implementation
        # 2. Create model with vectorized implementation
        # 3. Run apply_disorder on both
        # 4. Compare outputs
    
    def test_arbitrary_q_mode_equivalence(self):
        """Test arbitrary q-vector mode produces identical results."""
        self.skipTest("Waiting for vectorized implementation")
        
        # Will test:
        # 1. Create random q-vectors
        # 2. Run both implementations
        # 3. Compare outputs
    
    def test_gradient_preservation(self):
        """Test gradients flow correctly in vectorized version."""
        self.skipTest("Waiting for vectorized implementation")
        
        # Will test:
        # 1. Create models with requires_grad=True
        # 2. Forward pass through both
        # 3. Backward pass
        # 4. Compare gradient magnitudes
    
    def test_edge_cases(self):
        """Test edge cases are handled identically."""
        self.skipTest("Waiting for vectorized implementation")
        
        # Will test:
        # 1. Single q-point
        # 2. Very large grid
        # 3. High resolution limits
        # 4. NaN handling
    
    def test_memory_efficiency(self):
        """Test vectorized version uses memory efficiently."""
        self.skipTest("Waiting for vectorized implementation")
        
        # Will test:
        # 1. Peak memory usage
        # 2. Memory scaling with problem size
        # 3. No memory leaks


class StateComparisonTest(unittest.TestCase):
    """Test state capture and comparison for debugging vectorization."""
    
    def capture_model_state(self, model: Any) -> Dict[str, Any]:
        """
        Capture complete model state for comparison.
        
        Args:
            model: Model to capture state from
            
        Returns:
            Dictionary containing model state
        """
        state = {}
        
        # Capture all tensor attributes
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr = getattr(model, attr_name)
                if torch.is_tensor(attr):
                    state[attr_name] = attr.detach().cpu().clone()
                elif isinstance(attr, (int, float, str, bool)):
                    state[attr_name] = attr
                elif isinstance(attr, (list, tuple)) and len(attr) < 100:
                    state[attr_name] = attr
        
        return state
    
    def compare_states(self, 
                      state1: Dict[str, Any],
                      state2: Dict[str, Any],
                      rtol: float = 1e-12,
                      atol: float = 1e-14) -> Dict[str, bool]:
        """
        Compare two model states.
        
        Args:
            state1: First state dictionary
            state2: Second state dictionary
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Dictionary mapping attribute names to equivalence status
        """
        comparison = {}
        
        # Check all keys in both states
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            if key not in state1:
                comparison[key] = False  # Missing in state1
            elif key not in state2:
                comparison[key] = False  # Missing in state2
            else:
                val1 = state1[key]
                val2 = state2[key]
                
                if torch.is_tensor(val1) and torch.is_tensor(val2):
                    # Compare tensors
                    if val1.shape != val2.shape:
                        comparison[key] = False
                    else:
                        comparison[key] = torch.allclose(val1, val2, rtol=rtol, atol=atol)
                else:
                    # Direct comparison for non-tensors
                    comparison[key] = (val1 == val2)
        
        return comparison


def create_test_suite():
    """Create test suite for vectorization validation."""
    suite = unittest.TestSuite()
    
    # Add equivalence tests
    suite.addTest(unittest.makeSuite(VectorizationEquivalenceTest))
    
    # Add state comparison tests
    suite.addTest(unittest.makeSuite(StateComparisonTest))
    
    return suite


if __name__ == '__main__':
    # Run test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    if not result.wasSuccessful():
        exit(1)
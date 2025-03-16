import os
import unittest
import torch
import numpy as np
from typing import Tuple, Dict, Optional, List, Any

from tests.test_base import TestBase
from eryx.models_torch import OnePhonon

class TestBatchedImplementation(TestBase):
    """Test case for batched implementation of OnePhonon model."""
    
    def setUp(self):
        """Set up test environment."""
        # Call parent setUp
        super().setUp()
        
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
        # Test parameters
        self.test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    
    def create_test_models(self, use_batching: bool = True) -> OnePhonon:
        """Create a test model with specified batching setting."""
        return OnePhonon(
            **self.test_params,
            device=self.device,
            use_batching=use_batching
        )
        
    def _create_test_model(self):
        """Create a small model instance for testing."""
        # Create model with minimal dimensions
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        return OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],  # Small grid for testing
            expand_p1=True,
            use_batching=True,
            device=self.device
        )
        
    def test_fully_collapsed_tensor_conversion(self):
        """Test conversion between original and fully collapsed tensor formats."""
        # Create a model instance with small dimensions for testing
        model = self._create_test_model()
        
        # Create a test tensor with original 3D shape
        h_dim, k_dim, l_dim = 2, 3, 4
        features = 5
        original_tensor = torch.randn(h_dim, k_dim, l_dim, features, device=self.device)
        
        # Convert to fully collapsed shape
        collapsed_tensor = model.to_batched_shape(original_tensor)
        
        # Verify shape is correct
        expected_shape = (h_dim * k_dim * l_dim, features)
        self.assertEqual(collapsed_tensor.shape, expected_shape)
        
        # Convert back to original shape
        model.test_k_dim = k_dim
        model.test_l_dim = l_dim
        restored_tensor = model.to_original_shape(collapsed_tensor)
        
        # Verify shape and values are preserved
        self.assertEqual(restored_tensor.shape, original_tensor.shape)
        self.assertTrue(torch.allclose(restored_tensor, original_tensor))
    
    def test_tensor_format_conversion(self):
        """Test conversion between original and batched tensor formats."""
        # Create model
        model = self.create_test_models(use_batching=True)
        
        # Create test tensor in original format
        h_dim, k_dim, l_dim = 2, 3, 4
        feature_dim = 5
        original_tensor = torch.rand((h_dim, k_dim, l_dim, feature_dim), device=self.device)
        
        # Store test dimensions for proper restoration
        model.test_k_dim = k_dim
        model.test_l_dim = l_dim
        
        # Convert to batched format
        batched_tensor = model.to_batched_shape(original_tensor)
        
        # Verify dimensions
        self.assertEqual(batched_tensor.shape, (h_dim * k_dim * l_dim, feature_dim))
        
        # Convert back to original format
        restored_tensor = model.to_original_shape(batched_tensor)
        
        # Verify dimensions are restored
        self.assertEqual(restored_tensor.shape, original_tensor.shape)
        
        # Verify values are preserved
        self.assertTrue(torch.allclose(original_tensor, restored_tensor))
    
    def test_fully_collapsed_index_conversion(self):
        """Test conversion between 3D indices and fully collapsed indices."""
        # Create a model instance
        model = self._create_test_model()
        
        # Get dimensions
        h_dim = int(model.hsampling[2])
        k_dim = int(model.ksampling[2])
        l_dim = int(model.lsampling[2])
        
        # Test case 1: Single index
        h, k, l = 1, 1, 1
        flat_idx = h * (k_dim * l_dim) + k * l_dim + l
        
        # Convert to 3D indices
        h_computed, k_computed, l_computed = model._flat_to_3d_indices(torch.tensor([flat_idx], device=self.device))
        
        # Verify conversion
        self.assertEqual(h_computed.item(), h)
        self.assertEqual(k_computed.item(), k)
        self.assertEqual(l_computed.item(), l)
        
        # Convert back to flat index
        flat_computed = model._3d_to_flat_indices(
            torch.tensor([h], device=self.device),
            torch.tensor([k], device=self.device),
            torch.tensor([l], device=self.device)
        )
        
        # Verify round-trip conversion
        self.assertEqual(flat_computed.item(), flat_idx)
        
        # Test case 2: Multiple indices
        h_indices = torch.tensor([0, 1, 0, 1], device=self.device)
        k_indices = torch.tensor([0, 0, 1, 1], device=self.device)
        l_indices = torch.tensor([0, 1, 1, 0], device=self.device)
        
        # Convert to flat indices
        flat_indices = model._3d_to_flat_indices(h_indices, k_indices, l_indices)
        
        # Convert back to 3D
        h_computed, k_computed, l_computed = model._flat_to_3d_indices(flat_indices)
        
        # Verify round-trip conversion
        self.assertTrue(torch.all(h_computed == h_indices))
        self.assertTrue(torch.all(k_computed == k_indices))
        self.assertTrue(torch.all(l_computed == l_indices))
    
    def test_batched_vs_nonbatched_kvec_brillouin(self):
        """Test that batched k-vector generation matches non-batched results."""
        # Create two models: one with batching, one without
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        model_batched = OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],
            expand_p1=True,
            use_batching=True,
            device=self.device
        )
        
        model_nonbatched = OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],
            expand_p1=True,
            use_batching=False,
            device=self.device
        )
        
        # Generate k-vectors with both implementations
        model_batched._build_kvec_Brillouin()
        model_nonbatched._build_kvec_Brillouin()
        
        # Convert batched result to original shape for comparison
        kvec_batched_original = model_batched.to_original_shape(model_batched.kvec)
        kvec_norm_batched_original = model_batched.to_original_shape(model_batched.kvec_norm)
        
        # Compare results
        self.assertTrue(torch.allclose(kvec_batched_original, model_nonbatched.kvec, rtol=1e-5, atol=1e-8))
        self.assertTrue(torch.allclose(kvec_norm_batched_original, model_nonbatched.kvec_norm, rtol=1e-5, atol=1e-8))
        
        # Verify gradient requirements
        self.assertTrue(model_batched.kvec.requires_grad)
        self.assertTrue(model_batched.kvec_norm.requires_grad)
        
        # Verify shapes
        h_dim = int(model_batched.hsampling[2])
        k_dim = int(model_batched.ksampling[2])
        l_dim = int(model_batched.lsampling[2])
        
        self.assertEqual(model_batched.kvec.shape, (h_dim * k_dim * l_dim, 3))
        self.assertEqual(model_batched.kvec_norm.shape, (h_dim * k_dim * l_dim, 1))
        self.assertEqual(model_nonbatched.kvec.shape, (h_dim, k_dim, l_dim, 3))
        self.assertEqual(model_nonbatched.kvec_norm.shape, (h_dim, k_dim, l_dim, 1))
    
    def test_at_kvec_from_miller_points_fully_collapsed(self):
        """Test that _at_kvec_from_miller_points works with fully collapsed indices."""
        # Create batched and non-batched models
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        model_batched = OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],
            expand_p1=True,
            use_batching=True,
            device=self.device
        )
        
        model_nonbatched = OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],
            expand_p1=True,
            use_batching=False,
            device=self.device
        )
        
        # Get dimensions
        h_dim = int(model_batched.hsampling[2])
        k_dim = int(model_batched.ksampling[2])
        l_dim = int(model_batched.lsampling[2])
        
        # Test with traditional format
        h, k, l = 1, 1, 1
        indices_nonbatched = model_nonbatched._at_kvec_from_miller_points((h, k, l))
        indices_batched = model_batched._at_kvec_from_miller_points((h, k, l))
        
        # Verify both implementations return the same indices
        self.assertTrue(torch.all(indices_nonbatched == indices_batched))
        
        # Test with fully collapsed format
        flat_idx = h * (k_dim * l_dim) + k * l_dim + l
        indices_from_flat = model_batched._at_kvec_from_miller_points(flat_idx)
        
        # Verify flat index input produces the same result as tuple input
        self.assertTrue(torch.all(indices_batched == indices_from_flat))
        
        # Test with edge cases
        # Case 1: Origin (0, 0, 0)
        self.assertTrue(torch.all(
            model_batched._at_kvec_from_miller_points((0, 0, 0)) == 
            model_batched._at_kvec_from_miller_points(0)
        ))
        
        # Case 2: Maximum indices
        max_h, max_k, max_l = h_dim - 1, k_dim - 1, l_dim - 1
        max_flat = max_h * (k_dim * l_dim) + max_k * l_dim + max_l
        self.assertTrue(torch.all(
            model_batched._at_kvec_from_miller_points((max_h, max_k, max_l)) == 
            model_batched._at_kvec_from_miller_points(max_flat)
        ))

if __name__ == '__main__':
    unittest.main()

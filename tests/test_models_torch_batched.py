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
        self.assertEqual(batched_tensor.shape, (h_dim, k_dim * l_dim, feature_dim))
        
        # Convert back to original format
        restored_tensor = model.to_original_shape(batched_tensor)
        
        # Verify dimensions are restored
        self.assertEqual(restored_tensor.shape, original_tensor.shape)
        
        # Verify values are preserved
        self.assertTrue(torch.allclose(original_tensor, restored_tensor))
    
    def test_index_conversion(self):
        """Test conversion between 3D and flat indices."""
        # Create model with explicit sampling parameters for testing
        model = self.create_test_models(use_batching=True)
        
        # Override sampling parameters for this test to ensure consistency
        model.lsampling = [-2, 2, 2]  # Ensure l_dim is 2
        
        # Create test indices
        h_indices = torch.tensor([0, 1, 0, 1], device=self.device)
        k_indices = torch.tensor([0, 0, 1, 1], device=self.device)
        l_indices = torch.tensor([0, 1, 0, 1], device=self.device)
        
        # Convert to flat indices
        flat_indices = model._3d_to_flat_indices(h_indices, k_indices, l_indices)
        
        # Convert back to 3D indices
        k_restored, l_restored = model._flat_to_3d_indices(flat_indices)
        
        # Verify round-trip conversion
        self.assertTrue(torch.all(k_indices == k_restored), 
                       f"k_indices {k_indices} != k_restored {k_restored}")
        self.assertTrue(torch.all(l_indices == l_restored),
                       f"l_indices {l_indices} != l_restored {l_restored}")
        
        # Test specific cases
        l_dim = int(model.lsampling[2])
        
        # Test (0,0,0) -> flat -> (0,0)
        flat_idx = model._3d_to_flat_indices(
            torch.tensor([0], device=self.device),
            torch.tensor([0], device=self.device),
            torch.tensor([0], device=self.device)
        )
        self.assertEqual(flat_idx.item(), 0)
        
        # Test (0,1,0) -> flat -> (1,0)
        flat_idx = model._3d_to_flat_indices(
            torch.tensor([0], device=self.device),
            torch.tensor([1], device=self.device),
            torch.tensor([0], device=self.device)
        )
        self.assertEqual(flat_idx.item(), l_dim)
    
    def test_batched_kvector_generation(self):
        """Test k-vector generation in batched and non-batched modes."""
        # Create models with different batching settings
        batched_model = self.create_test_models(use_batching=True)
        nonbatched_model = self.create_test_models(use_batching=False)
        
        # Generate k-vectors
        batched_model._build_kvec_Brillouin()
        nonbatched_model._build_kvec_Brillouin()
        
        # Convert batched output to original format for comparison
        batched_kvec_original = batched_model.to_original_shape(batched_model.kvec)
        batched_norm_original = batched_model.to_original_shape(batched_model.kvec_norm)
        
        # Compare values
        self.assertTrue(torch.allclose(batched_kvec_original, nonbatched_model.kvec))
        self.assertTrue(torch.allclose(batched_norm_original, nonbatched_model.kvec_norm))
        
        # Verify shapes
        h_dim = int(batched_model.hsampling[2])
        k_dim = int(batched_model.ksampling[2])
        l_dim = int(batched_model.lsampling[2])
        
        self.assertEqual(batched_model.kvec.shape, (h_dim, k_dim * l_dim, 3))
        self.assertEqual(batched_model.kvec_norm.shape, (h_dim, k_dim * l_dim, 1))
        self.assertEqual(nonbatched_model.kvec.shape, (h_dim, k_dim, l_dim, 3))
        self.assertEqual(nonbatched_model.kvec_norm.shape, (h_dim, k_dim, l_dim, 1))
    
    def test_at_kvec_from_miller_points(self):
        """Test _at_kvec_from_miller_points with different input formats."""
        # Create model
        model = self.create_test_models(use_batching=True)
        
        # Test traditional format input (h, k, l)
        traditional_input = (0, 0, 0)
        traditional_output = model._at_kvec_from_miller_points(traditional_input)
        
        # Test batched format input (h, flat_idx)
        batched_input = (0, 0)  # Equivalent to (0, 0, 0)
        batched_output = model._at_kvec_from_miller_points(batched_input)
        
        # Verify both formats produce the same output
        self.assertTrue(torch.all(traditional_output == batched_output))
        
        # Test edge cases
        # Maximum values
        h_dim = int(model.hsampling[2])
        k_dim = int(model.ksampling[2])
        l_dim = int(model.lsampling[2])
        
        max_traditional = (h_dim-1, k_dim-1, l_dim-1)
        max_batched = (h_dim-1, (k_dim-1) * l_dim + (l_dim-1))
        
        max_traditional_output = model._at_kvec_from_miller_points(max_traditional)
        max_batched_output = model._at_kvec_from_miller_points(max_batched)
        
        # Verify outputs match for equivalent inputs
        self.assertTrue(torch.all(max_traditional_output == max_batched_output))

if __name__ == '__main__':
    unittest.main()

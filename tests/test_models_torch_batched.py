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
            'hsampling': [-4, 4, 3],
            'ksampling': [-17, 17, 3],
            'lsampling': [-29, 29, 3],
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
        
    def test_batched_vs_nonbatched_compute_K(self):
        """Test that compute_K_batched produces equivalent results to compute_K."""
        # Import necessary components
        from eryx.pdb_torch import GaussianNetworkModel
        import time
        
        # Create a small GaussianNetworkModel instance
        # For testing, we'll use a mock GNM with minimal dimensions
        gnm = GaussianNetworkModel()
        gnm.n_asu = 2
        gnm.n_atoms_per_asu = 3
        gnm.n_cell = 2
        gnm.id_cell_ref = 0
        gnm.device = self.device
        
        # Create a sample hessian tensor
        hessian = torch.randn(
            gnm.n_asu, gnm.n_atoms_per_asu,
            gnm.n_cell, gnm.n_asu, gnm.n_atoms_per_asu,
            dtype=torch.complex64, device=self.device
        )
        
        # Ensure the hessian is Hermitian for numerical stability
        for i_asu in range(gnm.n_asu):
            for j_asu in range(gnm.n_asu):
                for i_cell in range(gnm.n_cell):
                    hessian[i_asu, :, i_cell, j_asu, :] = 0.5 * (
                        hessian[i_asu, :, i_cell, j_asu, :] + 
                        hessian[j_asu, :, i_cell, i_asu, :].transpose(-2, -1).conj()
                    )
        
        # Mock the crystal methods needed for compute_K
        class MockCrystal:
            def __init__(self, device):
                self.device = device
                
            def get_unitcell_origin(self, unitcell):
                # Return a simple tensor as the unit cell origin
                return torch.tensor([float(unitcell[0]), 0.0, 0.0], device=self.device)
                
            def id_to_hkl(self, cell_id):
                # Return a simple list as the unit cell indices
                return [cell_id, 0, 0]
        
        gnm.crystal = MockCrystal(self.device)
        
        # Test case 1: Single k-vector
        k_vec = torch.tensor([[0.1, 0.2, 0.3]], device=self.device)
        
        # Compute K matrix using original method
        K_single = gnm.compute_K(hessian, kvec=k_vec[0])
        
        # Compute using batched method
        K_batch = gnm.compute_K_batched(hessian, k_vec)
        
        # Get the first matrix from batch result
        if K_batch.dim() > 4:  # If reshaped output
            K_batch_single = K_batch[0]
        else:  # If 2D output
            n_asu = gnm.n_asu
            n_atoms = gnm.n_atoms_per_asu
            K_batch_single = K_batch[0].reshape(n_asu, n_atoms, n_asu, n_atoms)
        
        # Compare results - should be very close
        self.assertTrue(torch.allclose(K_single, K_batch_single, rtol=1e-5, atol=1e-7))
        
        # Test case 2: Multiple k-vectors
        k_vecs = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], device=self.device)
        
        # Time the non-batched computation
        start_time = time.time()
        # Compute K matrices one by one using original method
        K_list = []
        for i in range(k_vecs.shape[0]):
            K_list.append(gnm.compute_K(hessian, kvec=k_vecs[i]))
        non_batched_time = time.time() - start_time
        
        # Time the batched computation
        start_time = time.time()
        # Compute K matrices in batch using batched method
        K_batched = gnm.compute_K_batched(hessian, k_vecs)
        batched_time = time.time() - start_time
        
        # Print timing comparison
        print(f"\nCompute_K timing comparison (3 k-vectors):")
        print(f"  Non-batched: {non_batched_time:.6f} seconds")
        print(f"  Batched:     {batched_time:.6f} seconds")
        print(f"  Speedup:     {non_batched_time/batched_time:.2f}x")
        
        # Compare results for each k-vector
        for i in range(k_vecs.shape[0]):
            if K_batched.dim() > 4:  # If reshaped output
                K_batch_i = K_batched[i]
            else:  # If 2D output
                K_batch_i = K_batched[i].reshape(n_asu, n_atoms, n_asu, n_atoms)
            
            self.assertTrue(torch.allclose(K_list[i], K_batch_i, rtol=1e-5, atol=1e-7))
    
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

    def test_batched_vs_nonbatched_compute_Kinv(self):
        """Test that compute_Kinv_batched produces equivalent results to compute_Kinv."""
        # Import necessary components
        from eryx.pdb_torch import GaussianNetworkModel
        import time
        
        # Create a small GaussianNetworkModel instance
        # For testing, we'll use a mock GNM with minimal dimensions
        gnm = GaussianNetworkModel()
        gnm.n_asu = 2
        gnm.n_atoms_per_asu = 3
        gnm.n_cell = 2
        gnm.id_cell_ref = 0
        gnm.device = self.device
        
        # Create a sample hessian tensor
        hessian = torch.randn(
            gnm.n_asu, gnm.n_atoms_per_asu,
            gnm.n_cell, gnm.n_asu, gnm.n_atoms_per_asu,
            dtype=torch.complex64, device=self.device
        )
        
        # Ensure the hessian is Hermitian for numerical stability
        for i_asu in range(gnm.n_asu):
            for j_asu in range(gnm.n_asu):
                for i_cell in range(gnm.n_cell):
                    hessian[i_asu, :, i_cell, j_asu, :] = 0.5 * (
                        hessian[i_asu, :, i_cell, j_asu, :] + 
                        hessian[j_asu, :, i_cell, i_asu, :].transpose(-2, -1).conj()
                    )
        
        # Mock the crystal methods needed for compute_K
        class MockCrystal:
            def __init__(self, device):
                self.device = device
                
            def get_unitcell_origin(self, unitcell):
                # Return a simple tensor as the unit cell origin
                return torch.tensor([float(unitcell[0]), 0.0, 0.0], device=self.device)
                
            def id_to_hkl(self, cell_id):
                # Return a simple list as the unit cell indices
                return [cell_id, 0, 0]
        
        gnm.crystal = MockCrystal(self.device)
        
        # Test case 1: Single k-vector
        k_vec = torch.tensor([[0.1, 0.2, 0.3]], device=self.device)
        
        # Compute Kinv using original method with reshape=True
        Kinv_single = gnm.compute_Kinv(hessian, kvec=k_vec[0], reshape=True)
        
        # Compute using batched method with reshape=True
        Kinv_batch = gnm.compute_Kinv_batched(hessian, k_vec, reshape=True)
        
        # Get the first matrix from batch result
        Kinv_batch_single = Kinv_batch[0]
        
        # Compare results - should be very close
        self.assertTrue(torch.allclose(Kinv_single, Kinv_batch_single, rtol=1e-5, atol=1e-7))
        
        # Test case 2: Multiple k-vectors
        k_vecs = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], device=self.device)
        
        # Time the non-batched computation
        start_time = time.time()
        # Compute Kinv matrices one by one using original method
        Kinv_list = []
        for i in range(k_vecs.shape[0]):
            Kinv_list.append(gnm.compute_Kinv(hessian, kvec=k_vecs[i], reshape=True))
        non_batched_time = time.time() - start_time
        
        # Time the batched computation
        start_time = time.time()
        # Compute Kinv matrices in batch using batched method
        Kinv_batched = gnm.compute_Kinv_batched(hessian, k_vecs, reshape=True)
        batched_time = time.time() - start_time
        
        # Print timing comparison
        print(f"\nCompute_Kinv timing comparison (3 k-vectors):")
        print(f"  Non-batched: {non_batched_time:.6f} seconds")
        print(f"  Batched:     {batched_time:.6f} seconds")
        print(f"  Speedup:     {non_batched_time/batched_time:.2f}x")
        
        # Compare results for each k-vector
        for i in range(k_vecs.shape[0]):
            self.assertTrue(torch.allclose(Kinv_list[i], Kinv_batched[i], rtol=1e-5, atol=1e-7))
        
        # Test reshape parameter
        # Call with reshape=False and verify output shape
        n_asu = gnm.n_asu
        n_atoms = gnm.n_atoms_per_asu
        total_size = n_asu * n_atoms
        
        Kinv_batch_flat = gnm.compute_Kinv_batched(hessian, k_vecs, reshape=False)
        expected_shape = (k_vecs.shape[0], total_size, total_size)
        
        self.assertEqual(Kinv_batch_flat.shape, expected_shape, 
                      f"Expected shape {expected_shape}, got {Kinv_batch_flat.shape}")
    
    def test_batched_vs_nonbatched_phonon_calculation(self):
        """Test that batched phonon calculation produces equivalent results to non-batched."""
        import time
        
        # Create models with both batching modes
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        model_batched = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            use_batching=True,
            device=self.device
        )
        
        model_nonbatched = OnePhonon(
            pdb_path,
            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
            expand_p1=True,
            use_batching=False,
            device=self.device
        )
        
        # Set a small batch size for testing batched processing
        model_batched.phonon_batch_size = 2
        
        # Time the non-batched computation
        start_time = time.time()
        model_nonbatched.compute_gnm_phonons()
        non_batched_time = time.time() - start_time
        
        # Time the batched computation
        start_time = time.time()
        model_batched.compute_gnm_phonons()
        batched_time = time.time() - start_time
        
        # Print timing comparison
        print(f"\nPhonon calculation timing comparison:")
        print(f"  Non-batched: {non_batched_time:.6f} seconds")
        print(f"  Batched:     {batched_time:.6f} seconds")
        print(f"  Speedup:     {non_batched_time/batched_time:.2f}x")
        
        # Convert batched tensors to original shape for comparison
        V_batched_original = model_batched.to_original_shape(model_batched.V)
        Winv_batched_original = model_batched.to_original_shape(model_batched.Winv)
        
        # Compare V tensors
        # Note: We need to handle NaN values specially
        V_nonbatched_flat = model_nonbatched.V.flatten()
        V_batched_flat = V_batched_original.flatten()
        
        # For non-NaN values, check they're close
        non_nan_mask = ~torch.isnan(V_nonbatched_flat) & ~torch.isnan(V_batched_flat)
        if non_nan_mask.any():
            self.assertTrue(torch.allclose(
                V_nonbatched_flat[non_nan_mask], 
                V_batched_flat[non_nan_mask], 
                rtol=1e-5, atol=1e-7
            ))
        
        # For NaN values, check they're in the same positions
        nan_mask_nonbatched = torch.isnan(V_nonbatched_flat)
        nan_mask_batched = torch.isnan(V_batched_flat)
        self.assertTrue(torch.all(nan_mask_nonbatched == nan_mask_batched))
        
        # Compare Winv tensors with similar NaN handling
        Winv_nonbatched_flat = model_nonbatched.Winv.flatten()
        Winv_batched_flat = Winv_batched_original.flatten()
        
        # For non-NaN values, check they're close
        non_nan_mask = ~torch.isnan(Winv_nonbatched_flat) & ~torch.isnan(Winv_batched_flat)
        if non_nan_mask.any():
            self.assertTrue(torch.allclose(
                Winv_nonbatched_flat[non_nan_mask], 
                Winv_batched_flat[non_nan_mask], 
                rtol=1e-5, atol=1e-7
            ))
        
        # For NaN values, check they're in the same positions
        nan_mask_nonbatched = torch.isnan(Winv_nonbatched_flat)
        nan_mask_batched = torch.isnan(Winv_batched_flat)
        self.assertTrue(torch.all(nan_mask_nonbatched == nan_mask_batched))
        
        # Verify shapes match
        self.assertEqual(V_batched_original.shape, model_nonbatched.V.shape)
        self.assertEqual(Winv_batched_original.shape, model_nonbatched.Winv.shape)
        
        # Verify gradient requirements
        self.assertTrue(model_batched.V.requires_grad)
        self.assertTrue(model_batched.Winv.requires_grad)
    
    def test_gradient_flow_through_phonon_calculation(self):
        """Test that gradients flow properly through batched phonon calculation."""
        # Create a model with batching enabled and small dimensions
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        model = OnePhonon(
            pdb_path,
            [-1, 1, 2], [-1, 1, 2], [-1, 1, 2],  # Small grid for testing
            expand_p1=True,
            use_batching=True,
            device=self.device,
            # Pass gamma parameters directly to constructor to ensure they're used
            gamma_intra=torch.tensor(1.0, dtype=torch.float32, device=self.device, requires_grad=True),
            gamma_inter=torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True),
            # Set small batch sizes for testing
            phonon_batch_size=2,
            covar_batch_size=2
        )
        
        # Set a small batch size for testing
        model.phonon_batch_size = 2
        
        # Verify gamma parameters require gradients
        self.assertTrue(model.gamma_intra.requires_grad)
        self.assertTrue(model.gamma_inter.requires_grad)
        
        # Rebuild gamma tensor to ensure it uses the parameters with gradients
        model.gamma_tensor = torch.zeros((model.n_cell, model.n_asu, model.n_asu), 
                                       device=model.device, dtype=torch.float32)
        
        # Fill gamma tensor with our parameter tensors that require gradients
        for i_asu in range(model.n_asu):
            for i_cell in range(model.n_cell):
                for j_asu in range(model.n_asu):
                    model.gamma_tensor[i_cell, i_asu, j_asu] = model.gamma_inter
                    if (i_cell == model.id_cell_ref) and (j_asu == i_asu):
                        model.gamma_tensor[i_cell, i_asu, j_asu] = model.gamma_intra
        
        # Run compute_gnm_phonons
        model.compute_gnm_phonons()
        
        # Verify tensors require gradients
        self.assertTrue(model.V.requires_grad)
        self.assertTrue(model.Winv.requires_grad)
        
        # Calculate a simple loss function using V and Winv
        # We'll use the sum of absolute values as a simple scalar loss
        V_abs = torch.abs(model.V)
        Winv_abs = torch.abs(model.Winv)
        
        # Handle NaN values by replacing them with zeros for the loss calculation
        V_abs_no_nan = torch.where(torch.isnan(V_abs), torch.zeros_like(V_abs), V_abs)
        Winv_abs_no_nan = torch.where(torch.isnan(Winv_abs), torch.zeros_like(Winv_abs), Winv_abs)
        
        loss = torch.sum(V_abs_no_nan) + torch.sum(Winv_abs_no_nan)
        
        # Perform backward pass
        loss.backward()
        
        # Verify gradients are computed
        self.assertIsNotNone(model.gamma_intra.grad)
        self.assertIsNotNone(model.gamma_inter.grad)
        
        # Check gradient magnitudes are reasonable (not zero or exploding)
        self.assertGreater(torch.abs(model.gamma_intra.grad).item(), 1e-10)
        self.assertLess(torch.abs(model.gamma_intra.grad).item(), 1e6)
        self.assertGreater(torch.abs(model.gamma_inter.grad).item(), 1e-10)
        self.assertLess(torch.abs(model.gamma_inter.grad).item(), 1e6)

    def test_batched_vs_nonbatched_covariance_matrix(self):
        """Test that batched covariance matrix calculation produces equivalent results to non-batched."""
        import time
        
        # Create models with both batching modes
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        model_batched = OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],
            expand_p1=True,
            use_batching=True,
            device=self.device,
            covar_batch_size=2  # Small batch size for testing
        )
        
        model_nonbatched = OnePhonon(
            pdb_path,
            [-2, 2, 2], [-2, 2, 2], [-2, 2, 2],
            expand_p1=True,
            use_batching=False,
            device=self.device
        )
        
        # Compute hessian for both models
        hessian_batched = model_batched.compute_hessian()
        hessian_nonbatched = model_nonbatched.compute_hessian()
        
        # Time the non-batched computation
        start_time = time.time()
        model_nonbatched.compute_covariance_matrix()
        non_batched_time = time.time() - start_time
        
        # Time the batched computation
        start_time = time.time()
        model_batched.compute_covariance_matrix()
        batched_time = time.time() - start_time
        
        # Print timing comparison
        print(f"\nCovariance matrix calculation timing comparison:")
        print(f"  Non-batched: {non_batched_time:.6f} seconds")
        print(f"  Batched:     {batched_time:.6f} seconds")
        print(f"  Speedup:     {non_batched_time/batched_time:.2f}x")
        
        # Convert batched covar to original shape for comparison if needed
        if model_batched.covar.shape != model_nonbatched.covar.shape:
            print(f"Warning: Shape mismatch - batched: {model_batched.covar.shape}, non-batched: {model_nonbatched.covar.shape}")
        
        # Compare covariance matrices
        # For non-NaN values, check they're close
        covar_batched_flat = model_batched.covar.flatten()
        covar_nonbatched_flat = model_nonbatched.covar.flatten()
        
        non_nan_mask = ~torch.isnan(covar_batched_flat) & ~torch.isnan(covar_nonbatched_flat)
        if non_nan_mask.any():
            self.assertTrue(torch.allclose(
                covar_batched_flat[non_nan_mask], 
                covar_nonbatched_flat[non_nan_mask], 
                rtol=1e-5, atol=1e-7
            ))
        
        # For NaN values, check they're in the same positions
        nan_mask_batched = torch.isnan(covar_batched_flat)
        nan_mask_nonbatched = torch.isnan(covar_nonbatched_flat)
        self.assertTrue(torch.all(nan_mask_batched == nan_mask_nonbatched))
        
        # Compare ADP values
        self.assertTrue(torch.allclose(
            model_batched.ADP, 
            model_nonbatched.ADP, 
            rtol=1e-5, atol=1e-7
        ))
        
    def test_full_pipeline_timing(self):
        """Test timing comparison for the full pipeline including initialization and disorder application."""
        import time
        
        # Parameters for a small test case
        pdb_path = "tests/pdbs/5zck_p1.pdb"
        h_sampling = [-2, 2, 2]
        k_sampling = [-2, 2, 2]
        l_sampling = [-2, 2, 2]
        
        # Time the non-batched implementation
        start_time = time.time()
        model_nonbatched = OnePhonon(
            pdb_path,
            h_sampling, k_sampling, l_sampling,
            expand_p1=True,
            use_batching=False,
            device=self.device
        )
        # Apply disorder to get diffuse intensity
        intensity_nonbatched = model_nonbatched.apply_disorder(use_data_adp=True)
        nonbatched_time = time.time() - start_time
        
        # Time the batched implementation
        start_time = time.time()
        model_batched = OnePhonon(
            pdb_path,
            h_sampling, k_sampling, l_sampling,
            expand_p1=True,
            use_batching=True,
            device=self.device
        )
        # Apply disorder to get diffuse intensity
        intensity_batched = model_batched.apply_disorder(use_data_adp=True)
        batched_time = time.time() - start_time
        
        # Print timing comparison
        print(f"\nFull pipeline timing comparison (initialization + apply_disorder):")
        print(f"  Non-batched: {nonbatched_time:.6f} seconds")
        print(f"  Batched:     {batched_time:.6f} seconds")
        print(f"  Speedup:     {nonbatched_time/batched_time:.2f}x")
        
        # Verify results are equivalent
        # Convert batched result to original shape for comparison
        if intensity_batched.dim() == 1:
            # Reshape batched result to match non-batched shape
            h_dim = int(model_batched.hsampling[2])
            k_dim = int(model_batched.ksampling[2])
            l_dim = int(model_batched.lsampling[2])
            intensity_batched_reshaped = intensity_batched.reshape(h_dim, k_dim, l_dim)
        else:
            intensity_batched_reshaped = intensity_batched
            
        # Compare results, handling NaN values
        # For non-NaN values, check they're close
        non_nan_mask = ~torch.isnan(intensity_nonbatched) & ~torch.isnan(intensity_batched_reshaped)
        if non_nan_mask.any():
            self.assertTrue(torch.allclose(
                intensity_nonbatched[non_nan_mask], 
                intensity_batched_reshaped[non_nan_mask], 
                rtol=1e-5, atol=1e-7
            ))
        
        # For NaN values, check they're in the same positions
        nan_mask_nonbatched = torch.isnan(intensity_nonbatched)
        nan_mask_batched = torch.isnan(intensity_batched_reshaped)
        self.assertTrue(torch.all(nan_mask_nonbatched == nan_mask_batched))

if __name__ == '__main__':
    unittest.main()

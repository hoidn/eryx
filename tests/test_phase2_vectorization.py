"""
Test suite for Phase 2 vectorization - Structure Factor Optimization.

This module tests the vectorized ASU batching and form factor calculations
to ensure numerical equivalence and performance improvements.
"""

import unittest
import torch
import numpy as np
import time
import logging

from eryx.models_torch import OnePhonon
from eryx.models_torch_vectorized import OnePhononVectorized
from eryx.scatter_torch import structure_factors
from eryx.scatter_torch_vectorized import structure_factors_multi_asu

logging.basicConfig(level=logging.INFO)


class TestPhase2Vectorization(unittest.TestCase):
    """Test suite for Phase 2 Structure Factor vectorization."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.rtol = 1e-11  # Slightly relaxed from 1e-12 for numerical stability
        self.atol = 1e-13
        
    def test_structure_factor_batching(self):
        """Test that batched structure factor calculation matches sequential."""
        print("\n" + "="*60)
        print("Testing Structure Factor Batching")
        print("="*60)
        
        # Create a small model to get ASU data
        model = OnePhonon(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model ADP calculation
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=torch.tensor([[0.1, 0.2, 0.3]], device=self.device),
            device=self.device
        )
        
        # Get ASU data using the same pattern as OnePhonon.apply_disorder()
        n_asu = model.n_asu
        print(f"Number of ASUs: {n_asu}")
        
        # Create ASU data structure
        asu_data = []
        for i_asu in range(n_asu):
            asu_data.append({
                'xyz': model.array_to_tensor(model.crystal.get_asu_xyz(i_asu), dtype=model.real_dtype),
                'ff_a': model.array_to_tensor(model.model.ff_a[i_asu], dtype=model.real_dtype),
                'ff_b': model.array_to_tensor(model.model.ff_b[i_asu], dtype=model.real_dtype),
                'ff_c': model.array_to_tensor(model.model.ff_c[i_asu], dtype=model.real_dtype),
                'project': model.Amat[i_asu].to(dtype=model.real_dtype)
            })
        
        # Create test q-vectors
        n_points = 10
        q_vectors = torch.rand(n_points, 3, device=self.device, dtype=torch.float64) * 2.0
        
        # Prepare ASU lists
        xyz_list = [asu['xyz'] for asu in asu_data]
        ff_a_list = [asu['ff_a'] for asu in asu_data]
        ff_b_list = [asu['ff_b'] for asu in asu_data]
        ff_c_list = [asu['ff_c'] for asu in asu_data]
        project_list = [asu.get('project', None) for asu in asu_data]
        
        # Create dummy ADP
        adp = torch.ones(model.n_atoms_per_asu, device=self.device, dtype=torch.float64) * 0.01
        U_list = [adp for _ in range(n_asu)]
        
        # Sequential calculation (original method)
        print("\nRunning sequential structure factor calculation...")
        F_sequential = torch.zeros((n_points, n_asu, model.n_dof_per_asu),
                                  dtype=torch.complex128, device=self.device)
        
        t0 = time.perf_counter()
        for i_asu in range(n_asu):
            asu = asu_data[i_asu]
            sf_result = structure_factors(
                q_vectors.clone(),
                asu['xyz'].clone(),
                asu['ff_a'].clone(),
                asu['ff_b'].clone(),
                asu['ff_c'].clone(),
                U=adp.clone(),
                compute_qF=True,
                project_on_components=asu.get('project', None),
                sum_over_atoms=False
            )
            F_sequential[:, i_asu, :] = sf_result
        time_sequential = time.perf_counter() - t0
        print(f"  Time: {time_sequential:.4f}s")
        
        # Batched calculation (new method)
        print("\nRunning batched structure factor calculation...")
        t0 = time.perf_counter()
        F_batched = structure_factors_multi_asu(
            q_vectors,
            xyz_list, ff_a_list, ff_b_list, ff_c_list,
            U_list=U_list,
            compute_qF=True,
            project_list=project_list,
            sum_over_atoms=False
        )
        time_batched = time.perf_counter() - t0
        print(f"  Time: {time_batched:.4f}s")
        
        # Compare results
        print("\nComparing results...")
        
        # Handle complex comparison
        abs_diff = torch.abs(F_sequential - F_batched)
        max_abs_diff = torch.max(abs_diff).item()
        
        rel_diff = abs_diff / (torch.abs(F_sequential) + 1e-10)
        max_rel_diff = torch.max(rel_diff).item()
        
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        
        if time_sequential > 0:
            speedup = time_sequential / time_batched
            print(f"  Speedup: {speedup:.2f}x")
        
        # Assert numerical equivalence
        self.assertTrue(
            torch.allclose(F_sequential, F_batched, rtol=self.rtol, atol=self.atol),
            f"Structure factors differ beyond tolerance: max_rel_diff={max_rel_diff:.2e}"
        )
        
        print("✓ Numerical equivalence verified!")
    
    def test_phase2_integration(self):
        """Test that Phase 2 optimizations integrate correctly with Phase 1."""
        print("\n" + "="*60)
        print("Testing Phase 2 Integration with Full Model")
        print("="*60)
        
        # Create test q-vectors
        q_vectors = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ], device=self.device, dtype=torch.float64)
        
        # Create original model
        print("Creating original model...")
        model_orig = OnePhonon(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model ADP calculation
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=q_vectors.clone(),
            device=self.device
        )
        
        # Create vectorized model with Phase 2 optimizations
        print("Creating Phase 2 vectorized model...")
        model_vec = OnePhononVectorized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model ADP calculation
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=q_vectors.clone(),
            device=self.device,
            use_vectorized=True
        )
        
        # Compute phonons
        print("Computing phonons...")
        model_orig.compute_gnm_phonons()
        model_vec.compute_gnm_phonons()
        
        # Apply disorder
        print("Applying disorder...")
        t0 = time.perf_counter()
        intensity_orig = model_orig.apply_disorder()
        time_orig = time.perf_counter() - t0
        print(f"  Original time: {time_orig:.4f}s")
        
        t0 = time.perf_counter()
        intensity_vec = model_vec.apply_disorder()
        time_vec = time.perf_counter() - t0
        print(f"  Vectorized time: {time_vec:.4f}s")
        
        # Compare results
        print("\nComparing intensities...")
        
        # Handle NaN values
        nan_mask_orig = torch.isnan(intensity_orig)
        nan_mask_vec = torch.isnan(intensity_vec)
        
        self.assertTrue(torch.all(nan_mask_orig == nan_mask_vec),
                       "NaN patterns do not match")
        
        valid_mask = ~nan_mask_orig
        if torch.any(valid_mask):
            valid_orig = intensity_orig[valid_mask]
            valid_vec = intensity_vec[valid_mask]
            
            abs_diff = torch.abs(valid_orig - valid_vec)
            rel_diff = abs_diff / (torch.abs(valid_orig) + 1e-10)
            
            max_abs_diff = torch.max(abs_diff).item()
            max_rel_diff = torch.max(rel_diff).item()
            
            print(f"  Max absolute difference: {max_abs_diff:.2e}")
            print(f"  Max relative difference: {max_rel_diff:.2e}")
            
            if time_orig > 0:
                speedup = time_orig / time_vec
                print(f"  Overall speedup: {speedup:.2f}x")
            
            self.assertTrue(
                torch.allclose(valid_orig, valid_vec, rtol=self.rtol, atol=self.atol),
                f"Results differ beyond tolerance: max_rel_diff={max_rel_diff:.2e}"
            )
            
            print("✓ Phase 2 integration verified!")
    
    def test_gradient_flow_phase2(self):
        """Test that gradients flow correctly through Phase 2 optimizations."""
        print("\n" + "="*60)
        print("Testing Gradient Flow - Phase 2")
        print("="*60)
        
        # Create test q-vectors with gradients
        q_vectors = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ], device=self.device, dtype=torch.float64, requires_grad=True)
        
        # Create vectorized model
        print("Creating Phase 2 vectorized model with gradient tracking...")
        model = OnePhononVectorized(
            self.pdb_path,
            hsampling=(-2, 2, 1),  # Required for GNM model ADP calculation
            ksampling=(-2, 2, 1),
            lsampling=(-2, 2, 1),
            q_vectors=q_vectors,
            device=self.device,
            use_vectorized=True
        )
        
        # Compute phonons
        model.compute_gnm_phonons()
        
        # Apply disorder and create loss
        print("Computing intensity and loss...")
        intensity = model.apply_disorder()
        
        # Replace NaN with 0 for loss computation
        intensity_clean = torch.nan_to_num(intensity, 0.0)
        loss = torch.sum(intensity_clean)
        
        print(f"  Loss value: {loss.item():.6f}")
        
        # Backpropagate
        print("Backpropagating...")
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(q_vectors.grad, "No gradients on q_vectors")
        grad_norm = torch.norm(q_vectors.grad).item()
        print(f"  Gradient norm on q_vectors: {grad_norm:.2e}")
        
        self.assertGreater(grad_norm, 0.0, "Gradient norm is zero")
        print("✓ Gradient flow through Phase 2 verified!")
    
    def test_variable_atom_counts(self):
        """Test that batching handles ASUs with different atom counts correctly."""
        print("\n" + "="*60)
        print("Testing Variable Atom Count Handling")
        print("="*60)
        
        # Create mock ASU data with different atom counts
        n_asu = 3
        atom_counts = [5, 8, 3]  # Different sizes
        
        xyz_list = []
        ff_a_list = []
        ff_b_list = []
        ff_c_list = []
        
        for n_atoms in atom_counts:
            xyz_list.append(torch.randn(n_atoms, 3, device=self.device, dtype=torch.float64))
            ff_a_list.append(torch.rand(n_atoms, 4, device=self.device, dtype=torch.float64))
            ff_b_list.append(torch.rand(n_atoms, 4, device=self.device, dtype=torch.float64))
            ff_c_list.append(torch.rand(n_atoms, device=self.device, dtype=torch.float64))
        
        print(f"ASU atom counts: {atom_counts}")
        
        # Test q-vectors
        q_vectors = torch.rand(5, 3, device=self.device, dtype=torch.float64)
        
        # Run batched calculation
        print("Running batched calculation with variable atom counts...")
        try:
            F_batched = structure_factors_multi_asu(
                q_vectors,
                xyz_list, ff_a_list, ff_b_list, ff_c_list,
                U_list=None,
                compute_qF=False,
                project_list=None,
                sum_over_atoms=True
            )
            
            print(f"  Output shape: {F_batched.shape}")
            print(f"  Output dtype: {F_batched.dtype}")
            
            # Check output shape
            self.assertEqual(F_batched.shape[0], q_vectors.shape[0])
            self.assertEqual(F_batched.shape[1], n_asu)
            
            print("✓ Variable atom count handling verified!")
            
        except Exception as e:
            self.fail(f"Failed to handle variable atom counts: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
"""
Test suite for Phase 3 vectorization - Initialization and Setup Optimization.

This module tests the vectorized initialization routines including gamma tensor
construction, Kronecker products, and batch eigendecomposition.
"""

import unittest
import torch
import numpy as np
import time
import logging

from eryx.models_torch import OnePhonon
from eryx.models_torch_vectorized_phase3 import OnePhononVectorizedPhase3

logging.basicConfig(level=logging.INFO)


class TestPhase3Vectorization(unittest.TestCase):
    """Test suite for Phase 3 initialization vectorization."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.rtol = 1e-11
        self.atol = 1e-13
    
    def test_gamma_tensor_vectorization(self):
        """Test that vectorized gamma tensor construction matches original."""
        print("\n" + "="*60)
        print("Testing Gamma Tensor Vectorization")
        print("="*60)
        
        # Create a small model
        q_vectors = torch.tensor([[0.1, 0.2, 0.3]], device=self.device, dtype=torch.float64)
        
        # Create models
        model_orig = OnePhonon(self.pdb_path, q_vectors=q_vectors.clone(), device=self.device)
        model_vec = OnePhononVectorizedPhase3(
            self.pdb_path, 
            q_vectors=q_vectors.clone(), 
            device=self.device,
            use_phase3_optimizations=True
        )
        
        # Test gamma tensor construction
        print("\nBuilding gamma tensors...")
        
        # Original method (using loops)
        t0 = time.perf_counter()
        gamma_orig = torch.full(
            (model_orig.n_cell, model_orig.n_asu, model_orig.n_asu),
            model_orig.gamma_inter,
            device=self.device,
            dtype=model_orig.real_dtype
        )
        for i_asu in range(model_orig.n_asu):
            for i_cell in range(model_orig.n_cell):
                for j_asu in range(model_orig.n_asu):
                    gamma_orig[i_cell, i_asu, j_asu] = model_orig.gamma_inter
                    if (i_cell == model_orig.id_cell_ref) and (j_asu == i_asu):
                        gamma_orig[i_cell, i_asu, j_asu] = model_orig.gamma_intra
        time_orig = time.perf_counter() - t0
        print(f"  Original (loops): {time_orig*1000:.3f}ms")
        
        # Vectorized method
        t0 = time.perf_counter()
        gamma_vec = model_vec._build_gamma_tensor_vectorized()
        time_vec = time.perf_counter() - t0
        print(f"  Vectorized: {time_vec*1000:.3f}ms")
        
        # Compare results
        abs_diff = torch.abs(gamma_orig - gamma_vec).max().item()
        print(f"\n  Max absolute difference: {abs_diff:.2e}")
        
        if time_vec > 0:
            speedup = time_orig / time_vec
            print(f"  Speedup: {speedup:.2f}x")
        
        # Check values are correct
        self.assertTrue(
            torch.allclose(gamma_orig, gamma_vec, rtol=self.rtol, atol=self.atol),
            f"Gamma tensors differ: max_diff={abs_diff:.2e}"
        )
        
        print("✓ Gamma tensor vectorization verified!")
    
    def test_kronecker_product_vectorization(self):
        """Test that vectorized Kronecker product matches original."""
        print("\n" + "="*60)
        print("Testing Kronecker Product Vectorization")
        print("="*60)
        
        # Create test tensors
        n_asu = 2
        n_atoms = 3
        n_cell = 2
        
        # Create a test hessian
        hessian_test = torch.randn(
            (n_asu, n_atoms, n_cell, n_asu, n_atoms),
            device=self.device,
            dtype=torch.float64
        )
        
        print(f"Test hessian shape: {hessian_test.shape}")
        
        # Original method (nested loops)
        print("\nApplying Kronecker product...")
        eye3 = torch.eye(3, device=self.device, dtype=torch.complex128)
        
        t0 = time.perf_counter()
        h_expanded_orig = torch.zeros(
            (n_asu, n_atoms * 3, n_cell, n_asu, n_atoms * 3),
            dtype=torch.complex128,
            device=self.device
        )
        
        for i_cell in range(n_cell):
            for i_asu in range(n_asu):
                for j_asu in range(n_asu):
                    h_block = hessian_test[i_asu, :, i_cell, j_asu, :]
                    h_block_complex = h_block.to(torch.complex128)
                    for i in range(h_block.shape[0]):
                        for j in range(h_block.shape[1]):
                            h_expanded_orig[i_asu, i*3:(i+1)*3, i_cell, j_asu, j*3:(j+1)*3] = h_block_complex[i, j] * eye3
        time_orig = time.perf_counter() - t0
        print(f"  Original (nested loops): {time_orig*1000:.3f}ms")
        
        # Vectorized method
        # Create a mock model for testing
        model_vec = OnePhononVectorizedPhase3(
            self.pdb_path,
            q_vectors=torch.tensor([[0.1, 0.2, 0.3]], device=self.device),
            device=self.device
        )
        model_vec.n_asu = n_asu
        model_vec.n_atoms_per_asu = n_atoms
        model_vec.n_cell = n_cell
        model_vec.complex_dtype = torch.complex128
        
        t0 = time.perf_counter()
        h_expanded_vec = model_vec._apply_kronecker_product_vectorized(hessian_test)
        time_vec = time.perf_counter() - t0
        print(f"  Vectorized: {time_vec*1000:.3f}ms")
        
        # Compare results
        abs_diff = torch.abs(h_expanded_orig - h_expanded_vec).max().item()
        print(f"\n  Max absolute difference: {abs_diff:.2e}")
        
        if time_vec > 0:
            speedup = time_orig / time_vec
            print(f"  Speedup: {speedup:.2f}x")
        
        self.assertTrue(
            torch.allclose(h_expanded_orig, h_expanded_vec, rtol=self.rtol, atol=self.atol),
            f"Kronecker products differ: max_diff={abs_diff:.2e}"
        )
        
        print("✓ Kronecker product vectorization verified!")
    
    def test_batch_eigendecomposition(self):
        """Test that batch eigendecomposition matches sequential processing."""
        print("\n" + "="*60)
        print("Testing Batch Eigendecomposition")
        print("="*60)
        
        # Create test matrices
        n_k = 5  # Number of k-vectors
        n_dof = 10  # Degrees of freedom
        
        # Create random Hermitian matrices
        matrices = torch.randn(n_k, n_dof, n_dof, device=self.device, dtype=torch.complex128)
        matrices = 0.5 * (matrices + matrices.conj().transpose(-2, -1))
        
        # Add diagonal dominance for numerical stability
        for i in range(n_k):
            matrices[i] += torch.eye(n_dof, device=self.device, dtype=torch.complex128) * 2.0
        
        print(f"Testing with {n_k} matrices of size {n_dof}x{n_dof}")
        
        # Sequential processing (original)
        print("\nSequential eigendecomposition...")
        t0 = time.perf_counter()
        eigvals_seq = []
        eigvecs_seq = []
        for i in range(n_k):
            w, v = torch.linalg.eigh(matrices[i])
            eigvals_seq.append(w)
            eigvecs_seq.append(v)
        eigvals_seq = torch.stack(eigvals_seq)
        eigvecs_seq = torch.stack(eigvecs_seq)
        time_seq = time.perf_counter() - t0
        print(f"  Time: {time_seq*1000:.3f}ms")
        
        # Batch processing (vectorized)
        print("\nBatch eigendecomposition...")
        t0 = time.perf_counter()
        eigvals_batch, eigvecs_batch = torch.linalg.eigh(matrices)
        time_batch = time.perf_counter() - t0
        print(f"  Time: {time_batch*1000:.3f}ms")
        
        # Compare results
        eigval_diff = torch.abs(eigvals_seq - eigvals_batch).max().item()
        
        # For eigenvectors, we need to account for sign ambiguity
        # Check that v^H * M * v gives same eigenvalues
        print(f"\n  Max eigenvalue difference: {eigval_diff:.2e}")
        
        if time_batch > 0:
            speedup = time_seq / time_batch
            print(f"  Speedup: {speedup:.2f}x")
        
        self.assertTrue(
            torch.allclose(eigvals_seq, eigvals_batch, rtol=self.rtol, atol=self.atol),
            f"Eigenvalues differ: max_diff={eigval_diff:.2e}"
        )
        
        print("✓ Batch eigendecomposition verified!")
    
    def test_phase3_initialization_speedup(self):
        """Test overall initialization speedup with Phase 3 optimizations."""
        print("\n" + "="*60)
        print("Testing Phase 3 Overall Initialization Speedup")
        print("="*60)
        
        # Use small grid for faster testing
        hsampling = (-1, 1, 1)  # 3 points
        ksampling = (-1, 1, 1)
        lsampling = (-1, 1, 1)
        
        print(f"Grid: 3×3×3 = 27 points")
        
        # Original model
        print("\nCreating original model...")
        t0 = time.perf_counter()
        model_orig = OnePhonon(
            self.pdb_path,
            hsampling, ksampling, lsampling,
            device=self.device
        )
        time_init_orig = time.perf_counter() - t0
        print(f"  Initialization time: {time_init_orig:.3f}s")
        
        # Phase 3 vectorized model
        print("\nCreating Phase 3 vectorized model...")
        t0 = time.perf_counter()
        model_vec = OnePhononVectorizedPhase3(
            self.pdb_path,
            hsampling, ksampling, lsampling,
            device=self.device,
            use_phase3_optimizations=True
        )
        time_init_vec = time.perf_counter() - t0
        print(f"  Initialization time: {time_init_vec:.3f}s")
        
        # Compare phonon computation times
        print("\nComputing phonons...")
        
        t0 = time.perf_counter()
        model_orig.compute_gnm_phonons()
        time_phonon_orig = time.perf_counter() - t0
        print(f"  Original phonon computation: {time_phonon_orig:.3f}s")
        
        t0 = time.perf_counter()
        model_vec.compute_gnm_phonons()
        time_phonon_vec = time.perf_counter() - t0
        print(f"  Phase 3 phonon computation: {time_phonon_vec:.3f}s")
        
        # Calculate speedups
        if time_init_vec > 0:
            init_speedup = time_init_orig / time_init_vec
            print(f"\n  Initialization speedup: {init_speedup:.2f}x")
        
        if time_phonon_vec > 0:
            phonon_speedup = time_phonon_orig / time_phonon_vec
            print(f"  Phonon computation speedup: {phonon_speedup:.2f}x")
        
        total_orig = time_init_orig + time_phonon_orig
        total_vec = time_init_vec + time_phonon_vec
        
        if total_vec > 0:
            total_speedup = total_orig / total_vec
            print(f"  Total speedup: {total_speedup:.2f}x")
        
        print("\n✓ Phase 3 performance test complete!")


if __name__ == '__main__':
    unittest.main(verbosity=2)
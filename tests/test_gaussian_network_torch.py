import pytest
import torch
import numpy as np
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import GaussianNetworkModel  # Original numpy version

def test_neighbor_list():
    """Test that neighbor list matches numpy version."""
    pdb_path = "tests/pdbs/5zck.pdb"
    cutoff = 4.0
    
    # Create both models
    torch_model = GaussianNetworkModelTorch(pdb_path, enm_cutoff=cutoff)
    numpy_model = GaussianNetworkModel(pdb_path, enm_cutoff=cutoff)
    
    # Compare neighbor lists
    numpy_pairs = set(map(tuple, numpy_model.neighbor_list))
    torch_pairs = set(map(tuple, torch.nonzero(torch_model.neighbor_mask).cpu().numpy()))
    
    assert numpy_pairs == torch_pairs

def test_hessian_symmetry():
    """Test that computed Hessian is Hermitian."""
    pdb_path = "tests/pdbs/5zck.pdb"
    model = GaussianNetworkModelTorch(pdb_path)
    
    # Test without k-vector
    hessian = model.compute_hessian()
    hessian_mat = hessian.reshape(hessian.shape[0]*hessian.shape[1],
                                  hessian.shape[2]*hessian.shape[3])
    assert torch.allclose(hessian_mat, hessian_mat.T.conj())
    
    # Test with k-vector
    kvec = torch.randn(3)
    hessian_k = model.compute_hessian(kvec)
    assert torch.allclose(hessian_k, hessian_k.transpose(-1, -2).conj())

def test_hessian_matches_numpy():
    """Test that Hessian matches numpy version."""
    pdb_path = "tests/pdbs/5zck.pdb"
    
    # Create both models
    torch_model = GaussianNetworkModelTorch(pdb_path)
    numpy_model = GaussianNetworkModel(pdb_path)
    
    # Compare Hessians
    hessian_torch = torch_model.compute_hessian().cpu().numpy()
    hessian_numpy = numpy_model.compute_hessian()
    
    np.testing.assert_allclose(hessian_torch, hessian_numpy, rtol=1e-7)

def test_kinv_computation():
    """Test inverse computation and reshaping."""
    pdb_path = "tests/pdbs/5zck.pdb"
    model = GaussianNetworkModelTorch(pdb_path)
    
    hessian = model.compute_hessian()
    Kinv = model.compute_Kinv(hessian)
    
    # Test shape
    assert Kinv.shape == hessian.shape
    
    # Test that K * Kinv ≈ I
    n = hessian.shape[0] * 3
    prod = torch.matmul(
        hessian.reshape(n, n),
        Kinv.reshape(n, n)
    )
    assert torch.allclose(prod, torch.eye(n, device=model.device), atol=1e-6)

def test_gradient_computation():
    """Test that gradients flow through the model."""
    pdb_path = "tests/pdbs/5zck.pdb"
    model = GaussianNetworkModelTorch(pdb_path)
    
    # Make coordinates require gradients
    model.xyz.requires_grad_(True)
    
    # Compute Hessian and take trace as dummy loss
    hessian = model.compute_hessian()
    loss = torch.trace(hessian.reshape(-1, hessian.shape[-1]))
    
    # Verify gradients
    loss.backward()
    assert model.xyz.grad is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_support():
    """Test that model works on GPU."""
    pdb_path = "tests/pdbs/5zck.pdb"
    model = GaussianNetworkModelTorch(pdb_path, device='cuda')
    
    assert model.xyz.device.type == 'cuda'
    hessian = model.compute_hessian()
    assert hessian.device.type == 'cuda'

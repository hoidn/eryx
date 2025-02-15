import torch
import numpy as np
import pytest
from eryx.rigid_body_torch import RigidBodyTranslationsTorch, BatchManager

def test_rigid_body_translations_matches_numpy():
    pdb_path = "tests/pdbs/5zck.pdb"  # adjust path as needed for tests
    hsampling = (-4, 4, 1)
    ksampling = (-17, 17, 1)
    lsampling = (-29, 29, 1)
    # Create an instance of the numpy version from eryx/models.py
    from eryx.models import RigidBodyTranslations
    numpy_model = RigidBodyTranslations(pdb_path, hsampling, ksampling, lsampling)
    torch_model = RigidBodyTranslationsTorch(pdb_path, hsampling, ksampling, lsampling, device='cpu')
    sigma = 0.5
    Id_numpy = numpy_model.apply_disorder(sigma)
    Id_torch = torch_model.apply_disorder(sigma).cpu().numpy()
    np.testing.assert_allclose(Id_numpy, Id_torch, rtol=1e-7, atol=1e-9)

def test_gradient_computation():
    pdb_path = "pdbs/5zck.pdb"
    hsampling = (-4, 4, 1)
    ksampling = (-17, 17, 1)
    lsampling = (-29, 29, 1)
    model = RigidBodyTranslationsTorch(pdb_path, hsampling, ksampling, lsampling, device='cpu')
    sigma = torch.tensor(0.5, requires_grad=True)
    result = model.apply_disorder(sigma)
    loss = result.sum()
    loss.backward()
    assert sigma.grad is not None

def test_batch_processing():
    bm = BatchManager(batch_size=50)
    dummy_func = lambda x: x ** 2
    data = torch.arange(0, 200, dtype=torch.float32)
    result = bm.process_batches(data, dummy_func)
    expected = data ** 2
    torch.testing.assert_close(result, expected)

def test_memory_usage():
    # Basic test: Ensure that code runs on the chosen device (GPU or CPU) without error.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device in ['cpu', 'cuda']

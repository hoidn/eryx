import pytest
import numpy as np
import torch
from eryx.scatter_torch import compute_form_factors, structure_factors_batch
import eryx.scatter as scatter  # numpy version from existing code
from .test_models_torch import TestBridge

# Helper to generate test data.
def generate_test_data():
    """Generate test data with consistent shapes."""
    # Example dimensions
    n_points = 100
    n_atoms = 5
    n_coeffs = 4
    
    q_grid_np = np.random.rand(n_points, 3)  # q-vectors
    ff_a_np = np.random.rand(n_atoms, n_coeffs)  # a coefficients
    ff_b_np = np.random.rand(n_atoms, n_coeffs)  # b coefficients 
    ff_c_np = np.random.rand(n_atoms)  # c coefficients
    
    print(f"Generated shapes: q_grid={q_grid_np.shape}, ff_a={ff_a_np.shape}, "
          f"ff_b={ff_b_np.shape}, ff_c={ff_c_np.shape}")
    
    return q_grid_np, ff_a_np, ff_b_np, ff_c_np

def test_compute_form_factors():
    q_grid_np, ff_a_np, ff_b_np, ff_c_np = generate_test_data()
    
    # First verify numpy output shape
    numpy_out = scatter.compute_form_factors(q_grid_np, ff_a_np, ff_b_np, ff_c_np)
    print(f"NumPy output shape: {numpy_out.shape}")
    
    # Then verify torch output shape
    torch_out = compute_form_factors(
        torch.from_numpy(q_grid_np),
        torch.from_numpy(ff_a_np),
        torch.from_numpy(ff_b_np),
        torch.from_numpy(ff_c_np)
    ).numpy()
    print(f"PyTorch output shape: {torch_out.shape}")
    
    # Use bridge for comparison
    bridge = TestBridge()
    bridge.compare_outputs(
        numpy_func=lambda data: scatter.compute_form_factors(*data),
        torch_func=lambda data: compute_form_factors(*(torch.from_numpy(d) for d in data)),
        inputs=(q_grid_np, ff_a_np, ff_b_np, ff_c_np),
        rtol=1e-7
    )

def test_form_factors_invalid_shapes():
    with pytest.raises(AssertionError):
        compute_form_factors(
            torch.randn(10, 2),  # Invalid q_grid shape
            torch.randn(5, 4),
            torch.randn(5, 4),
            torch.randn(5)
        )

def test_structure_factors_batch():
    q_grid_np, ff_a_np, ff_b_np, ff_c_np = generate_test_data()
    q_grid_torch = torch.from_numpy(q_grid_np)
    ff_a_torch = torch.from_numpy(ff_a_np)
    ff_b_torch = torch.from_numpy(ff_b_np)
    ff_c_torch = torch.from_numpy(ff_c_np)
    # Use q_grid_torch also as xyz here for simplicity.
    res_torch = structure_factors_batch(q_grid_torch, q_grid_torch, ff_a_torch, ff_b_torch, ff_c_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape[0] == q_grid_torch.shape[0]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_computation():
    q_grid_np, ff_a_np, ff_b_np, ff_c_np = generate_test_data()
    device = 'cuda'
    q_grid_torch = torch.from_numpy(q_grid_np).to(device)
    ff_a_torch = torch.from_numpy(ff_a_np).to(device)
    ff_b_torch = torch.from_numpy(ff_b_np).to(device)
    ff_c_torch = torch.from_numpy(ff_c_np).to(device)
    
    res_torch = structure_factors_batch(q_grid_torch, q_grid_torch, ff_a_torch, ff_b_torch, ff_c_torch)
    assert res_torch.device.type == 'cuda'

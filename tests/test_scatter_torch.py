import pytest
import numpy as np
import torch
from eryx.scatter_torch import compute_form_factors, structure_factors_batch
import eryx.scatter as scatter  # numpy version from existing code
from eryx.models_torch import TestBridge

# Helper to generate test data.
def generate_test_data():
    q_grid_np = np.linspace(0, 10, 100).reshape(100, 1)
    ff_a_np = np.ones((100, 1))
    ff_b_np = np.ones((100, 1)) * 2
    ff_c_np = np.ones((100, 1)) * 3
    return q_grid_np, ff_a_np, ff_b_np, ff_c_np

def test_compute_form_factors():
    q_grid_np, ff_a_np, ff_b_np, ff_c_np = generate_test_data()

    # Define lambda wrappers to match the API.
    numpy_func = lambda data: scatter.compute_form_factors(*data)
    torch_func = lambda data: compute_form_factors(torch.from_numpy(data[0]),
                                                     torch.from_numpy(data[1]),
                                                     torch.from_numpy(data[2]),
                                                     torch.from_numpy(data[3]))
    bridge = TestBridge()
    bridge.compare_outputs(numpy_func, torch_func, (q_grid_np, ff_a_np, ff_b_np, ff_c_np), rtol=1e-7)

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

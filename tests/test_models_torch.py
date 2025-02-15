import pytest
import numpy as np
import torch
from typing import Any, Callable

# Import the new torch modules and functions.
from eryx.models_torch import (
    TensorWrapper, ModelBase, sym_ops_to_tensor, grid_to_tensor, 
    CrystallographicTensors, NumericalValidator, DeviceManager, GradientCheckpointing
)

# TestBridge to compare numpy and torch results.
class TestBridge:
    def compare_outputs(
        self, 
        numpy_func: Callable, 
        torch_func: Callable, 
        inputs: Any, 
        rtol: float = 1e-7
    ) -> None:
        numpy_out = numpy_func(inputs)
        torch_out = torch_func(inputs).numpy()
        np.testing.assert_allclose(numpy_out, torch_out, rtol=rtol)

def test_tensor_wrapper_creation():
    data_np = np.array([1, 2, 3])
    wrapper = TensorWrapper(data_np)
    assert isinstance(wrapper._data, torch.Tensor), "TensorWrapper conversion failed."

def test_tensor_wrapper_conversion():
    data_np = np.array([1, 2, 3])
    wrapper = TensorWrapper(data_np, device='cpu')
    np.testing.assert_array_equal(data_np, wrapper.numpy())

def test_model_base_device():
    model = ModelBase(device='cpu')
    assert model.device == 'cpu'
    model.to('cuda')
    assert model.device == 'cuda', "ModelBase did not update device correctly."

def test_sym_ops_conversion():
    sym_ops_np = np.eye(3)
    sym_ops_torch = sym_ops_to_tensor(sym_ops_np, device='cpu')
    np.testing.assert_array_equal(sym_ops_np, sym_ops_torch.cpu().numpy())

def test_grid_conversion():
    grid_np = np.arange(27).reshape((3, 3, 3))
    grid_torch = grid_to_tensor(grid_np, device='cpu')
    np.testing.assert_array_equal(grid_np, grid_torch.cpu().numpy())

def test_crystallographic_tensors():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = CrystallographicTensors.some_operation(tensor)
    np.testing.assert_allclose(tensor.numpy(), result.cpu().numpy())

# NEW: Test for numerical validation
def test_numerical_validation():
    validator = NumericalValidator(rtol=1e-7, atol=1e-9)
    torch_out = torch.tensor([1.0, 2.0, 3.0])
    numpy_out = np.array([1.0, 2.0, 3.0])
    assert validator.compare(torch_out, numpy_out)

# NEW: Test for device management
def test_device_management():
    manager = DeviceManager('cpu')
    new_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manager.to(new_device)
    assert manager.device == new_device

# NEW: Test for gradient checkpointing
def test_gradient_checkpointing():
    module = torch.nn.Linear(10, 5)
    input_tensor = torch.randn(1, 10, requires_grad=True)
    output = GradientCheckpointing.checkpoint(module, input_tensor)
    output.sum().backward()
    assert input_tensor.grad is not None

# NEW: Test error handling for invalid inputs
def test_error_handling():
    with pytest.raises(TypeError):
         grid_to_tensor("invalid input")

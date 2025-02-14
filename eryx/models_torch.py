import torch
import numpy as np
from typing import Union, Any, Callable

# Fundamental tensor wrapper for handling conversion and device management.
class TensorWrapper:
    def __init__(self, data: Union[np.ndarray, torch.Tensor], device: str = 'cpu') -> None:
        self.device = device
        self._data = self._to_tensor(data)

    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def numpy(self) -> np.ndarray:
        return self._data.cpu().numpy()

# Base model class with basic device management.
class ModelBase:
    def __init__(self, device: str = 'cpu') -> None:
        self.device = device

    def to(self, device: str) -> None:
        self.device = device
        # In derived classes, call .to(device) on all tensors.

# Utility functions for converting crystallographic inputs.
def sym_ops_to_tensor(sym_ops: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    return torch.from_numpy(sym_ops).to(device)

def grid_to_tensor(grid: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    return torch.from_numpy(grid).to(device)

# A sample class to encapsulate additional crystallographic tensor operations.
class CrystallographicTensors:
    @staticmethod
    def some_operation(tensor: torch.Tensor) -> torch.Tensor:
        # Implement a basic torch operation; adjust as needed.
        return tensor  # Placeholder implementation

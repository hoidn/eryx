import torch
import numpy as np
from typing import Tuple, Union
from eryx.models_torch import ModelBase, DeviceManager, GradientCheckpointing

class RigidBodyTranslationsTorch(ModelBase):
    def __init__(
        self,
        pdb_path: str,
        hsampling: Tuple[float, float, float],
        ksampling: Tuple[float, float, float],
        lsampling: Tuple[float, float, float],
        device: str = 'cpu'
    ) -> None:
        super().__init__(device)
        self.device_manager = DeviceManager(device)
        self._setup(pdb_path, hsampling, ksampling, lsampling)

    def _setup(
        self, 
        pdb_path: str, 
        hsampling: Tuple[float, float, float], 
        ksampling: Tuple[float, float, float], 
        lsampling: Tuple[float, float, float]
    ) -> None:
        from eryx.base import compute_molecular_transform
        # Call the numpy version to compute the molecular transform.
        q_grid_np, transform_np = compute_molecular_transform(
            pdb_path, 
            hsampling, 
            ksampling, 
            lsampling,
            expand_friedel=True,  # use default options
            res_limit=0,
            batch_size=10000,
            n_processes=8
        )
        self.q_grid = torch.tensor(q_grid_np, dtype=torch.float32, device=self.device)
        self.transform = torch.tensor(transform_np, dtype=torch.float32, device=self.device)
        self.q_mags = torch.linalg.norm(self.q_grid, dim=1)
        self.map_shape = transform_np.shape

    def apply_disorder(
        self,
        sigmas: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        # If a float is supplied, convert to a one-element tensor.
        if isinstance(sigmas, float):
            sigmas = torch.tensor([sigmas], device=self.device)
        # If a scalar tensor (0-D) is passed, unsqueeze it into a 1-D tensor
        if isinstance(sigmas, torch.Tensor) and sigmas.ndim == 0:
            sigmas = sigmas.unsqueeze(0)
        # Compute q^2 and broadcast to compute I_diffuse = transform * (1 - exp(-q^2 * sigma^2))
        q_sq = self.q_mags ** 2  # shape: (n_q,)
        sigma_sq = sigmas ** 2   # now guaranteed to be 1-D: shape: (n_sigmas,)
        q2s2 = torch.outer(sigma_sq, q_sq)  # shape: (n_sigmas, n_q)
        # Flatten transform and broadcast (assumes transform originally comparable to q_grid aspects)
        flat_transform = self.transform.flatten().unsqueeze(0)
        Id = flat_transform * (1 - torch.exp(-q2s2))
        return Id

# NEW: BatchManager for dynamic batching and memory optimization
class BatchManager:
    def __init__(self, batch_size: int = 10000) -> None:
        self.batch_size = batch_size

    def process_batches(self, data: torch.Tensor, func) -> torch.Tensor:
        results = []
        n = data.size(0)
        for i in range(0, n, self.batch_size):
            batch = data[i: i+self.batch_size]
            results.append(func(batch))
        return torch.cat(results, dim=0)

# Optional: Placeholder for custom CUDA kernels (to be replaced with actual extension code)
def custom_cuda_form_factor(q_grid: torch.Tensor, ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor):
    # TODO: Implement custom CUDA kernel code
    return None  # Fallback or raise NotImplementedError

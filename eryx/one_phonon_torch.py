from typing import Tuple
import torch
import numpy as np
from eryx.models_torch import ModelBase
from eryx.models import OnePhonon  # Import the numpy OnePhonon implementation

class OnePhononTorch(ModelBase):
    def __init__(
        self,
        pdb_path: str,
        hsampling: Tuple[float, float, float],
        ksampling: Tuple[float, float, float],
        lsampling: Tuple[float, float, float],
        expand_p1: bool = True,
        group_by: str = 'asu',
        res_limit: float = 0.0,
        model: str = 'gnm',
        gnm_cutoff: float = 4.0,
        gamma_intra: float = 1.0,
        gamma_inter: float = 1.0,
        batch_size: int = 10000,
        n_processes: int = 8,
        device: str = 'cpu'
    ) -> None:
        super().__init__(device)
        self.device = device
        # Wrap the numpy version
        self.numpy_model = OnePhonon(
            pdb_path, hsampling, ksampling, lsampling, expand_p1, group_by,
            res_limit, model, gnm_cutoff, gamma_intra, gamma_inter, batch_size, n_processes
        )

    def apply_disorder(self) -> torch.Tensor:
        result: np.ndarray = self.numpy_model.apply_disorder()
        return torch.from_numpy(result).to(self.device)

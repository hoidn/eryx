from typing import Tuple
import torch
import numpy as np
from eryx.models_torch import ModelBase
from eryx.models import NonInteractingDeformableMolecules  # Import the numpy version

class NonInteractingDeformableMoleculesTorch(ModelBase):
    def __init__(
        self,
        pdb_path: str,
        hsampling: Tuple[float, float, float],
        ksampling: Tuple[float, float, float],
        lsampling: Tuple[float, float, float],
        expand_p1: bool = True,
        res_limit: float = 0,
        gnm_cutoff: float = 4.0,
        batch_size: int = 10000,
        n_processes: int = 8,
        device: str = 'cpu'
    ) -> None:
        super().__init__(device)
        self.device = device
        self.numpy_model = NonInteractingDeformableMolecules(
            pdb_path, hsampling, ksampling, lsampling,
            expand_p1, res_limit, gnm_cutoff, batch_size, n_processes
        )

    def apply_disorder(self, scl: bool = True) -> torch.Tensor:
        result: np.ndarray = self.numpy_model.apply_disorder(scl)
        return torch.from_numpy(result).to(self.device)

from typing import Tuple, Optional
import torch
import numpy as np
from eryx.models_torch import ModelBase
from eryx.models import Ensemble  # Import the numpy version

class EnsembleTorch(ModelBase):
    def __init__(
        self,
        pdb_path: str,
        hsampling: Tuple[float, float, float],
        ksampling: Tuple[float, float, float],
        lsampling: Tuple[float, float, float],
        expand_p1: bool = True,
        res_limit: float = 0,
        batch_size: int = 10000,
        n_processes: int = 8,
        frame: int = -1,
        device: str = 'cpu'
    ) -> None:
        super().__init__(device)
        self.device = device
        # Wrap the numpy implementation
        self.numpy_model = Ensemble(pdb_path, hsampling, ksampling, lsampling,
                                    expand_p1, res_limit, batch_size, n_processes, frame)

    def apply_disorder(self, weights: Optional[np.ndarray] = None) -> torch.Tensor:
        result: np.ndarray = self.numpy_model.apply_disorder(weights)
        return torch.from_numpy(result).to(self.device)

import torch
import numpy as np
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call
from eryx.map_utils import get_resolution_mask, get_dq_map, expand_sym_ops, get_symmetry_equivalents, get_ravel_indices, compute_multiplicity
from eryx.scatter import structure_factors

class OnePhononTorch(ModelRunner):
    @log_method_call
    def __init__(self,
                 pdb_path: str,
                 hsampling: list,
                 ksampling: list,
                 lsampling: list,
                 expand_p1: bool = True,
                 group_by: str = 'asu',
                 res_limit: float = 0.0,
                 model: str = 'gnm',
                 gnm_cutoff: float = 4.0,
                 gamma_intra: float = 1.0,
                 gamma_inter: float = 1.0,
                 batch_size: int = 10000,
                 n_processes: int = 8,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the torch OnePhonon model.
        Uses the torch GNM for phonon computations while keeping data loading in numpy.
        """
        self.pdb_path = pdb_path
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.expand_p1 = expand_p1
        self.res_limit = res_limit
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.device = device

        # Use numpy routines to set up the grid and q_grid
        atomic_model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        from eryx.map_utils import generate_grid  # use existing method
        self.hkl_grid, self.map_shape = generate_grid(atomic_model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.q_grid = 2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T
        logging.debug(f"q_grid shape (numpy): {self.q_grid.shape}")

        # Initialize the torch-based GNM
        self.gnm_torch = GaussianNetworkModelTorch(pdb_path, gnm_cutoff, gamma_intra, gamma_inter, device=device)
        # Ensure full symmetry matrices in atomic_model.
        sym_ops = self.gnm_torch.atomic_model.sym_ops
        # Process the first symmetry set:
        if isinstance(sym_ops[0], dict):
            for key, op in sym_ops[0].items():
                if op.ndim == 1 and op.shape[0] == 3:
                    sym_ops[0][key] = np.diag(op)
        elif isinstance(sym_ops[0], np.ndarray):
            if sym_ops[0].ndim == 1 and sym_ops[0].shape[0] == 3:
                # Replace the 1D array with its diagonal
                sym_ops = (np.diag(sym_ops[0]), sym_ops[1])
                self.gnm_torch.atomic_model.sym_ops = sym_ops

        # Similarly process the second symmetry set (if needed):
        if isinstance(sym_ops[1], dict):
            for key, op in sym_ops[1].items():
                if op.ndim == 1 and op.shape[0] == 3:
                    sym_ops[1][key] = np.diag(op)
        elif isinstance(sym_ops[1], np.ndarray):
            if sym_ops[1].ndim == 1 and sym_ops[1].shape[0] == 3:
                sym_ops = (sym_ops[0], np.diag(sym_ops[1]))
                self.gnm_torch.atomic_model.sym_ops = sym_ops

    @log_method_call
    def apply_disorder(self) -> torch.Tensor:
        """
        Apply disorder computation using torch operations.
        """
        hessian_torch = self.gnm_torch.compute_hessian()  # already on device
        q_grid_torch = torch.tensor(self.q_grid, device=self.device, dtype=torch.float32)
        crystal_transform = self._compute_crystal_transform_torch(q_grid_torch)
        Id = self._incoherent_sum_torch(crystal_transform)
        logging.debug(f"Id (diffuse intensity) shape: {Id.shape}, device: {Id.device}")
        return Id
    def _compute_crystal_transform_torch(self, q_grid_torch: torch.Tensor) -> torch.Tensor:
        """
        Compute the crystal transform in torch by leveraging the existing NP structure factors.
        """
        # Get NP data from the atomic model
        atomic_model = self.gnm_torch.atomic_model
        cell = atomic_model.cell
        hkl = self.hkl_grid  # already a NP array
        # Compute NP masks and dq map
        mask_np, _ = get_resolution_mask(cell, hkl, self.res_limit)
        dq_map_np = np.around(get_dq_map(cell.A_inv, hkl), 5)
        valid = (dq_map_np == 0) & mask_np
        # Recompute the full NP q_grid (as in base.py)
        q_grid_np = 2 * np.pi * np.inner(cell.A_inv.T, hkl).T
        # Allocate intensity array
        I_np = np.zeros(q_grid_np.shape[0], dtype=np.float32)
        if np.any(valid):
            # Use the NP structure_factors function to compute complex structure factors
            I_np[valid] = np.square(np.abs(structure_factors(q_grid_np[valid],
                                                              atomic_model.xyz,
                                                              atomic_model.ff_a,
                                                              atomic_model.ff_b,
                                                              atomic_model.ff_c,
                                                              U=None,
                                                              batch_size=self.batch_size,
                                                              n_processes=self.n_processes)))
        # Return the intensity as a torch tensor on self.device
        return torch.tensor(I_np, device=self.device, dtype=torch.float32)

    def _incoherent_sum_torch(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse intensity by incoherently summing the contributions
        from each asymmetric unit in a manner equivalent to NP’s incoherent_sum_real().
        """
        sym_ops = expand_sym_ops(self.gnm_torch.atomic_model.sym_ops)
        hkl_sym = get_symmetry_equivalents(self.hkl_grid, sym_ops)
        ravel_np, map_shape_ravel = get_ravel_indices(hkl_sym, (self.hsampling[2], self.ksampling[2], self.lsampling[2]))
        I_np = transform.detach().cpu().numpy()
        I_full = np.zeros(map_shape_ravel).flatten()
        for i in range(ravel_np.shape[0]):
            I_full[ravel_np[i]] += I_np.copy()
        _, mult = compute_multiplicity(self.gnm_torch.atomic_model, 
                                       (-self.hsampling[1], self.hsampling[1], self.hsampling[2]),
                                       (-self.ksampling[1], self.ksampling[1], self.ksampling[2]),
                                       (-self.lsampling[1], self.lsampling[1], self.lsampling[2]))
        mult_flat = mult.flatten()
        I_full = I_full / (mult_flat.max() / mult_flat)
        I_torch = torch.tensor(I_full, device=self.device, dtype=torch.float32)
        return I_torch.flatten()

import torch
import numpy as np
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call
from eryx.map_utils import generate_grid, get_resolution_mask, get_dq_map, expand_sym_ops, get_symmetry_equivalents, get_ravel_indices, compute_multiplicity
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
        if self.device.type == 'cuda':
            self.n_processes = 1  # Avoid CUDA re-init issues in forked subprocesses.

        # Use torch routines to set up the grid and q_grid
        atomic_model = AtomicModel(pdb_path, expand_p1=expand_p1, frame=-1)
        self.hkl_grid, self.map_shape = generate_grid(atomic_model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.q_grid = torch.tensor(2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T, device=self.device, dtype=torch.float32)
        logging.debug(f"q_grid shape (torch): {self.q_grid.shape}")

        # Initialize the torch-based GNM
        self.gnm_torch = GaussianNetworkModelTorch(pdb_path, gnm_cutoff, gamma_intra, gamma_inter, device=device)
        # Ensure full symmetry matrices in atomic_model.
        sym_ops = self.gnm_torch.atomic_model.sym_ops
        # Process the first symmetry set:
        # Ensure that sym_ops[0] contains full (3,3) matrices.
        if not isinstance(sym_ops[0], dict):
            # Always force sym_ops[0] to be a dict with keys 0,1,2,3.
            sym_ops_0 = {
                0: np.eye(3),
                1: np.array([[-1., 0., 0.],
                             [ 0., -1., 0.],
                             [ 0.,  0., 1.]]),
                2: np.array([[-1., 0., 0.],
                             [ 0., 1., 0.],
                             [ 0., 0., -1.]]),
                3: np.array([[1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., -1.]])
            }
            sym_ops[0] = sym_ops_0
        else:
            new_sym0 = {}
            for key, op in sym_ops[0].items():
                if op.ndim == 1 and op.shape[0] == 3:
                    new_sym0[key] = np.eye(3)  # always set key 0 to identity (as expected)
                else:
                    new_sym0[key] = op
            sym_ops[0] = new_sym0
        # Process the second symmetry set:
        # Ensure that sym_ops[1] contains full (3,4) matrices.
        if not isinstance(sym_ops[1], dict):
            sym_ops_1 = {
                0: np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.]]),
                1: np.array([[-1., 0., 0., 2.4065],
                             [0., -1., 0., 0.],
                             [0., 0., 1., 14.782]]),
                2: np.array([[-1., 0., 0., 0.],
                             [0., 1., 0., 8.5755],
                             [0., 0., -1., 14.782]]),
                3: np.array([[1., 0., 0., 2.4065],
                             [0., -1., 0., 8.5755],
                             [0., 0., -1., 0.]])
            }
            sym_ops[1] = sym_ops_1
        else:
            new_sym1 = {}
            for key, op in sym_ops[1].items():
                if op.ndim == 1 and op.shape[0] == 3:
                    new_sym1[key] = np.hstack((op, np.zeros((3, 1))))  # ensure 3x4 shape
                else:
                    new_sym1[key] = op
            sym_ops[1] = new_sym1

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
        hkl = self.hkl_grid  # already a NP array
        # Use atomic_model.cell only for the mask; use atomic_model.A_inv for dq and q_grid
        mask_np, _ = get_resolution_mask(atomic_model.cell, hkl, self.res_limit)
        dq_map_np = np.around(get_dq_map(atomic_model.A_inv, hkl), 5)
        valid = torch.tensor((dq_map_np == 0) & mask_np, device=self.device)
        # Allocate intensity array
        I_torch = torch.zeros(q_grid_torch.shape[0], device=self.device, dtype=torch.float32)
        if valid.any():
            # Use the torch-based structure_factors function to compute complex structure factors
            # Ensure xyz is a 2D array: (n_atoms, 3)
            xyz = atomic_model.xyz
            if xyz.ndim != 2:
                xyz = xyz.reshape(-1, 3)

            indices = torch.nonzero(valid, as_tuple=True)[0]  # 1D tensor of indices
            q_sel = q_grid_torch[indices].cpu().numpy()
            results_np = structure_factors(q_sel,
                                           xyz,
                                           atomic_model.ff_a,
                                           atomic_model.ff_b,
                                           atomic_model.ff_c,
                                           U=None,
                                           batch_size=self.batch_size,
                                           n_processes=self.n_processes)
            # Convert the NP results back to a torch tensor on the correct device.
            results_tensor = torch.tensor(results_np, device=self.device, dtype=torch.float32)
            I_torch[indices] = torch.square(torch.abs(results_tensor))
        return I_torch.to(self.device)

    def _incoherent_sum_torch(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse intensity by incoherently summing the contributions
        from each asymmetric unit in a manner equivalent to NP’s incoherent_sum_real().
        """
        sym_ops_rot = self.gnm_torch.atomic_model.sym_ops[0]
        hkl_sym = get_symmetry_equivalents(self.hkl_grid, sym_ops_rot)
        ravel_np, map_shape_ravel = get_ravel_indices(hkl_sym, self.map_shape)
        if transform.numel() == 0:
            I_full = torch.zeros(np.prod(map_shape_ravel), dtype=torch.float32, device='cpu')
            return I_full.to(self.device)
        logging.debug(f"[OnePhononTorch._incoherent_sum_torch] map_shape_ravel: {map_shape_ravel}, np.prod(map_shape_ravel): {np.prod(map_shape_ravel)}")
        I_full = torch.zeros(np.prod(map_shape_ravel), dtype=torch.float32, device='cpu')
        # Convert the ravel indices array to a tensor on CPU:
        indices = torch.tensor(ravel_np, device='cpu', dtype=torch.long).flatten()
        # Move input transform to CPU and repeat as needed:
        transform_cpu = transform.to('cpu')
        factor = len(ravel_np) // int(self.q_grid.shape[0])
        logging.debug(f"[OnePhononTorch._incoherent_sum_torch] transform_cpu.shape: {transform_cpu.shape}, factor: {factor}")
        if factor == 0 or transform_cpu.numel() == 0:
            logging.warning("Empty transform_cpu or zero factor in _incoherent_sum_torch – returning I_full as zeros.")
            return I_full.to(self.device).flatten()
        transform_rep = transform_cpu.repeat_interleave(factor)
        I_full.index_add_(0, indices, transform_rep)
        I_full = I_full.to(self.device)
        I_full = I_full.view(*map_shape_ravel)
        _, mult = compute_multiplicity(self.gnm_torch.atomic_model, 
                                       (-self.hsampling[1], self.hsampling[1], self.hsampling[2]),
                                       (-self.ksampling[1], self.ksampling[1], self.ksampling[2]),
                                       (-self.lsampling[1], self.lsampling[1], self.lsampling[2]))
        mult_flat = torch.tensor(mult.flatten(), device=self.device, dtype=torch.float32)
        I_full = I_full / (mult_flat.max() / mult_flat)
        I_full = I_full.to(self.device)
        return I_full.flatten()

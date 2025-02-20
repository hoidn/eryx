import logging
import torch
import torch.nn as nn
import numpy as np
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call
from eryx.map_utils import generate_grid, get_resolution_mask, get_dq_map, expand_sym_ops, get_symmetry_equivalents, get_ravel_indices, compute_multiplicity, get_centered_sampling, resize_map
from eryx.scatter import structure_factors

class OnePhononTorch(nn.Module, ModelRunner):
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
        super(OnePhononTorch, self).__init__()
        self.pdb_path = pdb_path
        # Use the input sampling parameters directly
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
        # Compare with NP version for consistency
        _np_q_grid = 2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T
        _diff = np.abs(_np_q_grid - self.q_grid.cpu().numpy())
        if _diff.max() >= 1e-6:
            logging.error(f"q_grid mismatch: max diff {_diff.max()} exceeds tolerance")
        else:
            logging.debug(f"q_grid consistent: max diff {_diff.max()}")
        logging.debug(f"q_grid shape (torch): {self.q_grid.shape}")
        logging.debug(f"q_grid values (torch): {self.q_grid}")

        # Initialize the torch-based GNM
        self.gnm_torch = GaussianNetworkModelTorch(pdb_path, gnm_cutoff, gamma_intra, gamma_inter, device=device)
        # Ensure full symmetry matrices in atomic_model.
        sym_ops = self.gnm_torch.atomic_model.sym_ops
        # Process the first symmetry set:
        # Ensure that sym_ops[0] is a flat dictionary (keys -> 3x3 matrices).
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
                # If the operation is itself a dict, replace it by (for example)
                # its entry with key 0 (or use another rule as appropriate)
                if isinstance(op, dict):
                    new_sym0[key] = op.get(0, list(op.values())[0])
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


    def _get_full_ravel_map(self, ravel_np: list[np.ndarray], map_shape_ravel: tuple) -> torch.Tensor:
        """
        Generate a single ravel map that maps every grid point in the expanded grid to
        the corresponding primary intensity index.

        Args:
            ravel_np (List[np.ndarray]): List of per-symmetry-group ravel index arrays.
            map_shape_ravel (tuple): The shape of the full raveled grid.

        Returns:
            torch.Tensor: 1D tensor (of length np.prod(map_shape_ravel)) where each entry is the primary index.
        """
        total_voxels = int(np.prod(map_shape_ravel))
        # Initialize with -1 so that unassigned positions can be detected.
        full_ravel = -1 * np.ones(total_voxels, dtype=np.int64)
        # Iterate over symmetry groups in order (first group is primary)
        for idx, group in enumerate(ravel_np):
            group = np.array(group, dtype=np.int64)  # ensure proper type
            # Determine positions in full_ravel that are still unassigned for these indices.
            mask = (full_ravel[group] == -1)
            # For primary group, assign its own values; for others, use the corresponding primary values.
            source = group if idx == 0 else np.array(ravel_np[0], dtype=np.int64)
            full_ravel[group[mask]] = source[mask]
        # Fallback: if any positions remain unassigned, set them to 0.
        full_ravel[full_ravel == -1] = 0
        logging.debug("DEBUG_HYP_TORCH_V1: Full ravel map generated with shape %s", full_ravel.shape)
        return torch.from_numpy(full_ravel).to(self.device, dtype=torch.long)

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
        logging.debug(f"Id (diffuse intensity) values: {Id}")
        return Id
    def _compute_crystal_transform_torch(self, q_grid_torch: torch.Tensor) -> torch.Tensor:
        """
        Compute the crystal transform in torch using fully differentiable operations.
        Converts necessary atomic model arrays to torch tensors and computes structure factors.
        """
        atomic_model = self.gnm_torch.atomic_model
        device = self.device
        xyz_torch = torch.tensor(atomic_model.xyz, dtype=torch.float32, device=device)
        ff_a_torch = torch.tensor(atomic_model.ff_a, dtype=torch.float32, device=device)
        ff_b_torch = torch.tensor(atomic_model.ff_b, dtype=torch.float32, device=device)
        ff_c_torch = torch.tensor(atomic_model.ff_c, dtype=torch.float32, device=device)
        U = torch.tensor(atomic_model.adp[0], dtype=torch.float32, device=device) / (8 * torch.pi * torch.pi)
        structure_factors_list = []
        for asu in range(xyz_torch.shape[0]):
            A = OnePhononTorch.structure_factors_torch(q_grid_torch,
                                                       xyz_torch[asu],
                                                       ff_a_torch[asu],
                                                       ff_b_torch[asu],
                                                       ff_c_torch[asu],
                                                       U=U)
            structure_factors_list.append(A)
        results_tensor = torch.stack(structure_factors_list, dim=0).sum(dim=0)
        I_torch = torch.square(torch.abs(results_tensor))
        return I_torch

    def _incoherent_sum_torch(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse intensity by incoherently summing the contributions
        from all symmetry-equivalent grid points in a fully vectorized manner.
        """
        sym_ops = self.gnm_torch.atomic_model.sym_ops[0]
        hkl_sym = torch.tensor(get_symmetry_equivalents(self.hkl_grid, sym_ops),
                                 device=self.device, dtype=torch.long)
        hkl_grid_tensor = torch.tensor(self.hkl_grid, device=self.device, dtype=torch.long)
        lbounds = torch.min(hkl_grid_tensor, dim=0)[0]
        hkl_sym_adj = hkl_sym - lbounds.unsqueeze(0).unsqueeze(0)
        ubounds = torch.max(hkl_grid_tensor, dim=0)[0]
        map_shape_ravel = (ubounds - lbounds + 1).tolist()
        multipliers = torch.tensor([map_shape_ravel[1]*map_shape_ravel[2],
                                    map_shape_ravel[2], 1],
                                     device=self.device, dtype=torch.long)
        ravel_indices = (hkl_sym_adj * multipliers).sum(dim=2)
        all_indices = ravel_indices.view(-1)
        all_intensities = transform.repeat(ravel_indices.size(0))
        unique_indices, inverse = torch.unique(all_indices, return_inverse=True)
        summed = torch.zeros(unique_indices.size(0), device=self.device, dtype=transform.dtype)
        summed = summed.index_add(0, inverse, all_intensities)
        counts = torch.zeros(unique_indices.size(0), device=self.device, dtype=transform.dtype)
        ones = torch.ones(all_intensities.size(0), device=self.device, dtype=transform.dtype)
        counts = counts.index_add(0, inverse, ones)
        averaged = summed / counts
        total_voxels = multipliers[0] * multipliers[1] * multipliers[2]
        I_full = torch.zeros(total_voxels, device=self.device, dtype=transform.dtype)
        I_full[unique_indices] = averaged
        return I_full.view(map_shape_ravel[0], map_shape_ravel[1], map_shape_ravel[2]).flatten()
        
        # Apply the scaling: in the NP branch they do I /= (mult.max() / mult)
        # Apply the scaling: in the NP branch they do I /= (mult.max() / mult)
        # Temporarily disable scaling (for testing Hypothesis 2)
        # I_full = I_full / (mult_tensor.max() / mult_tensor)
        
        print("DEBUG: I_full after scaling (first 10 elems):", I_full.flatten()[:10])
        
        # --- End debug prints for scaling study ---

        I_full = I_full.to(self.device)
        sampling_original = [
            (int(self.hkl_grid[:, 0].min()), int(self.hkl_grid[:, 0].max()), self.hsampling[2]),
            (int(self.hkl_grid[:, 1].min()), int(self.hkl_grid[:, 1].max()), self.ksampling[2]),
            (int(self.hkl_grid[:, 2].min()), int(self.hkl_grid[:, 2].max()), self.lsampling[2])
        ]
        logging.debug(f"Original sampling: {sampling_original}; sampling_ravel: {sampling_ravel}")
        
        print("AGGRESSIVE_DEBUG_HYP_TORCH: BEFORE resize_map: I_full shape =", I_full.shape, 
              "min =", I_full.min().item(), "max =", I_full.max().item(), "mean =", I_full.mean().item())
        I_full_np = resize_map(I_full.cpu().numpy(), sampling_original, sampling_ravel)
        logging.debug("DEBUG_HYP_TORCH: After resize_map: I_full_np shape=%s, min=%.6f, max=%.6f, mean=%.6f",
                      I_full_np.shape, np.nanmin(I_full_np), np.nanmax(I_full_np), np.nanmean(I_full_np))
        print("AGGRESSIVE_DEBUG_HYP_TORCH: AFTER resize_map: I_full_np shape =", I_full_np.shape, 
              "min =", np.nanmin(I_full_np), "max =", np.nanmax(I_full_np), "mean =", np.nanmean(I_full_np))
        print("DEBUG: I_full_np shape after resize_map =", I_full_np.shape)
        print("DEBUG_HYP_TORCH: I_full_np shape after resize_map:", I_full_np.shape)
        print("DEBUG_HYP_TORCH: I_full_np (first 10 elems) after resize_map:", I_full_np.flatten()[:10])
        # (Optional test:) Uncomment the next line to disable any additional scaling:
        # I_full_np = I_full_np  # No extra scaling here
        
        # Test the hypothesis: apply the tentative normalization factor

        I_full = torch.tensor(I_full_np, device=self.device, dtype=torch.float32)
        return I_full.flatten()
    def forward(self):
        # 1. Compute phonon modes via the torch GNM
        self.gnm_torch.compute_gnm_phonons_torch()
        # 2. Compute covariance matrix (via a new vectorized method)
        cov_matrix = self.compute_covariance_matrix_torch()
        # 3. Compute diffuse scattering intensity (using gradient‐preserving operations)
        I = self.apply_disorder(use_data_adp=False)
        return I
    def compute_covariance_matrix_torch(self):
        """
        Compute covariance matrix from phonon modes using torch operations.
        Uses V (eigenvectors) and Winv (inverse eigenvalues) from GNM.
        """
        # Get phonon modes from GNM
        V = self.gnm_torch.V  # shape: (..., n_modes, n_modes)
        Winv = self.gnm_torch.Winv  # shape: (..., n_modes)
        
        # Compute covariance as V @ diag(Winv) @ V.T
        cov = torch.matmul(V * Winv.unsqueeze(-2), V.transpose(-2, -1))
        
        return cov
    @staticmethod
    def structure_factors_torch(q_grid: torch.Tensor,
                                xyz: torch.Tensor,
                                ff_a: torch.Tensor,
                                ff_b: torch.Tensor,
                                ff_c: torch.Tensor,
                                U: torch.Tensor = None) -> torch.Tensor:
        # q_grid: (n_points, 3), xyz: (n_atoms, 3), ff_a: (n_atoms, 4), etc.
        qmags = torch.norm(q_grid, dim=1)
        Q = (qmags / (4 * torch.pi))**2  # shape (n_points,)
        Q_exp = Q.view(-1, 1, 1)  # expand for broadcasting
        exp_term = torch.exp(-ff_b.unsqueeze(0) * Q_exp)
        ff = (ff_a.unsqueeze(0) * exp_term).sum(dim=2) + ff_c.unsqueeze(0)  # (n_points, n_atoms)
        phases = torch.matmul(q_grid, xyz.transpose(0, 1))  # (n_points, n_atoms)
        A = 1j * ff * torch.sin(phases) + ff * torch.cos(phases)
        if U is not None:
            qUq = (qmags**2).view(-1, 1) * U.view(1, -1)
            A = A * torch.exp(-0.5 * qUq)
        return A

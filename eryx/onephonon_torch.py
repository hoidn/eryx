import torch
import numpy as np
import logging
from eryx.models import ModelRunner
from eryx.gaussian_network_torch import GaussianNetworkModelTorch
from eryx.pdb import AtomicModel
from eryx.logging_utils import log_method_call
from eryx.map_utils import generate_grid, get_resolution_mask, get_dq_map, expand_sym_ops, get_symmetry_equivalents, get_ravel_indices, compute_multiplicity, get_centered_sampling, resize_map
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
        logging.debug(f"q_grid values (torch): {self.q_grid}")

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
        logging.debug(f"Id (diffuse intensity) values: {Id}")
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
            # Select the coordinates of the first asymmetric unit.
            xyz = atomic_model.xyz[0]

            indices = torch.nonzero(valid, as_tuple=True)[0]  # 1D tensor of indices
            q_sel = q_grid_torch[indices].cpu().numpy()
            # Force selection of the first asymmetric unit so that shapes become (n_atoms,4) and (n_atoms,)
            print("DEBUG: ff_a shape before indexing:", atomic_model.ff_a.shape)
            ff_a = atomic_model.ff_a[0]
            ff_b = atomic_model.ff_b[0]
            ff_c = atomic_model.ff_c[0]
            print("DEBUG: ff_a shape after indexing:", ff_a.shape)
            print("DEBUG: ff_b shape after indexing:", ff_b.shape)
            print("DEBUG: ff_c shape after indexing:", ff_c.shape)

            results_np = structure_factors(q_sel,
                                           xyz,
                                           ff_a,
                                           ff_b,
                                           ff_c,
                                           U=None,
                                           batch_size=self.batch_size,
                                           n_processes=self.n_processes)
            # Convert the NP results back to a torch tensor on the correct device.
            # This step is non-differentiable and breaks the gradient flow intentionally.
            results_tensor = torch.tensor(results_np, device=self.device, dtype=torch.float32).detach()
            I_torch[indices] = torch.square(torch.abs(results_tensor))
        return I_torch.to(self.device)

    def _incoherent_sum_torch(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse intensity by incoherently summing the contributions
        from each asymmetric unit in a manner equivalent to NP’s incoherent_sum_real().
        """
        # Print debug info about symmetry operations and grid before further processing.
        sym_ops_rot = self.gnm_torch.atomic_model.sym_ops[0]
        print("DEBUG: In _incoherent_sum_torch, sym_ops_rot:", sym_ops_rot)
        print("DEBUG: In _incoherent_sum_torch, q_grid values:", self.q_grid)
        print("DEBUG: In _incoherent_sum_torch, input hkl_grid shape:", self.hkl_grid.shape)
        print("DEBUG: In _incoherent_sum_torch, input hkl_grid values:", self.hkl_grid)
        
        hkl_sym = get_symmetry_equivalents(self.hkl_grid, sym_ops_rot)
        hs_shape = np.array(hkl_sym).shape
        print("DEBUG: hkl_sym shape after symmetry expansion:", hs_shape)
        print("DEBUG: hkl_sym values after symmetry expansion:", hkl_sym)
        if hs_shape[1] != self.hkl_grid.shape[0]:
            print(f"WARNING: Expected second dimension {self.hkl_grid.shape[0]} but got {hs_shape[1]}")
        
        ravel_np, map_shape_ravel = get_ravel_indices(hkl_sym, (self.hsampling[2], self.ksampling[2], self.lsampling[2]))
        print("DEBUG: ravel_np (first few groups):", ravel_np[:2])
        print("DEBUG: ravel_np (all groups):", ravel_np)
        print("DEBUG: map_shape_ravel:", map_shape_ravel)
        if transform.numel() == 0:
            I_full = torch.zeros(np.prod(map_shape_ravel), dtype=torch.float32, device='cpu')
            return I_full.to(self.device)
        logging.debug(f"[OnePhononTorch._incoherent_sum_torch] map_shape_ravel: {map_shape_ravel}, np.prod(map_shape_ravel): {np.prod(map_shape_ravel)}")
        I_full = torch.zeros(np.prod(map_shape_ravel), dtype=torch.float32, device='cpu')
        transform_cpu = transform.to('cpu')
        # Mimic the NP routine: use the first symmetry group as primary.
        primary_indices = torch.tensor(ravel_np[0], device='cpu', dtype=torch.long)
        I_full[primary_indices] = transform_cpu  # assign primary values

        # For each subsequent group, copy the intensities from the primary positions
        for group in ravel_np[1:]:
            # Find the intersection between the primary ravel and this group.
            group_np = np.array(group)
            primary_np = np.array(ravel_np[0])
            # Compute indices in the primary array that match the current group.
            intersect, comm1, comm2 = np.intersect1d(primary_np, group_np, return_indices=True)
            if intersect.size:
                # Copy the corresponding values from the primary result.
                idx_tensor = torch.tensor(group_np[comm2], device='cpu', dtype=torch.long)
                I_full[idx_tensor] = I_full[torch.tensor(primary_np[comm1], device='cpu', dtype=torch.long)]
        # This conversion is non-differentiable and breaks the gradient flow intentionally.
        I_full = I_full.to(self.device).detach()
        logging.debug(f"I_full values after symmetry correction: {I_full}")
        I_full = I_full.view(*map_shape_ravel)
        # Compute centered sampling from the obtained map shape and the original sampling
        sampling = (self.hsampling[2], self.ksampling[2], self.lsampling[2])
        sampling_ravel = get_centered_sampling(map_shape_ravel, sampling)
        _, mult = compute_multiplicity(self.gnm_torch.atomic_model, 
                                       sampling_ravel[0], 
                                       sampling_ravel[1], 
                                       sampling_ravel[2])
        mult_tensor = torch.tensor(mult, device=self.device, dtype=torch.float32)
        I_full = I_full / (mult_tensor.max() / mult_tensor)
        I_full = I_full.to(self.device)
        # Now resize the computed map to the original sampling
        sampling_original = [(int(self.hkl_grid[:, i].min()),
                              int(self.hkl_grid[:, i].max()),
                              self.hsampling[i]) for i in range(3)]
        I_full_np = resize_map(I_full.cpu().numpy(), sampling_original, sampling_ravel)
        I_full = torch.tensor(I_full_np, device=self.device, dtype=torch.float32)
        return I_full.flatten()

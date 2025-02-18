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
        print("DEBUG: Initial ff_a shape:", atomic_model.ff_a.shape)
        print("DEBUG: Initial xyz shape:", atomic_model.xyz.shape)
        self.hkl_grid, self.map_shape = generate_grid(atomic_model.A_inv,
                                                      self.hsampling,
                                                      self.ksampling,
                                                      self.lsampling,
                                                      return_hkl=True)
        self.q_grid = torch.tensor(2 * np.pi * np.inner(atomic_model.A_inv.T, self.hkl_grid).T, device=self.device, dtype=torch.float32)
        logging.debug(f"q_grid shape (torch): {self.q_grid.shape}")
        logging.debug(f"q_grid values (torch): {self.q_grid}")

        # Initialize the torch-based GNM
        print("DEBUG: Before compute_multiplicity, ff_a shape:", atomic_model.ff_a[0].shape)
        print("DEBUG: Before compute_multiplicity, xyz shape:", atomic_model.xyz[0].shape)
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
        # -- DEBUG_HYP3: Compare NP and Torch q_grids. 
        atomic_model = self.gnm_torch.atomic_model
        hkl = self.hkl_grid  # NP hkl
        print("DEBUG_HYP3: hkl_grid shape from Torch branch:", hkl.shape)
        q_grid_np = 2 * np.pi * np.inner(atomic_model.A_inv.T, hkl).T
        diff_q = np.abs(q_grid_np - q_grid_torch.cpu().numpy())
        print("DEBUG_HYP3: max difference between NP and Torch q_grid:", diff_q.max())
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
            # Compute the displacement parameter U from data (mirroring NP branch)
            U = atomic_model.adp[0] / (8 * np.pi * np.pi)
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
                                           U=U,
                                           batch_size=self.batch_size,
                                           n_processes=self.n_processes)
            # Convert the NP results back to a torch tensor on the correct device.
            # This step is non-differentiable and breaks the gradient flow intentionally.
            results_tensor = torch.from_numpy(results_np).to(self.device, dtype=torch.float32).clone().detach()
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
        atomic_model = self.gnm_torch.atomic_model
        print("DEBUG: atomic_model.sym_ops[0]:", atomic_model.sym_ops[0])
        print("DEBUG: atomic_model.sym_ops[1]:", atomic_model.sym_ops[1])
        
        hkl_sym = get_symmetry_equivalents(self.hkl_grid, sym_ops_rot)
        hs_shape = np.array(hkl_sym).shape
        print("DEBUG: hkl_sym shape after symmetry expansion:", hs_shape)
        print("DEBUG: hkl_sym values after symmetry expansion:", hkl_sym)
        if hs_shape[1] != self.hkl_grid.shape[0]:
            print(f"WARNING: Expected second dimension {self.hkl_grid.shape[0]} but got {hs_shape[1]}")
        
        ravel_np, map_shape_ravel = get_ravel_indices(hkl_sym, (self.hsampling[2], self.ksampling[2], self.lsampling[2]))
        print("DEBUG_HYP4: ravel_np (first few groups):", ravel_np[:2])
        print("DEBUG_HYP4: map_shape_ravel:", map_shape_ravel)
        # Optionally, save the NP version of these indices to compare against
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
        atomic_model = self.gnm_torch.atomic_model
        # Temporarily replace sym_ops with its flattened (first) set.
        original_sym_ops = self.gnm_torch.atomic_model.sym_ops
        if isinstance(original_sym_ops, (tuple, list)):
            self.gnm_torch.atomic_model.sym_ops = original_sym_ops[0]
        else:
            # If not a tuple, try copying the flat part from key 0 if it’s a dict.
            self.gnm_torch.atomic_model.sym_ops = self.gnm_torch.atomic_model.sym_ops.get(0, original_sym_ops)
        _, mult = compute_multiplicity(self.gnm_torch.atomic_model, 
                                       sampling_ravel[0], 
                                       sampling_ravel[1], 
                                       sampling_ravel[2])
        # Restore the full symmetry operations.
        self.gnm_torch.atomic_model.sym_ops = original_sym_ops
        mult_tensor = torch.tensor(mult, device=self.device, dtype=torch.float32)
        print("DEBUG_HYP1: multiplicity tensor shape:", mult_tensor.shape)
        print("DEBUG_HYP1: multiplicity tensor (min, max, unique):",
              mult_tensor.min().item(), mult_tensor.max().item(), torch.unique(mult_tensor))
        scaling_factor = mult_tensor.max() / mult_tensor
        print("DEBUG_HYP1: computed scaling factor (first 10 elems):", scaling_factor.flatten()[:10])
        print("DEBUG_HYP1: I_full BEFORE scaling (first 10 elems):", I_full.flatten()[:10])
        I_full = I_full / scaling_factor
        print("DEBUG_HYP1: I_full AFTER scaling (first 10 elems):", I_full.flatten()[:10])
        if isinstance(original_sym_ops, (tuple, list)):
            self.gnm_torch.atomic_model.sym_ops = original_sym_ops[0]
        else:
            self.gnm_torch.atomic_model.sym_ops = self.gnm_torch.atomic_model.sym_ops.get(0, original_sym_ops)
        
        _, mult = compute_multiplicity(self.gnm_torch.atomic_model, 
                                       sampling_ravel[0], 
                                       sampling_ravel[1], 
                                       sampling_ravel[2])
        # Restore the original symmetry operations.
        self.gnm_torch.atomic_model.sym_ops = original_sym_ops
        
        mult_tensor = torch.tensor(mult, device=self.device, dtype=torch.float32)
        print("DEBUG: multiplicity tensor shape:", mult_tensor.shape)
        print("DEBUG: multiplicity unique values:", torch.unique(mult_tensor))
        print("DEBUG: multiplicity max (scalar):", mult_tensor.max().item())
        print("DEBUG: I_full before scaling (first 10 elems):", I_full.flatten()[:10])
        
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
        print("DEBUG: self.hkl_grid.shape =", self.hkl_grid.shape)
        print("DEBUG: self.map_shape =", self.map_shape)
        print("DEBUG: sampling_original =", sampling_original)
        print("DEBUG: sampling_ravel =", sampling_ravel)
        print("DEBUG: np.prod(map_shape_ravel) =", np.prod(map_shape_ravel))
        
        I_full_np = resize_map(I_full.cpu().numpy(), sampling_original, sampling_ravel)
        print("DEBUG: I_full_np shape after resize_map =", I_full_np.shape)
        print("DEBUG_HYP2: I_full_np shape after resize_map:", I_full_np.shape)
        print("DEBUG_HYP2: I_full_np (first 10 elems) after resize_map:", I_full_np.flatten()[:10])
        # (Optional test:) Uncomment the next line to disable any additional scaling:
        # I_full_np = I_full_np  # No extra scaling here
        
        I_full = torch.tensor(I_full_np, device=self.device, dtype=torch.float32)
        return I_full.flatten()

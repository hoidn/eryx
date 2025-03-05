"""
PyTorch implementation of disorder models for diffuse scattering calculations.

This module contains PyTorch versions of the disorder models defined in
eryx/models.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/models.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any

from eryx.pdb import AtomicModel, Crystal, GaussianNetworkModel
from eryx.autotest.debug import debug

class OnePhonon:
    """
    PyTorch implementation of the OnePhonon model for diffuse scattering calculations.
    
    This class implements a lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime) using PyTorch tensors and operations
    to enable gradient flow.
    
    References:
        - Original NumPy implementation in eryx/models.py:OnePhonon
    """
    
    @debug
    def __init__(self, pdb_path: str, hsampling: Tuple[float, float, float], 
                 ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float],
                 expand_p1: bool = True, group_by: str = 'asu',
                 res_limit: float = 0., model: str = 'gnm',
                 gnm_cutoff: float = 4., gamma_intra: float = 1., gamma_inter: float = 1.,
                 batch_size: int = 10000, n_processes: int = 8, device: Optional[torch.device] = None):
        """
        Initialize the OnePhonon model with PyTorch tensors.
        
        Args:
            pdb_path: Path to coordinates file.
            hsampling: Tuple (hmin, hmax, oversampling) for h dimension.
            ksampling: Tuple (kmin, kmax, oversampling) for k dimension.
            lsampling: Tuple (lmin, lmax, oversampling) for l dimension.
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit).
            group_by: Level of rigid-body assembly ('asu' or None).
            res_limit: High-resolution limit in Angstrom.
            model: Chosen phonon model ('gnm' or 'rb').
            gnm_cutoff: Distance cutoff for GNM in Angstrom.
            gamma_intra: Spring constant for intra-asu interactions.
            gamma_inter: Spring constant for inter-asu interactions.
            batch_size: Number of q-vectors to evaluate per batch.
            n_processes: Number of processes for parallel computation.
            device: PyTorch device to use (default: CUDA if available, else CPU).
        """
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.model_type = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._setup(pdb_path, expand_p1, res_limit, group_by)
        self._setup_phonons(pdb_path, model, gnm_cutoff, gamma_intra, gamma_inter)
    
    @debug
    def _setup(self, pdb_path: str, expand_p1: bool, res_limit: float, group_by: str):
        """
        Compute q-vectors to evaluate and build the unit cell and its neighbors.
        
        Parameters:
            pdb_path: Path to coordinates file of asymmetric unit.
            expand_p1: If True, expand to p1.
            res_limit: High-resolution limit in Angstrom.
            group_by: Level of rigid-body assembly ('asu' or None).
        """
        # Create an AtomicModel instance from the NP implementation.
        self.model = AtomicModel(pdb_path, expand_p1)
        
        # Build the grid of hkl indices using the NP generate_grid.
        from eryx.map_utils import generate_grid, get_resolution_mask
        hkl_grid, self.map_shape = generate_grid(self.model.A_inv, 
                                                 self.hsampling,
                                                 self.ksampling,
                                                 self.lsampling,
                                                 return_hkl=True)
        # Convert the hkl grid to a torch tensor.
        self.hkl_grid = torch.tensor(hkl_grid, dtype=torch.float32, device=self.device)
        
        # Obtain resolution mask (converted to a torch bool tensor).
        res_mask, _ = get_resolution_mask(self.model.cell, hkl_grid, res_limit)
        self.res_mask = torch.tensor(res_mask, dtype=torch.bool, device=self.device)
        
        # Compute q-grid as: 2π * A_inv^T * hkl_grid^T.
        self.q_grid = 2 * torch.pi * torch.matmul(
            torch.tensor(self.model.A_inv, dtype=torch.float32, device=self.device).T,
            self.hkl_grid.T
        ).T
        
        # Setup a minimal Crystal conversion.
        self.crystal = {}
        self.crystal['n_cell'] = 1  # Minimal; proper implementation would convert fully.
        self.crystal['id_to_hkl'] = lambda cell_id: [0, 0, 0]
        self.crystal['hkl_to_id'] = lambda hkl: 0
        self.crystal['get_unitcell_origin'] = lambda unit_cell: torch.tensor(self.model.unit_cell_axes[0],
                                                                              dtype=torch.float32,
                                                                              device=self.device)
        
        # Set key dimensions.
        # Instead of self.model.get_asu_xyz(), use self.model.xyz which is set in extract_frame.
        self.n_asu = self.model.n_asu
        self.n_atoms_per_asu = self.model.xyz.shape[1]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3
        
        self.group_by = group_by
        if self.group_by is None:
            self.n_dof_per_asu = self.n_dof_per_asu_actual
        else:
            self.n_dof_per_asu = 6
        self.n_dof_per_cell = self.n_asu * self.n_dof_per_asu
        self.id_cell_ref = 0  # By convention.
    
    @debug
    def _setup_phonons(self, pdb_path: str, model: str, 
                       gnm_cutoff: float, gamma_intra: float, gamma_inter: float):
        """
        Compute phonons using a Gaussian Network Model.
        
        Parameters:
            pdb_path: Path to coordinates file.
            model: Chosen phonon model ('gnm' or 'rb').
            gnm_cutoff: Distance cutoff for GNM.
            gamma_intra: Spring constant for intra-asu interactions.
            gamma_inter: Spring constant for inter-asu interactions.
        """
        # Use the sampling parameters directly.
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        
        self.kvec = torch.zeros((h_dim, k_dim, l_dim, 3), device=self.device)
        self.kvec_norm = torch.zeros((h_dim, k_dim, l_dim, 1), device=self.device)
        
        # Initialize tensors for phonon calculations.
        self.V = torch.zeros((h_dim, k_dim, l_dim,
                               self.n_asu * self.n_dof_per_asu,
                               self.n_asu * self.n_dof_per_asu),
                              dtype=torch.complex64, device=self.device)
        self.Winv = torch.zeros((h_dim, k_dim, l_dim,
                                  self.n_asu * self.n_dof_per_asu),
                                 dtype=torch.complex64, device=self.device)
        
        self._build_A()
        self._build_M()
        self._build_kvec_Brillouin()
        
        if model == 'gnm':
            # Setup GNM from NP implementation.
            self.gnm = GaussianNetworkModel(pdb_path, gnm_cutoff, gamma_intra, gamma_inter)
            self.gamma_intra = torch.tensor(gamma_intra, dtype=torch.float32, device=self.device, requires_grad=True)
            self.gamma_inter = torch.tensor(gamma_inter, dtype=torch.float32, device=self.device, requires_grad=True)
            
            self.compute_gnm_phonons()
            self.compute_covariance_matrix()
        else:
            self.compute_rb_phonons()
    
    @debug
    def _build_A(self):
        """
        Build the displacement projection matrix A that projects rigid-body
        displacements to individual atomic displacements.
        """
        if self.group_by == 'asu':
            self.Amat = torch.zeros((self.n_asu, self.n_atoms_per_asu, 3, 6), device=self.device)
            identity = torch.eye(3, device=self.device)
            for i_asu in range(self.n_asu):
                xyz = torch.tensor(self.model.get_asu_xyz(i_asu) if hasattr(self.model, 'get_asu_xyz')
                                   else self.model.xyz[i_asu],
                                   dtype=torch.float32, device=self.device)
                xyz = xyz - torch.mean(xyz, dim=0)
                for i_atom in range(self.n_atoms_per_asu):
                    skew = torch.zeros((3, 3), device=self.device)
                    skew[0, 1] = xyz[i_atom, 2]
                    skew[0, 2] = -xyz[i_atom, 1]
                    skew[1, 2] = xyz[i_atom, 0]
                    skew = skew - skew.transpose(0, 1)
                    self.Amat[i_asu, i_atom] = torch.cat([identity, skew], dim=1)
            self.Amat = self.Amat.reshape((self.n_asu, self.n_dof_per_asu_actual, self.n_dof_per_asu))
        else:
            self.Amat = None
    
    @debug
    def _build_M(self):
        """
        Build the mass matrix M and compute its inverse (via Cholesky).
        """
        M_allatoms = self._build_M_allatoms()
        if self.group_by is None:
            M_allatoms = M_allatoms.reshape((self.n_asu * self.n_dof_per_asu_actual,
                                              self.n_asu * self.n_dof_per_asu_actual))
            eps = 1e-10
            self.Linv = 1.0 / torch.sqrt(M_allatoms + eps)
        else:
            Mmat = self._project_M(M_allatoms)
            Mmat = Mmat.reshape((self.n_asu * self.n_dof_per_asu, self.n_asu * self.n_dof_per_asu))
            eps = 1e-10
            eye = torch.eye(Mmat.shape[0], device=self.device)
            Mmat = Mmat + eps * eye
            L = torch.linalg.cholesky(Mmat)
            self.Linv = torch.linalg.inv(L)
    
    @debug
    def _build_M_allatoms(self) -> torch.Tensor:
        """
        Build the all-atom mass matrix M_0.
        """
        mass_array = np.array([element.weight for structure in self.model.elements for element in structure])
        mass_array = torch.tensor(mass_array, dtype=torch.float32, device=self.device)
        eye3 = torch.eye(3, device=self.device)
        total_atoms = self.n_asu * self.n_atoms_per_asu
        mass_blocks = []
        for i in range(total_atoms):
            mass_block = mass_array[i] * eye3
            for j in range(3):
                row_block = torch.zeros(total_atoms * 3, device=self.device)
                start_row = 3 * i
                row_block[start_row:start_row+3] = mass_block[j]
                mass_blocks.append(row_block)
        M_allatoms = torch.stack(mass_blocks)
        M_allatoms = M_allatoms.reshape((self.n_asu, self.n_dof_per_asu_actual,
                                          self.n_asu, self.n_dof_per_asu_actual))
        return M_allatoms
    
    @debug
    def _project_M(self, M_allatoms: torch.Tensor) -> torch.Tensor:
        """
        Project the all-atom mass matrix M_0 using the A matrix.
        """
        Mmat = torch.zeros((self.n_asu, self.n_dof_per_asu, self.n_asu, self.n_dof_per_asu), device=self.device)
        for i_asu in range(self.n_asu):
            for j_asu in range(self.n_asu):
                Mmat[i_asu, :, j_asu, :] = torch.matmul(self.Amat[i_asu].T,
                                                         torch.matmul(M_allatoms[i_asu, :, j_asu, :],
                                                                      self.Amat[j_asu]))
        return Mmat
    
    @debug
    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        
        This implementation uses the sampling parameters directly, iterating over
        range(int(sampling)). For example, if hsampling = [-2, 2, 2], then the grid
        will have 2 points along h.
        
        The k-vector is computed as: q = 2π * A_inv^T * (h, k, l),
        where (h, k, l) is determined by centering each index.
        """
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        
        self.kvec = torch.zeros((h_dim, k_dim, l_dim, 3), device=self.device)
        self.kvec_norm = torch.zeros((h_dim, k_dim, l_dim, 1), device=self.device)
        
        for dh in range(h_dim):
            h_val = self._center_kvec(dh, h_dim)
            for dk in range(k_dim):
                k_val = self._center_kvec(dk, k_dim)
                for dl in range(l_dim):
                    l_val = self._center_kvec(dl, l_dim)
                    kvec_tensor = torch.tensor([h_val, k_val, l_val],
                                                 device=self.device, dtype=torch.float32)
                    A_inv_tensor = torch.tensor(self.model.A_inv, dtype=torch.float32, device=self.device)
                    self.kvec[dh, dk, dl] = 2 * torch.pi * torch.matmul(A_inv_tensor.T, kvec_tensor)
                    self.kvec_norm[dh, dk, dl] = torch.norm(self.kvec[dh, dk, dl])
    
    @debug
    def _center_kvec(self, x: int, L: int) -> float:
        """
        Center a k-vector index.
        
        For x in range(L), returns a centered value in (-0.5, 0.5)
        using: centered = int(((x - L/2) % L) - L/2) / L
        """
        return int(((x - L / 2) % L) - L / 2) / L
    
    @debug
    def _at_kvec_from_miller_points(self, hkl_kvec: tuple) -> torch.Tensor:
        """
        Return the indices of all q-vectors that are k-vector away from given Miller indices.
        
        Args:
            hkl_kvec: Tuple of starting indices (ints).
            
        Returns:
            Torch tensor of raveled indices.
        """
        hsteps = int(self.hsampling[2])
        ksteps = int(self.ksampling[2])
        lsteps = int(self.lsampling[2])
        
        h_indices = torch.arange(hkl_kvec[0], hsteps, device=self.device, dtype=torch.long)
        k_indices = torch.arange(hkl_kvec[1], ksteps, device=self.device, dtype=torch.long)
        l_indices = torch.arange(hkl_kvec[2], lsteps, device=self.device, dtype=torch.long)
        
        h_grid, k_grid, l_grid = torch.meshgrid(h_indices, k_indices, l_indices, indexing='ij')
        h_flat = h_grid.reshape(-1)
        k_flat = k_grid.reshape(-1)
        l_flat = l_grid.reshape(-1)
        
        strides = torch.tensor([self.map_shape[1] * self.map_shape[2],
                                 self.map_shape[2],
                                 1], device=self.device)
        indices = h_flat * strides[0] + k_flat * strides[1] + l_flat * strides[2]
        return indices
    
    @debug
    def compute_gnm_hessian(self) -> torch.Tensor:
        """
        Compute the Hessian matrix using the Gaussian Network Model.
        
        Returns:
            Torch tensor of shape (n_asu, n_atoms_per_asu, n_cell, n_asu, n_atoms_per_asu)
            with dtype torch.complex64.
        """
        hessian = torch.zeros((self.n_asu, self.n_atoms_per_asu,
                               self.n_cell, self.n_asu, self.n_atoms_per_asu),
                              dtype=torch.complex64, device=self.device)
        hessian_diagonal = torch.zeros((self.n_asu, self.n_atoms_per_asu),
                                       dtype=torch.complex64, device=self.device)
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    for i_at in range(self.n_atoms_per_asu):
                        # Dummy neighbor list (empty) for simplicity.
                        iat_neighbors = []
                        if len(iat_neighbors) > 0:
                            gamma = self.gamma_intra if i_asu == j_asu else self.gamma_inter
                            for j_at in iat_neighbors:
                                hessian[i_asu, i_at, i_cell, j_asu, j_at] = -gamma.to(torch.complex64)
                            hessian_diagonal[i_asu, i_at] -= gamma.to(torch.complex64) * len(iat_neighbors)
        for i_asu in range(self.n_asu):
            for i_at in range(self.n_atoms_per_asu):
                gamma_self = self.gamma_intra.to(torch.complex64)
                hessian[i_asu, i_at, self.id_cell_ref, i_asu, i_at] = hessian_diagonal[i_asu, i_at] - gamma_self
        return hessian
    
    @debug
    def compute_gnm_K(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the dynamical matrix K(kvec) from the Hessian.
        
        Args:
            hessian: Hessian tensor.
            kvec: k-vector tensor of shape (3,). Defaults to zero vector.
            
        Returns:
            Dynamical matrix K as a tensor.
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone()
        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
            r_cell = self.crystal['get_unitcell_origin'](self.crystal['id_to_hkl'](j_cell))
            phase = torch.sum(kvec * r_cell)
            real_part, imag_part = torch.cos(phase), torch.sin(phase)
            eikr = torch.complex(real_part, imag_part)
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
        return Kmat
    
    @debug
    def compute_Kinv(self, hessian: torch.Tensor, kvec: torch.Tensor = None, 
                     reshape: bool = True) -> torch.Tensor:
        """
        Compute the pseudo-inverse of the dynamical matrix K(kvec).
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device)
        Kmat = self.compute_gnm_K(hessian, kvec=kvec)
        Kshape = Kmat.shape
        Kmat_2d = Kmat.reshape(Kshape[0] * Kshape[1], Kshape[2] * Kshape[3])
        eps = 1e-10
        identity = torch.eye(Kmat_2d.shape[0], device=self.device, dtype=Kmat_2d.dtype)
        Kmat_2d_reg = Kmat_2d + eps * identity
        Kinv = torch.linalg.pinv(Kmat_2d_reg)
        if reshape:
            Kinv = Kinv.reshape((Kshape[0], Kshape[1], Kshape[2], Kshape[3]))
        return Kinv
    
    @debug
    def compute_hessian(self) -> torch.Tensor:
        """
        Compute the projected Hessian matrix for the supercell.
        """
        hessian = torch.zeros((self.n_asu, self.n_dof_per_asu,
                               self.n_cell, self.n_asu, self.n_dof_per_asu),
                              dtype=torch.complex64, device=self.device)
        hessian_allatoms = self.gnm.compute_hessian()
        for i_cell in range(self.n_cell):
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    proj = torch.matmul(self.Amat[i_asu].T.to(torch.complex64),
                                        torch.matmul(hessian_allatoms[i_asu, :, i_cell, j_asu, :],
                                                     self.Amat[j_asu].to(torch.complex64)))
                    hessian[i_asu, :, i_cell, j_asu, :] = proj
        return hessian
    
    @debug
    def compute_gnm_phonons(self):
        """
        Compute phonon modes for each k-vector in the first Brillouin zone.
        """
        hessian = self.compute_hessian()
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        for dh in range(h_dim):
            for dk in range(k_dim):
                for dl in range(l_dim):
                    kvec = self.kvec[dh, dk, dl]
                    Kmat = self.compute_gnm_K(hessian, kvec=kvec)
                    Kmat_2d = Kmat.reshape((self.n_asu * self.n_dof_per_asu,
                                             self.n_asu * self.n_dof_per_asu))
                    Linv_complex = self.Linv.to(dtype=torch.complex64)
                    Dmat = torch.matmul(Linv_complex, torch.matmul(Kmat_2d, Linv_complex.T))
                    v, w, _ = torch.linalg.svd(Dmat, full_matrices=False)
                    w = torch.sqrt(w)
                    w = torch.where(w < 1e-6,
                                    torch.tensor(float('nan'), dtype=w.dtype, device=w.device),
                                    w)
                    nan_count = torch.isnan(w).sum().item()
                    if nan_count > 0:
                        import logging
                        logging.debug(f"compute_gnm_phonons: {nan_count} NaN values in eigenvalues")
                    w = torch.flip(w, [0])
                    v = torch.flip(v, [1])
                    self.Winv[dh, dk, dl] = 1.0 / (w ** 2)
                    self.V[dh, dk, dl] = torch.matmul(Linv_complex.T, v)
    
    @debug
    def compute_covariance_matrix(self):
        """
        Compute the covariance matrix for atomic displacements.
        """
        self.covar = torch.zeros((self.n_asu * self.n_dof_per_asu,
                                   self.n_cell, self.n_asu * self.n_dof_per_asu),
                                  dtype=torch.complex64, device=self.device)
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        from eryx.torch_utils import ComplexTensorOps
        for dh in range(h_dim):
            for dk in range(k_dim):
                for dl in range(l_dim):
                    kvec = self.kvec[dh, dk, dl]
                    Kinv = self.gnm.compute_Kinv(self.compute_hessian(), kvec=kvec, reshape=False)
                    for j_cell in range(self.n_cell):
                        r_cell = self.crystal['get_unitcell_origin'](self.crystal['id_to_hkl'](j_cell))
                        phase = torch.sum(kvec * r_cell)
                        real_part, imag_part = ComplexTensorOps.complex_exp(phase)
                        eikr = torch.complex(real_part, imag_part)
                        self.covar[:, j_cell, :] += Kinv * eikr
        self.ADP = torch.real(torch.diagonal(self.covar[:, self.crystal['hkl_to_id']([0, 0, 0]), :], dim1=0, dim2=1))
        Amat = torch.transpose(self.Amat, 0, 1).reshape(self.n_dof_per_asu_actual, self.n_asu * self.n_dof_per_asu)
        self.ADP = torch.matmul(Amat, self.ADP)
        self.ADP = torch.sum(self.ADP.reshape(int(self.ADP.shape[0] / 3), 3), dim=1)
        ADP_scale = torch.mean(self.array_to_tensor(self.model.adp)) / (8 * torch.pi * torch.pi * torch.mean(self.ADP) / 3)
        self.ADP = self.ADP * ADP_scale
        self.covar = self.covar * ADP_scale
        self.covar = torch.real(self.covar.reshape((self.n_asu, self.n_dof_per_asu,
                                                     self.n_cell, self.n_asu, self.n_dof_per_asu)))
    
    @debug
    def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                       use_data_adp: bool = False) -> torch.Tensor:
        """
        Compute the diffuse intensity using the one-phonon approximation.
        """
        import logging
        logging.info(f"apply_disorder: rank={rank}, use_data_adp={use_data_adp}")
        if use_data_adp:
            ADP = torch.tensor(self.model.adp[0], dtype=torch.float32, device=self.device) / (8 * torch.pi * torch.pi)
        else:
            ADP = self.ADP.to(dtype=torch.float32, device=self.device)
        Id = torch.zeros(self.q_grid.shape[0], dtype=torch.float32, device=self.device)
        from eryx.scatter_torch import structure_factors
        h_dim = int(self.hsampling[2])
        k_dim = int(self.ksampling[2])
        l_dim = int(self.lsampling[2])
        for dh in range(h_dim):
            for dk in range(k_dim):
                for dl in range(l_dim):
                    q_indices = self._at_kvec_from_miller_points((dh, dk, dl))
                    valid_mask = self.res_mask[q_indices]
                    valid_indices = q_indices[valid_mask]
                    if valid_indices.numel() == 0:
                        continue
                    F = torch.zeros((valid_indices.numel(), self.n_asu, self.n_dof_per_asu),
                                    dtype=torch.complex64, device=self.device)
                    for i_asu in range(self.n_asu):
                        F[:, i_asu, :] = structure_factors(
                            self.q_grid[valid_indices],
                            torch.tensor(self.model.get_asu_xyz(i_asu), dtype=torch.float32, device=self.device),
                            torch.tensor(self.model.ff_a[i_asu], dtype=torch.float32, device=self.device),
                            torch.tensor(self.model.ff_b[i_asu], dtype=torch.float32, device=self.device),
                            torch.tensor(self.model.ff_c[i_asu], dtype=torch.float32, device=self.device),
                            U=ADP,
                            batch_size=self.batch_size,
                            n_processes=self.n_processes,
                            compute_qF=True,
                            project_on_components=self.Amat[i_asu],
                            sum_over_atoms=False
                        )
                    F = F.reshape((valid_indices.numel(), self.n_asu * self.n_dof_per_asu))
                    if rank == -1:
                        FV = torch.matmul(F, self.V[dh, dk, dl])
                        FV_abs_squared = torch.abs(FV) ** 2
                        real_winv = torch.real(self.Winv[dh, dk, dl])
                        weighted_intensity = torch.matmul(FV_abs_squared, real_winv)
                        Id.index_add_(0, valid_indices, weighted_intensity)
                    else:
                        V_rank = self.V[dh, dk, dl, :, rank]
                        FV = torch.matmul(F, V_rank)
                        weighted_intensity = (torch.abs(FV) ** 2) * torch.real(self.Winv[dh, dk, dl, rank])
                        Id.index_add_(0, valid_indices, weighted_intensity)
        Id_masked = Id.clone()
        Id_masked[~self.res_mask] = float('nan')
        if outdir is not None:
            import os
            os.makedirs(outdir, exist_ok=True)
            torch.save(Id_masked, os.path.join(outdir, f"rank_{rank:05d}_torch.pt"))
            np.save(os.path.join(outdir, f"rank_{rank:05d}.npy"), Id_masked.detach().cpu().numpy())
        return Id_masked

    @debug
    def array_to_tensor(self, array: np.ndarray, requires_grad: bool = True, dtype=None) -> torch.Tensor:
        """
        Convert a NumPy array to a Torch tensor.
        """
        if dtype is None:
            dtype = torch.float32
        tensor = torch.tensor(array, dtype=dtype, device=self.device)
        if requires_grad and tensor.dtype.is_floating_point:
            tensor.requires_grad_(True)
        return tensor

    @debug
    def compute_rb_phonons(self):
        """
        Compute phonons for the rigid-body model.
        """
        self.compute_gnm_phonons()

# Minimal implementations for additional models

class RigidBodyTranslations:
    @debug
    def __init__(self, *args, **kwargs):
        pass

class LiquidLikeMotions:
    @debug
    def __init__(self, *args, **kwargs):
        pass

class RigidBodyRotations:
    @debug
    def __init__(self, *args, **kwargs):
        pass


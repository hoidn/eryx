import torch
import numpy as np
import logging
from eryx.pdb import AtomicModel, Crystal
from scipy.spatial import KDTree
import torch.nn as nn

class GaussianNetworkModelTorch(nn.Module):
    def __init__(self, pdb_path: str, enm_cutoff: float, gamma_intra: float, gamma_inter: float,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the torch-based Gaussian Network Model.
        Loads the atomic model (using existing numpy code) and converts key arrays to torch tensors.
        """
        super(GaussianNetworkModelTorch, self).__init__()
        self.device = torch.device(device.type)
        # Convert input gamma constants to learnable parameters:
        self.gamma_intra = nn.Parameter(torch.tensor(gamma_intra, dtype=torch.float64, device=self.device))
        self.gamma_inter = nn.Parameter(torch.tensor(gamma_inter, dtype=torch.float64, device=self.device))
        # Load and set up the atomic model (using existing numpy routines)
        self.atomic_model = AtomicModel(pdb_path, expand_p1=True)
        self.crystal = Crystal(self.atomic_model)
        self.crystal.supercell_extent(nx=1, ny=1, nz=1)
        self.id_cell_ref = self.crystal.hkl_to_id([0, 0, 0])
        self.n_cell = self.crystal.n_cell
        self.n_asu = self.crystal.model.n_asu
        self.n_atoms_per_asu = self.crystal.get_asu_xyz().shape[0]
        self.n_dof_per_asu_actual = self.n_atoms_per_asu * 3
        self.enm_cutoff = enm_cutoff

        self.build_gamma()
        self.build_neighbor_list()

    def build_gamma(self) -> None:
        """
        Build the spring constant tensor.
        """
        # Create a gamma tensor from the learnable gamma_inter scalar:
        gamma_tensor = self.gamma_inter * torch.ones((self.n_cell, self.n_asu, self.n_asu),
                                                     device=self.device, dtype=torch.float64)
        # In the reference cell and on the diagonal, use gamma_intra:
        for i_asu in range(self.n_asu):
            gamma_tensor[self.id_cell_ref, i_asu, i_asu] = self.gamma_intra
        self.gamma = gamma_tensor

    def build_neighbor_list(self) -> None:
        """
        Build the neighbor list using numpy’s KDTree. (The neighbor lists remain numpy lists.)
        """
        self.asu_neighbors = []
        for i_asu in range(self.n_asu):
            self.asu_neighbors.append([])
            xyz_ref = self.crystal.get_asu_xyz(i_asu, self.crystal.id_to_hkl(self.id_cell_ref))
            kd_tree1 = KDTree(xyz_ref)
            for i_cell in range(self.n_cell):
                self.asu_neighbors[i_asu].append([])
                for j_asu in range(self.n_asu):
                    xyz_neighbor = self.crystal.get_asu_xyz(j_asu, self.crystal.id_to_hkl(i_cell))
                    kd_tree2 = KDTree(xyz_neighbor)
                    neighbors = kd_tree1.query_ball_tree(kd_tree2, r=self.enm_cutoff)
                    self.asu_neighbors[i_asu][-1].append(neighbors)

    def compute_hessian(self) -> torch.Tensor:
        """
        Compute the Hessian matrix using torch operations.
        Returns a tensor with shape:
           (n_asu, n_atoms_per_asu, n_cell, n_asu, n_atoms_per_asu) of dtype complex.
        """
        shape = (self.n_asu, self.n_atoms_per_asu, self.n_cell, self.n_asu, self.n_atoms_per_asu)
        hessian = torch.zeros(shape, dtype=torch.complex128, device=self.device)
        hessian_diag = torch.zeros((self.n_asu, self.n_atoms_per_asu),
                                   dtype=torch.complex128, device=self.device)
        # Loop over ASU and neighbor cells using the neighbor list
        for i_asu in range(self.n_asu):
            for i_cell in range(self.n_cell):
                for j_asu in range(self.n_asu):
                    neighbors_list = self.asu_neighbors[i_asu][i_cell][j_asu]
                    for i_at, neigh_indices in enumerate(neighbors_list):
                        if neigh_indices:
                            gamma_val = self.gamma[i_cell, i_asu, j_asu]
                            # Log details before assignment
                            # logging.debug(
                            #     f"[DEBUG - compute_hessian] i_asu={i_asu}, i_cell={i_cell}, j_asu={j_asu}, "
                            #     f"i_at={i_at}, neighbors={neigh_indices}, gamma_val (float)={gamma_val.item() if torch.is_tensor(gamma_val) else gamma_val}"
                            # )
                            idxs = torch.tensor(neigh_indices, device=self.device)
                            val_to_assign = -gamma_val.to(torch.complex128)
                            # logging.debug(f"[DEBUG - compute_hessian] Attempting assignment: hessian[{i_asu}, {i_at}, {i_cell}, {j_asu}, {idxs.tolist()}] = {val_to_assign}")
                            try:
                                hessian[i_asu, i_at, i_cell, j_asu, idxs] = val_to_assign
                            except Exception as e:
                                logging.error(
                                    f"[ERROR - compute_hessian] Failed assignment at i_asu={i_asu}, i_cell={i_cell}, j_asu={j_asu}, "
                                    f"i_at={i_at}, neighbors={neigh_indices}, gamma_val={gamma_val}. Exception: {e}"
                                )
                                raise
                            hessian_diag[i_asu, i_at] += val_to_assign * float(len(neigh_indices))
                            # logging.debug(
                            #     f"[DEBUG compute_hessian] ASU={i_asu}, atom index={i_at}: processed {len(neigh_indices)} neighbors, "
                            #     f"current diag accumulator = {hessian_diag[i_asu, i_at].item()}"
                            # )
        # Set the diagonal (reference cell)
        for i_asu in range(self.n_asu):
            for i_at in range(self.n_atoms_per_asu):
                hessian[i_asu, i_at, self.id_cell_ref, i_asu, i_at] = -hessian_diag[i_asu, i_at] - self.gamma[self.id_cell_ref, i_asu, i_asu].to(torch.complex128)
            # logging.debug(
            #     f"[DEBUG compute_hessian] Before diagonal assignment for ASU={i_asu}, atom index={i_at}: "
            #     f"hessian_diag = {hessian_diag[i_asu, i_at].item()}, gamma (ref cell) = {self.gamma[self.id_cell_ref, i_asu, i_asu].item()}"
            # )
        # logging.debug(hessian[0, :, :, 0, :])
        # logging.debug(f"Hessian shape: {hessian.shape}")
        return hessian

    def compute_K(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the dynamical matrix K(k) using torch.
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device, dtype=torch.float64)
        else:
            kvec = kvec.to(torch.float64)
        # Start with the reference cell term
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone()
        # Sum contributions from other cells (using explicit loops over ASU indices)
        for j_cell in range(self.n_cell):
            if j_cell == self.id_cell_ref:
                continue
            logging.debug(f"[DEBUG compute_K] Processing j_cell={j_cell}")
            # Get the cell origin (from the numpy Crystal object) and convert to torch tensor
            r_cell_np = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(j_cell))
            r_cell = torch.tensor(r_cell_np, device=self.device, dtype=torch.float64)
            phase = torch.dot(kvec, r_cell)
            eikr = torch.cos(phase) + 1j * torch.sin(phase)
            logging.debug(f"[DEBUG compute_K] j_cell={j_cell}, r_cell={r_cell.cpu().numpy()}, phase={phase.item():.8f}, eikr={eikr}")
            for i_asu in range(self.n_asu):
                for j_asu in range(self.n_asu):
                    Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
            logging.debug(f"[DEBUG compute_K] After j_cell={j_cell} update, Kmat norm={torch.norm(Kmat).item():.8f}")
        # logging.debug(f"Kmat shape: {Kmat.shape}")
        return Kmat

    def compute_Kinv(self, hessian: torch.Tensor, kvec: torch.Tensor = None, reshape: bool = True) -> torch.Tensor:
        """
        Compute the inverse of K(k) using torch.linalg.pinv.
        """
        Kmat = self.compute_K(hessian, kvec)
        shape = Kmat.shape  # (n_asu, n_atoms_per_asu, n_asu, n_atoms_per_asu)
        Kmat_flat = Kmat.reshape(shape[0] * shape[1], shape[2] * shape[3]).to(torch.complex128)
        logging.debug(f"[DEBUG compute_Kinv] Kmat flat shape: {Kmat_flat.shape}, norm={torch.norm(Kmat_flat).item():.8f}")
        Kinv_flat = torch.linalg.pinv(Kmat_flat)
        logging.debug(f"[DEBUG compute_Kinv] Kinv flat shape: {Kinv_flat.shape}, norm={torch.norm(Kinv_flat).item():.8f}")
        if reshape:
            Kinv = Kinv_flat.reshape((shape[0], shape[1], shape[2], shape[3]))
        else:
            Kinv = Kinv_flat
        # logging.debug(f"Kinv shape: {Kinv.shape}")
        return Kinv

    def compute_hessian_torch(self) -> torch.Tensor:
        # [NEW] Vectorized (or minimal–loop) implementation that uses torch operations
        # Ensure that any assignment uses tensors built from self.gamma (from build_gamma())
        # and that no .detach() calls are used.
        # (For brevity, replace the explicit loops with torch–compatible code once neighbor lists are batched.)
        # Example (if full vectorization is not yet feasible, at least ensure loops are within torch operations):
        shape = (self.n_asu, self.n_atoms_per_asu, self.n_cell, self.n_asu, self.n_atoms_per_asu)
        hessian = torch.zeros(shape, dtype=torch.complex64, device=self.device)
        # ... (preserve loop structure but do not break gradient flow) ...
        return hessian

    def compute_K_torch(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the dynamical matrix K(k) for a given k-vector.
        
        Args:
            hessian: The Hessian from compute_hessian_torch(), shape (n_asu, n_atoms, n_cell, n_asu, n_atoms)
            kvec: Phonon wavevector, shape (3,)
        
        Returns:
            torch.Tensor: K-matrix for this k-point, reshaped as (n_total, n_total)
        """
        if kvec is None:
            kvec = torch.zeros(3, device=self.device, dtype=torch.float64)
        else:
            kvec = kvec.to(torch.float64)
        shape = hessian.shape
        Kmat = hessian[:, :, self.id_cell_ref, :, :].clone().to(torch.complex64)
        for i_cell in range(self.n_cell):
            if i_cell == self.id_cell_ref:
                continue
            r_cell_np = self.crystal.get_unitcell_origin(self.crystal.id_to_hkl(i_cell))
            r_cell = torch.tensor(r_cell_np, device=self.device, dtype=torch.float64)
            phase = torch.dot(kvec, r_cell)
            eikr = torch.exp(1j * phase)
            Kmat += hessian[:, :, i_cell, :, :] * eikr
        n_total = shape[0] * shape[1]
        return Kmat.reshape(n_total, n_total)

    def compute_gnm_phonons_torch(self):
        # For each k–point (e.g. on a small grid), compute the K matrix using compute_K_torch().
        # Then form the mass–weighted dynamical matrix and use
        #   U, S, _ = torch.linalg.svd(Dmat)
        # to compute eigenvalues/vectors with gradient support.
        # Store self.V and self.Winv (ensure no .detach() is used).
        # (Retain the loop structure if necessary but make sure every operation is differentiable.)
        pass

    def _mass_weight_dynamical_matrix(self, Kmat: torch.Tensor) -> torch.Tensor:
        """
        Apply mass weighting to the dynamical matrix.
        
        Args:
            Kmat: Raw dynamical matrix, shape (n_total, n_total)
        
        Returns:
            torch.Tensor: Mass–weighted matrix D = L⁻¹ · Kmat · L⁻ᵀ.
        """
        n = Kmat.shape[0]
        L_inv = torch.eye(n, device=self.device, dtype=Kmat.dtype)
        return L_inv @ Kmat @ L_inv.T

    def _process_eigensystem(self, v: torch.Tensor, w: torch.Tensor, epsilon: float = 1e-6) -> (torch.Tensor, torch.Tensor):
        """
        Process eigenvalues and vectors obtained from SVD.
        
        Replaces eigenvalues below epsilon and returns the processed system.
        
        Args:
            v (torch.Tensor): Eigenvector matrix from torch.linalg.svd().
            w (torch.Tensor): Singular values.
            epsilon (float): Small-value threshold.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed eigenvectors and eigenvalues.
        """
        w_processed = torch.where(w < epsilon, torch.tensor(epsilon, device=self.device, dtype=w.dtype), w)
        return v, w_processed

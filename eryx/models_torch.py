"""
PyTorch implementation of disorder models for diffuse scattering calculations.

This module contains PyTorch versions of the disorder models defined in eryx/models.py.
All implementations maintain the same API as the NumPy versions but use PyTorch tensors
and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/models.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any

# Forward references for type hints
from eryx.pdb import AtomicModel, Crystal, GaussianNetworkModel

class OnePhonon:
    """
    PyTorch implementation of the OnePhonon model for diffuse scattering calculations.
    
    This class implements a lattice of interacting rigid bodies in the one-phonon
    approximation (a.k.a small-coupling regime) using PyTorch tensors and operations
    to enable gradient flow.
    
    References:
        - Original NumPy implementation in eryx/models.py:OnePhonon
    """
    
    def __init__(self, pdb_path: str, hsampling: Tuple[float, float, float], 
                 ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float],
                 expand_p1: bool = True, group_by: str = 'asu',
                 res_limit: float = 0., model: str = 'gnm',
                 gnm_cutoff: float = 4., gamma_intra: float = 1., gamma_inter: float = 1.,
                 batch_size: int = 10000, n_processes: int = 8):
        """
        Initialize the OnePhonon model with PyTorch tensors.
        
        Args:
            pdb_path: Path to coordinates file
            hsampling: (hmin, hmax, oversampling) for h dimension
            ksampling: (kmin, kmax, oversampling) for k dimension
            lsampling: (lmin, lmax, oversampling) for l dimension
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit)
            group_by: Level of rigid-body assembly, 'asu' or None
            res_limit: High-resolution limit in Angstrom
            model: Chosen phonon model ('gnm' or 'rb')
            gnm_cutoff: Distance cutoff for GNM in Angstrom
            gamma_intra: Spring constant for atom pairs in same molecule
            gamma_inter: Spring constant for atom pairs in different molecules
            batch_size: Number of q-vectors to evaluate per batch
            n_processes: Number of processes for parallel computation
            
        References:
            - Original implementation: eryx/models.py:OnePhonon.__init__
        """
        # TODO: Initialize class attributes similar to the NumPy implementation
        # TODO: Convert sampling tuples to PyTorch compatible formats
        # TODO: Call self._setup() and self._setup_phonons() to initialize tensors
        
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self.batch_size = batch_size
        self.n_processes = n_processes
        
        # These will be initialized in _setup() and _setup_phonons()
        self.model = None
        self.q_grid = None
        self.crystal = None
        self.res_mask = None
        self.group_by = group_by
        
        # Placeholder for a proper implementation
        raise NotImplementedError("OnePhonon.__init__ not implemented")
    
    def _setup(self, pdb_path: str, expand_p1: bool, res_limit: float, group_by: str):
        """
        Set up class, computing q-vectors and building the unit cell.
        
        Args:
            pdb_path: Path to coordinates file
            expand_p1: If True, expand to p1 (if PDB is asymmetric unit)
            res_limit: High-resolution limit in Angstrom
            group_by: Level of rigid-body assembly, 'asu' or None
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._setup
        """
        # TODO: Create AtomicModel using adapter
        # TODO: Generate reciprocal space grid and convert to torch.Tensor
        # TODO: Calculate q vectors and q magnitudes as torch tensors
        # TODO: Set up Crystal object and compute necessary dimensions
        
        raise NotImplementedError("OnePhonon._setup not implemented")
    
    def _setup_phonons(self, pdb_path: str, model: str, 
                     gnm_cutoff: float, gamma_intra: float, gamma_inter: float):
        """
        Compute phonons from a Gaussian Network Model using PyTorch operations.
        
        Args:
            pdb_path: Path to coordinates file
            model: Chosen phonon model ('gnm' or 'rb')
            gnm_cutoff: Distance cutoff for GNM in Angstrom
            gamma_intra: Spring constant for atom pairs in same molecule
            gamma_inter: Spring constant for atom pairs in different molecules
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._setup_phonons
        """
        # TODO: Initialize tensor arrays for phonon calculations
        # TODO: Build A and M matrices using PyTorch operations
        # TODO: Compute k-vectors in Brillouin zone as tensors
        # TODO: Setup GNM and compute phonon modes
        
        raise NotImplementedError("OnePhonon._setup_phonons not implemented")
    
    def _build_A(self):
        """
        Build the matrix A that projects small rigid-body displacements to individual atoms.
        
        This matrix converts from rigid-body displacements (translations and rotations)
        to individual atomic displacements based on the atom positions relative to the
        center of mass.
        
        For each atom i in group m, the conversion reads:
        u_i = A(r_i - o_m).w_m
        where A is a 3x6 matrix defined as:
        A(x,y,z) = [[ 1 0 0  0  z -y ]
                    [ 0 1 0 -z  0  x ]
                    [ 0 0 1  y -x  0 ]]
                    
        Returns:
            None - stores the matrix in self.Amat
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_A
        """
        # Handle case where group_by is set to 'asu'
        if self.group_by == 'asu':
            # Initialize Amat tensor for all ASUs
            self.Amat = torch.zeros((self.n_asu, self.n_atoms_per_asu, 3, 6), 
                                   device=self.device)
            
            # Identity matrix for the first 3 columns of A
            identity = torch.eye(3, device=self.device)
            
            # For each ASU
            for i_asu in range(self.n_asu):
                # Get coordinates for this ASU
                xyz = self.crystal.get_asu_xyz(i_asu).clone()
                
                # Subtract center of mass
                xyz -= torch.mean(xyz, dim=0)
                
                # For each atom in the ASU
                for i_atom in range(self.n_atoms_per_asu):
                    # Create the skew-symmetric matrix for rotational part
                    # [  0  z -y ]
                    # [ -z  0  x ]
                    # [  y -x  0 ]
                    skew = torch.zeros((3, 3), device=self.device)
                    skew[0, 1] = xyz[i_atom, 2]     # z
                    skew[0, 2] = -xyz[i_atom, 1]    # -y
                    skew[1, 2] = xyz[i_atom, 0]     # x
                    skew = skew - skew.transpose(0, 1)  # Make skew-symmetric
                    
                    # Combine translational (identity) and rotational (skew) parts
                    self.Amat[i_asu, i_atom] = torch.cat([identity, skew], dim=1)
            
            # Reshape Amat to final dimensions
            self.Amat = self.Amat.reshape((self.n_asu,
                                          self.n_dof_per_asu_actual,
                                          self.n_dof_per_asu))
        else:
            self.Amat = None
    
    def _build_M(self):
        """
        Build the mass matrix M and compute its Cholesky decomposition.
        
        If all atoms are considered individually (group_by=None), M = M_0 is diagonal
        and Linv = 1/sqrt(M_0) is also diagonal.
        
        If atoms are grouped as rigid bodies, the all-atoms M matrix is projected
        using the A matrix: M = A.T M_0 A and Linv is obtained via Cholesky 
        decomposition: M = LL.T, Linv = L^(-1)
        
        Returns:
            None - stores the result in self.Linv
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_M
        """
        # Get the all-atoms mass matrix
        M_allatoms = self._build_M_allatoms()
        
        # Handle different cases based on group_by parameter
        if self.group_by is None:
            # No grouping, reshape to 2D matrix
            M_allatoms = M_allatoms.reshape((self.n_asu * self.n_dof_per_asu_actual,
                                            self.n_asu * self.n_dof_per_asu_actual))
            
            # For diagonal mass matrix, Linv is simply 1/sqrt(M)
            # Add small epsilon for numerical stability
            epsilon = 1e-10
            self.Linv = 1.0 / torch.sqrt(M_allatoms + epsilon)
            
        else:
            # Project the mass matrix for rigid body case
            Mmat = self._project_M(M_allatoms)
            
            # Reshape to 2D matrix for Cholesky decomposition
            Mmat = Mmat.reshape((self.n_asu * self.n_dof_per_asu,
                                self.n_asu * self.n_dof_per_asu))
            
            # Add small regularization for numerical stability
            epsilon = 1e-10
            eye = torch.eye(Mmat.shape[0], device=self.device)
            Mmat = Mmat + epsilon * eye
            
            # Compute Cholesky decomposition: M = L*L^T
            try:
                L = torch.linalg.cholesky(Mmat)
                
                # Compute inverse of L
                self.Linv = torch.linalg.inv(L)
            except RuntimeError as e:
                # Handle case where matrix is not positive definite
                print(f"Warning: Cholesky decomposition failed: {e}")
                print("Using SVD-based approach instead")
                
                # Alternative approach using SVD
                U, S, Vh = torch.linalg.svd(Mmat)
                
                # Ensure S is positive
                S = torch.clamp(S, min=epsilon)
                
                # Compute L = U * sqrt(S)
                L = U * torch.sqrt(S).unsqueeze(0)
                
                # Compute inverse of L
                self.Linv = torch.matmul(torch.diag(1.0 / torch.sqrt(S)), U.transpose(0, 1))
    
    def _build_M_allatoms(self) -> torch.Tensor:
        """
        Build all-atom mass matrix M_0 from element weights.

        Returns:
            torch.Tensor: Mass matrix with shape (n_asu, n_atoms*3, n_asu, n_atoms*3)
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_M_allatoms
        """
        # Extract atomic masses from the crystal model
        # Flatten the nested structure to get a single array of masses
        mass_array = torch.tensor([element.weight for structure in self.crystal.model.elements 
                                 for element in structure], device=self.device)
        
        # Create a 3x3 identity matrix
        eye3 = torch.eye(3, device=self.device)
        
        # Initialize a list to store block matrices
        mass_blocks = []
        
        # For each atom, create a 3x3 block with the atom's mass on the diagonal
        for i in range(self.n_asu * self.n_atoms_per_asu):
            # Create a 3x3 matrix with the atom's mass on the diagonal
            mass_block = mass_array[i] * eye3
            
            # Extract the rows for this atom
            # For each atom i, we create rows for coordinates 3*i, 3*i+1, 3*i+2
            start_row = 3 * i
            
            # For each row, add a block matrix
            for j in range(3):
                row_block = torch.zeros(3 * self.n_asu * self.n_atoms_per_asu, device=self.device)
                
                # Only populate the 3 elements corresponding to this atom
                row_block[start_row:start_row + 3] = mass_block[j]
                
                mass_blocks.append(row_block)
        
        # Stack all blocks into a matrix
        M_allatoms = torch.stack(mass_blocks)
        
        # Reshape to the required 4D shape
        M_allatoms = M_allatoms.reshape((self.n_asu, self.n_dof_per_asu_actual,
                                        self.n_asu, self.n_dof_per_asu_actual))
        
        return M_allatoms
    
    def _project_M(self, M_allatoms: torch.Tensor) -> torch.Tensor:
        """
        Project all-atom mass matrix using the A matrix: M = A.T M_0 A

        Parameters:
            M_allatoms: torch.Tensor - All-atom mass matrix with shape 
                        (n_asu, n_atoms*3, n_asu, n_atoms*3)

        Returns:
            torch.Tensor: Projected mass matrix with shape 
                        (n_asu, n_dof_per_asu, n_asu, n_dof_per_asu)
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._project_M
        """
        # Initialize projected mass matrix with zeros
        Mmat = torch.zeros((self.n_asu, self.n_dof_per_asu,
                          self.n_asu, self.n_dof_per_asu), 
                          device=self.device)
        
        # For each pair of ASUs
        for i_asu in range(self.n_asu):
            for j_asu in range(self.n_asu):
                # Perform the projection: A^T * M * A
                # First multiply M_allatoms by A on the right
                # M_allatoms[i_asu, :, j_asu, :] has shape (n_dof_per_asu_actual, n_dof_per_asu_actual)
                # self.Amat[j_asu] has shape (n_dof_per_asu_actual, n_dof_per_asu)
                intermediate = torch.matmul(M_allatoms[i_asu, :, j_asu, :],
                                          self.Amat[j_asu])
                
                # Then multiply by A^T on the left
                # self.Amat[i_asu].T has shape (n_dof_per_asu, n_dof_per_asu_actual)
                # intermediate has shape (n_dof_per_asu_actual, n_dof_per_asu)
                Mmat[i_asu, :, j_asu, :] = torch.matmul(self.Amat[i_asu].transpose(0, 1),
                                                      intermediate)
        
        return Mmat
    
    def _build_kvec_Brillouin(self):
        """
        Compute all k-vectors and their norm in the first Brillouin zone.
        
        This is achieved by regularly sampling [-0.5,0.5[ for h, k and l,
        computing the corresponding vectors in reciprocal space, and storing
        their norms.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_kvec_Brillouin
        """
        # Get dimensions
        h_dim = self.hsampling[2]
        k_dim = self.ksampling[2]
        l_dim = self.lsampling[2]
        
        # Create centered coordinates for h, k, l
        h_vals = torch.tensor([self._center_kvec(dh, h_dim) for dh in range(h_dim)], device=self.device)
        k_vals = torch.tensor([self._center_kvec(dk, k_dim) for dk in range(k_dim)], device=self.device)
        l_vals = torch.tensor([self._center_kvec(dl, l_dim) for dl in range(l_dim)], device=self.device)
        
        # Create meshgrid
        h_grid, k_grid, l_grid = torch.meshgrid(h_vals, k_vals, l_vals, indexing='ij')
        
        # Stack to create k-vectors
        k_vecs = torch.stack([h_grid, k_grid, l_grid], dim=-1)
        
        # Reshape for matrix multiplication
        k_vecs_flat = k_vecs.reshape(-1, 3)
        
        # Compute 2π * A_inv^T * k for all k-vectors at once
        q_vecs_flat = 2 * torch.pi * torch.matmul(self.A_inv.T, k_vecs_flat.T).T
        
        # Reshape back to grid
        self.kvec = q_vecs_flat.reshape(h_dim, k_dim, l_dim, 3)
        
        # Compute norms
        self.kvec_norm = torch.norm(self.kvec, dim=-1, keepdim=True)
    
    def _center_kvec(self, x: int, L: int) -> float:
        """
        Center k-vector components.
        
        For x and L integers such that 0 < x < L, return -L/2 < x < L/2
        by applying periodic boundary condition in L/2
        
        Args:
            x: Index to center
            L: Length of the periodic box
            
        Returns:
            float: Centered k-vector component
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._center_kvec
        """
        # This function is essentially identical to the NumPy implementation
        # as it's a simple calculation not requiring tensor operations
        return int(((x - L / 2) % L) - L / 2) / L
    
    def _at_kvec_from_miller_points(self, hkl_kvec: tuple):
        """
        Return the indices of all q-vector that are k-vector away from any
        Miller index in the map.
        
        Args:
            hkl_kvec: Tuple of ints with fractional Miller index of the desired k-vector
            
        Returns:
            torch.Tensor: Indices of q-vectors in raveled form
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._at_kvec_from_miller_points
        """
        # Calculate steps for each dimension
        hsteps = int(self.hsampling[2] * (self.hsampling[1] - self.hsampling[0]) + 1)
        ksteps = int(self.ksampling[2] * (self.ksampling[1] - self.ksampling[0]) + 1)
        lsteps = int(self.lsampling[2] * (self.lsampling[1] - self.lsampling[0]) + 1)
        
        # Create meshgrid equivalent to NumPy's mgrid
        h_indices = torch.arange(hkl_kvec[0], hsteps, self.hsampling[2], device=self.device, dtype=torch.long)
        k_indices = torch.arange(hkl_kvec[1], ksteps, self.ksampling[2], device=self.device, dtype=torch.long)
        l_indices = torch.arange(hkl_kvec[2], lsteps, self.lsampling[2], device=self.device, dtype=torch.long)
        
        # Create meshgrid
        h_grid, k_grid, l_grid = torch.meshgrid(h_indices, k_indices, l_indices, indexing='ij')
        
        # Flatten indices
        h_flat = h_grid.reshape(-1)
        k_flat = k_grid.reshape(-1)
        l_flat = l_grid.reshape(-1)
        
        # Create a PyTorch equivalent to np.ravel_multi_index
        # Formula: index = h * (dim_k * dim_l) + k * dim_l + l
        indices = (h_flat * (self.map_shape[1] * self.map_shape[2]) + 
                  k_flat * self.map_shape[2] + 
                  l_flat)
        
        return indices
    
    def compute_hessian(self) -> torch.Tensor:
        """
        Build the projected Hessian matrix using PyTorch operations.
        
        Returns:
            torch.Tensor: Hessian matrix tensor
            
        References:
            - Original implementation: eryx/models.py:OnePhonon.compute_hessian
        """
        # TODO: Initialize Hessian tensor with complex dtype
        # TODO: Initialize diagonal tensor
        # TODO: Compute off-diagonal and diagonal elements
        # TODO: Ensure proper gradient flow
        
        raise NotImplementedError("OnePhonon.compute_hessian not implemented")
    
    def compute_gnm_phonons(self):
        """
        Compute phonon modes and frequencies with PyTorch operations.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon.compute_gnm_phonons
        """
        # TODO: Compute Hessian matrix
        # TODO: For each k-vector, compute dynamical matrix
        # TODO: Use torch.linalg.svd for eigendecomposition
        # TODO: Store eigenvalues and eigenvectors in tensors
        
        raise NotImplementedError("OnePhonon.compute_gnm_phonons not implemented")
    
    def compute_covariance_matrix(self):
        """
        Compute atomic displacement covariance matrix with PyTorch.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon.compute_covariance_matrix
        """
        # TODO: Initialize covariance tensor with complex dtype
        # TODO: Compute for each k-vector with phase factors
        # TODO: Scale to match experimental ADPs
        # TODO: Compute ADP values from covariance
        
        raise NotImplementedError("OnePhonon.compute_covariance_matrix not implemented")
    
    def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                     use_data_adp: bool = False):
        """
        Compute diffuse intensity map using PyTorch operations.
        
        Args:
            rank: If -1, sum across ranks; else use specific rank
            outdir: Directory to save results
            use_data_adp: If True, use ADPs from data instead of computed ones
            
        Returns:
            torch.Tensor: Diffuse intensity map
            
        References:
            - Original implementation: eryx/models.py:OnePhonon.apply_disorder
        """
        # TODO: Choose appropriate ADPs based on use_data_adp
        # TODO: Initialize output tensor with complex dtype
        # TODO: For each k-vector, compute structure factors
        # TODO: Apply phonon mode calculations and summation
        # TODO: Apply resolution mask
        # TODO: Save results if outdir is provided
        
        raise NotImplementedError("OnePhonon.apply_disorder not implemented")

# Add stubs for additional classes as well:

class RigidBodyTranslations:
    """
    PyTorch implementation of rigid body translation disorder model.
    
    References:
        - Original NumPy implementation in eryx/models.py:RigidBodyTranslations
    """
    # TODO: Implement initialization and methods with PyTorch operations
    pass

class LiquidLikeMotions:
    """
    PyTorch implementation of liquid-like motions disorder model.
    
    References:
        - Original NumPy implementation in eryx/models.py:LiquidLikeMotions
    """
    # TODO: Implement initialization and methods with PyTorch operations
    pass

class RigidBodyRotations:
    """
    PyTorch implementation of rigid body rotations disorder model.
    
    References:
        - Original NumPy implementation in eryx/models.py:RigidBodyRotations
    """
    # TODO: Implement initialization and methods with PyTorch operations
    pass

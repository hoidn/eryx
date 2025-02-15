import torch
from typing import Tuple, Optional, Union
from .models_torch import ModelBase, DeviceManager
import numpy as np

class GaussianNetworkModelTorch(ModelBase):
    """PyTorch implementation of Gaussian Network Model.
    
    A model where atoms within a cutoff distance are connected by springs,
    with different spring constants for intra- and inter-molecular pairs.
    """
    
    def __init__(
        self,
        pdb_path: str,
        enm_cutoff: float = 4.0,
        gamma_intra: float = 1.0,
        gamma_inter: float = 1.0,
        device: str = 'cpu'
    ) -> None:
        super().__init__(device)
        self.device_manager = DeviceManager(device)
        self.enm_cutoff = enm_cutoff
        self.gamma_intra = gamma_intra
        self.gamma_inter = gamma_inter
        self._setup(pdb_path)

    def _setup(self, pdb_path: str) -> None:
        """Initialize model from PDB file."""
        from .pdb import AtomicModel  # Import here to avoid circular imports
        
        # Use numpy AtomicModel to load initial data
        model = AtomicModel(pdb_path)
        self.n_atoms_per_asu = model.xyz.shape[0]
        
        # Convert coordinates to tensor
        self.xyz = torch.from_numpy(model.xyz).to(self.device)
        
        # Build neighbor list using numpy initially
        self._build_neighbor_list()

    def _build_neighbor_list(self) -> None:
        """Build list of atom pairs within cutoff distance."""
        # Compute pairwise distances
        diffs = self.xyz.unsqueeze(1) - self.xyz.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)
        
        # Create mask for pairs within cutoff
        self.neighbor_mask = distances <= self.enm_cutoff
        
        # Zero out self-interactions
        self.neighbor_mask.fill_diagonal_(False)

    def compute_hessian(
        self,
        kvec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Hessian matrix for the network.
        
        Parameters
        ----------
        kvec : Optional[torch.Tensor]
            A 1D tensor of shape (3,) representing the k-vector for phase factors.
            If provided, the Hessian will include complex phase factors.
        
        Returns
        -------
        torch.Tensor
            A tensor of shape (n_atoms, 3, n_atoms, 3); if kvec is given, dtype is complex.
        """
        # Get pairs of atoms that are neighbors
        pairs = torch.nonzero(self.neighbor_mask)
        
        # Compute normalized displacement vectors
        diffs = self.xyz[pairs[:, 0]] - self.xyz[pairs[:, 1]]
        distances = torch.linalg.norm(diffs, dim=1, keepdim=True)
        directions = diffs / distances
        
        # Compute spring constants (gamma)
        gamma = torch.full_like(distances, self.gamma_inter, device=self.device)
        same_molecule = pairs[:, 0] // self.n_atoms_per_asu == pairs[:, 1] // self.n_atoms_per_asu
        gamma[same_molecule] = self.gamma_intra
        
        # Build Hessian blocks
        n_atoms = len(self.xyz)
        dtype = torch.complex128 if kvec is not None else torch.float64
        hessian = torch.zeros((n_atoms, 3, n_atoms, 3), dtype=dtype, device=self.device)
        
        for idx, (i, j) in enumerate(pairs):
            # Outer product of direction vectors
            block = torch.matmul(directions[idx].unsqueeze(1), directions[idx].unsqueeze(0))
            block *= gamma[idx]
            
            # Add phase factor if kvec provided
            if kvec is not None:
                phase = torch.dot(kvec, diffs[idx])
                phase_factor = torch.exp(1j * phase)
                block = block * phase_factor
            
            # Add blocks to Hessian
            hessian[i, :, i, :] += block
            hessian[j, :, j, :] += block
            hessian[i, :, j, :] -= block
            hessian[j, :, i, :] -= block.conj() if kvec is not None else block
            
        # Add small regularization term for numerical stability
        eye = torch.eye(3, dtype=dtype, device=self.device)
        eye_full = torch.zeros((n_atoms, 3, n_atoms, 3), dtype=dtype, device=self.device)
        for i in range(n_atoms):
            eye_full[i, :, i, :] = eye
        hessian += 1e-12 * eye_full
            
        return hessian

    def compute_Kinv(
        self,
        hessian: torch.Tensor,
        kvec: Optional[torch.Tensor] = None,
        reshape: bool = True
    ) -> torch.Tensor:
        """
        Compute inverse of dynamical matrix.
        
        Parameters
        ----------
        hessian : torch.Tensor
            Hessian matrix from compute_hessian
        kvec : torch.Tensor, optional
            k-vector for phase factors
        reshape : bool
            If True, reshape output to match hessian shape
            
        Returns
        -------
        Kinv : torch.Tensor
            Inverse of dynamical matrix
        """
        if kvec is not None:
            hessian = self.compute_hessian(kvec)
            
        # Reshape for matrix operations
        shape = hessian.shape
        if reshape:
            hessian = hessian.reshape(shape[0]*3, -1)
            
        # Add small diagonal term for numerical stability
        eye = torch.eye(hessian.shape[0], device=self.device)
        if hessian.dtype == torch.complex128:
            eye = eye.to(torch.complex128)
        hessian = hessian + 1e-12 * eye
        
        # Compute inverse
        Kinv = torch.linalg.pinv(hessian)
        
        if reshape:
            Kinv = Kinv.reshape(shape)
            
        return Kinv

    def compute_K(
        self,
        hessian: torch.Tensor,
        kvec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute dynamical matrix K.
        
        Parameters
        ----------
        hessian : torch.Tensor
            Hessian matrix from compute_hessian
        kvec : torch.Tensor, optional
            k-vector for phase factors
            
        Returns
        -------
        K : torch.Tensor
            Dynamical matrix
        """
        if kvec is not None:
            return self.compute_hessian(kvec)
        return hessian

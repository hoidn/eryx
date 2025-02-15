import torch
import numpy as np
from typing import Union

# Torch implementation of compute_form_factors.
def compute_form_factors(q_grid: torch.Tensor, 
                        ff_a: torch.Tensor,
                        ff_b: torch.Tensor, 
                        ff_c: torch.Tensor) -> torch.Tensor:
    """
    Compute atomic form factors.
    
    Parameters
    ----------
    q_grid : torch.Tensor, shape (n_points, 3)
        q-vectors in Angstrom
    ff_a : torch.Tensor, shape (n_atoms, 4)
        a coefficients of atomic form factors
    ff_b : torch.Tensor, shape (n_atoms, 4)
        b coefficients of atomic form factors
    ff_c : torch.Tensor, shape (n_atoms,)
        c coefficients of atomic form factors
    
    Returns
    -------
    fj : torch.Tensor, shape (n_points, n_atoms)
        Atomic form factors
    """
    # Validate input shapes
    if q_grid.shape[1] != 3:
        raise ValueError(f"q_grid must have shape (n_points, 3), got shape {q_grid.shape}")
    assert ff_a.shape == ff_b.shape, "ff_a and ff_b must have same shape"
    assert ff_a.shape[0] == ff_c.shape[0], "Number of atoms must match across coefficients"
    
    Q = torch.square(torch.linalg.norm(q_grid, dim=1) / (4 * np.pi))
    Q = Q.unsqueeze(1).unsqueeze(2)  # Now shape: (n_points, 1, 1)
    
    exp_term = torch.exp(-1 * ff_b.unsqueeze(0) * Q)  # Shape: (n_points, n_atoms, 4)
    fj = ff_a.unsqueeze(0) * exp_term                # Shape: (n_points, n_atoms, 4)
    fj = torch.sum(fj, dim=2) + ff_c                    # Shape: (n_points, n_atoms)
    
    return fj

# Torch implementation of structure_factors_batch.
def structure_factors_batch(q_grid: torch.Tensor, 
                            xyz: torch.Tensor, 
                            ff_a: torch.Tensor, 
                            ff_b: torch.Tensor, 
                            ff_c: torch.Tensor, 
                            U: Union[None, torch.Tensor] = None,
                            compute_qF: bool = False, 
                            project_on_components: Union[None, torch.Tensor] = None, 
                            batch_size: int = 100000, 
                            n_processes: int = 8) -> torch.Tensor:
    # This is a simplified placeholder.
    # Replace with the proper batched computation using torch operations.
    result = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    return result

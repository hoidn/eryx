"""
PyTorch vectorized implementation of structure factor calculations for Phase 2.

This module contains optimized versions of structure factor calculations that
process multiple ASUs simultaneously using batch operations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union


def compute_form_factors_batched(q_batch: torch.Tensor, ff_a_batch: torch.Tensor,
                                 ff_b_batch: torch.Tensor, ff_c_batch: torch.Tensor) -> torch.Tensor:
    """
    Calculate atomic form factors for batched ASU data.
    
    Args:
        q_batch: Tensor of shape (n_batch, 3) containing q-vectors in Angstrom
        ff_a_batch: Tensor of shape (n_batch, max_atoms, 4) with a coefficients
        ff_b_batch: Tensor of shape (n_batch, max_atoms, 4) with b coefficients  
        ff_c_batch: Tensor of shape (n_batch, max_atoms) with c coefficients
        
    Returns:
        Tensor of shape (n_batch, max_atoms) with atomic form factors
    """
    real_dtype = torch.float64
    q_batch = q_batch.to(dtype=real_dtype)
    ff_a_batch = ff_a_batch.to(dtype=real_dtype)
    ff_b_batch = ff_b_batch.to(dtype=real_dtype)
    ff_c_batch = ff_c_batch.to(dtype=real_dtype)
    
    # Compute q^2 / (16π^2) for each q-vector
    q_squared = torch.sum(q_batch * q_batch, dim=-1, keepdim=True) / (16 * torch.pi * torch.pi)  # [n_batch, 1]
    q_squared = q_squared.unsqueeze(-1)  # [n_batch, 1, 1]
    
    # Compute Gaussian form factor components
    exponents = -ff_b_batch * q_squared  # [n_batch, max_atoms, 4]
    exponents = torch.clamp(exponents, min=-50.0, max=50.0)
    exp_terms = torch.exp(exponents)  # [n_batch, max_atoms, 4]
    gaussian_terms = ff_a_batch * exp_terms  # [n_batch, max_atoms, 4]
    
    # Sum over the last dimension (4 Gaussian terms)
    ff_gauss = torch.sum(gaussian_terms, dim=-1)  # [n_batch, max_atoms]
    
    # Add the constant term c
    form_factors = ff_gauss + ff_c_batch  # [n_batch, max_atoms]
    
    return form_factors


def structure_factors_multi_asu(
    q_grid: torch.Tensor,
    xyz_list: List[torch.Tensor],
    ff_a_list: List[torch.Tensor],
    ff_b_list: List[torch.Tensor],
    ff_c_list: List[torch.Tensor],
    U_list: Optional[List[torch.Tensor]] = None,
    compute_qF: bool = False,
    project_list: Optional[List[torch.Tensor]] = None,
    sum_over_atoms: bool = False
) -> torch.Tensor:
    """
    Compute structure factors for multiple ASUs in a single batched operation.
    
    This function processes all ASUs simultaneously, eliminating the sequential
    loop over ASUs for improved GPU utilization.
    
    Args:
        q_grid: Q-vector tensor with shape [n_points, 3]
        xyz_list: List of atomic coordinates, each [n_atoms_i, 3]
        ff_a_list: List of form factor coefficients, each [n_atoms_i, 4]
        ff_b_list: List of form factor coefficients, each [n_atoms_i, 4]
        ff_c_list: List of form factor coefficients, each [n_atoms_i]
        U_list: Optional list of ADPs, each [n_atoms_i]
        compute_qF: If True, compute q-weighted structure factors
        project_list: Optional list of projection matrices
        sum_over_atoms: If True, sum over atoms
        
    Returns:
        Structure factors tensor with shape [n_points, n_asu, n_dof_per_asu]
    """
    device = q_grid.device
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    
    n_points = q_grid.shape[0]
    n_asu = len(xyz_list)
    
    # Ensure all inputs are on correct device and dtype
    q_grid = q_grid.to(device=device, dtype=real_dtype)
    
    # Find maximum number of atoms across all ASUs for padding
    max_atoms = max(xyz.shape[0] for xyz in xyz_list)
    
    # Stack and pad all ASU data to create uniform tensors
    xyz_padded = torch.zeros((n_asu, max_atoms, 3), device=device, dtype=real_dtype)
    ff_a_padded = torch.zeros((n_asu, max_atoms, 4), device=device, dtype=real_dtype)
    ff_b_padded = torch.zeros((n_asu, max_atoms, 4), device=device, dtype=real_dtype)
    ff_c_padded = torch.zeros((n_asu, max_atoms), device=device, dtype=real_dtype)
    atom_mask = torch.zeros((n_asu, max_atoms), device=device, dtype=torch.bool)
    
    for i, (xyz, ff_a, ff_b, ff_c) in enumerate(zip(xyz_list, ff_a_list, ff_b_list, ff_c_list)):
        n_atoms = xyz.shape[0]
        xyz_padded[i, :n_atoms] = xyz.to(device=device, dtype=real_dtype)
        ff_a_padded[i, :n_atoms] = ff_a.to(device=device, dtype=real_dtype)
        ff_b_padded[i, :n_atoms] = ff_b.to(device=device, dtype=real_dtype)
        ff_c_padded[i, :n_atoms] = ff_c.to(device=device, dtype=real_dtype)
        atom_mask[i, :n_atoms] = True
    
    # Handle U (ADP) parameters if provided
    if U_list is not None:
        U_padded = torch.zeros((n_asu, max_atoms), device=device, dtype=real_dtype)
        for i, U in enumerate(U_list):
            if U is not None:
                n_atoms = U.shape[0]
                U_padded[i, :n_atoms] = U.to(device=device, dtype=real_dtype)
    else:
        U_padded = None
    
    # Compute form factors for all ASUs at once
    # Reshape for batch processing: combine n_points and n_asu dimensions
    q_expanded = q_grid.unsqueeze(1).expand(-1, n_asu, -1)  # [n_points, n_asu, 3]
    q_batch = q_expanded.reshape(n_points * n_asu, 3)  # [n_points*n_asu, 3]
    
    # Repeat ASU data for each q-point
    ff_a_batch = ff_a_padded.unsqueeze(0).expand(n_points, -1, -1, -1).reshape(n_points*n_asu, max_atoms, 4)
    ff_b_batch = ff_b_padded.unsqueeze(0).expand(n_points, -1, -1, -1).reshape(n_points*n_asu, max_atoms, 4)
    ff_c_batch = ff_c_padded.unsqueeze(0).expand(n_points, -1, -1).reshape(n_points*n_asu, max_atoms)
    
    # Compute form factors
    form_factors = compute_form_factors_batched(q_batch, ff_a_batch, ff_b_batch, ff_c_batch)
    form_factors = form_factors.reshape(n_points, n_asu, max_atoms)  # [n_points, n_asu, max_atoms]
    
    # Compute phase factors: exp(i * q . r)
    # Use einsum for efficient batch dot product
    xyz_expanded = xyz_padded.unsqueeze(0).expand(n_points, -1, -1, -1)  # [n_points, n_asu, max_atoms, 3]
    q_for_phase = q_expanded.unsqueeze(2)  # [n_points, n_asu, 1, 3]
    
    # Compute q . r for all atoms and ASUs
    q_dot_r = torch.sum(q_for_phase * xyz_expanded, dim=-1)  # [n_points, n_asu, max_atoms]
    
    # Apply Debye-Waller factor if U is provided
    if U_padded is not None:
        U_expanded = U_padded.unsqueeze(0).expand(n_points, -1, -1)  # [n_points, n_asu, max_atoms]
        q_squared = torch.sum(q_expanded**2, dim=-1, keepdim=True)  # [n_points, n_asu, 1]
        dwf = torch.exp(-0.5 * q_squared * U_expanded)  # [n_points, n_asu, max_atoms]
        form_factors = form_factors * dwf
    
    # Compute complex exponential
    phase_real = torch.cos(q_dot_r)  # [n_points, n_asu, max_atoms]
    phase_imag = torch.sin(q_dot_r)  # [n_points, n_asu, max_atoms]
    
    # Apply form factors and mask
    mask_expanded = atom_mask.unsqueeze(0).expand(n_points, -1, -1)  # [n_points, n_asu, max_atoms]
    A_real = form_factors * phase_real * mask_expanded
    A_imag = form_factors * phase_imag * mask_expanded
    
    # Handle compute_qF option
    if compute_qF:
        # Multiply by q-vector for each component
        # This creates [n_points, n_asu, max_atoms, 3] tensor
        q_for_qF = q_expanded.unsqueeze(2).expand(-1, -1, max_atoms, -1)  # [n_points, n_asu, max_atoms, 3]
        
        # Expand A_real and A_imag
        A_real_expanded = A_real.unsqueeze(-1)  # [n_points, n_asu, max_atoms, 1]
        A_imag_expanded = A_imag.unsqueeze(-1)  # [n_points, n_asu, max_atoms, 1]
        
        A_real_qF = A_real_expanded * q_for_qF  # [n_points, n_asu, max_atoms, 3]
        A_imag_qF = A_imag_expanded * q_for_qF  # [n_points, n_asu, max_atoms, 3]
        
        # Reshape to combine atom and component dimensions
        A_real = A_real_qF.reshape(n_points, n_asu, max_atoms * 3)
        A_imag = A_imag_qF.reshape(n_points, n_asu, max_atoms * 3)
    
    # Apply projection if provided
    if project_list is not None:
        # Stack projection matrices (assuming they're all the same size)
        n_dof = project_list[0].shape[1]
        projected_real = torch.zeros((n_points, n_asu, n_dof), device=device, dtype=real_dtype)
        projected_imag = torch.zeros((n_points, n_asu, n_dof), device=device, dtype=real_dtype)
        
        for i, proj in enumerate(project_list):
            if proj is not None:
                proj = proj.to(device=device, dtype=real_dtype)
                # Apply projection for this ASU
                if compute_qF:
                    # Truncate to actual atom count for this ASU
                    n_atoms_i = xyz_list[i].shape[0]
                    A_real_i = A_real[:, i, :n_atoms_i*3]  # [n_points, n_atoms_i*3]
                    A_imag_i = A_imag[:, i, :n_atoms_i*3]  # [n_points, n_atoms_i*3]
                else:
                    n_atoms_i = xyz_list[i].shape[0]
                    A_real_i = A_real[:, i, :n_atoms_i]  # [n_points, n_atoms_i]
                    A_imag_i = A_imag[:, i, :n_atoms_i]  # [n_points, n_atoms_i]
                
                projected_real[:, i] = torch.matmul(A_real_i, proj)
                projected_imag[:, i] = torch.matmul(A_imag_i, proj)
        
        A_real = projected_real
        A_imag = projected_imag
    else:
        # No projection - keep per-atom results or sum
        if sum_over_atoms:
            A_real = torch.sum(A_real, dim=-1)  # [n_points, n_asu]
            A_imag = torch.sum(A_imag, dim=-1)  # [n_points, n_asu]
    
    # Create complex result
    result = torch.complex(A_real, A_imag)
    
    return result


def prepare_asu_batch(asu_data_list: List[dict]) -> Tuple[List[torch.Tensor], ...]:
    """
    Prepare ASU data for batch processing.
    
    Args:
        asu_data_list: List of ASU data dictionaries
        
    Returns:
        Tuple of lists: (xyz_list, ff_a_list, ff_b_list, ff_c_list, project_list)
    """
    xyz_list = []
    ff_a_list = []
    ff_b_list = []
    ff_c_list = []
    project_list = []
    
    for asu in asu_data_list:
        xyz_list.append(asu['xyz'])
        ff_a_list.append(asu['ff_a'])
        ff_b_list.append(asu['ff_b'])
        ff_c_list.append(asu['ff_c'])
        project_list.append(asu.get('project', None))
    
    return xyz_list, ff_a_list, ff_b_list, ff_c_list, project_list
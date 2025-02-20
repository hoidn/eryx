import pytest
import numpy as np
import torch
from eryx.onephonon_torch import OnePhononTorch
from eryx.models import OnePhonon  # reference (numpy) implementation

@pytest.fixture
def device():
    return torch.device("cpu")

def test_phonon_modes_equivalence(device):
    """
    Compare the full onephonon diffuse intensity computed by the torch and numpy implementations.
    The computed intensities should match within a relative tolerance of 1e-5.
    """
    pdb_path = "tests/pdbs/5zck.pdb"  # example pdb file
    hsampling = [-1, 2, 10]
    ksampling = [-1, 2, 10]
    lsampling = [-1, 2, 10]
    
    model_torch = OnePhononTorch(pdb_path, hsampling, ksampling, lsampling, device=device)
    I_torch = model_torch.forward().cpu().numpy()
    
    model_np = OnePhonon(pdb_path, hsampling, ksampling, lsampling)
    I_np = model_np.apply_disorder(rank=-1)
    
    assert np.allclose(I_torch, I_np, rtol=1e-5), "Diffuse intensities mismatch between torch and numpy implementations."

def test_structure_factors_match(device):
    """
    Validate that the structure factors computed with torch operations match the numpy reference
    within a relative tolerance of 1e-5.
    """
    pdb_path = "tests/pdbs/5zck.pdb"
    hsampling = [-1, 2, 10]
    ksampling = [-1, 2, 10]
    lsampling = [-1, 2, 10]
    
    model_torch = OnePhononTorch(pdb_path, hsampling, ksampling, lsampling, device=device)
    q_grid = model_torch.q_grid.to(device)
    torch_sf = model_torch._compute_crystal_transform_torch(q_grid).cpu().detach().numpy()
    
    from eryx.scatter import structure_factors
    from eryx.pdb import AtomicModel
    atomic_model = AtomicModel(pdb_path, expand_p1=True)
    np_sf = structure_factors(2 * np.pi * np.inner(atomic_model.A_inv.T, model_torch.hkl_grid).T,
                              atomic_model.xyz[0],
                              atomic_model.ff_a[0],
                              atomic_model.ff_b[0],
                              atomic_model.ff_c[0],
                              U=atomic_model.adp[0]/(8*np.pi*np.pi))
    assert np.allclose(torch.abs(torch_sf), np.abs(np_sf), rtol=1e-5), "Structure factors differ between torch and numpy."
def test_adp_scale_factor(device):
    """
    Test that the ADP scale factor computed by OnePhononTorch matches
    the reference (numpy) implementation.
    """
    pdb_path = "tests/pdbs/5zck.pdb"
    hsampling = [-1, 2, 10]
    ksampling = [-1, 2, 10]
    lsampling = [-1, 2, 10]
    model_torch = OnePhononTorch(pdb_path, hsampling, ksampling, lsampling, device=device)
    model_np = OnePhonon(pdb_path, hsampling, ksampling, lsampling)  # assuming _compute_adp_scale() exists in OnePhonon
    scale_torch = model_torch._compute_adp_scale_factor()
    scale_np = model_np._compute_adp_scale()
    assert torch.allclose(scale_torch.cpu(), torch.tensor(scale_np, dtype=scale_torch.dtype), rtol=1e-5)


def test_scaled_variances(device):
    """
    Test that the variances from the GNM (after scaling) approximately
    match the experimental variances.
    """
    pdb_path = "tests/pdbs/5zck.pdb"
    hsampling = [-1, 2, 10]
    ksampling = [-1, 2, 10]
    lsampling = [-1, 2, 10]
    model = OnePhononTorch(pdb_path, hsampling, ksampling, lsampling, device=device)
    scale = model._compute_adp_scale_factor()
    kinv_diag = model._compute_adp_diagonal()
    model_vars = scale * kinv_diag
    exp_vars = torch.tensor(model.gnm_torch.atomic_model.adp[0],
                            device=model.device,
                            dtype=model_vars.dtype) / (8 * torch.pi * torch.pi)
    assert torch.allclose(model_vars.cpu(), exp_vars.cpu(), rtol=1e-4)
import torch
import pytest
from eryx.onephonon_torch import OnePhononTorch
from eryx.models import OnePhonon

def test_adp_scale_factor():
    """
    Test that the ADP scale factor computed by OnePhononTorch matches the numpy reference.
    """
    # Create a OnePhononTorch instance with test parameters
    model_torch = OnePhononTorch(
        pdb_path="tests/pdbs/5zck.pdb",
        hsampling=[-1, 2, 10],
        ksampling=[-1, 2, 10],
        lsampling=[-1, 2, 10],
        device=torch.device("cpu")
    )
    # Create a numpy-based OnePhonon instance serving as the reference
    model_np = OnePhonon(
        pdb_path="tests/pdbs/5zck.pdb",
        hsampling=[-1, 2, 10],
        ksampling=[-1, 2, 10],
        lsampling=[-1, 2, 10]
    )
    scale_torch = model_torch._compute_adp_scale_factor()
    scale_np = model_np._compute_adp_scale()   # Assuming _compute_adp_scale() is defined in OnePhonon
    assert torch.allclose(
        scale_torch.cpu(), 
        torch.tensor(scale_np, dtype=scale_torch.dtype),
        rtol=1e-5
    ), f"Scale factors differ: torch={scale_torch.item()}, numpy={scale_np}"

def test_scaled_variances():
    """
    Test that the scaled model variances match the experimental ADPs.
    """
    model = OnePhononTorch(
        pdb_path="tests/pdbs/5zck.pdb",
        hsampling=[-1, 2, 10],
        ksampling=[-1, 2, 10],
        lsampling=[-1, 2, 10],
        device=torch.device("cpu")
    )
    scale = model._compute_adp_scale_factor()
    kinv_diag = model._compute_kinv_diagonal()
    model_vars = scale * kinv_diag
    exp_vars_tensor = torch.tensor(
        model.gnm_torch.atomic_model.adp[0],
        dtype=model_vars.dtype
    ) / (8 * torch.pi * torch.pi)
    assert torch.allclose(
        model_vars.cpu(),
        exp_vars_tensor,
        rtol=1e-4
    ), f"Scaled variances differ: model_vars={model_vars.cpu()}, experimental={exp_vars_tensor}"

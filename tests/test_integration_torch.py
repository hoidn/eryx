import torch
import numpy as np
import pytest

# Import numpy versions from the existing models
from eryx.models import (
    RigidBodyTranslations,
    LiquidLikeMotions,
    Ensemble,
    NonInteractingDeformableMolecules,
    OnePhonon
)

# Import the new torch implementations
from eryx.rigid_body_torch import RigidBodyTranslationsTorch
from eryx.liquid_like_motions_torch import LiquidLikeMotionsTorch
from eryx.ensemble_torch import EnsembleTorch
from eryx.deformable_molecules_torch import NonInteractingDeformableMoleculesTorch
from eryx.one_phonon_torch import OnePhononTorch

# Define common parameters and sample pdb file path.
PDB_PATH = "tests/pdbs/5zck.pdb"
HSAMPLING = (-4, 4, 1)
KSAMPLING = (-17, 17, 1)
LSAMPLING = (-29, 29, 1)


def test_rigid_body_translations_integration():
    sigma = 0.5
    # Instantiate numpy version
    numpy_model = RigidBodyTranslations(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING)
    numpy_output = numpy_model.apply_disorder(sigma)
    
    # Instantiate torch version
    torch_model = RigidBodyTranslationsTorch(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, device='cpu')
    torch_output = torch_model.apply_disorder(sigma)
    
    # Compare outputs: convert numpy output to tensor for torch.allclose
    assert torch.allclose(torch_output.cpu(), torch.from_numpy(numpy_output), rtol=1e-7, equal_nan=True)


def test_liquid_like_motions_integration():
    sigma = 0.5
    # Instantiate numpy version
    numpy_model = LiquidLikeMotions(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True)
    numpy_output = numpy_model.apply_disorder(sigma, 1.0)
    
    # Instantiate torch version
    torch_model = LiquidLikeMotionsTorch(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True, device='cpu')
    torch_output = torch_model.apply_disorder(sigma, 1.0)
    
    assert torch.allclose(torch_output.cpu(), torch.from_numpy(numpy_output), rtol=1e-7, equal_nan=True)


def test_ensemble_integration():
    # For Ensemble, use default weights (which should produce equal weighting)
    weights = None
    # Instantiate numpy version
    numpy_model = Ensemble(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True)
    numpy_output = numpy_model.apply_disorder(weights)
    
    # Instantiate torch version
    torch_model = EnsembleTorch(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True, device='cpu')
    torch_output = torch_model.apply_disorder(weights)
    
    assert torch.allclose(torch_output.cpu(), torch.from_numpy(numpy_output), rtol=1e-7)


def test_deformable_molecules_integration():
    # Instantiate numpy version
    numpy_model = NonInteractingDeformableMolecules(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True)
    numpy_output = numpy_model.apply_disorder()
    
    # Instantiate torch version
    torch_model = NonInteractingDeformableMoleculesTorch(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True, device='cpu')
    torch_output = torch_model.apply_disorder()
    
    assert torch.allclose(torch_output.cpu(), torch.from_numpy(numpy_output), rtol=1e-7)


def test_one_phonon_integration():
    # Instantiate numpy version
    numpy_model = OnePhonon(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True)
    numpy_output = numpy_model.apply_disorder()  # Expected to return the one-phonon intensity map
    
    # Instantiate torch version
    torch_model = OnePhononTorch(PDB_PATH, HSAMPLING, KSAMPLING, LSAMPLING, expand_p1=True, device='cpu')
    torch_output = torch_model.apply_disorder()
    
    assert torch.allclose(torch_output.cpu(), torch.from_numpy(numpy_output), rtol=1e-7)

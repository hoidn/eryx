"""
Eryx: A package for diffuse scattering simulation and analysis.

This package provides tools for simulating diffuse scattering from protein crystals,
with both NumPy and PyTorch implementations. The PyTorch implementation enables
gradient-based optimization of simulation parameters.
"""

# NumPy implementation
from eryx.models import OnePhonon, RigidBodyTranslations, LiquidLikeMotions, RigidBodyRotations
from eryx.scatter import compute_form_factors, structure_factors, structure_factors_batch
from eryx.base import compute_molecular_transform, compute_crystal_transform

# Import PyTorch implementations if available
try:
    import torch
    HAS_TORCH = True
    
    # Import PyTorch implementations with _torch suffix to avoid naming conflicts
    from eryx.models_torch import OnePhonon as OnePhonon_torch
    from eryx.models_torch import RigidBodyTranslations as RigidBodyTranslations_torch
    from eryx.models_torch import LiquidLikeMotions as LiquidLikeMotions_torch
    from eryx.models_torch import RigidBodyRotations as RigidBodyRotations_torch
    
    from eryx.scatter_torch import compute_form_factors as compute_form_factors_torch
    from eryx.scatter_torch import structure_factors as structure_factors_torch
    from eryx.scatter_torch import structure_factors_batch as structure_factors_batch_torch
    
    from eryx.base_torch import compute_molecular_transform as compute_molecular_transform_torch
    from eryx.base_torch import compute_crystal_transform as compute_crystal_transform_torch
    
    # Import adapters for convenience
    from eryx.adapters import PDBToTensor, GridToTensor, TensorToNumpy, ModelAdapters
    
    # Import utility classes from torch_utils.py
    from eryx.torch_utils import ComplexTensorOps
    from eryx.torch_utils import EigenOps
    from eryx.torch_utils import GradientUtils
    
except ImportError:
    HAS_TORCH = False

__version__ = "0.1.0"

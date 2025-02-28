Specification Template for PyTorch Port Stub Generation

Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

High-Level Objective

Create a parallel PyTorch project structure with comprehensive stub implementations for all components that need to be ported from NumPy to PyTorch

Mid-Level Objectives

Create the directory structure for the PyTorch implementation
Generate stub files for all modules that need PyTorch equivalents
Implement adapter components to bridge NumPy and PyTorch
Create comprehensive function stubs with detailed TODOs, docstrings, and type hints
Set up scaffolding for testing components

Implementation Notes
Code Organization

Create _torch.py files parallel to existing NumPy implementations (e.g., scatter.py → scatter_torch.py)
Function and class names should match original NumPy versions but may include _torch suffix if necessary to avoid import conflicts
Include imports for both NumPy and PyTorch in stub files for easier reference during implementation

Documentation Standards

Each module should have a detailed module docstring explaining its purpose
Each function/method should have a detailed docstring with:

Brief description
Parameter descriptions with types
Return value descriptions with types
Note about PyTorch conversion from NumPy
References to the original NumPy implementation


Use Google style docstrings for consistency

Type Hints

Use full type hints throughout
Use torch.Tensor instead of np.ndarray for tensor arguments
Include shape information in comments (e.g., # shape: (batch_size, n_points, 3))
Use Optional for parameters that can be None
Use Union for parameters that can have multiple types

TODO Comments

Each stub should include detailed TODO comments explaining the implementation approach
TODOs should reference the corresponding NumPy implementation by file and line number
TODOs should highlight specific PyTorch considerations (e.g., gradient flow, tensor operations)

Tensor vs. Array

Clearly mark functions that operate on tensors vs. arrays
Note when tensor-to-array conversion might be needed
Include considerations for gradient flow and detachment

Error Handling

Include placeholder error handling with NotImplementedError
Document expected error cases in comments

Context
Beginning context

Existing NumPy implementation in eryx/ directory
Empty target structure for PyTorch port

Ending context

Complete parallel directory structure with _torch.py files
Comprehensive function stubs for all components to be ported
Adapter components for NumPy-PyTorch conversion
Test scaffolding set up

Low-Level Tasks

Ordered from start to finish


Create Core Module Stubs

aiderCopyCREATE eryx/models_torch.py
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
        Build the matrix A that projects small rigid-body displacements using PyTorch.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_A
        """
        # TODO: Implement tensor-based projection matrix construction
        # TODO: Handle the group_by='asu' case with PyTorch matrix operations
        # TODO: Ensure gradient flow through all operations
        
        raise NotImplementedError("OnePhonon._build_A not implemented")
    
    def _build_M(self):
        """
        Build the mass matrix M using PyTorch operations.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_M
        """
        # TODO: Get all-atoms mass matrix with _build_M_allatoms()
        # TODO: Project if needed based on group_by parameter
        # TODO: Implement Cholesky decomposition with PyTorch for Linv
        
        raise NotImplementedError("OnePhonon._build_M not implemented")
    
    def _build_M_allatoms(self):
        """
        Build all-atom mass matrix using PyTorch operations.
        
        Returns:
            torch.Tensor: Mass matrix for all atoms
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_M_allatoms
        """
        # TODO: Convert mass array to tensor
        # TODO: Create block diagonal mass matrix
        # TODO: Reshape to the correct dimensions
        
        raise NotImplementedError("OnePhonon._build_M_allatoms not implemented")
    
    def _project_M(self, M_allatoms: torch.Tensor) -> torch.Tensor:
        """
        Project all-atom mass matrix using PyTorch tensor operations.
        
        Args:
            M_allatoms: All-atom mass matrix tensor
            
        Returns:
            torch.Tensor: Projected mass matrix
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._project_M
        """
        # TODO: Initialize output tensor with correct shape
        # TODO: Implement matrix multiplication with PyTorch for projection
        # TODO: Ensure gradient flow through operations
        
        raise NotImplementedError("OnePhonon._project_M not implemented")
    
    def _build_kvec_Brillouin(self):
        """
        Compute k-vectors and their norm in the first Brillouin zone using PyTorch.
        
        References:
            - Original implementation: eryx/models.py:OnePhonon._build_kvec_Brillouin
        """
        # TODO: Generate k-vector grid using PyTorch's meshgrid
        # TODO: Compute k-vector norms with torch.norm
        # TODO: Store as tensors for differentiable computations
        
        raise NotImplementedError("OnePhonon._build_kvec_Brillouin not implemented")
    
    def _center_kvec(self, x: int, L: int) -> float:
        """
        Center k-vector components.
        
        Args:
            x: Index to center
            L: Length of periodic box
            
        Returns:
            float: Centered k-vector component
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._center_kvec
        """
        # This function can remain the same as it's a simple calculation
        # that doesn't need tensor operations
        return int(((x - L / 2) % L) - L / 2) / L
    
    def _at_kvec_from_miller_points(self, hkl_kvec: Tuple[int, int, int]):
        """
        Return indices of q-vectors that are k-vector away from Miller indices.
        
        Args:
            hkl_kvec: Fractional Miller index tuple
            
        Returns:
            torch.Tensor: Indices of q-vectors
            
        References:
            - Original implementation: eryx/models.py:OnePhonon._at_kvec_from_miller_points
        """
        # TODO: Calculate index grid
        # TODO: Convert to PyTorch tensor for output
        # TODO: Handle ravel operation with PyTorch
        
        raise NotImplementedError("OnePhonon._at_kvec_from_miller_points not implemented")
    
    def compute_hessian(self):
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

Create Scatter Module Stub

aiderCopyCREATE eryx/scatter_torch.py
"""
PyTorch implementation of structure factor calculations for diffuse scattering.

This module contains PyTorch versions of the structure factor calculations defined
in eryx/scatter.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/scatter.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union

def compute_form_factors(q_grid: torch.Tensor, ff_a: torch.Tensor, 
                         ff_b: torch.Tensor, ff_c: torch.Tensor) -> torch.Tensor:
    """
    Calculate atomic form factors at the input q-vectors using PyTorch.
    
    Args:
        q_grid: PyTorch tensor of shape (n_points, 3) containing q-vectors in Angstrom
        ff_a: PyTorch tensor of shape (n_atoms, 4) with a coefficients of atomic form factors
        ff_b: PyTorch tensor of shape (n_atoms, 4) with b coefficients of atomic form factors
        ff_c: PyTorch tensor of shape (n_atoms,) with c coefficients of atomic form factors
        
    Returns:
        PyTorch tensor of shape (n_points, n_atoms) with atomic form factors
        
    References:
        - Original implementation: eryx/scatter.py:compute_form_factors
    """
    # TODO: Convert the NumPy implementation to PyTorch operations:
    # 1. Compute Q = ||q_grid||^2 / (4*pi)^2 using torch operations
    # 2. Compute exponential terms using torch.exp()
    # 3. Perform tensor multiplications and summations
    # 4. Return the transposed form factors tensor
    
    raise NotImplementedError("compute_form_factors not implemented")

def structure_factors_batch(q_grid: torch.Tensor, xyz: torch.Tensor, 
                           ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor, 
                           U: Optional[torch.Tensor] = None,
                           compute_qF: bool = False, 
                           project_on_components: Optional[torch.Tensor] = None,
                           sum_over_atoms: bool = True) -> torch.Tensor:
    """
    Compute structure factors for an atomic model at the given q-vectors using PyTorch.
    
    Args:
        q_grid: PyTorch tensor of shape (n_points, 3) with q-vectors in Angstrom
        xyz: PyTorch tensor of shape (n_atoms, 3) with atomic positions in Angstrom
        ff_a: PyTorch tensor of shape (n_atoms, 4) with a coefficients
        ff_b: PyTorch tensor of shape (n_atoms, 4) with b coefficients 
        ff_c: PyTorch tensor of shape (n_atoms,) with c coefficients
        U: Optional PyTorch tensor of shape (n_atoms,) with isotropic displacement parameters
        compute_qF: If True, return structure factors times q-vectors
        project_on_components: Optional projection matrix
        sum_over_atoms: If True, sum over atoms; otherwise return per-atom values
        
    Returns:
        PyTorch tensor containing structure factors
        
    References:
        - Original implementation: eryx/scatter.py:structure_factors_batch
    """
    # TODO: Convert the NumPy implementation to PyTorch operations:
    # 1. Initialize U as zeros tensor if None
    # 2. Compute form factors using compute_form_factors()
    # 3. Calculate q magnitudes using torch.norm()
    # 4. Compute qUq with broadcasting
    # 5. Calculate sine and cosine components using torch operations
    # 6. Apply Debye-Waller factor with torch.exp()
    # 7. Handle compute_qF, project_on_components, and sum_over_atoms options
    
    raise NotImplementedError("structure_factors_batch not implemented")

def structure_factors(q_grid: torch.Tensor, xyz: torch.Tensor, 
                     ff_a: torch.Tensor, ff_b: torch.Tensor, ff_c: torch.Tensor, 
                     U: Optional[torch.Tensor] = None,
                     batch_size: int = 100000, n_processes: int = 1,
                     compute_qF: bool = False, 
                     project_on_components: Optional[torch.Tensor] = None,
                     sum_over_atoms: bool = True) -> torch.Tensor:
    """
    Batched version of structure factor calculation using PyTorch.
    
    Args:
        q_grid: PyTorch tensor of shape (n_points, 3) with q-vectors in Angstrom
        xyz: PyTorch tensor of shape (n_atoms, 3) with atomic positions in Angstrom
        ff_a: PyTorch tensor of shape (n_atoms, 4) with a coefficients
        ff_b: PyTorch tensor of shape (n_atoms, 4) with b coefficients 
        ff_c: PyTorch tensor of shape (n_atoms,) with c coefficients
        U: Optional PyTorch tensor of shape (n_atoms,) with isotropic displacement parameters
        batch_size: Number of q-vectors to evaluate per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        compute_qF: If True, return structure factors times q-vectors
        project_on_components: Optional projection matrix
        sum_over_atoms: If True, sum over atoms; otherwise return per-atom values
        
    Returns:
        PyTorch tensor containing structure factors
        
    References:
        - Original implementation: eryx/scatter.py:structure_factors
    """
    # TODO: Convert the NumPy implementation to PyTorch operations:
    # 1. Compute number of batches based on q_grid size and batch_size
    # 2. Create output tensor of appropriate shape
    # 3. Process each batch with structure_factors_batch
    # 4. Handle device placement for efficient GPU usage
    # Note: multiprocessing (n_processes) should be ignored in PyTorch implementation as
    # PyTorch has its own parallelization mechanisms
    
    raise NotImplementedError("structure_factors not implemented")

Create Map Utilities Stub

aiderCopyCREATE eryx/map_utils_torch.py
"""
PyTorch implementation of map utility functions for diffuse scattering.

This module contains PyTorch versions of the map utility functions defined
in eryx/map_utils.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/map_utils.py
"""

import numpy as np
import torch
import gemmi  # Keep gemmi imports for crystallographic data
from typing import Tuple, List, Dict, Optional, Union, Any

def generate_grid(A_inv: torch.Tensor, hsampling: Tuple[float, float, float], 
                 ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float], 
                 return_hkl: bool = False) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Generate a grid of q-vectors based on the desired extents and spacing in hkl space.
    
    Args:
        A_inv: PyTorch tensor of shape (3, 3) with fractional cell orthogonalization matrix
        hsampling: Tuple (hmin, hmax, oversampling) for h dimension
        ksampling: Tuple (kmin, kmax, oversampling) for k dimension
        lsampling: Tuple (lmin, lmax, oversampling) for l dimension
        return_hkl: If True, return hkl indices rather than q-vectors
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points, 3) with q-vectors or hkl indices
            - Tuple with shape of 3D map
            
    References:
        - Original implementation: eryx/map_utils.py:generate_grid
    """
    # TODO: Implement PyTorch version using torch.meshgrid or equivalent
    # TODO: Calculate hsteps, ksteps, lsteps
    # TODO: Generate hkl_grid using torch operations 
    # TODO: Reshape and reorder dimensions correctly
    # TODO: Calculate q_grid if return_hkl is False
    
    raise NotImplementedError("generate_grid not implemented")

def get_symmetry_equivalents(hkl_grid: torch.Tensor, sym_ops: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Get symmetry equivalent Miller indices of input hkl_grid.
    
    Args:
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        sym_ops: Dictionary mapping integer keys to rotation matrices as PyTorch tensors
        
    Returns:
        PyTorch tensor of shape (n_asu, n_points, 3) with stacked hkl indices
        
    References:
        - Original implementation: eryx/map_utils.py:get_symmetry_equivalents
    """
    # TODO: Initialize output tensor
    # TODO: Apply symmetry operations using torch.matmul
    # TODO: Stack and reshape results
    
    raise NotImplementedError("get_symmetry_equivalents not implemented")

def get_ravel_indices(hkl_grid_sym: torch.Tensor, 
                    sampling: Tuple[float, float, float]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Map 3D hkl indices to corresponding 1D indices after raveling.
    
    Args:
        hkl_grid_sym: PyTorch tensor of shape (n_asu, n_points, 3) with symmetry equivalents
        sampling: Tuple with sampling rates along (h, k, l)
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_asu, n_points) with raveled indices
            - Tuple with shape of expanded/raveled map
            
    References:
        - Original implementation: eryx/map_utils.py:get_ravel_indices
    """
    # TODO: Reshape input for processing
    # TODO: Convert to integer indices with scaling
    # TODO: Find bounds and calculate map shape
    # TODO: Implement ravel_multi_index equivalent using PyTorch
    
    raise NotImplementedError("get_ravel_indices not implemented")

def compute_resolution(cell: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    """
    Compute reflections' resolution in 1/Angstrom using PyTorch.
    
    Args:
        cell: PyTorch tensor of shape (6,) with unit cell parameters
        hkl: PyTorch tensor of shape (n_refl, 3) with Miller indices
        
    Returns:
        PyTorch tensor of shape (n_refl,) with resolution in Angstrom
        
    References:
        - Original implementation: eryx/map_utils.py:compute_resolution
    """
    # TODO: Extract cell parameters
    # TODO: Convert to radians using torch operations
    # TODO: Implement calculation using torch functions
    # TODO: Handle potential divide by zero with torch.where
    
    raise NotImplementedError("compute_resolution not implemented")

def get_resolution_mask(cell: torch.Tensor, hkl_grid: torch.Tensor, 
                       res_limit: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a boolean mask for resolution limits.
    
    Args:
        cell: PyTorch tensor of shape (6,) with cell parameters
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        res_limit: High resolution limit in Angstrom
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points,) with boolean mask
            - PyTorch tensor of shape (n_points,) with resolution map
            
    References:
        - Original implementation: eryx/map_utils.py:get_resolution_mask
    """
    # TODO: Compute resolution map
    # TODO: Create mask by comparing to res_limit
    
    raise NotImplementedError("get_resolution_mask not implemented")

def get_dq_map(A_inv: torch.Tensor, hkl_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute distance to nearest Bragg peak for grid points.
    
    Args:
        A_inv: PyTorch tensor of shape (3, 3) with cell orthogonalization matrix
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        
    Returns:
        PyTorch tensor of shape (n_points,) with distances
        
    References:
        - Original implementation: eryx/map_utils.py:get_dq_map
    """
    # TODO: Find closest integral hkl points using torch.round
    # TODO: Convert to q-vectors
    # TODO: Compute distances using torch.norm
    # TODO: Round to specified precision
    
    raise NotImplementedError("get_dq_map not implemented")

def get_centered_sampling(map_shape: Tuple[int, int, int], 
                         sampling: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    """
    Get sampling tuples for map centered about the origin.
    
    Args:
        map_shape: Tuple with map dimensions
        sampling: Tuple with fractional sampling rates
        
    Returns:
        List of sampling tuples for h, k, l dimensions
        
    References:
        - Original implementation: eryx/map_utils.py:get_centered_sampling
    """
    # TODO: Calculate extent for each dimension
    # TODO: Create tuples for min, max, sampling rate
    # Note: This function may not need PyTorch as it's just calculating parameters
    
    raise NotImplementedError("get_centered_sampling not implemented")

def resize_map(new_map: torch.Tensor, 
              old_sampling: List[Tuple[float, float, float]], 
              new_sampling: List[Tuple[float, float, float]]) -> torch.Tensor:
    """
    Resize map if symmetrization changed dimensions.
    
    Args:
        new_map: PyTorch tensor of shape (dim_h, dim_k, dim_l) with map data
        old_sampling: List of (min, max, rate) tuples for original grid
        new_sampling: List of (min, max, rate) tuples for new map
        
    Returns:
        PyTorch tensor with resized map
        
    References:
        - Original implementation: eryx/map_utils.py:resize_map
    """
    # TODO: Check sampling differences with small tolerance
    # TODO: Calculate cropping dimensions
    # TODO: Apply cropping with PyTorch slicing operations
    
    raise NotImplementedError("resize_map not implemented")

Create Base Module Stub

aiderCopyCREATE eryx/base_torch.py
"""
PyTorch implementation of transform calculations for diffuse scattering.

This module contains PyTorch versions of the transform calculation functions defined
in eryx/base.py. All implementations maintain the same API as the NumPy versions
but use PyTorch tensors and operations to enable gradient flow.

References:
    - Original NumPy implementation in eryx/base.py
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union, Any

def compute_molecular_transform(pdb_path: str, 
                              hsampling: Tuple[float, float, float], 
                              ksampling: Tuple[float, float, float], 
                              lsampling: Tuple[float, float, float], 
                              U: Optional[torch.Tensor] = None, 
                              expand_p1: bool = True,
                              expand_friedel: bool = True, 
                              res_limit: float = 0, 
                              batch_size: int = 10000, 
                              n_processes: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the molecular transform as the incoherent sum of asymmetric units using PyTorch.
    
    Args:
        pdb_path: Path to coordinates file
        hsampling: Tuple (hmin, hmax, oversampling) for h dimension
        ksampling: Tuple (kmin, kmax, oversampling) for k dimension
        lsampling: Tuple (lmin, lmax, oversampling) for l dimension
        U: Optional tensor with isotropic displacement parameters
        expand_p1: If True, expand PDB to unit cell
        expand_friedel: If True, expand to full sphere in reciprocal space
        res_limit: High resolution limit in Angstrom
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points, 3) with q-vectors
            - PyTorch tensor of shape (dim_h, dim_k, dim_l) with intensity map
            
    References:
        - Original implementation: eryx/base.py:compute_molecular_transform
    """
    # TODO: Load model with adapter to convert to PyTorch tensors
    # TODO: Generate grid using map_utils_torch.generate_grid
    # TODO: Apply resolution mask
    # TODO: Choose calculation approach based on space group and expand_friedel
    # TODO: Call appropriate summation function
    
    raise NotImplementedError("compute_molecular_transform not implemented")

def compute_crystal_transform(pdb_path: str, 
                             hsampling: Tuple[float, float, float], 
                             ksampling: Tuple[float, float, float], 
                             lsampling: Tuple[float, float, float], 
                             U: Optional[torch.Tensor] = None, 
                             expand_p1: bool = True, 
                             res_limit: float = 0, 
                             batch_size: int = 5000, 
                             n_processes: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute crystal transform as coherent sum of asymmetric units using PyTorch.
    
    Args:
        pdb_path: Path to coordinates file
        hsampling: Tuple (hmin, hmax, oversampling) for h dimension
        ksampling: Tuple (kmin, kmax, oversampling) for k dimension
        lsampling: Tuple (lmin, lmax, oversampling) for l dimension
        U: Optional tensor with isotropic displacement parameters
        expand_p1: If True, expand PDB to unit cell
        res_limit: High resolution limit in Angstrom
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        Tuple containing:
            - PyTorch tensor of shape (n_points, 3) with q-vectors
            - PyTorch tensor of shape (dim_h, dim_k, dim_l) with intensity map
            
    References:
        - Original implementation: eryx/base.py:compute_crystal_transform
    """
    # TODO: Load model with adapter
    # TODO: Generate grid and q-vectors
    # TODO: Apply resolution and dq masks
    # TODO: Compute structure factors for Bragg reflections
    # TODO: Reshape results to 3D map
    
    raise NotImplementedError("compute_crystal_transform not implemented")

def incoherent_sum_real(model: Any, 
                       hkl_grid: torch.Tensor, 
                       sampling: Tuple[float, float, float], 
                       U: Optional[torch.Tensor] = None, 
                       mask: Optional[torch.Tensor] = None, 
                       batch_size: int = 10000, 
                       n_processes: int = 8) -> torch.Tensor:
    """
    Compute incoherent sum of scattering from all ASUs in real space using PyTorch.
    
    Args:
        model: AtomicModel instance (converted to use PyTorch)
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        sampling: Tuple with sampling rates
        U: Optional tensor with displacement parameters
        mask: Optional tensor with boolean mask
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        PyTorch tensor of shape (dim_h, dim_k, dim_l) with intensity map
        
    References:
        - Original implementation: eryx/base.py:incoherent_sum_real
    """
    # TODO: Generate ASU mask and combine with resolution mask
    # TODO: Compute scattering for unique reciprocal wedge
    # TODO: Expand symmetry operations and hkl indices
    # TODO: Get raveled indices
    # TODO: Perform symmetrization using PyTorch operations
    # TODO: Account for multiplicity
    # TODO: Resize map if needed
    
    raise NotImplementedError("incoherent_sum_real not implemented")

def incoherent_sum_reciprocal(model: Any, 
                            hkl_grid: torch.Tensor, 
                            sampling: Tuple[float, float, float], 
                            U: Optional[torch.Tensor] = None, 
                            batch_size: int = 10000, 
                            n_processes: int = 8) -> torch.Tensor:
    """
    Compute incoherent sum of scattering in reciprocal space using PyTorch.
    
    Args:
        model: AtomicModel instance (converted to use PyTorch)
        hkl_grid: PyTorch tensor of shape (n_points, 3) with hkl indices
        sampling: Tuple with sampling rates
        U: Optional tensor with displacement parameters
        batch_size: Number of q-vectors per batch
        n_processes: Number of processes (ignored in PyTorch implementation)
        
    Returns:
        PyTorch tensor of shape (n_points,) with intensity values
        
    References:
        - Original implementation: eryx/base.py:incoherent_sum_reciprocal
    """
    # TODO: Get symmetry equivalents of hkl indices
    # TODO: Compute raveled indices
    # TODO: Initialize tensors for symmetry-equivalent structure factors
    # TODO: Compute structure factors for the first ASU
    # TODO: Map to other ASUs and compute additional structure factors
    # TODO: Sum intensities over all symmetry-equivalents
    
    raise NotImplementedError("incoherent_sum_reciprocal not implemented")

Create Adapter Components

aiderCopyCREATE eryx/adapters.py
"""
Adapter components to bridge NumPy and PyTorch implementations.

This module contains adapter classes to convert between NumPy arrays and PyTorch tensors,
as well as domain-specific adapters for the diffuse scattering calculations.
All adapters preserve the computational graph for gradient backpropagation.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union, Any
import gemmi  # For crystallographic data structures

class PDBToTensor:
    """
    Adapter to convert PDB data from NumPy arrays to PyTorch tensors.
    
    This class handles the conversion of AtomicModel and related classes from
    the NumPy implementation to PyTorch tensors suitable for gradient-based calculations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the adapter.
        
        Args:
            device: The PyTorch device to place tensors on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_atomic_model(self, model: Any) -> Dict[str, Any]:
        """
        Convert an AtomicModel to PyTorch tensors.
        
        Args:
            model: AtomicModel instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the model attributes
        """
        # TODO: Convert all NumPy arrays to PyTorch tensors
        # TODO: Preserve crystallographic information
        # TODO: Handle complex attributes like symmetry operations
        # TODO: Return a dictionary with all the converted tensors
        
        raise NotImplementedError("convert_atomic_model not implemented")
    
    def convert_crystal(self, crystal: Any) -> Dict[str, Any]:
        """
        Convert a Crystal object to PyTorch tensors.
        
        Args:
            crystal: Crystal instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the crystal attributes
        """
        # TODO: Convert relevant attributes to PyTorch tensors
        # TODO: Handle unit cell information
        # TODO: Preserve crystallographic metadata
        
        raise NotImplementedError("convert_crystal not implemented")
    
    def convert_gnm(self, gnm: Any) -> Dict[str, Any]:
        """
        Convert a GaussianNetworkModel to PyTorch tensors.
        
        Args:
            gnm: GaussianNetworkModel instance from eryx.pdb
            
        Returns:
            Dictionary containing PyTorch tensor versions of the GNM attributes
        """
        # TODO: Convert gamma matrix to tensor
        # TODO: Convert neighbor lists to tensor format
        # TODO: Preserve parameter information
        
        raise NotImplementedError("convert_gnm not implemented")
    
    def array_to_tensor(self, array: np.ndarray, requires_grad: bool = True) -> torch.Tensor:
        """
        Convert a NumPy array to a PyTorch tensor.
        
        Args:
            array: NumPy array to convert
            requires_grad: Whether the tensor requires gradients
            
        Returns:
            PyTorch tensor with the same data
        """
        if array is None:
            return None
            
        tensor = torch.from_numpy(array).to(self.device)
        tensor.requires_grad = requires_grad
        return tensor
    
    def convert_dict_of_arrays(self, dict_arrays: Dict[Any, np.ndarray], 
                              requires_grad: bool = True) -> Dict[Any, torch.Tensor]:
        """
        Convert a dictionary of NumPy arrays to PyTorch tensors.
        
        Args:
            dict_arrays: Dictionary mapping keys to NumPy arrays
            requires_grad: Whether tensors require gradients
            
        Returns:
            Dictionary mapping the same keys to PyTorch tensors
        """
        return {k: self.array_to_tensor(v, requires_grad) for k, v in dict_arrays.items()}


class GridToTensor:
    """
    Adapter to convert grid data from NumPy arrays to PyTorch tensors.
    
    This class handles the conversion of reciprocal space grids and related data
    from the NumPy implementation to PyTorch tensors.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the adapter.
        
        Args:
            device: The PyTorch device to place tensors on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_grid(self, q_grid: np.ndarray, map_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Convert a grid of q-vectors to PyTorch tensor.
        
        Args:
            q_grid: NumPy array of shape (n_points, 3) with q-vectors
            map_shape: Tuple with 3D map shape
            
        Returns:
            Tuple containing:
                - PyTorch tensor of q-vectors
                - Tuple with map shape (unchanged)
        """
        # TODO: Convert q_grid to PyTorch tensor
        # TODO: Set requires_grad to True
        # TODO: Return tensor and shape
        
        raise NotImplementedError("convert_grid not implemented")
    
    def convert_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        Convert a boolean mask to PyTorch tensor.
        
        Args:
            mask: NumPy boolean array
            
        Returns:
            PyTorch boolean tensor
        """
        # TODO: Convert mask to PyTorch tensor
        # TODO: Ensure boolean dtype
        
        raise NotImplementedError("convert_mask not implemented")
    
    def convert_symmetry_ops(self, sym_ops: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        """
        Convert symmetry operations to PyTorch tensors.
        
        Args:
            sym_ops: Dictionary mapping IDs to rotation matrices
            
        Returns:
            Dictionary mapping IDs to tensor rotation matrices
        """
        # TODO: Convert each symmetry operation matrix to tensor
        # TODO: Maintain dictionary structure
        
        raise NotImplementedError("convert_symmetry_ops not implemented")


class TensorToNumpy:
    """
    Adapter to convert PyTorch tensors back to NumPy arrays.
    
    This class handles the conversion of PyTorch tensors to NumPy arrays
    for visualization, saving, or compatibility with existing code.
    """
    
    def __init__(self):
        """
        Initialize the adapter.
        """
        pass
    
    def tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy array.
        
        Args:
            tensor: PyTorch tensor to convert
            
        Returns:
            NumPy array with the same data
        """
        if tensor is None:
            return None
            
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        return tensor.cpu().numpy()
    
    def convert_dict_of_tensors(self, dict_tensors: Dict[Any, torch.Tensor]) -> Dict[Any, np.ndarray]:
        """
        Convert a dictionary of PyTorch tensors to NumPy arrays.
        
        Args:
            dict_tensors: Dictionary mapping keys to PyTorch tensors
            
        Returns:
            Dictionary mapping the same keys to NumPy arrays
        """
        return {k: self.tensor_to_array(v) for k, v in dict_tensors.items()}
    
    def convert_intensity_map(self, intensity: torch.Tensor, map_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert an intensity map tensor to a NumPy array.
        
        Args:
            intensity: PyTorch tensor with intensity values
            map_shape: Tuple with desired 3D shape
            
        Returns:
            NumPy array with intensity map reshaped to 3D
        """
        # TODO: Detach tensor if it requires gradients
        # TODO: Convert to CPU NumPy array
        # TODO: Reshape to 3D if necessary
        
        raise NotImplementedError("convert_intensity_map not implemented")


class ModelAdapters:
    """
    Adapters for the various model classes in eryx.
    
    This class contains methods to convert between the NumPy and PyTorch
    versions of the various model classes used in diffuse scattering calculations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the adapters.
        
        Args:
            device: The PyTorch device to place tensors on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pdb_to_tensor = PDBToTensor(device)
        self.grid_to_tensor = GridToTensor(device)
        self.tensor_to_numpy = TensorToNumpy()
    
    def adapt_one_phonon_inputs(self, np_model: Any) -> Dict[str, Any]:
        """
        Adapt inputs for the OnePhonon model from NumPy to PyTorch.
        
        Args:
            np_model: OnePhonon instance from eryx.models
            
        Returns:
            Dictionary with PyTorch tensor versions of inputs
        """
        # TODO: Extract necessary inputs
        # TODO: Convert to PyTorch tensors
        # TODO: Structure for easy passing to PyTorch implementation
        
        raise NotImplementedError("adapt_one_phonon_inputs not implemented")
    
    def adapt_one_phonon_outputs(self, torch_outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Adapt outputs from the PyTorch OnePhonon model back to NumPy.
        
        Args:
            torch_outputs: Dictionary with PyTorch tensor outputs
            
        Returns:
            NumPy array with intensity map
        """
        # TODO: Extract intensity map
        # TODO: Convert to NumPy array
        # TODO: Reshape if necessary
        
        raise NotImplementedError("adapt_one_phonon_outputs not implemented")
    
    def adapt_rigid_body_translations_inputs(self, np_model: Any) -> Dict[str, Any]:
        """
        Adapt inputs for the RigidBodyTranslations model from NumPy to PyTorch.
        
        Args:
            np_model: RigidBodyTranslations instance from eryx.models
            
        Returns:
            Dictionary with PyTorch tensor versions of inputs
        """
        # TODO: Extract necessary inputs
        # TODO: Convert to PyTorch tensors
        # TODO: Structure for easy passing to PyTorch implementation
        
        raise NotImplementedError("adapt_rigid_body_translations_inputs not implemented")
    
    # Add similar methods for other model classes

Create PyTorch Utilities

aiderCopyCREATE eryx/torch_utils.py
"""
PyTorch-specific utilities for diffuse scattering calculations.

This module contains PyTorch-specific utilities and helper functions for
diffuse scattering calculations, including complex number operations,
differentiable eigendecomposition, and other tensor operations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union, Any

class ComplexTensorOps:
    """
    Operations for complex tensors.
    
    PyTorch's complex tensor support is still evolving, so this class provides
    helper methods for complex tensor operations to ensure differentiability.
    """
    
    @staticmethod
    def complex_exp(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute complex exponential e^(i*phase).
        
        Args:
            phase: PyTorch tensor with phase values
            
        Returns:
            Tuple of (real, imaginary) parts
        """
        return torch.cos(phase), torch.sin(phase)
    
    @staticmethod
    def complex_mul(a_real: torch.Tensor, a_imag: torch.Tensor, 
                   b_real: torch.Tensor, b_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiply complex numbers in rectangular form.
        
        Args:
            a_real: Real part of first operand
            a_imag: Imaginary part of first operand
            b_real: Real part of second operand
            b_imag: Imaginary part of second operand
            
        Returns:
            Tuple of (real, imaginary) parts of the product
        """
        real = a_real * b_real - a_imag * b_imag
        imag = a_real * b_imag + a_imag * b_real
        return real, imag
    
    @staticmethod
    def complex_abs_squared(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """
        Compute squared magnitude of complex numbers.
        
        Args:
            real: Real part
            imag: Imaginary part
            
        Returns:
            Squared magnitude |z|^2
        """
        return real**2 + imag**2
    
    @staticmethod
    def complex_exp_dwf(q_vec: torch.Tensor, u_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute complex exponential for Debye-Waller factor.
        
        Args:
            q_vec: Q-vector tensor
            u_vec: Displacement tensor
            
        Returns:
            Tuple of (real, imaginary) parts of e^(-0.5*qUq)
        """
        qUq = torch.sum(q_vec * u_vec * q_vec, dim=-1)
        dwf = torch.exp(-0.5 * qUq)
        return dwf, torch.zeros_like(dwf)


class FFTOps:
    """
    FFT operations for diffuse scattering calculations.
    
    This class provides FFT operations specifically tailored for diffuse
    scattering calculations, ensuring differentiability and proper handling
    of complex numbers.
    """
    
    @staticmethod
    def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Convolve signal with kernel using FFT.
        
        Args:
            signal: Input signal tensor
            kernel: Convolution kernel tensor
            
        Returns:
            Convolved signal tensor
        """
        # TODO: Normalize kernel
        # TODO: Compute FFTs
        # TODO: Multiply in frequency domain
        # TODO: Compute inverse FFT
        # TODO: Return real part
        
        raise NotImplementedError("fft_convolve not implemented")
    
    @staticmethod
    def fft_3d(input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D FFT of input tensor.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Output tensor with FFT result
        """
        # TODO: Use torch.fft.fftn with appropriate normalization
        # TODO: Handle complex numbers properly
        
        raise NotImplementedError("fft_3d not implemented")
    
    @staticmethod
    def ifft_3d(input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D inverse FFT of input tensor.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Output tensor with IFFT result
        """
        # TODO: Use torch.fft.ifftn with appropriate normalization
        # TODO: Handle complex numbers properly
        
        raise NotImplementedError("ifft_3d not implemented")


class EigenOps:
    """
    Differentiable eigendecomposition operations.
    
    This class provides differentiable implementations of eigenvalue
    decomposition and related operations for the diffuse scattering calculations.
    """
    
    @staticmethod
    def svd_decomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SVD decomposition with gradient support.
        
        Args:
            matrix: Input matrix tensor
            
        Returns:
            Tuple of (U, S, V) tensors
        """
        # TODO: Use torch.linalg.svd
        # TODO: Ensure proper gradient flow
        
        raise NotImplementedError("svd_decomposition not implemented")
    
    @staticmethod
    def eigen_decomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition with gradient support.
        
        Args:
            matrix: Input matrix tensor
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) tensors
        """
        # TODO: Use appropriate PyTorch function
        # TODO: Handle complex matrices if necessary
        # TODO: Ensure proper gradient flow
        
        raise NotImplementedError("eigen_decomposition not implemented")
    
    @staticmethod
    def solve_linear_system(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve linear system Ax = b with gradient support.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        # TODO: Use torch.linalg.solve
        # TODO: Handle edge cases (singular matrices)
        # TODO: Ensure proper gradient flow
        
        raise NotImplementedError("solve_linear_system not implemented")


class GradientUtils:
    """
    Utilities for gradient computation and manipulation.
    
    This class provides utilities for computing and manipulating gradients
    in the diffuse scattering calculations.
    """
    
    @staticmethod
    def finite_differences(func: callable, input_tensor: torch.Tensor, 
                          eps: float = 1e-6) -> torch.Tensor:
        """
        Compute gradients using finite differences for validation.
        
        Args:
            func: Function to differentiate
            input_tensor: Input tensor
            eps: Step size for finite differences
            
        Returns:
            Gradient tensor
        """
        # TODO: Implement central difference scheme
        # TODO: Handle multidimensional inputs
        
        raise NotImplementedError("finite_differences not implemented")
    
    @staticmethod
    def validate_gradients(analytical_grad: torch.Tensor, 
                          numerical_grad: torch.Tensor, 
                          rtol: float = 1e-4, 
                          atol: float = 1e-6) -> bool:
        """
        Validate analytical gradients against numerical gradients.
        
        Args:
            analytical_grad: Analytically computed gradients
            numerical_grad: Numerically computed gradients
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if gradients match within tolerance
        """
        # TODO: Compute element-wise relative and absolute errors
        # TODO: Check if errors are within tolerance
        
        raise NotImplementedError("validate_gradients not implemented")
    
    @staticmethod
    def gradient_norm(gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 norm of gradient.
        
        Args:
            gradient: Gradient tensor
            
        Returns:
            Gradient norm
        """
        return torch.norm(gradient, p=2)

Create Run Script with PyTorch Support

aiderCopyCREATE eryx/run_torch.py
#!/usr/bin/env python3
"""
PyTorch implementation of diffuse scattering simulation.

This script provides a PyTorch-based implementation of the diffuse scattering
simulation, parallel to the NumPy implementation in run_debug.py. It enables
gradient-based optimization of simulation parameters.
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    filename="torch_debug_output.log",
    filemode="w"
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logging.getLogger("").addHandler(console)

def setup_logging():
    """
    Set up logging configuration.
    """
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename="torch_debug_output.log",
        filemode="w"
    )
    # Also output to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def run_torch(device: Optional[torch.device] = None):
    """
    Run PyTorch version of the diffuse scattering simulation.
    
    Args:
        device: PyTorch device to use (default: CUDA if available, else CPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"Starting PyTorch branch computation on {device}")
    
    # Same parameters as NumPy version
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # TODO: Import the PyTorch OnePhonon implementation
    # TODO: Create OnePhonon instance with parameters
    # TODO: Apply disorder model
    # TODO: Extract and save results
    # TODO: Log debug information
    
    # Placeholder for actual implementation
    logging.info("PyTorch implementation not implemented yet")
    
    # When implemented, save results for comparison
    # torch.save(Id_torch, "torch_diffuse_intensity.pt")
    # np.save("torch_diffuse_intensity.npy", Id_torch.detach().cpu().numpy())

def run_np():
    """
    Run NumPy version of the diffuse scattering simulation.
    
    This is the same function as in run_debug.py, included for direct comparison.
    """
    # Use a small grid for testing; adjust parameters as necessary.
    logging.info("Starting NP branch computation")
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Import original NumPy implementation
    from eryx.models import OnePhonon
    
    onephonon_np = OnePhonon(
        pdb_path,
        [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0
    )
    Id_np = onephonon_np.apply_disorder(use_data_adp=True)
    logging.debug(f"NP: hkl_grid shape = {onephonon_np.hkl_grid.shape}")
    logging.debug("NP: hkl_grid coordinate ranges:")
    logging.debug(f"  Dimension 0: min = {onephonon_np.hkl_grid[:,0].min()}, max = {onephonon_np.hkl_grid[:,0].max()}")
    logging.debug(f"  Dimension 1: min = {onephonon_np.hkl_grid[:,1].min()}, max = {onephonon_np.hkl_grid[:,1].max()}")
    logging.debug(f"  Dimension 2: min = {onephonon_np.hkl_grid[:,2].min()}, max = {onephonon_np.hkl_grid[:,2].max()}")
    logging.debug(f"NP: q_grid range: min = {onephonon_np.q_grid.min()}, max = {onephonon_np.q_grid.max()}")
    logging.info("NP branch diffuse intensity stats: min=%s, max=%s", np.nanmin(Id_np), np.nanmax(Id_np))
    # Save for later comparison
    np.save("np_diffuse_intensity.npy", Id_np)

def compare_results():
    """
    Compare the results from NumPy and PyTorch implementations.
    """
    # TODO: Load NumPy and PyTorch results
    # TODO: Compute statistics (MSE, correlation)
    # TODO: Log comparison results
    
    raise NotImplementedError("compare_results not implemented")

if __name__ == "__main__":
    setup_logging()
    
    # After setting up and importing everything, call the run routines.
    run_np()
    run_torch()
    logging.info("Completed debug run. Please check debug_output.log, np_diffuse_intensity.npy and torch_diffuse_intensity.npy")

Create Test Utilities for PyTorch Tests

aiderCopyCREATE eryx/autotest/torch_testing.py
"""
Testing utilities for PyTorch implementation.

This module extends the autotest framework with PyTorch-specific testing
capabilities, including tensor comparison, gradient checking, and
PyTorch-NumPy conversion for testing.
"""

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .testing import Testing
from .logger import Logger
from .functionmapping import FunctionMapping

class TorchTesting(Testing):
    """
    Extended testing framework for PyTorch implementations.
    
    This class extends the Testing class with PyTorch-specific utilities,
    including tensor comparison and gradient checking.
    """
    
    def __init__(self, logger: Logger, function_mapping: FunctionMapping, 
                rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize the PyTorch testing framework.
        
        Args:
            logger: Logger instance for test logging
            function_mapping: FunctionMapping instance for function lookup
            rtol: Relative tolerance for tensor comparison
            atol: Absolute tolerance for tensor comparison
        """
        super().__init__(logger, function_mapping)
        self.rtol = rtol
        self.atol = atol
    
    def testTorchCallable(self, log_path_prefix: str, torch_func: Callable) -> bool:
        """
        Test a PyTorch function against NumPy implementation.
        
        Args:
            log_path_prefix: Path prefix for log files
            torch_func: PyTorch function to test
            
        Returns:
            True if test passes, False otherwise
        """
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            for i in range(len(logs) // 2):
                args = logs[2 * i]['args']
                kwargs = logs[2 * i]['kwargs']
                expected_output = logs[2 * i + 1]['result']
                try:
                    # Deserialize and convert to PyTorch
                    deserialized_args = self._numpy_to_torch(self.logger.serializer.deserialize(args))
                    deserialized_kwargs = self._numpy_to_torch(self.logger.serializer.deserialize(kwargs))
                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
                    
                    # Run PyTorch function
                    actual_output = torch_func(*deserialized_args, **deserialized_kwargs)
                    
                    # Convert PyTorch output to NumPy for comparison
                    numpy_actual_output = self._torch_to_numpy(actual_output)
                    
                    # Compare outputs
                    if not self._compare_outputs(numpy_actual_output, deserialized_expected_output):
                        print(f"Test failed for {log_file}")
                        return False
                except Exception as e:
                    print(f"Error testing PyTorch function: {e}")
                    return False
        return True
    
    def _numpy_to_torch(self, obj: Any) -> Any:
        """
        Convert NumPy arrays to PyTorch tensors recursively.
        
        Args:
            obj: Input object potentially containing NumPy arrays
            
        Returns:
            Object with NumPy arrays converted to PyTorch tensors
        """
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj.copy())
        elif isinstance(obj, list):
            return [self._numpy_to_torch(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_torch(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._numpy_to_torch(v) for k, v in obj.items()}
        else:
            return obj
    
    def _torch_to_numpy(self, obj: Any) -> Any:
        """
        Convert PyTorch tensors to NumPy arrays recursively.
        
        Args:
            obj: Input object potentially containing PyTorch tensors
            
        Returns:
            Object with PyTorch tensors converted to NumPy arrays
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, list):
            return [self._torch_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._torch_to_numpy(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._torch_to_numpy(v) for k, v in obj.items()}
        else:
            return obj
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """
        Compare outputs with tolerance for numerical differences.
        
        Args:
            actual: Actual output
            expected: Expected output
            
        Returns:
            True if outputs match within tolerance, False otherwise
        """
        if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            # Compare arrays with tolerance
            if actual.shape != expected.shape:
                print(f"Shape mismatch: {actual.shape} vs {expected.shape}")
                return False
            
            # Handle NaN values
            nan_mask_actual = np.isnan(actual)
            nan_mask_expected = np.isnan(expected)
            if not np.array_equal(nan_mask_actual, nan_mask_expected):
                print("NaN pattern mismatch")
                return False
            
            # Compare non-NaN values
            non_nan_mask = ~nan_mask_actual
            if np.any(non_nan_mask):
                return np.allclose(actual[non_nan_mask], expected[non_nan_mask], 
                                 rtol=self.rtol, atol=self.atol)
            return True
        
        elif isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                print(f"Length mismatch: {len(actual)} vs {len(expected)}")
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, tuple) and isinstance(expected, tuple):
            if len(actual) != len(expected):
                print(f"Length mismatch: {len(actual)} vs {len(expected)}")
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                print(f"Key mismatch: {set(actual.keys())} vs {set(expected.keys())}")
                return False
            return all(self._compare_outputs(actual[k], expected[k]) for k in actual)
        
        else:
            # Direct comparison for other types
            return actual == expected
    
    def check_gradients(self, torch_func: Callable, inputs: List[torch.Tensor], 
                       eps: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
        """
        Check gradients of PyTorch function using finite differences.
        
        Args:
            torch_func: PyTorch function to check
            inputs: List of input tensors
            eps: Step size for finite differences
            
        Returns:
            Tuple containing:
                - True if gradients match, False otherwise
                - Dictionary with gradient statistics
        """
        # TODO: Clone inputs and set requires_grad=True
        # TODO: Compute forward pass and backward pass
        # TODO: Compute numerical gradients with finite differences
        # TODO: Compare analytical and numerical gradients
        # TODO: Return results and statistics
        
        raise NotImplementedError("check_gradients not implemented")
    
    def create_tensor_test_case(self, log_path_prefix: str, 
                              torch_func: Callable, numpy_func: Callable) -> Callable:
        """
        Create a test case for comparing PyTorch and NumPy implementations.
        
        Args:
            log_path_prefix: Path prefix for log files
            torch_func: PyTorch function to test
            numpy_func: NumPy function to compare against
            
        Returns:
            Test function that can be called directly
        """
        def test_case():
            # Create test data
            # TODO: Generate test inputs
            # TODO: Run both functions
            # TODO: Compare outputs
            # TODO: Return True if outputs match, False otherwise
            
            raise NotImplementedError("test_case not implemented")
        
        return test_case

Create an Example PyTorch Test

aiderCopyCREATE tests/test_scatter_torch.py
"""
Tests for the PyTorch implementation of structure factor calculations.

This module contains tests for the PyTorch implementation of structure factor
calculations in eryx/scatter_torch.py, comparing results with the original NumPy
implementation in eryx/scatter.py.
"""

import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from eryx.scatter import compute_form_factors as np_compute_form_factors
from eryx.scatter import structure_factors_batch as np_structure_factors_batch
from eryx.scatter import structure_factors as np_structure_factors
from eryx.scatter_torch import compute_form_factors as torch_compute_form_factors
from eryx.scatter_torch import structure_factors_batch as torch_structure_factors_batch
from eryx.scatter_torch import structure_factors as torch_structure_factors

# Set up testing framework
logger = Logger()
function_mapping = FunctionMapping(log_directory="np_ground_truth")
torch_testing = TorchTesting(logger, function_mapping)

def instrument_numpy_functions():
    """
    Instrument NumPy functions to capture inputs and outputs for testing.
    """
    from eryx.autotest.debug import Debug
    debug = Debug().decorate
    
    # Apply debug decorator to functions
    global np_compute_form_factors, np_structure_factors_batch, np_structure_factors
    np_compute_form_factors = debug(np_compute_form_factors)
    np_structure_factors_batch = debug(np_structure_factors_batch)
    np_structure_factors = debug(np_structure_factors)

def generate_test_data():
    """
    Generate test data by running NumPy implementations.
    """
    # TODO: Generate random test data or load from files
    # TODO: Run NumPy implementations to capture inputs/outputs
    
    pass

def test_compute_form_factors():
    """
    Test compute_form_factors PyTorch implementation against NumPy version.
    """
    # Skip the actual test as the implementation is not ready yet
    # When implementation is ready, uncomment:
    # assert torch_testing.testTorchCallable("np_ground_truth/eryx.scatter.compute_form_factors", 
    #                                      torch_compute_form_factors)
    pass

def test_structure_factors_batch():
    """
    Test structure_factors_batch PyTorch implementation against NumPy version.
    """
    # Skip the actual test as the implementation is not ready yet
    # When implementation is ready, uncomment:
    # assert torch_testing.testTorchCallable("np_ground_truth/eryx.scatter.structure_factors_batch", 
    #                                      torch_structure_factors_batch)
    pass

def test_structure_factors():
    """
    Test structure_factors PyTorch implementation against NumPy version.
    """
    # Skip the actual test as the implementation is not ready yet
    # When implementation is ready, uncomment:
    # assert torch_testing.testTorchCallable("np_ground_truth/eryx.scatter.structure_factors", 
    #                                      torch_structure_factors)
    pass

def test_structure_factors_gradients():
    """
    Test gradient computation for structure_factors PyTorch implementation.
    """
    # TODO: When implementation is ready, create test inputs
    # TODO: Compute gradients and validate with finite differences
    
    pass

if __name__ == "__main__":
    # When ready to run the tests:
    # instrument_numpy_functions()
    # generate_test_data()
    # test_compute_form_factors()
    # test_structure_factors_batch()
    # test_structure_factors()
    # test_structure_factors_gradients()
    print("Tests disabled until implementations are ready")

Set up API configuration in init.py

aiderCopyUPDATE eryx/__init__.py
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
    
except ImportError:
    HAS_TORCH = False

__version__ = "0.1.0"
These are the comprehensive stub implementations for the PyTorch port of the diffuse scattering simulation. The stubs include detailed docstrings, type hints, TODOs, and references to the original NumPy implementations.
CopyRetryClaude does not have the ability to run the code it generates yet.FBcontinueEditMapped out PyTorch implementation strategy with comprehensive planning.3sI need to continue with the spec prompt. Let me summarize what I've done so far:

I've created the file structure for the core computational parts of the PyTorch implementation
I've added detailed stubs with TODOs, docstrings, and references to the original NumPy implementations
I've created adapter components to bridge between NumPy and PyTorch
I've set up utilities for testing with the autotest framework

Now I need to complete the spec prompt with:

Any additional directories or files that might be needed
Any specific implementation guidance that should be included
Clear instructions for what each of the tasks actually does

I should also emphasize the use of autotest for generating ground truth data.Here's the completion of the spec prompt:
Testing Setup
Using Autotest for Ground Truth
To effectively validate the PyTorch implementations against the NumPy originals, we'll use the autotest framework to:

Capture Ground Truth Data

Instrument NumPy functions to log inputs and outputs
Store serialized data for each function call
Use this data to validate PyTorch implementations


Ground Truth Generation Process
pythonCopy# Example script to generate ground truth data
from eryx.autotest.debug import Debug
from eryx.models import OnePhonon
from eryx.scatter import structure_factors

# Create debug decorator
debug = Debug().decorate

# Instrument functions
structure_factors = debug(structure_factors)

# Run simulation with instrumented functions
model = OnePhonon("test.pdb", (-4,4,3), (-4,4,3), (-4,4,3))
result = model.apply_disorder()

# Ground truth is now captured in log files


Test Directory Structure
Ensure test files have proper organization:
Copytests/
├── fixtures/                    # Test data
│   ├── pdb_samples/             # Small PDB files for testing
│   └── expected_outputs/        # Pre-computed expected results
├── utils/                       # Test utilities
│   ├── tensor_utils.py          # Tensor comparison utilities
│   └── gradient_checker.py      # Gradient validation utilities
├── unit/                        # Unit tests
│   ├── test_scatter_torch.py    # Tests for structure factors
│   ├── test_models_torch.py     # Tests for disorder models
│   └── ...                      # Other component tests
└── integration/                 # Integration tests
    ├── test_full_simulation.py  # End-to-end test
    └── test_gradients.py        # Gradient flow tests
Additional Requirements

Type Consistency

All PyTorch implementations must maintain the same function signatures as NumPy versions
Return types should be torch.Tensor where NumPy returns np.ndarray


Device Management

All implementations should support explicit device placement
Default to CUDA if available, CPU otherwise
Support moving tensors between devices


Gradient Validation

Each differentiable function should have gradient tests
Use finite differences to validate analytical gradients


Performance Considerations

Implement batching where possible
Minimize unnecessary tensor copying
Use in-place operations where appropriate



Final Deliverables Checklist

Stub Files

Complete _torch.py files with detailed stubs
Comprehensive TODOs in all functions
Type hints and docstrings that match NumPy versions


Adapter Components

NumPy-to-PyTorch conversion utilities
Domain-specific adapters for model classes


Test Scaffolding

Unit test stubs for all components
Test data generation utilities
PyTorch-specific test utilities


Documentation

Module-level docstrings explaining PyTorch adaptations
Function-level docstrings referencing NumPy equivalents
Implementation notes on differentiability



This detailed spec provides a comprehensive foundation for implementing a PyTorch-based differentiable simulation of diffuse scattering calculations, enabling gradient-based optimization of model parameters while maintaining compatibility with the existing NumPy implementation.

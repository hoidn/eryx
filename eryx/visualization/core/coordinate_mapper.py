"""
Coordinate mapping utilities for visualization.

This module handles transformations between different coordinate systems:
- Miller indices (h,k,l) in reciprocal space
- Q-vectors in Cartesian reciprocal space  
- Array indices in 3D intensity grids
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

# Try to import the existing map_utils functions
try:
    from eryx.map_utils import generate_grid
    HAS_MAP_UTILS = True
except ImportError:
    HAS_MAP_UTILS = False
    logging.warning("Could not import eryx.map_utils, will use internal implementation")

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """
    Handle transformations between coordinate systems.
    
    This class provides conversions between:
    - Miller indices (h,k,l) 
    - Q-vectors in reciprocal space
    - Array indices in 3D grids
    """
    
    def __init__(self, 
                 A_inv: np.ndarray,
                 shape: Optional[Tuple[int, int, int]] = None,
                 sampling: Optional[Tuple[Tuple[float, float, int], ...]] = None):
        """
        Initialize with orthogonalization matrix and grid parameters.
        
        Parameters
        ----------
        A_inv : np.ndarray, shape (3, 3)
            Fractional cell orthogonalization matrix (fractionalization matrix).
            Converts fractional coordinates to Cartesian coordinates.
        shape : tuple of int, optional
            Grid shape (hsteps, ksteps, lsteps)
        sampling : tuple of tuples, optional
            Sampling parameters: ((hmin, hmax, h_oversample),
                                 (kmin, kmax, k_oversample),  
                                 (lmin, lmax, l_oversample))
        """
        self.A_inv = np.asarray(A_inv)
        if self.A_inv.shape != (3, 3):
            raise ValueError(f"A_inv must be (3,3) matrix, got shape {self.A_inv.shape}")
        
        self.shape = shape
        self.sampling = sampling
        
        # Cache commonly used matrices
        self.A_inv_T = self.A_inv.T
        self.A_inv_T_inv = None  # Computed lazily when needed
        
        # Cache grid if parameters are provided
        self._hkl_grid_cache = None
        self._q_grid_cache = None
        
        if sampling is not None:
            self._validate_sampling(sampling)
            
    def _validate_sampling(self, sampling):
        """Validate sampling parameters."""
        if len(sampling) != 3:
            raise ValueError("Sampling must have 3 elements (h, k, l)")
        
        for i, samp in enumerate(sampling):
            if len(samp) != 3:
                raise ValueError(f"Sampling[{i}] must be (min, max, oversampling)")
            if samp[2] <= 0:
                raise ValueError(f"Oversampling rate must be positive, got {samp[2]}")
    
    def hkl_to_cartesian(self, hkl: np.ndarray) -> np.ndarray:
        """
        Convert Miller indices to Cartesian q-vectors.
        
        Parameters
        ----------
        hkl : np.ndarray
            Miller indices, shape (..., 3) or (3,)
            
        Returns
        -------
        q : np.ndarray
            Q-vectors in reciprocal space, same shape as input
            Units are 2π/Å (reciprocal Angstroms)
        """
        hkl = np.asarray(hkl)
        
        # Handle both single vectors and arrays
        if hkl.ndim == 1:
            if hkl.shape[0] != 3:
                raise ValueError(f"HKL vector must have 3 components, got {hkl.shape[0]}")
            q = 2 * np.pi * np.dot(self.A_inv_T, hkl)
        else:
            # For arrays of vectors, use inner product
            q = 2 * np.pi * np.inner(self.A_inv_T, hkl).T
            
        logger.debug(f"Converted HKL shape {hkl.shape} to q shape {q.shape}")
        return q
    
    def cartesian_to_hkl(self, q: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian q-vectors to Miller indices.
        
        Parameters
        ----------
        q : np.ndarray
            Q-vectors in reciprocal space, shape (..., 3) or (3,)
            
        Returns
        -------
        hkl : np.ndarray
            Miller indices, same shape as input
        """
        q = np.asarray(q)
        
        # Compute inverse transformation matrix if not cached
        if self.A_inv_T_inv is None:
            self.A_inv_T_inv = np.linalg.inv(self.A_inv_T)
            logger.debug("Computed A_inv_T_inv for inverse transformation")
        
        # Scale by 1/(2π) and apply inverse transformation
        scaling_factor = 1.0 / (2.0 * np.pi)
        
        if q.ndim == 1:
            if q.shape[0] != 3:
                raise ValueError(f"Q vector must have 3 components, got {q.shape[0]}")
            hkl = np.dot(self.A_inv_T_inv, q * scaling_factor)
        else:
            hkl = np.matmul(self.A_inv_T_inv, (q * scaling_factor).T).T
            
        logger.debug(f"Converted q shape {q.shape} to HKL shape {hkl.shape}")
        return hkl
    
    def get_grid_coordinates(self, 
                           sampling: Optional[Tuple[Tuple[float, float, int], ...]] = None,
                           return_hkl: bool = False) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Generate full coordinate grid.
        
        Parameters
        ----------
        sampling : tuple of tuples, optional
            Overrides instance sampling if provided
        return_hkl : bool
            If True, return Miller indices; if False, return q-vectors
            
        Returns
        -------
        coordinates : np.ndarray, shape (n_points, 3)
            Grid coordinates (either hkl or q-vectors)
        map_shape : tuple
            Shape of the 3D grid (hsteps, ksteps, lsteps)
        """
        # Use provided sampling or fall back to instance sampling
        if sampling is None:
            sampling = self.sampling
        if sampling is None:
            raise ValueError("No sampling parameters provided")
        
        # Check if we can use cached values
        if sampling == self.sampling:
            if return_hkl and self._hkl_grid_cache is not None:
                return self._hkl_grid_cache, self.shape
            elif not return_hkl and self._q_grid_cache is not None:
                return self._q_grid_cache, self.shape
        
        # Use existing implementation if available, otherwise use internal
        if HAS_MAP_UTILS:
            logger.debug("Using eryx.map_utils.generate_grid")
            hsampling, ksampling, lsampling = sampling
            grid, map_shape = generate_grid(self.A_inv, hsampling, ksampling, 
                                           lsampling, return_hkl=return_hkl)
        else:
            logger.debug("Using internal grid generation")
            grid, map_shape = self._generate_grid_internal(sampling, return_hkl)
        
        # Cache if using instance sampling
        if sampling == self.sampling:
            self.shape = map_shape
            if return_hkl:
                self._hkl_grid_cache = grid
            else:
                self._q_grid_cache = grid
        
        return grid, map_shape
    
    def _generate_grid_internal(self, 
                               sampling: Tuple[Tuple[float, float, int], ...],
                               return_hkl: bool = False) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Internal implementation of grid generation.
        
        This matches the logic from eryx.map_utils.generate_grid.
        """
        hsampling, ksampling, lsampling = sampling
        
        # Calculate number of steps
        hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)
        ksteps = int(ksampling[2] * (ksampling[1] - ksampling[0]) + 1)
        lsteps = int(lsampling[2] * (lsampling[1] - lsampling[0]) + 1)
        
        logger.debug(f"Grid dimensions: h={hsteps}, k={ksteps}, l={lsteps}")
        
        # Create grid using mgrid with complex step for linspace behavior
        # Note: Order is (l, k, h) for memory efficiency
        hkl_grid = np.mgrid[lsampling[0]:lsampling[1]:lsteps*1j,
                           ksampling[0]:ksampling[1]:ksteps*1j,
                           hsampling[0]:hsampling[1]:hsteps*1j]
        
        # Get shape in (h, k, l) order by reversing
        map_shape = hkl_grid.shape[1:][::-1]  # (hsteps, ksteps, lsteps)
        
        # Reshape to (n_points, 3) and reorder from (l,k,h) to (h,k,l)
        hkl_grid = hkl_grid.T.reshape(-1, 3)
        hkl_grid = hkl_grid[:, [2, 1, 0]]  # Swap columns: [l,k,h] -> [h,k,l]
        
        if return_hkl:
            return hkl_grid, map_shape
        else:
            # Convert to q-vectors
            q_grid = self.hkl_to_cartesian(hkl_grid)
            return q_grid, map_shape
    
    def index_to_hkl(self, i: int, j: int, k: int,
                    sampling: Optional[Tuple[Tuple[float, float, int], ...]] = None) -> Tuple[float, float, float]:
        """
        Convert array indices to Miller indices.
        
        Parameters
        ----------
        i, j, k : int
            Array indices in the 3D grid
        sampling : tuple of tuples, optional
            Sampling parameters, uses instance sampling if not provided
            
        Returns
        -------
        h, k, l : float
            Miller indices corresponding to the array position
        """
        if sampling is None:
            sampling = self.sampling
        if sampling is None:
            raise ValueError("No sampling parameters available")
        
        hsampling, ksampling, lsampling = sampling
        
        # Calculate number of steps
        hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)
        ksteps = int(ksampling[2] * (ksampling[1] - ksampling[0]) + 1) 
        lsteps = int(lsampling[2] * (lsampling[1] - lsampling[0]) + 1)
        
        # Validate indices
        if not (0 <= i < hsteps):
            raise ValueError(f"Index i={i} out of range [0, {hsteps})")
        if not (0 <= j < ksteps):
            raise ValueError(f"Index j={j} out of range [0, {ksteps})")
        if not (0 <= k < lsteps):
            raise ValueError(f"Index k={k} out of range [0, {lsteps})")
        
        # Convert indices to Miller values using linspace logic
        h = hsampling[0] + (hsampling[1] - hsampling[0]) * i / (hsteps - 1)
        k_miller = ksampling[0] + (ksampling[1] - ksampling[0]) * j / (ksteps - 1)
        l = lsampling[0] + (lsampling[1] - lsampling[0]) * k / (lsteps - 1)
        
        return h, k_miller, l
    
    def hkl_to_index(self, h: float, k: float, l: float,
                    sampling: Optional[Tuple[Tuple[float, float, int], ...]] = None) -> Tuple[int, int, int]:
        """
        Convert Miller indices to nearest array indices.
        
        Parameters
        ----------
        h, k, l : float
            Miller indices
        sampling : tuple of tuples, optional
            Sampling parameters, uses instance sampling if not provided
            
        Returns
        -------
        i, j, k : int
            Nearest array indices in the 3D grid
        """
        if sampling is None:
            sampling = self.sampling
        if sampling is None:
            raise ValueError("No sampling parameters available")
        
        hsampling, ksampling, lsampling = sampling
        
        # Calculate number of steps
        hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)
        ksteps = int(ksampling[2] * (ksampling[1] - ksampling[0]) + 1)
        lsteps = int(lsampling[2] * (lsampling[1] - lsampling[0]) + 1)
        
        # Convert Miller indices to fractional indices
        i_frac = (h - hsampling[0]) / (hsampling[1] - hsampling[0]) * (hsteps - 1)
        j_frac = (k - ksampling[0]) / (ksampling[1] - ksampling[0]) * (ksteps - 1)
        k_frac = (l - lsampling[0]) / (lsampling[1] - lsampling[0]) * (lsteps - 1)
        
        # Round to nearest integer and clip to valid range
        i = np.clip(int(np.round(i_frac)), 0, hsteps - 1)
        j = np.clip(int(np.round(j_frac)), 0, ksteps - 1)
        k_idx = np.clip(int(np.round(k_frac)), 0, lsteps - 1)
        
        return i, j, k_idx
    
    def get_voxel_coordinates(self, 
                            sampling: Optional[Tuple[Tuple[float, float, int], ...]] = None,
                            as_array: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                            np.ndarray]:
        """
        Get 3D coordinates for each voxel in the grid.
        
        Parameters
        ----------
        sampling : tuple of tuples, optional
            Sampling parameters, uses instance sampling if not provided
        as_array : bool
            If True, return stacked array; if False, return separate arrays
            
        Returns
        -------
        If as_array=False:
            x, y, z : np.ndarray, each shape map_shape
                Coordinate arrays for visualization
        If as_array=True:
            xyz : np.ndarray, shape (n_points, 3)
                Stacked coordinate array
        """
        # Get q-vectors
        q_grid, map_shape = self.get_grid_coordinates(sampling, return_hkl=False)
        
        if as_array:
            return q_grid
        else:
            # Reshape to 3D grid
            x = q_grid[:, 0].reshape(map_shape)
            y = q_grid[:, 1].reshape(map_shape)
            z = q_grid[:, 2].reshape(map_shape)
            return x, y, z
    
    def get_resolution(self, hkl: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """
        Compute resolution in Angstroms for given Miller indices.
        
        Parameters
        ----------
        hkl : np.ndarray, shape (..., 3)
            Miller indices
        cell : np.ndarray, shape (6,)
            Unit cell parameters (a, b, c, alpha, beta, gamma) in Angstrom/degrees
            
        Returns
        -------
        resolution : np.ndarray
            Resolution in Angstroms for each reflection
        """
        # Import the existing function if available
        try:
            from eryx.map_utils import compute_resolution
            return compute_resolution(cell, hkl)
        except ImportError:
            logger.warning("Could not import compute_resolution, using simplified version")
            # Simplified resolution calculation (d-spacing)
            q = self.hkl_to_cartesian(hkl)
            q_mag = np.linalg.norm(q, axis=-1)
            with np.errstate(divide='ignore'):
                d_spacing = 2 * np.pi / q_mag
            return d_spacing
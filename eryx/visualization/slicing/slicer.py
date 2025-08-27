"""
Volume slicing utilities for extracting 2D slices and partial volumes.

This module provides the VolumeSlicer class for extracting slices from
3D intensity data at arbitrary cutting planes.
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class VolumeSlicer:
    """
    Extract slices and partial volumes from 3D data.
    
    This class provides methods for slicing 3D volumes along arbitrary planes,
    masking regions above/below planes, and interpolating values on plane surfaces.
    """
    
    def __init__(self, volume: np.ndarray):
        """
        Initialize with 3D intensity data.
        
        Parameters
        ----------
        volume : np.ndarray, shape (nx, ny, nz)
            3D intensity data
        """
        self.volume = np.asarray(volume)
        if self.volume.ndim != 3:
            raise ValueError(f"Volume must be 3D, got shape {self.volume.shape}")
        
        self.shape = self.volume.shape
        
        # Set up interpolator for smooth slicing
        self._setup_interpolator()
        
        logger.debug(f"VolumeSlicer initialized with volume shape {self.shape}")
    
    def _setup_interpolator(self):
        """Set up interpolator for the volume."""
        # Create coordinate arrays
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        z = np.arange(self.shape[2])
        
        # Handle NaN values by replacing with nearest valid value
        volume_clean = self.volume.copy()
        if np.any(np.isnan(volume_clean)):
            # Simple nearest neighbor fill for NaN values
            mask = np.isnan(volume_clean)
            volume_clean[mask] = 0  # Replace with 0 for now
            logger.debug(f"Replaced {mask.sum()} NaN values for interpolation")
        
        # Create interpolator
        try:
            self.interpolator = RegularGridInterpolator(
                (x, y, z), volume_clean,
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            self.has_interpolator = True
        except Exception as e:
            logger.warning(f"Could not create interpolator: {e}")
            self.has_interpolator = False
    
    def slice_at_plane(self, plane: np.ndarray, 
                      grid_resolution: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a 2D slice at an arbitrary plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients [a, b, c, d] for ax + by + cz + d = 0
        grid_resolution : int, optional
            Resolution of the extracted slice. Defaults to max dimension.
            
        Returns
        -------
        slice_data : np.ndarray, shape (grid_resolution, grid_resolution)
            Interpolated intensity values on the plane
        coords : np.ndarray, shape (grid_resolution, grid_resolution, 3)
            3D coordinates of each point in the slice
        """
        if grid_resolution is None:
            grid_resolution = max(self.shape)
        
        # Extract plane normal and normalize
        normal = plane[:3] / np.linalg.norm(plane[:3])
        
        # Find two orthogonal vectors in the plane
        # Use Gram-Schmidt to find orthogonal basis
        if abs(normal[2]) < 0.9:
            v1 = np.array([0, 0, 1])
        else:
            v1 = np.array([1, 0, 0])
        
        # First basis vector in plane
        u = v1 - np.dot(v1, normal) * normal
        u = u / np.linalg.norm(u)
        
        # Second basis vector (perpendicular to both normal and u)
        v = np.cross(normal, u)
        
        # Find a point on the plane (closest to volume center)
        center = np.array(self.shape) / 2
        d = plane[3]
        # Point on plane closest to center
        t = -(np.dot(normal, center) + d)
        plane_center = center + t * normal
        
        # Create grid on the plane
        extent = np.sqrt(sum(s**2 for s in self.shape))
        coords_1d = np.linspace(-extent/2, extent/2, grid_resolution)
        uu, vv = np.meshgrid(coords_1d, coords_1d)
        
        # Calculate 3D coordinates
        coords_3d = (plane_center[np.newaxis, np.newaxis, :] +
                    uu[:, :, np.newaxis] * u[np.newaxis, np.newaxis, :] +
                    vv[:, :, np.newaxis] * v[np.newaxis, np.newaxis, :])
        
        # Interpolate values at these points
        if self.has_interpolator:
            points = coords_3d.reshape(-1, 3)
            values = self.interpolator(points)
            slice_data = values.reshape(grid_resolution, grid_resolution)
        else:
            # Fallback: nearest neighbor
            slice_data = self._extract_nearest_neighbor(coords_3d)
        
        return slice_data, coords_3d
    
    def _extract_nearest_neighbor(self, coords: np.ndarray) -> np.ndarray:
        """
        Extract values using nearest neighbor interpolation.
        
        Parameters
        ----------
        coords : np.ndarray, shape (..., 3)
            3D coordinates
            
        Returns
        -------
        values : np.ndarray, shape coords.shape[:-1]
            Values at the nearest voxel
        """
        # Round to nearest integer coordinates
        indices = np.round(coords).astype(int)
        
        # Clip to volume bounds
        indices[..., 0] = np.clip(indices[..., 0], 0, self.shape[0] - 1)
        indices[..., 1] = np.clip(indices[..., 1], 0, self.shape[1] - 1)
        indices[..., 2] = np.clip(indices[..., 2], 0, self.shape[2] - 1)
        
        # Extract values
        values = self.volume[indices[..., 0], indices[..., 1], indices[..., 2]]
        
        # Mark out-of-bounds as NaN
        out_of_bounds = (
            (coords[..., 0] < 0) | (coords[..., 0] >= self.shape[0]) |
            (coords[..., 1] < 0) | (coords[..., 1] >= self.shape[1]) |
            (coords[..., 2] < 0) | (coords[..., 2] >= self.shape[2])
        )
        values[out_of_bounds] = np.nan
        
        return values
    
    def mask_above_plane(self, plane: np.ndarray) -> np.ndarray:
        """
        Hide voxels above the cutting plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients
            
        Returns
        -------
        masked_volume : np.ndarray, shape=volume.shape
            Volume with voxels above plane set to NaN
        """
        from .plane_calculator import PlaneCalculator
        
        calculator = PlaneCalculator(self.shape)
        distances = calculator.get_voxel_distances(plane)
        
        # Mask voxels above plane (positive distance)
        masked_volume = self.volume.copy()
        masked_volume[distances > 0] = np.nan
        
        n_masked = np.sum(distances > 0)
        logger.debug(f"Masked {n_masked}/{self.volume.size} voxels above plane")
        
        return masked_volume
    
    def mask_below_plane(self, plane: np.ndarray) -> np.ndarray:
        """
        Hide voxels below the cutting plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients
            
        Returns
        -------
        masked_volume : np.ndarray, shape=volume.shape
            Volume with voxels below plane set to NaN
        """
        from .plane_calculator import PlaneCalculator
        
        calculator = PlaneCalculator(self.shape)
        distances = calculator.get_voxel_distances(plane)
        
        # Mask voxels below plane (negative distance)
        masked_volume = self.volume.copy()
        masked_volume[distances < 0] = np.nan
        
        n_masked = np.sum(distances < 0)
        logger.debug(f"Masked {n_masked}/{self.volume.size} voxels below plane")
        
        return masked_volume
    
    def interpolate_on_plane(self, plane: np.ndarray,
                            thickness: float = 1.0) -> np.ndarray:
        """
        Interpolate intensity values on the plane surface.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients
        thickness : float
            Thickness of the slice to average over
            
        Returns
        -------
        interpolated : np.ndarray, shape=volume.shape
            Volume with only values near the plane, others NaN
        """
        from .plane_calculator import PlaneCalculator
        
        calculator = PlaneCalculator(self.shape)
        distances = calculator.get_voxel_distances(plane)
        
        # Create output array
        interpolated = np.full_like(self.volume, np.nan)
        
        # Find voxels within thickness of plane
        near_plane = np.abs(distances) <= thickness / 2
        
        if thickness > 0 and np.any(near_plane):
            # Weight by distance from plane (closer = higher weight)
            weights = 1 - np.abs(distances[near_plane]) / (thickness / 2)
            interpolated[near_plane] = self.volume[near_plane] * weights
        else:
            # Just copy exact plane voxels
            on_plane = np.abs(distances) < 0.5
            interpolated[on_plane] = self.volume[on_plane]
        
        n_interpolated = np.sum(~np.isnan(interpolated))
        logger.debug(f"Interpolated {n_interpolated} voxels on plane")
        
        return interpolated
    
    def get_partial_volume(self, plane: np.ndarray, 
                         side: str = 'below',
                         include_plane: bool = True) -> np.ndarray:
        """
        Extract partial volume on one side of a plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients
        side : str
            'below' or 'above' to specify which side
        include_plane : bool
            Whether to include voxels on the plane
            
        Returns
        -------
        partial : np.ndarray, shape=volume.shape
            Partial volume with other side set to NaN
        """
        if side == 'below':
            partial = self.mask_above_plane(plane)
        elif side == 'above':
            partial = self.mask_below_plane(plane)
        else:
            raise ValueError(f"side must be 'below' or 'above', got {side}")
        
        if include_plane:
            # Add plane interpolation
            plane_values = self.interpolate_on_plane(plane)
            # Combine with partial volume
            plane_mask = ~np.isnan(plane_values)
            partial[plane_mask] = plane_values[plane_mask]
        
        return partial
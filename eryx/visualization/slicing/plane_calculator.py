"""
Plane calculation utilities for slicing 3D volumes.

This module provides tools for calculating cutting planes through 3D data
and determining which voxels are above/below/on the plane.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PlaneCalculator:
    """
    Calculate cutting plane positions and intersections for 3D volumes.
    
    A plane in 3D space is defined by the equation: ax + by + cz + d = 0
    where (a, b, c) is the normal vector and d is the distance term.
    """
    
    def __init__(self, volume_shape: Tuple[int, int, int], 
                 center: Optional[np.ndarray] = None):
        """
        Initialize with volume dimensions.
        
        Parameters
        ----------
        volume_shape : tuple of int
            Shape of the 3D volume (nx, ny, nz)
        center : np.ndarray, optional
            Center point for plane calculations. Defaults to volume center.
        """
        self.shape = volume_shape
        self.center = center if center is not None else np.array(volume_shape) / 2.0
        
        # Create coordinate grids for voxel positions
        self._create_coordinate_grids()
        
        logger.debug(f"PlaneCalculator initialized with shape {volume_shape}, "
                    f"center {self.center}")
    
    def _create_coordinate_grids(self):
        """Create 3D coordinate grids for voxel positions."""
        # Create index grids
        i, j, k = np.mgrid[0:self.shape[0], 
                          0:self.shape[1], 
                          0:self.shape[2]]
        
        # Stack into (N, 3) array for efficient computation
        self.voxel_coords = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
        
        # Also keep 3D grids for visualization
        self.x_grid = i
        self.y_grid = j
        self.z_grid = k
    
    def calculate_plane_position(self, normal: np.ndarray, 
                                distance: float) -> np.ndarray:
        """
        Calculate plane equation coefficients.
        
        Parameters
        ----------
        normal : np.ndarray, shape (3,)
            Normal vector to the plane
        distance : float
            Signed distance from center to plane along normal
            
        Returns
        -------
        plane : np.ndarray, shape (4,)
            Plane coefficients [a, b, c, d] for equation ax + by + cz + d = 0
        """
        # Normalize the normal vector
        normal = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Normal vector cannot be zero")
        normal = normal / norm
        
        # Calculate d term: d = -normal · (center + distance*normal)
        point_on_plane = self.center + distance * normal
        d = -np.dot(normal, point_on_plane)
        
        # Return plane coefficients
        plane = np.append(normal, d)
        
        logger.debug(f"Plane equation: {plane[0]:.3f}x + {plane[1]:.3f}y + "
                    f"{plane[2]:.3f}z + {plane[3]:.3f} = 0")
        
        return plane
    
    def get_voxel_distances(self, plane: np.ndarray) -> np.ndarray:
        """
        Compute signed distances from all voxels to the plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients [a, b, c, d]
            
        Returns
        -------
        distances : np.ndarray, shape volume_shape
            Signed distances (positive = above plane, negative = below)
        """
        # Extract plane coefficients
        a, b, c, d = plane
        
        # Compute distances for all voxels
        # Distance = (ax + by + cz + d) / sqrt(a² + b² + c²)
        # Since normal is normalized, denominator is 1
        distances_flat = (a * self.voxel_coords[:, 0] + 
                         b * self.voxel_coords[:, 1] + 
                         c * self.voxel_coords[:, 2] + d)
        
        # Reshape to volume shape
        distances = distances_flat.reshape(self.shape)
        
        return distances
    
    def create_cutting_sequence(self, normal: np.ndarray, 
                               n_frames: int = 50,
                               extent_factor: float = 1.2) -> List[np.ndarray]:
        """
        Generate a sequence of cutting planes for animation.
        
        Parameters
        ----------
        normal : np.ndarray, shape (3,)
            Normal vector for all planes
        n_frames : int
            Number of frames in the sequence
        extent_factor : float
            How far beyond the volume to start/end (1.2 = 20% extra)
            
        Returns
        -------
        planes : list of np.ndarray
            List of plane coefficients for each frame
        """
        # Normalize normal vector
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate extent of volume along normal direction
        # Find corners of the volume
        corners = np.array([
            [0, 0, 0],
            [0, 0, self.shape[2]-1],
            [0, self.shape[1]-1, 0],
            [0, self.shape[1]-1, self.shape[2]-1],
            [self.shape[0]-1, 0, 0],
            [self.shape[0]-1, 0, self.shape[2]-1],
            [self.shape[0]-1, self.shape[1]-1, 0],
            [self.shape[0]-1, self.shape[1]-1, self.shape[2]-1]
        ])
        
        # Project corners onto normal to find extent
        projections = np.dot(corners - self.center, normal)
        min_proj = projections.min()
        max_proj = projections.max()
        
        # Extend range by extent_factor
        extent = max_proj - min_proj
        buffer = extent * (extent_factor - 1) / 2
        start_distance = min_proj - buffer
        end_distance = max_proj + buffer
        
        # Create sequence of distances
        distances = np.linspace(start_distance, end_distance, n_frames)
        
        # Generate planes
        planes = []
        for distance in distances:
            plane = self.calculate_plane_position(normal, distance)
            planes.append(plane)
        
        logger.info(f"Created sequence of {n_frames} planes from "
                   f"distance {start_distance:.2f} to {end_distance:.2f}")
        
        return planes
    
    def get_slice_at_plane(self, volume: np.ndarray, plane: np.ndarray,
                          thickness: float = 0.5) -> np.ndarray:
        """
        Extract a slice of the volume at the plane position.
        
        Parameters
        ----------
        volume : np.ndarray, shape volume_shape
            3D data volume
        plane : np.ndarray, shape (4,)
            Plane coefficients
        thickness : float
            Thickness of the slice in voxels
            
        Returns
        -------
        slice_data : np.ndarray
            Values at the plane (NaN where no data)
        """
        # Get distances to plane
        distances = self.get_voxel_distances(plane)
        
        # Create mask for voxels within thickness of plane
        mask = np.abs(distances) <= thickness
        
        # Extract slice
        slice_data = np.where(mask, volume, np.nan)
        
        return slice_data
    
    def get_mask_below_plane(self, plane: np.ndarray,
                            include_plane: bool = True,
                            thickness: float = 0.5) -> np.ndarray:
        """
        Create a boolean mask for voxels below (and optionally on) the plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients
        include_plane : bool
            Whether to include voxels on the plane
        thickness : float
            Thickness for determining "on plane" if include_plane=True
            
        Returns
        -------
        mask : np.ndarray, dtype=bool, shape volume_shape
            True for voxels below (and optionally on) the plane
        """
        distances = self.get_voxel_distances(plane)
        
        if include_plane:
            # Include voxels below plane and within thickness
            mask = distances <= thickness
        else:
            # Only voxels strictly below plane
            mask = distances < -thickness
        
        return mask
    
    def get_plane_intersection_points(self, plane: np.ndarray) -> np.ndarray:
        """
        Find where the plane intersects the volume boundaries.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients [a, b, c, d]
            
        Returns
        -------
        points : np.ndarray, shape (n_points, 3)
            Intersection points on volume boundaries
        """
        a, b, c, d = plane
        points = []
        
        # Check intersections with each face of the volume
        # Face x=0 and x=max
        for x in [0, self.shape[0]-1]:
            if abs(b) > 1e-10:
                for z in range(self.shape[2]):
                    y = -(a*x + c*z + d) / b
                    if 0 <= y < self.shape[1]:
                        points.append([x, y, z])
            if abs(c) > 1e-10:
                for y in range(self.shape[1]):
                    z = -(a*x + b*y + d) / c
                    if 0 <= z < self.shape[2]:
                        points.append([x, y, z])
        
        # Face y=0 and y=max
        for y in [0, self.shape[1]-1]:
            if abs(a) > 1e-10:
                for z in range(self.shape[2]):
                    x = -(b*y + c*z + d) / a
                    if 0 <= x < self.shape[0]:
                        points.append([x, y, z])
            if abs(c) > 1e-10:
                for x in range(self.shape[0]):
                    z = -(a*x + b*y + d) / c
                    if 0 <= z < self.shape[2]:
                        points.append([x, y, z])
        
        # Face z=0 and z=max
        for z in [0, self.shape[2]-1]:
            if abs(a) > 1e-10:
                for y in range(self.shape[1]):
                    x = -(b*y + c*z + d) / a
                    if 0 <= x < self.shape[0]:
                        points.append([x, y, z])
            if abs(b) > 1e-10:
                for x in range(self.shape[0]):
                    y = -(a*x + c*z + d) / b
                    if 0 <= y < self.shape[1]:
                        points.append([x, y, z])
        
        if points:
            # Remove duplicates
            points = np.unique(np.array(points), axis=0)
        else:
            points = np.array([]).reshape(0, 3)
        
        return points
    
    def decompose_normal_to_angles(self, normal: np.ndarray) -> Tuple[float, float]:
        """
        Decompose a normal vector into spherical angles.
        
        Parameters
        ----------
        normal : np.ndarray, shape (3,)
            Normal vector
            
        Returns
        -------
        theta : float
            Polar angle from z-axis (0 to pi)
        phi : float
            Azimuthal angle in xy-plane from x-axis (0 to 2*pi)
        """
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate angles
        theta = np.arccos(np.clip(normal[2], -1, 1))  # Angle from z-axis
        phi = np.arctan2(normal[1], normal[0])  # Angle in xy-plane
        
        # Ensure phi is in [0, 2*pi]
        if phi < 0:
            phi += 2 * np.pi
        
        return theta, phi
    
    def angles_to_normal(self, theta: float, phi: float) -> np.ndarray:
        """
        Convert spherical angles to a normal vector.
        
        Parameters
        ----------
        theta : float
            Polar angle from z-axis (0 to pi)
        phi : float
            Azimuthal angle in xy-plane from x-axis (0 to 2*pi)
            
        Returns
        -------
        normal : np.ndarray, shape (3,)
            Unit normal vector
        """
        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.array([x, y, z])
"""
Matplotlib-based slicer for generating animation frames.

This module provides visualization of 3D volumes being progressively sliced
by cutting planes, using matplotlib for rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MatplotlibSlicer:
    """
    Generate visualization frames of progressive slicing using matplotlib.
    
    This class creates frames showing a 3D volume being cut by a plane,
    with the portion above the plane hidden and the slice highlighted.
    """
    
    def __init__(self, volume: np.ndarray, coordinates: Optional[np.ndarray] = None):
        """
        Initialize the matplotlib slicer.
        
        Parameters
        ----------
        volume : np.ndarray, shape (nx, ny, nz)
            3D intensity data
        coordinates : np.ndarray, optional, shape (nx, ny, nz, 3)
            Real-space coordinates for each voxel (for proper scaling)
        """
        self.volume = np.asarray(volume)
        if self.volume.ndim != 3:
            raise ValueError(f"Volume must be 3D, got shape {self.volume.shape}")
        
        self.shape = self.volume.shape
        self.coordinates = coordinates
        
        # Prepare coordinate grids
        self._setup_coordinates()
        
        # Cache for performance
        self._cache = {}
        
        logger.debug(f"MatplotlibSlicer initialized with volume shape {self.shape}")
    
    def _setup_coordinates(self):
        """Set up coordinate grids for visualization."""
        if self.coordinates is None:
            # Use index coordinates
            x = np.arange(self.shape[0])
            y = np.arange(self.shape[1])
            z = np.arange(self.shape[2])
            self.x_grid, self.y_grid, self.z_grid = np.meshgrid(x, y, z, indexing='ij')
        else:
            # Use provided coordinates
            if self.coordinates.shape == self.shape + (3,):
                self.x_grid = self.coordinates[..., 0]
                self.y_grid = self.coordinates[..., 1]
                self.z_grid = self.coordinates[..., 2]
            else:
                logger.warning("Coordinates shape mismatch, using index coordinates")
                x = np.arange(self.shape[0])
                y = np.arange(self.shape[1])
                z = np.arange(self.shape[2])
                self.x_grid, self.y_grid, self.z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    def create_frame(self, plane: np.ndarray,
                    view_angles: Tuple[float, float] = (30, 45),
                    colormap: str = 'viridis',
                    show_plane: bool = True,
                    show_volume: bool = True,
                    figsize: Tuple[int, int] = (10, 8),
                    dpi: int = 100,
                    **kwargs) -> plt.Figure:
        """
        Create a single frame showing the volume cut by a plane.
        
        Parameters
        ----------
        plane : np.ndarray, shape (4,)
            Plane coefficients [a, b, c, d]
        view_angles : tuple of float
            (elevation, azimuth) viewing angles in degrees
        colormap : str
            Matplotlib colormap name
        show_plane : bool
            Whether to show the cutting plane
        show_volume : bool
            Whether to show the volume below the plane
        figsize : tuple of int
            Figure size in inches
        dpi : int
            Figure resolution
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        from .plane_calculator import PlaneCalculator
        from .slicer import VolumeSlicer
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate what to show
        calculator = PlaneCalculator(self.shape)
        slicer = VolumeSlicer(self.volume)
        
        # Get distances to plane
        distances = calculator.get_voxel_distances(plane)
        
        # Determine intensity range (excluding NaN)
        valid_intensities = self.volume[~np.isnan(self.volume)]
        if len(valid_intensities) > 0:
            vmin = np.percentile(valid_intensities, 1)
            vmax = np.percentile(valid_intensities, 99)
        else:
            vmin, vmax = 0, 1
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        if show_volume:
            # Show voxels below the plane
            below_mask = distances <= 0
            
            # Subsample for performance
            subsample = kwargs.get('subsample', 4)
            if subsample > 1:
                # Subsample the data
                x_sub = self.x_grid[::subsample, ::subsample, ::subsample]
                y_sub = self.y_grid[::subsample, ::subsample, ::subsample]
                z_sub = self.z_grid[::subsample, ::subsample, ::subsample]
                vol_sub = self.volume[::subsample, ::subsample, ::subsample]
                below_sub = below_mask[::subsample, ::subsample, ::subsample]
            else:
                x_sub = self.x_grid
                y_sub = self.y_grid
                z_sub = self.z_grid
                vol_sub = self.volume
                below_sub = below_mask
            
            # Flatten for scatter plot
            x_flat = x_sub[below_sub & ~np.isnan(vol_sub)]
            y_flat = y_sub[below_sub & ~np.isnan(vol_sub)]
            z_flat = z_sub[below_sub & ~np.isnan(vol_sub)]
            c_flat = vol_sub[below_sub & ~np.isnan(vol_sub)]
            
            if len(x_flat) > 0:
                # Normalize colors
                c_norm = (c_flat - vmin) / (vmax - vmin) if vmax > vmin else c_flat * 0
                c_norm = np.clip(c_norm, 0, 1)
                
                # Plot voxels
                scatter = ax.scatter(x_flat, y_flat, z_flat,
                                   c=c_norm, cmap=cmap,
                                   s=kwargs.get('voxel_size', 1),
                                   alpha=kwargs.get('volume_alpha', 0.6),
                                   edgecolors='none')
        
        if show_plane:
            # Draw the cutting plane
            self._draw_plane(ax, plane, alpha=kwargs.get('plane_alpha', 0.3))
            
            # Optionally show slice on plane
            if kwargs.get('show_slice', True):
                self._draw_slice_on_plane(ax, plane, slicer, cmap, vmin, vmax,
                                         alpha=kwargs.get('slice_alpha', 0.9))
        
        # Set view and labels
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set axis limits
        ax.set_xlim(0, self.shape[0])
        ax.set_ylim(0, self.shape[1])
        ax.set_zlim(0, self.shape[2])
        
        # Add title
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        
        # Add colorbar if requested
        if kwargs.get('colorbar', True) and 'scatter' in locals():
            plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.6)
        
        plt.tight_layout()
        
        return fig
    
    def _draw_plane(self, ax, plane: np.ndarray, alpha: float = 0.3):
        """
        Draw the cutting plane as a semi-transparent surface.
        
        Parameters
        ----------
        ax : matplotlib axes
            3D axes to draw on
        plane : np.ndarray, shape (4,)
            Plane coefficients
        alpha : float
            Transparency of the plane
        """
        a, b, c, d = plane
        
        # Create a grid on the plane within the volume bounds
        # Find extent of plane intersection with volume
        x_range = [0, self.shape[0]]
        y_range = [0, self.shape[1]]
        z_range = [0, self.shape[2]]
        
        # Create grid based on dominant normal component
        normal = plane[:3]
        abs_normal = np.abs(normal)
        
        if abs_normal[2] == max(abs_normal):
            # Plane is most perpendicular to z-axis
            x = np.linspace(x_range[0], x_range[1], 20)
            y = np.linspace(y_range[0], y_range[1], 20)
            X, Y = np.meshgrid(x, y)
            if abs(c) > 1e-10:
                Z = -(a * X + b * Y + d) / c
                # Clip to volume bounds
                mask = (Z >= z_range[0]) & (Z <= z_range[1])
                Z[~mask] = np.nan
            else:
                return  # Plane parallel to z-axis
        elif abs_normal[1] == max(abs_normal):
            # Plane is most perpendicular to y-axis
            x = np.linspace(x_range[0], x_range[1], 20)
            z = np.linspace(z_range[0], z_range[1], 20)
            X, Z = np.meshgrid(x, z)
            if abs(b) > 1e-10:
                Y = -(a * X + c * Z + d) / b
                mask = (Y >= y_range[0]) & (Y <= y_range[1])
                Y[~mask] = np.nan
            else:
                return
        else:
            # Plane is most perpendicular to x-axis
            y = np.linspace(y_range[0], y_range[1], 20)
            z = np.linspace(z_range[0], z_range[1], 20)
            Y, Z = np.meshgrid(y, z)
            if abs(a) > 1e-10:
                X = -(b * Y + c * Z + d) / a
                mask = (X >= x_range[0]) & (X <= x_range[1])
                X[~mask] = np.nan
            else:
                return
        
        # Draw the plane
        ax.plot_surface(X, Y, Z, alpha=alpha, color='gray', shade=False)
    
    def _draw_slice_on_plane(self, ax, plane: np.ndarray, slicer,
                            cmap, vmin: float, vmax: float, alpha: float = 0.9):
        """
        Draw the intensity values on the cutting plane.
        
        Parameters
        ----------
        ax : matplotlib axes
            3D axes to draw on
        plane : np.ndarray
            Plane coefficients
        slicer : VolumeSlicer
            Slicer object for extracting values
        cmap : colormap
            Matplotlib colormap
        vmin, vmax : float
            Intensity range for coloring
        alpha : float
            Transparency
        """
        # Get slice data
        slice_data, coords_3d = slicer.slice_at_plane(plane, grid_resolution=30)
        
        # Remove NaN values
        mask = ~np.isnan(slice_data)
        if not np.any(mask):
            return
        
        # Get coordinates and values
        x = coords_3d[:, :, 0]
        y = coords_3d[:, :, 1]
        z = coords_3d[:, :, 2]
        
        # Clip to volume bounds
        in_bounds = (
            (x >= 0) & (x < self.shape[0]) &
            (y >= 0) & (y < self.shape[1]) &
            (z >= 0) & (z < self.shape[2])
        )
        
        # Normalize colors
        colors = (slice_data - vmin) / (vmax - vmin) if vmax > vmin else slice_data * 0
        colors = np.clip(colors, 0, 1)
        colors = cmap(colors)
        colors[..., 3] = alpha  # Set alpha
        colors[~(mask & in_bounds)] = [0, 0, 0, 0]  # Transparent outside
        
        # Draw as surface
        ax.plot_surface(x, y, z, facecolors=colors, shade=False, linewidth=0)
    
    def render_composite(self, volume_mask: np.ndarray, 
                        slice_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a composite visualization combining volume and slice.
        
        Parameters
        ----------
        volume_mask : np.ndarray, dtype=bool
            Mask for which voxels to show
        slice_data : np.ndarray, optional
            2D slice data to overlay
            
        Returns
        -------
        image : np.ndarray
            Rendered image as numpy array
        """
        # This is a simplified version - full implementation would
        # render to an off-screen buffer
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot masked volume
        masked_volume = np.where(volume_mask, self.volume, np.nan)
        valid = ~np.isnan(masked_volume)
        
        if np.any(valid):
            x = self.x_grid[valid]
            y = self.y_grid[valid]
            z = self.z_grid[valid]
            c = masked_volume[valid]
            
            ax.scatter(x, y, z, c=c, cmap='viridis', s=1, alpha=0.6)
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return image
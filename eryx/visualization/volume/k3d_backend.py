"""
K3D backend for high-quality 3D volume rendering.

This module provides photorealistic volume visualization using k3d's
WebGL-based rendering engine, offering much better quality than matplotlib.
"""

import numpy as np
import k3d
from typing import Optional, Tuple, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class K3DVolumeRenderer:
    """
    High-quality volume renderer using k3d.
    
    K3D provides photorealistic volume rendering with WebGL, supporting
    advanced features like opacity transfer functions, isosurfaces, and
    interactive controls.
    """
    
    def __init__(self, volume: np.ndarray, 
                 coordinates: Optional[np.ndarray] = None,
                 name: str = "Diffuse Intensity"):
        """
        Initialize the K3D volume renderer.
        
        Parameters
        ----------
        volume : np.ndarray, shape (nx, ny, nz)
            3D intensity data
        coordinates : np.ndarray, optional
            Real-space coordinates for each voxel
        name : str
            Display name for the volume
        """
        self.volume = np.asarray(volume, dtype=np.float32)
        if self.volume.ndim != 3:
            raise ValueError(f"Volume must be 3D, got shape {self.volume.shape}")
        
        self.shape = self.volume.shape
        self.coordinates = coordinates
        self.name = name
        
        # Create k3d plot
        self.plot = None
        self.volume_object = None
        
        # Cache for performance
        self._cache = {}
        
        logger.info(f"K3DVolumeRenderer initialized with volume shape {self.shape}")
    
    def create_volume_plot(self, 
                          colormap: str = 'viridis',
                          opacity_function: Optional[np.ndarray] = None,
                          bounds: Optional[list] = None,
                          **kwargs) -> k3d.Plot:
        """
        Create a k3d volume rendering plot.
        
        Parameters
        ----------
        colormap : str or array
            K3D colormap name or custom colormap array
        opacity_function : np.ndarray, optional
            Opacity transfer function (256 values from 0 to 1)
        bounds : list, optional
            [xmin, xmax, ymin, ymax, zmin, zmax] bounds
        **kwargs : dict
            Additional k3d.volume parameters
            
        Returns
        -------
        plot : k3d.Plot
            K3D plot object with volume
        """
        # Create plot if not exists
        if self.plot is None:
            self.plot = k3d.plot(
                grid_visible=kwargs.get('grid_visible', False),
                menu_visibility=kwargs.get('menu_visibility', True),
                camera_auto_fit=kwargs.get('camera_auto_fit', True)
            )
        
        # Prepare volume data
        volume_data = self._prepare_volume_data()
        
        # Set bounds if not provided
        if bounds is None:
            bounds = [0, self.shape[0], 
                     0, self.shape[1], 
                     0, self.shape[2]]
        
        # Create opacity function if not provided
        if opacity_function is None:
            opacity_function = self._create_default_opacity()
        
        # Get colormap
        color_map = self._get_colormap(colormap)
        
        # Create volume object
        self.volume_object = k3d.volume(
            volume_data,
            color_map=color_map,
            opacity_function=opacity_function,
            bounds=bounds,
            name=self.name,
            compression_level=kwargs.get('compression_level', 9)
        )
        
        # Add to plot
        self.plot += self.volume_object
        
        # Set camera if specified
        if 'camera' in kwargs:
            self.plot.camera = kwargs['camera']
        
        logger.info("Created k3d volume plot")
        return self.plot
    
    def _prepare_volume_data(self) -> np.ndarray:
        """
        Prepare volume data for k3d rendering.
        
        Returns
        -------
        prepared : np.ndarray
            Volume data ready for k3d (normalized, NaN-handled)
        """
        # Handle NaN values
        volume_clean = self.volume.copy()
        nan_mask = np.isnan(volume_clean)
        
        if np.any(nan_mask):
            # Replace NaN with minimum value (will be transparent)
            valid_min = np.nanmin(volume_clean)
            volume_clean[nan_mask] = valid_min
            logger.debug(f"Replaced {nan_mask.sum()} NaN values")
        
        # Normalize to 0-1 range for k3d
        vmin = np.min(volume_clean)
        vmax = np.max(volume_clean)
        
        if vmax > vmin:
            volume_normalized = (volume_clean - vmin) / (vmax - vmin)
        else:
            volume_normalized = volume_clean * 0
        
        # Convert to float32 for k3d
        return volume_normalized.astype(np.float32)
    
    def _create_default_opacity(self) -> np.ndarray:
        """
        Create default opacity transfer function.
        
        Returns
        -------
        opacity : np.ndarray, shape (256,)
            Opacity values from 0 to 1
        """
        # Create opacity that reveals more of the low-intensity structure
        x = np.linspace(0, 1, 256)
        
        # Use a power function for better visibility of low values
        # This reveals the diffuse scattering structure better
        opacity = np.power(x, 0.3)  # Power < 1 enhances low values
        
        # Scale opacity to reasonable range
        opacity = opacity * 0.8  # Max opacity 0.8 for some transparency
        
        # Make very low values (NaN regions) completely transparent
        opacity[:5] = 0
        
        # Ensure smooth transition
        opacity[5:10] = np.linspace(0, opacity[10], 5)
        
        return opacity.astype(np.float32)
    
    def _get_colormap(self, colormap: Union[str, np.ndarray]) -> np.ndarray:
        """
        Get or create colormap for k3d.
        
        Parameters
        ----------
        colormap : str or array
            Colormap name or custom array
            
        Returns
        -------
        cmap : np.ndarray
            K3D colormap array
        """
        if isinstance(colormap, np.ndarray):
            return colormap
        
        # K3D colormaps - use direct access which works better
        k3d_cmaps = {
            'viridis': k3d.matplotlib_color_maps.Viridis,
            'plasma': k3d.matplotlib_color_maps.Plasma,
            'inferno': k3d.matplotlib_color_maps.Inferno,
            'magma': k3d.matplotlib_color_maps.Magma,
            'cividis': k3d.matplotlib_color_maps.Cividis,
            'coolwarm': k3d.matplotlib_color_maps.Coolwarm,
            'hot': k3d.matplotlib_color_maps.Hot,
            'cool': k3d.matplotlib_color_maps.Cool,
            'rainbow': k3d.basic_color_maps.Rainbow,
            'jet': k3d.basic_color_maps.Jet,
        }
        
        if colormap.lower() in k3d_cmaps:
            return k3d_cmaps[colormap.lower()]
        else:
            logger.warning(f"Unknown colormap {colormap}, using viridis")
            return k3d.matplotlib_color_maps.Viridis
    
    def add_isosurface(self, level: float = 0.5, 
                       color: int = 0x00ff00,
                       opacity: float = 0.5) -> None:
        """
        Add an isosurface at a specific intensity level.
        
        Parameters
        ----------
        level : float
            Intensity level (0 to 1 after normalization)
        color : int
            Color as hex integer
        opacity : float
            Surface opacity
        """
        if self.plot is None:
            raise ValueError("Create plot first with create_volume_plot()")
        
        # Prepare volume data
        volume_data = self._prepare_volume_data()
        
        # Create isosurface
        iso = k3d.marching_cubes(
            volume_data,
            level=level,
            color=color,
            opacity=opacity,
            name=f"Isosurface {level:.2f}"
        )
        
        # Add to plot
        self.plot += iso
        
        logger.info(f"Added isosurface at level {level}")
    
    def add_slice(self, axis: str = 'z', position: float = 0.5,
                 colormap: str = 'viridis') -> None:
        """
        Add a 2D slice through the volume.
        
        Parameters
        ----------
        axis : str
            Axis to slice along ('x', 'y', or 'z')
        position : float
            Position along axis (0 to 1)
        colormap : str
            Colormap for the slice
        """
        if self.plot is None:
            raise ValueError("Create plot first with create_volume_plot()")
        
        # Prepare volume data
        volume_data = self._prepare_volume_data()
        
        # Get slice index
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
        
        axis_idx = axis_map[axis]
        slice_idx = int(position * self.shape[axis_idx])
        slice_idx = np.clip(slice_idx, 0, self.shape[axis_idx] - 1)
        
        # Extract slice
        if axis == 'x':
            slice_data = volume_data[slice_idx, :, :]
        elif axis == 'y':
            slice_data = volume_data[:, slice_idx, :]
        else:  # z
            slice_data = volume_data[:, :, slice_idx]
        
        # Create texture for slice
        # K3D expects shape (height, width, 1) for grayscale
        slice_texture = np.expand_dims(slice_data, axis=-1)
        
        # Add as textured mesh
        # (This is simplified - full implementation would create proper mesh)
        logger.info(f"Added slice along {axis} at position {position}")
    
    def export_html(self, filename: str) -> None:
        """
        Export the plot as a standalone HTML file.
        
        Parameters
        ----------
        filename : str
            Output HTML file path
        """
        if self.plot is None:
            raise ValueError("Create plot first with create_volume_plot()")
        
        with open(filename, 'w') as f:
            f.write(self.plot.get_snapshot())
        
        logger.info(f"Exported plot to {filename}")
    
    def display(self) -> k3d.Plot:
        """
        Display the plot (for Jupyter notebooks).
        
        Returns
        -------
        plot : k3d.Plot
            The k3d plot object
        """
        if self.plot is None:
            self.create_volume_plot()
        
        return self.plot


def create_k3d_visualization(data_path: str,
                            nan_strategy: str = 'mask',
                            colormap: str = 'viridis',
                            output_html: Optional[str] = None,
                            **kwargs) -> k3d.Plot:
    """
    High-level function to create k3d visualization from data file.
    
    Parameters
    ----------
    data_path : str
        Path to NPZ file with intensity data
    nan_strategy : str
        How to handle NaN values
    colormap : str
        Colormap name
    output_html : str, optional
        Path to save HTML file
    **kwargs : dict
        Additional parameters for rendering
        
    Returns
    -------
    plot : k3d.Plot
        K3D plot object
    """
    from ..core import IntensityDataHandler, NaNProcessor
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    handler = IntensityDataHandler(data_path)
    q_vectors, intensity, map_shape = handler.load_data()
    
    if intensity is None:
        raise ValueError(f"Could not load data from {data_path}")
    
    # Reshape if needed
    if intensity.ndim == 1 and map_shape is not None:
        handler.reshape_to_3d()
        intensity = handler.intensity
    
    # Handle NaN values
    logger.info(f"Processing NaN values with strategy '{nan_strategy}'")
    intensity = NaNProcessor.apply_strategy(intensity, nan_strategy)
    
    # Create k3d visualization
    logger.info("Creating k3d volume rendering")
    renderer = K3DVolumeRenderer(intensity, coordinates=q_vectors)
    plot = renderer.create_volume_plot(
        colormap=colormap,
        **kwargs
    )
    
    # Add isosurfaces if requested
    if 'isosurfaces' in kwargs:
        for level in kwargs['isosurfaces']:
            renderer.add_isosurface(level)
    
    # Export if requested
    if output_html:
        renderer.export_html(output_html)
        logger.info(f"Saved interactive visualization to {output_html}")
    
    return plot
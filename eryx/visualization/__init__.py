"""
Advanced visualization module for diffuse scattering intensity data.

This module provides tools for creating animations and interactive
3D visualizations of diffuse scattering data from protein crystals.

Quick Start:
-----------
# K3D Interactive Visualization (Recommended)
from eryx.visualization import create_clipped_visualization
plot, volume = create_clipped_visualization(volume_data)
plot.clipping_planes = [[1, 0, 0, 0]]

# Matplotlib Animation (Alternative)  
from eryx.visualization import create_slicing_animation
anim = create_slicing_animation(
    data_path='results.npz',
    normal=[1, 1, 1]
)
"""

# Version
__version__ = "0.2.0"

# Import core components
from .core import IntensityDataHandler, CoordinateMapper, NaNProcessor

# Import K3D volume rendering (production ready)
try:
    from .volume.k3d import create_clipped_visualization, K3D_AVAILABLE
    _has_k3d = K3D_AVAILABLE
except ImportError:
    _has_k3d = False

# Import slicing animation components (in development)
try:
    from .slicing.animator import SlicingAnimator, create_slicing_animation
    from .slicing.plane_calculator import PlaneCalculator
    _has_slicing = True
except ImportError:
    _has_slicing = False

# All exported components
__all__ = [
    # Core
    "IntensityDataHandler",
    "CoordinateMapper", 
    "NaNProcessor",
    # Status flags
    "K3D_AVAILABLE",
]

# Add K3D components if available (recommended)
if _has_k3d:
    __all__.extend([
        "create_clipped_visualization",
    ])

# Add slicing components if available
if _has_slicing:
    __all__.extend([
        "SlicingAnimator",
        "create_slicing_animation",
        "PlaneCalculator",
    ])


def visualize_diffuse_3d(data, method='k3d', **kwargs):
    """
    High-level API for 3D visualization of diffuse intensity data.
    
    Parameters
    ----------
    data : np.ndarray or str
        3D volume data or path to data file
    method : str
        'k3d' (recommended) or 'matplotlib'
    **kwargs
        Animation options for method='matplotlib'. K3D takes no keyword options;
        configure the returned plot and volume objects instead.
        
    Returns
    -------
    (plot, volume) for K3D, or the output filename for Matplotlib animation.
    
    Examples
    --------
    # K3D with clipping (recommended)
    plot, rendered_volume = visualize_diffuse_3d(volume, method='k3d')
    plot.clipping_planes = [[1, 0, 0, 0]]
                               
    # Matplotlib animation  
    anim = visualize_diffuse_3d('data.npz', method='matplotlib',
                               normal=[1,1,1])
    """
    import numpy as np
    
    # Handle data loading
    if isinstance(data, str):
        handler = IntensityDataHandler(data)
        coords, volume, shape = handler.load_data()
        if volume is None:
            raise ValueError(f"Could not load intensity from {data}")
        if volume.ndim == 1 and shape is not None:
            volume = volume.reshape(shape)
    else:
        volume = np.asarray(data)
    
    if method.lower() == 'k3d':
        if volume.ndim != 3:
            raise ValueError("K3D requires a 3D array or an NPZ with map_shape")
        if not _has_k3d:
            raise ImportError("K3D not available. Install with: pip install k3d")
        return create_clipped_visualization(volume, **kwargs)
    
    elif method.lower() == 'matplotlib':
        if not _has_slicing:
            raise ImportError("Matplotlib slicing not available")
        # For matplotlib, we need coordinates
        if isinstance(data, str):
            return create_slicing_animation(data, **kwargs)
        else:
            raise ValueError("Matplotlib method requires data file path, not array")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'k3d' or 'matplotlib'")


__all__.append("visualize_diffuse_3d")

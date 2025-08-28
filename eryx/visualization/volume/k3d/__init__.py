"""
K3D visualization module for advanced 3D clipping and interactive rendering.

This module provides production-ready 3D visualization capabilities using K3D's
WebGL-based rendering engine with full clipping plane support.

Key Features:
- Interactive 3D volume rendering with real-time clipping planes
- Spherical and planar clipping modes for anisotropy analysis
- Browser-based animation and interactive controls
- Production-quality visualizations for diffuse intensity data

Quick Start:
-----------
import numpy as np
from eryx.visualization.volume.k3d import create_clipped_visualization

# Load your diffuse intensity data
data = np.load('torch_diffuse_intensity.npy')
volume_3d = data.reshape(41, 41, 41)  # Assuming cubic data

# Create interactive visualization with clipping
plot = create_clipped_visualization(
    volume_3d,
    bounds=[-2, 2, -2, 2, -2, 2],  # q-space bounds
    clipping_planes=[[1, 0, 0, 0]]  # qx > 0 half-space
)

# Export to HTML
with open('diffuse_intensity_clipped.html', 'w') as f:
    f.write(plot.get_snapshot())

Available Examples:
------------------
- k3d_spherical_clipping.py/.ipynb: Spherical clipping for isotropic analysis
- k3d_interactive_clipping.py/.ipynb: Interactive planar clipping controls  
- final_k3d_clipping_solution.py: Production-ready implementation
- FINAL_WORKING_SOLUTION.md: Complete documentation and examples
"""

try:
    import k3d
    K3D_AVAILABLE = True
except ImportError:
    K3D_AVAILABLE = False

__all__ = ['K3D_AVAILABLE']

if K3D_AVAILABLE:
    # Import main functions when k3d is available
    from .final_k3d_clipping_solution import create_clipped_visualization
    __all__.extend(['create_clipped_visualization'])
"""Optional K3D rendering.

create_clipped_visualization(volume_data) returns (plot, volume).
Configure properties on those objects after construction. See
docs/visualization/k3d/getting-started/QUICK_START.md for an executable example
with recorded map shape, axis order, and explicit display bounds.
"""

try:
    import k3d
    K3D_AVAILABLE = True
except ImportError:
    K3D_AVAILABLE = False

__all__ = ['K3D_AVAILABLE']

if K3D_AVAILABLE:
    # Import main functions when k3d is available
    from .final_k3d_clipping_solution import create_production_visualization
    # Also create an alias for backwards compatibility
    create_clipped_visualization = create_production_visualization
    __all__.extend(['create_production_visualization', 'create_clipped_visualization'])

"""
Advanced visualization module for diffuse scattering intensity data.

This module provides tools for creating animations and interactive
3D visualizations of diffuse scattering data from protein crystals.
"""

# Version
__version__ = "0.2.0"

# Import core components
from .core import IntensityDataHandler, CoordinateMapper, NaNProcessor

# Import slicing animation components
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
]

# Add slicing components if available
if _has_slicing:
    __all__.extend([
        "SlicingAnimator",
        "create_slicing_animation",
        "PlaneCalculator",
    ])
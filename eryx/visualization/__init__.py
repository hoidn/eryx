"""
Advanced visualization module for diffuse scattering intensity data.

This module provides tools for creating animations and interactive
3D visualizations of diffuse scattering data from protein crystals.
"""

# Version
__version__ = "0.1.0"

# Import core components
from .core import IntensityDataHandler, CoordinateMapper, NaNProcessor

# Core components
__all__ = [
    "IntensityDataHandler",
    "CoordinateMapper", 
    "NaNProcessor",
]
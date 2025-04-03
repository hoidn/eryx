"""
Visualization module for diffuse scattering data.

This module provides functions to visualize diffuse scattering data from
both grid-based and arbitrary q-vector calculations, with careful handling
of q-vector consistency issues.

IMPORTANT NOTE ON SAMPLING PARAMETERS:
There are two different interpretations of sampling parameters in the codebase:
1. Direct interpretation: h_dim = int(hsampling[2])
2. Formula-based: hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)

This can lead to mismatches between q-vectors and intensity arrays.
Always ensure your visualization uses the same q-vectors as your simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import warnings
import logging
import os
from typing import Dict, Tuple, List, Optional, Union, Any

# Check for PyTorch and import if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Some functionality may be limited.")
    
# Set up logging
logger = logging.getLogger(__name__)

def ensure_numpy(data: Any) -> np.ndarray:
    """
    Convert input data to NumPy array.
    
    Args:
        data: Input data (PyTorch tensor or NumPy array)
        
    Returns:
        NumPy array version of the input data
    """
    if HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)
        
def grid_to_nongrid(intensity_grid: Any, q_grid: Any, mask: Optional[Any] = None) -> Dict[str, np.ndarray]:
    """
    Convert grid-formatted data to non-grid format with dimension validation.
    
    Args:
        intensity_grid: Grid of intensity values (3D or flattened)
        q_grid: Grid of q-vectors [n_points, 3]
        mask: Optional mask for valid points
        
    Returns:
        Dictionary with:
            q_vectors: [n_valid_points, 3] array of q-vectors
            intensities: [n_valid_points] array of intensity values
    """
    # Convert to NumPy arrays
    intensity_np = ensure_numpy(intensity_grid)
    q_grid_np = ensure_numpy(q_grid)
    
    # Log dimensions for debugging
    logger.info(f"Q-grid shape: {q_grid_np.shape}, Intensity grid shape: {intensity_np.shape}")
    
    # If intensity_grid is multi-dimensional, flatten it
    if intensity_np.ndim > 1 and intensity_np.shape != q_grid_np.shape:
        original_shape = intensity_np.shape
        intensity_np = intensity_np.flatten()
        logger.info(f"Flattened intensity grid from {original_shape} to {intensity_np.shape}")
        
    # Validate dimensions
    if intensity_np.shape[0] != q_grid_np.shape[0]:
        logger.warning(f"Dimension mismatch between intensity ({intensity_np.shape[0]}) and q-grid ({q_grid_np.shape[0]})")
        logger.warning("This could indicate sampling parameter inconsistency issues")
        logger.warning("Attempting to align data by truncation/padding...")
        
        min_size = min(intensity_np.shape[0], q_grid_np.shape[0])
        if intensity_np.shape[0] > min_size:
            intensity_np = intensity_np[:min_size]
            logger.warning(f"Truncated intensity array to {min_size} points")
        if q_grid_np.shape[0] > min_size:
            q_grid_np = q_grid_np[:min_size]
            logger.warning(f"Truncated q-grid array to {min_size} points")
    
    # Apply mask if provided
    if mask is not None:
        mask_np = ensure_numpy(mask)
        
        # Ensure mask has correct size
        if mask_np.shape[0] != intensity_np.shape[0]:
            logger.warning(f"Mask size ({mask_np.shape[0]}) doesn't match data size ({intensity_np.shape[0]})")
            if mask_np.shape[0] > intensity_np.shape[0]:
                mask_np = mask_np[:intensity_np.shape[0]]
            else:
                # Extend mask with False values
                extended_mask = np.zeros(intensity_np.shape[0], dtype=bool)
                extended_mask[:mask_np.shape[0]] = mask_np
                mask_np = extended_mask
        
        # If mask is not boolean, convert it
        if mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)
            
        # Apply mask
        valid_indices = np.where(mask_np)[0]
        q_vectors = q_grid_np[valid_indices]
        intensities = intensity_np[valid_indices]
    else:
        # No mask, but filter out NaN values in intensities
        valid_indices = ~np.isnan(intensity_np)
        q_vectors = q_grid_np[valid_indices]
        intensities = intensity_np[valid_indices]
    
    logger.info(f"Converted data: {len(intensities)} valid points out of {len(intensity_np)}")
    
    return {
        'q_vectors': q_vectors,
        'intensities': intensities
    }

def load_simulation_data(output_dir: str = ".", 
                       np_file: str = "np_diffuse_intensity.npy",
                       torch_file: str = "torch_diffuse_intensity.npy",
                       explicit_q_file: str = "torch_explicit_q_diffuse_intensity.npy",
                       q_vectors_file: str = "q_vectors.npy",
                       mask_file: str = "resolution_mask.npy") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Safely load simulation results from different modes with q-vector consistency checks.
    
    Args:
        output_dir: Directory containing output files
        np_file: Filename for NumPy grid-based results
        torch_file: Filename for PyTorch grid-based results
        explicit_q_file: Filename for PyTorch explicit q-vector results
        q_vectors_file: Filename for saved q-vectors
        mask_file: Filename for resolution mask
        
    Returns:
        Dictionary with data from all available modes
    """
    # Initialize return dictionary
    data_dict = {}
    
    # Attempt to load saved q-vectors first (preferred approach)
    q_vectors = None
    mask = None
    
    try:
        q_vectors_path = os.path.join(output_dir, q_vectors_file)
        mask_path = os.path.join(output_dir, mask_file)
        
        if os.path.exists(q_vectors_path):
            q_vectors = np.load(q_vectors_path)
            logger.info(f"Loaded saved q-vectors with shape {q_vectors.shape}")
            
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                logger.info(f"Loaded resolution mask with {np.sum(mask)} valid points")
            else:
                logger.warning(f"Resolution mask file not found: {mask_path}")
                logger.warning("All points will be considered valid")
        else:
            logger.warning(f"Saved q-vectors file not found: {q_vectors_path}")
            logger.warning("Will attempt to regenerate q-vectors or align data dimensions")
    except Exception as e:
        logger.error(f"Error loading saved q-vectors: {e}")
    
    # Fallback: create q-vectors from a minimal model if needed
    if q_vectors is None and HAS_TORCH:
        try:
            logger.warning("Attempting to regenerate q-vectors (may cause inconsistencies)")
            
            from eryx.models_torch import OnePhonon
            
            # Use minimal parameters to reduce computation
            pdb_path = "tests/pdbs/5zck_p1.pdb"
            model = OnePhonon(
                pdb_path=pdb_path,
                hsampling=[-4, 4, 3],
                ksampling=[-17, 17, 3],
                lsampling=[-29, 29, 3],
                expand_p1=True,
                res_limit=0.0,
                gnm_cutoff=0.0,  # Minimal computation
                gamma_intra=0.0,
                gamma_inter=0.0
            )
            
            # Get q-vectors and mask
            q_vectors = model.q_grid.detach().cpu().numpy()
            mask = model.res_mask.detach().cpu().numpy()
            
            logger.info(f"Regenerated q-vectors with shape {q_vectors.shape}")
            logger.warning("Using regenerated q-vectors may cause inconsistencies")
            logger.warning("Consider saving q-vectors with your simulation results")
        except Exception as e:
            logger.error(f"Error regenerating q-vectors: {e}")
    
    # Load intensity data
    np_path = os.path.join(output_dir, np_file)
    torch_path = os.path.join(output_dir, torch_file)
    explicit_q_path = os.path.join(output_dir, explicit_q_file)
    
    # NumPy grid-based results
    if os.path.exists(np_path):
        try:
            np_intensity = np.load(np_path)
            logger.info(f"Loaded NumPy intensity data with shape {np_intensity.shape}")
            
            if q_vectors is not None:
                np_data = grid_to_nongrid(np_intensity, q_vectors, mask)
                data_dict['NumPy'] = np_data
            else:
                # Can't convert to non-grid format without q-vectors
                logger.warning("No q-vectors available for NumPy data conversion")
                data_dict['NumPy'] = {'intensities': np_intensity.flatten()}
        except Exception as e:
            logger.error(f"Error loading NumPy data: {e}")
    
    # PyTorch grid-based results
    if os.path.exists(torch_path):
        try:
            torch_intensity = np.load(torch_path)
            logger.info(f"Loaded PyTorch grid intensity data with shape {torch_intensity.shape}")
            
            if q_vectors is not None:
                torch_data = grid_to_nongrid(torch_intensity, q_vectors, mask)
                data_dict['PyTorch Grid'] = torch_data
            else:
                # Can't convert to non-grid format without q-vectors
                logger.warning("No q-vectors available for PyTorch data conversion")
                data_dict['PyTorch Grid'] = {'intensities': torch_intensity.flatten()}
        except Exception as e:
            logger.error(f"Error loading PyTorch grid data: {e}")
    
    # PyTorch explicit q-vector results
    if os.path.exists(explicit_q_path):
        try:
            explicit_q_intensity = np.load(explicit_q_path)
            logger.info(f"Loaded PyTorch explicit q-vector intensity data with shape {explicit_q_intensity.shape}")
            
            if q_vectors is not None:
                explicit_q_data = grid_to_nongrid(explicit_q_intensity, q_vectors, mask)
                data_dict['PyTorch Explicit-Q'] = explicit_q_data
            else:
                # Can't convert to non-grid format without q-vectors
                logger.warning("No q-vectors available for explicit q-vector data conversion")
                data_dict['PyTorch Explicit-Q'] = {'intensities': explicit_q_intensity.flatten()}
        except Exception as e:
            logger.error(f"Error loading PyTorch explicit q-vector data: {e}")
    
    if not data_dict:
        raise FileNotFoundError(f"No valid diffuse scattering data found in {output_dir}")
    
    return data_dict

def visualize_basic_scatter(data_dict: Dict[str, Dict[str, np.ndarray]], 
                          output_dir: str = "visualization_output",
                          max_points: int = 5000,
                          log_scale: bool = True):
    """
    Create basic scatter plots for each mode to validate data consistency.
    
    Args:
        data_dict: Dictionary with data from all modes
        output_dir: Directory to save visualization files
        max_points: Maximum number of points to plot
        log_scale: Whether to use log scale for intensities
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scatter plots for each mode
    for mode, data in data_dict.items():
        logger.info(f"Creating validation scatter plot for {mode}")
        
        # Skip if no q-vectors available
        if 'q_vectors' not in data:
            logger.warning(f"No q-vectors for {mode}, skipping visualization")
            continue
            
        # Get data
        q_vectors = data['q_vectors']
        intensities = data['intensities']
        
        # Remove NaN values
        valid_mask = ~np.isnan(intensities)
        q_vectors = q_vectors[valid_mask]
        intensities = intensities[valid_mask]
        
        if len(intensities) == 0:
            logger.warning(f"No valid intensity points for {mode}")
            continue
        
        # Handle log scale
        if log_scale:
            # Add small epsilon to avoid log(0)
            intensities = np.log1p(np.maximum(intensities, 0))
            cbar_label = 'Log(Intensity + 1)'
        else:
            cbar_label = 'Intensity'
        
        # Downsample if needed
        if len(intensities) > max_points:
            indices = np.random.choice(len(intensities), max_points, replace=False)
            q_vectors = q_vectors[indices]
            intensities = intensities[indices]
            logger.info(f"Downsampled to {max_points} points for visualization")
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(
            q_vectors[:, 0], q_vectors[:, 1], q_vectors[:, 2],
            c=intensities, cmap='viridis', s=30, alpha=0.7
        )
        
        # Set labels
        ax.set_xlabel('qx (Å⁻¹)')
        ax.set_ylabel('qy (Å⁻¹)')
        ax.set_zlabel('qz (Å⁻¹)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=cbar_label)
        
        # Set title
        plt.title(f'{mode} Diffuse Scattering - {len(intensities)} points')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{mode.lower().replace(' ', '_')}_scatter.png"), dpi=150)
        plt.close(fig)
        
        logger.info(f"Saved scatter plot for {mode}")
        
        # Create histogram of intensities as additional validation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(intensities, bins=100, alpha=0.7)
        ax.set_xlabel(cbar_label)
        ax.set_ylabel('Count')
        ax.set_title(f'{mode} Intensity Distribution')
        plt.savefig(os.path.join(output_dir, f"{mode.lower().replace(' ', '_')}_histogram.png"), dpi=150)
        plt.close(fig)
        
        logger.info(f"Saved histogram for {mode}")
    
    logger.info(f"Basic visualizations saved to {output_dir}")

def save_q_vectors(q_vectors: np.ndarray, 
                  mask: Optional[np.ndarray] = None,
                  output_dir: str = ".",
                  q_vectors_file: str = "q_vectors.npy",
                  mask_file: str = "resolution_mask.npy"):
    """
    Save q-vectors and resolution mask for future visualizations.
    
    This helps ensure consistency between simulations and visualizations.
    
    Args:
        q_vectors: Array of q-vectors
        mask: Optional resolution mask
        output_dir: Directory to save files
        q_vectors_file: Filename for q-vectors
        mask_file: Filename for resolution mask
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to NumPy if needed
    q_vectors = ensure_numpy(q_vectors)
    
    # Save q-vectors
    q_vectors_path = os.path.join(output_dir, q_vectors_file)
    np.save(q_vectors_path, q_vectors)
    logger.info(f"Saved q-vectors with shape {q_vectors.shape} to {q_vectors_path}")
    
    # Save mask if provided
    if mask is not None:
        mask = ensure_numpy(mask)
        mask_path = os.path.join(output_dir, mask_file)
        np.save(mask_path, mask)
        logger.info(f"Saved resolution mask with {np.sum(mask)} valid points to {mask_path}")
        
    logger.info("The saved q-vectors can be used for consistent visualization")

"""
NaN processing strategies for diffuse intensity data.

This module provides various strategies for handling NaN values that occur
at lattice points in diffuse scattering data.
"""

import numpy as np
from typing import Union, Tuple, Optional
import logging
from scipy import ndimage
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


class NaNProcessor:
    """
    Strategies for handling NaN values in intensity data.
    
    NaN values typically occur at Bragg peaks (lattice points) where the
    diffuse scattering model is invalid. Different strategies are provided
    depending on the visualization requirements.
    """
    
    # Available strategies
    STRATEGIES = ['mask', 'zero', 'interpolate', 'remove', 'min', 'mean']
    
    def __init__(self):
        """Initialize the NaN processor."""
        self.last_mask = None  # Cache the last NaN mask for reuse
    
    @staticmethod
    def apply_strategy(data: np.ndarray, 
                      strategy: str = 'mask',
                      **kwargs) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Apply selected NaN handling strategy.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with potential NaN values
        strategy : str
            Strategy name: 'mask', 'zero', 'interpolate', 'remove', 'min', 'mean'
        **kwargs : dict
            Additional arguments for specific strategies
            
        Returns
        -------
        processed_data : np.ndarray or np.ma.MaskedArray
            Processed data with NaNs handled according to strategy
        """
        if strategy not in NaNProcessor.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                           f"Available: {NaNProcessor.STRATEGIES}")
        
        # Count NaNs for logging
        n_nan = np.sum(np.isnan(data))
        n_total = data.size
        nan_percent = 100 * n_nan / n_total if n_total > 0 else 0
        
        if n_nan == 0:
            logger.debug("No NaN values found in data")
            return data
        
        logger.info(f"Processing {n_nan}/{n_total} ({nan_percent:.1f}%) NaN values "
                   f"using strategy '{strategy}'")
        
        if strategy == 'mask':
            return NaNProcessor._apply_mask(data)
        elif strategy == 'zero':
            return NaNProcessor._apply_zero(data)
        elif strategy == 'interpolate':
            method = kwargs.get('method', 'nearest')
            return NaNProcessor._apply_interpolate(data, method)
        elif strategy == 'remove':
            return NaNProcessor._apply_remove(data)
        elif strategy == 'min':
            return NaNProcessor._apply_min(data)
        elif strategy == 'mean':
            return NaNProcessor._apply_mean(data)
        else:
            raise ValueError(f"Strategy '{strategy}' not implemented")
    
    @staticmethod
    def _apply_mask(data: np.ndarray) -> np.ma.MaskedArray:
        """
        Mask NaN values using numpy masked arrays.
        
        Best for: Visualization where NaNs should be transparent/hidden.
        """
        masked = np.ma.masked_invalid(data)
        logger.debug(f"Created masked array with {masked.mask.sum()} masked values")
        return masked
    
    @staticmethod
    def _apply_zero(data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with zero.
        
        Best for: Quick visualization, computation where NaNs should not contribute.
        """
        result = np.nan_to_num(data, nan=0.0)
        logger.debug("Replaced NaN values with zeros")
        return result
    
    @staticmethod  
    def _apply_min(data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with minimum of valid data.
        
        Best for: Visualization where NaNs should appear as background.
        """
        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0:
            logger.warning("No valid data points, replacing NaNs with 0")
            return np.zeros_like(data)
        
        min_val = np.min(valid_data)
        result = np.where(np.isnan(data), min_val, data)
        logger.debug(f"Replaced NaN values with minimum value {min_val:.3e}")
        return result
    
    @staticmethod
    def _apply_mean(data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with mean of valid data.
        
        Best for: Statistical analysis where NaNs should be neutral.
        """
        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0:
            logger.warning("No valid data points, replacing NaNs with 0")
            return np.zeros_like(data)
        
        mean_val = np.mean(valid_data)
        result = np.where(np.isnan(data), mean_val, data)
        logger.debug(f"Replaced NaN values with mean value {mean_val:.3e}")
        return result
    
    @staticmethod
    def _apply_interpolate(data: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """
        Fill NaN values using spatial interpolation.
        
        Best for: High-quality visualization, publication figures.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        method : str
            Interpolation method: 'nearest', 'linear', 'cubic'
        """
        if data.ndim == 1:
            return NaNProcessor._interpolate_1d(data, method)
        elif data.ndim == 2:
            return NaNProcessor._interpolate_2d(data, method)
        elif data.ndim == 3:
            return NaNProcessor._interpolate_3d(data, method)
        else:
            logger.warning(f"Cannot interpolate {data.ndim}D data, using zero fill")
            return np.nan_to_num(data, nan=0.0)
    
    @staticmethod
    def _interpolate_1d(data: np.ndarray, method: str) -> np.ndarray:
        """Interpolate NaN values in 1D array."""
        mask = np.isnan(data)
        if np.all(mask):
            logger.warning("All values are NaN, cannot interpolate")
            return np.zeros_like(data)
        
        # Get indices
        indices = np.arange(len(data))
        valid_indices = indices[~mask]
        invalid_indices = indices[mask]
        
        if len(valid_indices) < 2:
            logger.warning("Not enough valid points for interpolation, using mean fill")
            return NaNProcessor._apply_mean(data)
        
        # Interpolate
        from scipy import interpolate
        if method == 'nearest':
            f = interpolate.interp1d(valid_indices, data[~mask], 
                                    kind='nearest', fill_value='extrapolate')
        elif method == 'linear':
            f = interpolate.interp1d(valid_indices, data[~mask], 
                                    kind='linear', fill_value='extrapolate')
        elif method == 'cubic' and len(valid_indices) >= 4:
            f = interpolate.interp1d(valid_indices, data[~mask], 
                                    kind='cubic', fill_value='extrapolate')
        else:
            f = interpolate.interp1d(valid_indices, data[~mask], 
                                    kind='linear', fill_value='extrapolate')
        
        result = data.copy()
        result[mask] = f(invalid_indices)
        
        logger.debug(f"Interpolated {mask.sum()} NaN values using {method} method")
        return result
    
    @staticmethod
    def _interpolate_2d(data: np.ndarray, method: str) -> np.ndarray:
        """Interpolate NaN values in 2D array."""
        mask = np.isnan(data)
        if np.all(mask):
            logger.warning("All values are NaN, cannot interpolate")
            return np.zeros_like(data)
        
        # Create coordinate grids
        y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        
        # Get valid points
        valid_points = np.column_stack([y[~mask], x[~mask]])
        valid_values = data[~mask]
        
        # Get invalid points
        invalid_points = np.column_stack([y[mask], x[mask]])
        
        if len(valid_points) < 3:
            logger.warning("Not enough valid points for 2D interpolation, using mean fill")
            return NaNProcessor._apply_mean(data)
        
        # Interpolate
        try:
            interpolated = griddata(valid_points, valid_values, 
                                  invalid_points, method=method)
            
            # Handle any remaining NaNs with nearest neighbor
            if np.any(np.isnan(interpolated)):
                interpolated_nn = griddata(valid_points, valid_values, 
                                         invalid_points, method='nearest')
                interpolated = np.where(np.isnan(interpolated), 
                                       interpolated_nn, interpolated)
            
            result = data.copy()
            result[mask] = interpolated
            
            logger.debug(f"Interpolated {mask.sum()} NaN values using {method} method")
            return result
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, using nearest neighbor")
            return NaNProcessor._interpolate_2d(data, 'nearest')
    
    @staticmethod
    def _interpolate_3d(data: np.ndarray, method: str) -> np.ndarray:
        """Interpolate NaN values in 3D array."""
        mask = np.isnan(data)
        if np.all(mask):
            logger.warning("All values are NaN, cannot interpolate")
            return np.zeros_like(data)
        
        # For 3D, use a simpler approach: nearest neighbor filling
        # Full 3D interpolation can be very memory intensive
        if method in ['linear', 'cubic']:
            logger.info(f"3D {method} interpolation is expensive, using optimized approach")
        
        # Use scipy's distance transform for nearest neighbor
        from scipy import ndimage
        
        # Create binary mask (1 for valid, 0 for NaN)
        valid_mask = ~mask
        
        # Find nearest valid neighbor for each NaN
        # This is more memory efficient than full griddata for 3D
        indices = ndimage.distance_transform_edt(~valid_mask, 
                                                return_distances=False, 
                                                return_indices=True)
        
        # Fill NaN values with nearest valid neighbors
        result = data[tuple(indices)]
        
        logger.debug(f"Filled {mask.sum()} NaN values using nearest neighbor in 3D")
        return result
    
    @staticmethod
    def _apply_remove(data: np.ndarray) -> np.ndarray:
        """
        Remove NaN values (flatten array).
        
        Best for: Statistical analysis, not for visualization.
        
        Note: This changes the array shape!
        """
        result = data[~np.isnan(data)]
        logger.debug(f"Removed {data.size - result.size} NaN values, "
                    f"shape changed from {data.shape} to {result.shape}")
        return result
    
    @staticmethod
    def create_mask(data: np.ndarray) -> np.ndarray:
        """
        Create boolean mask for valid (non-NaN) data points.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
            
        Returns
        -------
        mask : np.ndarray, dtype=bool
            True where data is valid (not NaN), False for NaN values
        """
        return ~np.isnan(data)
    
    @staticmethod
    def interpolate_nans(data: np.ndarray, 
                        method: str = 'nearest', 
                        max_iter: int = 10) -> np.ndarray:
        """
        Advanced NaN interpolation with iterative refinement.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with NaN values
        method : str
            Base interpolation method
        max_iter : int
            Maximum iterations for refinement
            
        Returns
        -------
        interpolated : np.ndarray
            Data with NaN values interpolated
        """
        result = data.copy()
        
        for iteration in range(max_iter):
            n_nan_before = np.sum(np.isnan(result))
            if n_nan_before == 0:
                logger.debug(f"All NaNs filled after {iteration} iterations")
                break
            
            # Apply interpolation
            result = NaNProcessor._apply_interpolate(result, method)
            
            n_nan_after = np.sum(np.isnan(result))
            if n_nan_after == n_nan_before:
                logger.debug(f"No improvement after iteration {iteration}, stopping")
                break
        
        # Final check - fill any remaining with nearest
        if np.any(np.isnan(result)):
            logger.debug("Filling remaining NaNs with nearest neighbor")
            result = NaNProcessor._apply_interpolate(result, 'nearest')
        
        return result
    
    @staticmethod
    def get_statistics(data: np.ndarray) -> dict:
        """
        Get statistics about NaN values in the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
            
        Returns
        -------
        stats : dict
            Dictionary with NaN statistics
        """
        n_nan = np.sum(np.isnan(data))
        n_total = data.size
        n_valid = n_total - n_nan
        
        stats = {
            'n_nan': int(n_nan),
            'n_valid': int(n_valid),
            'n_total': int(n_total),
            'nan_fraction': float(n_nan / n_total) if n_total > 0 else 0.0,
            'nan_percentage': float(100 * n_nan / n_total) if n_total > 0 else 0.0
        }
        
        if data.ndim == 3:
            # For 3D data, also report distribution across dimensions
            nan_mask = np.isnan(data)
            stats['nan_per_h_slice'] = [int(np.sum(nan_mask[i, :, :])) 
                                        for i in range(data.shape[0])]
            stats['nan_per_k_slice'] = [int(np.sum(nan_mask[:, i, :])) 
                                        for i in range(data.shape[1])]
            stats['nan_per_l_slice'] = [int(np.sum(nan_mask[:, :, i])) 
                                        for i in range(data.shape[2])]
        
        return stats
    
    @staticmethod
    def visualize_nan_pattern(data: np.ndarray, slice_idx: Optional[int] = None):
        """
        Create a visualization of NaN pattern in the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        slice_idx : int, optional
            For 3D data, which slice to visualize (middle slice if None)
            
        Returns
        -------
        pattern : np.ndarray
            Binary array (1 for NaN, 0 for valid) for visualization
        """
        nan_mask = np.isnan(data).astype(float)
        
        if data.ndim == 3 and slice_idx is not None:
            # Return specific slice
            return nan_mask[slice_idx, :, :]
        elif data.ndim == 3:
            # Return middle slice
            mid = data.shape[0] // 2
            return nan_mask[mid, :, :]
        else:
            return nan_mask
"""
Data handler for loading and preprocessing diffuse intensity data.

This module provides a unified interface for loading intensity data from
various formats (NPZ, NPY) and preprocessing it for visualization.
"""

import os
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class IntensityDataHandler:
    """
    Unified interface for loading and preprocessing intensity data.
    
    This class handles loading from NPZ/NPY files, shape inference,
    and basic preprocessing operations. It follows the patterns from
    visualize_diffuse.py but provides a more structured API.
    """
    
    # Map dataset types to expected filenames
    DATASET_FILENAMES = {
        'np': {
            'npz': 'np_results.npz',
            'npy': 'np_diffuse_intensity.npy'
        },
        'torch': {
            'npz': 'torch_grid_results.npz',
            'npy': 'torch_diffuse_intensity.npy'
        },
        'arbq': {
            'npz': 'torch_arbq_results.npz',
            'npy': 'arb_q_diffuse_intensity.npy'
        }
    }
    
    def __init__(self, data_source: Union[str, Path, np.ndarray]):
        """
        Initialize with a data source.
        
        Parameters
        ----------
        data_source : str, Path, or np.ndarray
            Either a path to NPZ/NPY file, a dataset type ('np', 'torch', 'arbq'),
            or an existing numpy array
        """
        self.data_source = data_source
        self.intensity = None
        self.q_vectors = None
        self.map_shape = None
        self._loaded = False
        
        # If data_source is an array, store it directly
        if isinstance(data_source, np.ndarray):
            self.intensity = data_source
            self.map_shape = data_source.shape if data_source.ndim == 3 else None
            self._loaded = True
    
    def load_data(self, dataset_type: Optional[str] = None) -> Tuple[Optional[np.ndarray], 
                                                                     Optional[np.ndarray], 
                                                                     Optional[Tuple[int, int, int]]]:
        """
        Load intensity data, q_vectors, and shape information.
        
        Parameters
        ----------
        dataset_type : str, optional
            If data_source is a directory, specify dataset type ('np', 'torch', 'arbq')
            
        Returns
        -------
        q_vectors : np.ndarray or None
            Q-vector array of shape (N, 3)
        intensity : np.ndarray or None
            Intensity values (1D or 3D)
        map_shape : tuple or None
            Shape tuple (H, K, L) for 3D reconstruction
        """
        if self._loaded and isinstance(self.data_source, np.ndarray):
            return self.q_vectors, self.intensity, self.map_shape
        
        # Determine file paths based on data source
        if self.data_source in self.DATASET_FILENAMES:
            # data_source is a dataset type
            dataset_type = self.data_source
            npz_file = self.DATASET_FILENAMES[dataset_type]['npz']
            npy_file = self.DATASET_FILENAMES[dataset_type]['npy']
        elif os.path.isfile(self.data_source):
            # data_source is a file path
            if self.data_source.endswith('.npz'):
                npz_file = self.data_source
                npy_file = None
            elif self.data_source.endswith('.npy'):
                npz_file = None
                npy_file = self.data_source
            else:
                raise ValueError(f"Unsupported file type: {self.data_source}")
        else:
            # Try to use dataset_type with standard names
            if dataset_type in self.DATASET_FILENAMES:
                npz_file = self.DATASET_FILENAMES[dataset_type]['npz']
                npy_file = self.DATASET_FILENAMES[dataset_type]['npy']
            else:
                raise ValueError(f"Cannot determine files from data_source: {self.data_source}")
        
        # Try loading NPZ first
        if npz_file and os.path.exists(npz_file):
            try:
                self._load_npz(npz_file)
                if self.intensity is not None:
                    logger.info(f"Successfully loaded data from {npz_file}")
                    self._loaded = True
                    return self.q_vectors, self.intensity, self.map_shape
            except Exception as e:
                logger.warning(f"Failed to load NPZ file {npz_file}: {e}")
        
        # Fallback to NPY
        if npy_file and os.path.exists(npy_file):
            try:
                self._load_npy(npy_file)
                logger.info(f"Successfully loaded data from {npy_file} (NPY fallback)")
                self._loaded = True
                return self.q_vectors, self.intensity, self.map_shape
            except Exception as e:
                logger.error(f"Failed to load NPY file {npy_file}: {e}")
        
        # If we get here, loading failed
        logger.error(f"Could not load data from {self.data_source}")
        return None, None, None
    
    def _load_npz(self, filepath: str):
        """Load data from NPZ file."""
        with np.load(filepath) as data:
            # Load intensity
            self.intensity = data.get('intensity')
            if self.intensity is not None:
                self.intensity = np.array(self.intensity)  # Ensure it's a numpy array
                logger.debug(f"Loaded intensity with shape {self.intensity.shape}")
            
            # Load q_vectors
            self.q_vectors = data.get('q_vectors')
            if self.q_vectors is not None:
                self.q_vectors = np.array(self.q_vectors)
                logger.debug(f"Loaded q_vectors with shape {self.q_vectors.shape}")
            
            # Load map_shape
            if 'map_shape' in data:
                map_shape_raw = data['map_shape']
                # Validate map_shape
                if self._validate_map_shape(map_shape_raw):
                    self.map_shape = tuple(map_shape_raw)
                    logger.debug(f"Loaded map_shape: {self.map_shape}")
                else:
                    logger.warning(f"Invalid map_shape in NPZ: {map_shape_raw}")
            
            # Try to infer map_shape from intensity if needed
            if self.map_shape is None and self.intensity is not None and self.intensity.ndim == 3:
                self.map_shape = self.intensity.shape
                logger.debug(f"Inferred map_shape from 3D intensity: {self.map_shape}")
    
    def _load_npy(self, filepath: str):
        """Load data from NPY file (intensity only)."""
        self.intensity = np.load(filepath)
        logger.debug(f"Loaded NPY intensity with shape {self.intensity.shape}")
        
        # NPY files don't contain q_vectors or map_shape
        self.q_vectors = None
        self.map_shape = None
        
        # If intensity is 3D, use its shape
        if self.intensity.ndim == 3:
            self.map_shape = self.intensity.shape
    
    def _validate_map_shape(self, shape) -> bool:
        """Validate that map_shape is a valid 3-element tuple of integers."""
        try:
            if not isinstance(shape, (tuple, list, np.ndarray)):
                return False
            if len(shape) != 3:
                return False
            return all(isinstance(dim, (int, np.integer)) for dim in shape)
        except Exception:
            return False
    
    def get_map_shape_from_reference(self, ref_files: Optional[list] = None) -> Optional[Tuple[int, int, int]]:
        """
        Try to infer map shape from reference files.
        
        Parameters
        ----------
        ref_files : list, optional
            List of reference files to check. Defaults to standard grid files.
            
        Returns
        -------
        map_shape : tuple or None
            Inferred shape tuple (H, K, L)
        """
        if ref_files is None:
            ref_files = ["torch_grid_results.npz", "np_results.npz"]
        
        for ref_file in ref_files:
            if os.path.exists(ref_file):
                try:
                    with np.load(ref_file) as data:
                        if 'map_shape' in data:
                            shape = tuple(data['map_shape'])
                            if self._validate_map_shape(shape):
                                logger.info(f"Inferred map shape {shape} from reference: {ref_file}")
                                return shape
                        elif 'intensity' in data and data['intensity'].ndim == 3:
                            shape = data['intensity'].shape
                            logger.info(f"Inferred map shape {shape} from intensity in {ref_file}")
                            return shape
                except Exception as e:
                    logger.warning(f"Could not read shape from {ref_file}: {e}")
        
        logger.warning("Could not find any valid reference file to infer map shape")
        return None
    
    def reshape_to_3d(self, force: bool = False) -> bool:
        """
        Attempt to reshape 1D intensity data to 3D.
        
        Parameters
        ----------
        force : bool
            If True, try to infer shape from reference files if not available
            
        Returns
        -------
        success : bool
            True if reshaping was successful
        """
        if self.intensity is None:
            logger.error("No intensity data loaded")
            return False
        
        if self.intensity.ndim == 3:
            logger.debug("Intensity is already 3D")
            return True
        
        if self.intensity.ndim != 1:
            logger.warning(f"Intensity has unexpected dimensions: {self.intensity.ndim}D")
            return False
        
        # Try to get map_shape if not available
        if self.map_shape is None and force:
            self.map_shape = self.get_map_shape_from_reference()
        
        if self.map_shape is None:
            logger.warning("No map_shape available for reshaping")
            return False
        
        # Check size compatibility
        expected_size = np.prod(self.map_shape)
        if self.intensity.size != expected_size:
            logger.error(f"Size mismatch: intensity has {self.intensity.size} elements, "
                        f"but map_shape {self.map_shape} expects {expected_size}")
            return False
        
        # Perform reshape
        try:
            self.intensity = self.intensity.reshape(self.map_shape)
            logger.info(f"Successfully reshaped intensity to {self.map_shape}")
            return True
        except ValueError as e:
            logger.error(f"Failed to reshape intensity: {e}")
            return False
    
    def preprocess(self, nan_strategy: str = 'mask') -> np.ndarray:
        """
        Apply preprocessing including NaN handling.
        
        Parameters
        ----------
        nan_strategy : str
            Strategy for handling NaN values ('mask', 'zero', 'interpolate', 'remove')
            Note: Actual implementation deferred to NaNProcessor class
            
        Returns
        -------
        processed_intensity : np.ndarray
            Preprocessed intensity data
        """
        if self.intensity is None:
            raise ValueError("No intensity data loaded")
        
        # For now, just return the intensity
        # NaN processing will be handled by NaNProcessor class
        logger.info(f"Preprocessing with nan_strategy='{nan_strategy}' "
                   "(actual processing will be done by NaNProcessor)")
        return self.intensity
    
    def subsample(self, factor: int = 1) -> np.ndarray:
        """
        Reduce data resolution by subsampling.
        
        Parameters
        ----------
        factor : int
            Subsampling factor (1 = no subsampling, 2 = half resolution, etc.)
            
        Returns
        -------
        subsampled : np.ndarray
            Subsampled intensity data
        """
        if self.intensity is None:
            raise ValueError("No intensity data loaded")
        
        if factor == 1:
            return self.intensity
        
        if self.intensity.ndim == 3:
            # Subsample 3D data
            subsampled = self.intensity[::factor, ::factor, ::factor]
            logger.info(f"Subsampled from {self.intensity.shape} to {subsampled.shape}")
            return subsampled
        elif self.intensity.ndim == 1:
            # Subsample 1D data
            subsampled = self.intensity[::factor]
            logger.info(f"Subsampled from {self.intensity.shape} to {subsampled.shape}")
            return subsampled
        else:
            logger.warning(f"Cannot subsample {self.intensity.ndim}D data")
            return self.intensity
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute statistics of the intensity data.
        
        Returns
        -------
        stats : dict
            Dictionary containing min, max, mean, std, and percentiles
        """
        if self.intensity is None:
            raise ValueError("No intensity data loaded")
        
        # Filter out NaN values for statistics
        valid_data = self.intensity[~np.isnan(self.intensity)]
        
        if valid_data.size == 0:
            logger.warning("No valid (non-NaN) data points")
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'percentile_01': np.nan,
                'percentile_25': np.nan,
                'percentile_50': np.nan,
                'percentile_75': np.nan,
                'percentile_99': np.nan,
                'n_valid': 0,
                'n_nan': self.intensity.size
            }
        
        stats = {
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'percentile_01': float(np.percentile(valid_data, 1)),
            'percentile_25': float(np.percentile(valid_data, 25)),
            'percentile_50': float(np.percentile(valid_data, 50)),
            'percentile_75': float(np.percentile(valid_data, 75)),
            'percentile_99': float(np.percentile(valid_data, 99)),
            'n_valid': int(valid_data.size),
            'n_nan': int(np.sum(np.isnan(self.intensity)))
        }
        
        logger.info(f"Statistics: {stats['n_valid']} valid points, "
                   f"{stats['n_nan']} NaN points, "
                   f"range [{stats['min']:.3e}, {stats['max']:.3e}]")
        
        return stats
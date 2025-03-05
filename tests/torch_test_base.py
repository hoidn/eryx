"""
Base test case class for PyTorch component testing.

This module provides a base test case class for testing PyTorch implementations
against NumPy ground truth in diffuse scattering simulations.
"""

import unittest
import numpy as np
import torch
import warnings
from typing import Dict, Any, Tuple, List, Optional, Union

from tests.torch_test_utils import TensorComparison, ModelState


class TorchComponentTestCase(unittest.TestCase):
    """
    Base test case for PyTorch component testing.
    
    This class provides common functionality for testing PyTorch implementations
    against NumPy ground truth, including model creation, state capture and injection,
    and tensor comparison.
    """
    
    def setUp(self):
        """Set up test environment with default configuration."""
        # Set device (CPU for consistent testing)
        self.device = torch.device('cpu')
        
        # Initialize utilities
        self.tensor_comparison = TensorComparison()
        
        # Set default test parameters
        self.default_test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],  # Smaller grid for faster testing
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Set up suppression for known warnings
        warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*missing entity_type.*")
        
        # Set default tolerances
        self.rtol = 1e-5
        self.atol = 1e-8
    
    def create_models(self, params: Optional[Dict[str, Any]] = None, 
                     device: Optional[torch.device] = None) -> Tuple[Any, Any]:
        """
        Create NumPy and PyTorch models with identical parameters.
        
        Args:
            params: Dictionary of model parameters, or None for defaults
            device: Device for PyTorch model, or None for self.device
            
        Returns:
            Tuple of (numpy_model, torch_model)
        """
        # Import model classes
        from eryx import models
        from eryx import models_torch
        
        NumpyOnePhonon = models.OnePhonon
        TorchOnePhonon = models_torch.OnePhonon
        
        # Use provided params or default test parameters
        if params is None:
            params = self.default_test_params.copy()
        
        # Use provided device or default
        if device is None:
            device = self.device
        
        # Create NumPy model
        np_model = NumpyOnePhonon(**params)
        
        # Create PyTorch model with device parameter
        torch_params = params.copy()
        torch_params['device'] = device
        torch_model = TorchOnePhonon(**torch_params)
        
        # Store models as instance attributes
        self.np_model = np_model
        self.torch_model = torch_model
        
        return np_model, torch_model
    
    def assert_tensors_equal(self, np_array: np.ndarray, 
                            torch_tensor: torch.Tensor, 
                            **kwargs) -> None:
        """
        Assert that NumPy array and PyTorch tensor are equivalent.
        
        Args:
            np_array: NumPy array to compare
            torch_tensor: PyTorch tensor to compare
            **kwargs: Additional arguments to pass to assert_tensors_equal
        """
        # Set default tolerances if not provided
        if 'rtol' not in kwargs:
            kwargs['rtol'] = self.rtol
        if 'atol' not in kwargs:
            kwargs['atol'] = self.atol
        
        # Use TensorComparison utility
        TensorComparison.assert_tensors_equal(np_array, torch_tensor, **kwargs)
    
    def capture_model_state(self, model: Any, **kwargs) -> Dict[str, Any]:
        """
        Capture model state.
        
        Args:
            model: Model instance to capture state from
            **kwargs: Additional arguments to pass to capture_model_state
            
        Returns:
            Dictionary with captured state
        """
        # Use ModelState utility
        return ModelState.capture_model_state(model, **kwargs)
    
    def inject_model_state(self, model: Any, state: Dict[str, Any], **kwargs) -> None:
        """
        Inject state into model.
        
        Args:
            model: Model instance to inject state into
            state: State dictionary from capture_model_state
            **kwargs: Additional arguments to pass to inject_model_state
        """
        # Use ModelState utility
        ModelState.inject_model_state(model, state, **kwargs)
    
    def prepare_test_environment(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Prepare test environment with models and common setup.
        
        Args:
            params: Dictionary of model parameters, or None for defaults
        """
        # Create models
        self.create_models(params)
        
        # Ensure both models have computed key attributes
        if not hasattr(self.np_model, 'kvec') or self.np_model.kvec is None:
            self.np_model._build_kvec_Brillouin()
            
        if not hasattr(self.torch_model, 'kvec') or self.torch_model.kvec is None:
            self.torch_model._build_kvec_Brillouin()
    
    def run_component_test(self, test_func: callable, *args, **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        Run a component test with detailed reporting.
        
        Args:
            test_func: Test function to run
            *args: Arguments to pass to test function
            **kwargs: Keyword arguments to pass to test function
            
        Returns:
            Tuple of (bool success, dict metrics)
        """
        # Run the test function
        success, metrics = test_func(*args, **kwargs)
        
        # Add test function name to metrics
        metrics['test_name'] = test_func.__name__
        
        # Print detailed metrics for debugging
        if not success:
            print(f"\nTest {test_func.__name__} failed:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        
        return success, metrics

import unittest
import os
import torch
import numpy as np
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping
from eryx.autotest.torch_testing import TorchTesting
from typing import Dict, List, Tuple, Any, Optional, Type

class TestBase(unittest.TestCase):
    def setUp(self):
        # Initialize logger, function mapping, and torch testing
        self.logger = Logger()
        self.function_mapping = FunctionMapping()
        
        # Set device from environment variable or use default
        device_name = os.environ.get('TORCH_TEST_DEVICE', 'cpu')
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu')
        
        # Initialize torch testing
        self.torch_testing = TorchTesting(self.logger, self.function_mapping)
        # Store device for later use
        
        # Set tolerances from environment variables or use defaults
        self.rtol = float(os.environ.get('TORCH_TEST_RTOL', '1e-5'))
        self.atol = float(os.environ.get('TORCH_TEST_ATOL', '1e-8'))
        
        # Enable/disable log verification
        self.verify_logs = os.environ.get('TORCH_VERIFY_LOGS', 'True').lower() == 'true'
        
    def _load_state(self, module_name: str, class_name: str, method_name: str, before: bool = True) -> Dict:
        # Construct log path for state
        state_type = "_state_before_" if before else "_state_after_"
        log_path = f"logs/{module_name}.{class_name}.{state_type}{method_name}.log"
        
        # Load state log
        state = self.logger.loadStateLog(log_path)
        if not state and self.verify_logs:
            self.fail(f"State log not found: {log_path}")
        return state
        
    def _init_from_state(self, torch_class: Type, state: Dict) -> Any:
        # Initialize object from state dictionary
        model = self.torch_testing.initializeFromState(torch_class, state)
        
        # Set device attribute if the model has one
        if hasattr(model, 'device'):
            model.device = self.device
            
            # Move tensors to the correct device
            for attr_name in dir(model):
                if attr_name.startswith('_'):
                    continue
                    
                try:
                    attr = getattr(model, attr_name)
                    if isinstance(attr, torch.Tensor):
                        setattr(model, attr_name, attr.to(self.device))
                except Exception:
                    pass
                    
        return model
        
    def _compare_states(self, expected: Dict, actual: Dict, 
                       attr_tolerances: Optional[Dict] = None) -> bool:
        # Use torch_testing.compareStates with appropriate tolerances
        attr_tolerances = attr_tolerances or {}
        return self.torch_testing.compareStates(expected, actual, attr_tolerances)
        
    def _get_method_args(self, module_name: str, method_name: str) -> Tuple[List, Dict]:
        # Find log file for the method
        log_path = f"logs/{module_name}.{method_name}.log"
        
        try:
            # Load log data and extract args/kwargs from first call
            logs = self.logger.loadLog(log_path)
            if not logs:
                return [], {}
                
            call_log = logs[0]  # Get first call
            args = self.logger.serializer.deserialize(call_log["args"])
            kwargs = self.logger.serializer.deserialize(call_log["kwargs"])
            
            # Skip 'self' for method calls
            if len(args) > 0 and hasattr(args[0], '__dict__'):
                args = args[1:]
            
            # Convert NumPy arrays to PyTorch tensors
            args_tensor = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    try:
                        args_tensor.append(torch.tensor(arg, device=self.device))
                    except Exception:
                        # Fall back to CPU if conversion fails
                        args_tensor.append(torch.tensor(arg))
                else:
                    args_tensor.append(arg)
                    
            kwargs_tensor = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    try:
                        kwargs_tensor[k] = torch.tensor(v, device=self.device)
                    except Exception:
                        # Fall back to CPU if conversion fails
                        kwargs_tensor[k] = torch.tensor(v)
                else:
                    kwargs_tensor[k] = v
            
            return args_tensor, kwargs_tensor
        except Exception as e:
            if self.verify_logs:
                self.fail(f"Failed to extract method arguments: {e}")
            return [], {}
        
    def _verify_tensor(self, tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                      requires_grad: bool = True, check_gradient: bool = False,
                      dtype: Optional[torch.dtype] = None):
        # Verify tensor is not None
        self.assertIsNotNone(tensor, "Tensor should not be None")
        
        # Verify shape if provided
        if expected_shape:
            self.assertEqual(tensor.shape, expected_shape, 
                           f"Expected shape {expected_shape}, got {tensor.shape}")
        
        # Verify dtype if provided
        if dtype:
            self.assertEqual(tensor.dtype, dtype,
                           f"Expected dtype {dtype}, got {tensor.dtype}")
        
        # Verify requires_grad for floating point tensors
        if requires_grad and tensor.dtype.is_floating_point:
            self.assertTrue(tensor.requires_grad, "Tensor should require gradients")
        
        # Verify gradient flow if requested
        if check_gradient and tensor.requires_grad:
            # Create a simple scalar loss
            loss = tensor.sum()
            loss.backward()
            
            # Verify gradient exists and is not all zeros
            self.assertIsNotNone(tensor.grad, "No gradient computed")
            self.assertFalse(torch.all(tensor.grad == 0), "Gradient is all zeros")
            
            # Reset gradient for future tests
            tensor.grad = None
        
    def _get_tolerances(self, method_name: str) -> Dict:
        # Return appropriate tolerances based on method name
        # Default to standard tolerances for most methods
        tolerances = {}
        
        # Higher tolerances for eigendecomposition methods
        if method_name in ['compute_gnm_phonons', 'compute_covariance_matrix']:
            tolerances = {
                'V': {'rtol': 1e-4, 'atol': 1e-6},
                'Winv': {'rtol': 1e-4, 'atol': 1e-6},
                'covar': {'rtol': 1e-4, 'atol': 1e-6}
            }
        # For disorder calculations
        elif method_name in ['apply_disorder']:
            tolerances = {'__all__': {'rtol': 1e-3, 'atol': 1e-5}}
        
        return tolerances
        
    def verify_required_logs(self, module_name: str, method_name: str, 
                           required_attrs: Optional[List[str]] = None):
        # Verify logs exist and contain required attributes
        import subprocess
        required_attrs = required_attrs or []
        attr_str = ",".join(required_attrs)
        
        result = subprocess.run(
            ["python", "scripts/verify_logs.py", 
             "--log-dir", "logs", 
             "--required-attrs", attr_str],
            capture_output=True, text=True
        )
        
        if method_name not in result.stdout:
            self.fail(f"Log for {method_name} not found in verification output")
            
        if "Invalid pairs" in result.stdout and method_name in result.stdout:
            self.fail(f"Invalid log pairs found for {method_name}")

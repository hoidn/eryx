"""
Testing utilities for PyTorch implementation.

This module extends the autotest framework with PyTorch-specific testing
capabilities, including tensor comparison, gradient checking, and
PyTorch-NumPy conversion for testing.
"""

import numpy as np
import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type, Set
from .testing import Testing
from .logger import Logger
from .functionmapping import FunctionMapping

class TorchTesting(Testing):
    """
    Extended testing framework for PyTorch implementations.
    
    This class extends the Testing class with PyTorch-specific utilities,
    including tensor comparison and gradient checking.
    """
    
    def __init__(self, logger: Logger, function_mapping: FunctionMapping, 
                rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize the PyTorch testing framework.
        
        Args:
            logger: Logger instance for test logging
            function_mapping: FunctionMapping instance for function lookup
            rtol: Relative tolerance for tensor comparison
            atol: Absolute tolerance for tensor comparison
        """
        super().__init__(logger, function_mapping)
        self.rtol = rtol
        self.atol = atol
        self.default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def testTorchCallable(self, log_path_prefix: str, torch_func: Callable) -> bool:
        """
        Test a PyTorch function against NumPy implementation.
        
        Args:
            log_path_prefix: Path prefix for log files
            torch_func: PyTorch function to test
            
        Returns:
            True if test passes, False otherwise
        """
        log_files = self.logger.searchLogDirectory(log_path_prefix)
        for log_file in log_files:
            logs = self.logger.loadLog(log_file)
            for i in range(len(logs) // 2):
                args = logs[2 * i]['args']
                kwargs = logs[2 * i]['kwargs']
                expected_output = logs[2 * i + 1]['result']
                try:
                    # Deserialize and convert to PyTorch
                    deserialized_args = self._numpy_to_torch(self.logger.serializer.deserialize(args))
                    deserialized_kwargs = self._numpy_to_torch(self.logger.serializer.deserialize(kwargs))
                    deserialized_expected_output = self.logger.serializer.deserialize(expected_output)
                    
                    # Run PyTorch function
                    actual_output = torch_func(*deserialized_args, **deserialized_kwargs)
                    
                    # Convert PyTorch output to NumPy for comparison
                    numpy_actual_output = self._torch_to_numpy(actual_output)
                    
                    # Compare outputs
                    if not self._compare_outputs(numpy_actual_output, deserialized_expected_output):
                        print(f"Test failed for {log_file}")
                        return False
                except Exception as e:
                    print(f"Error testing PyTorch function: {e}")
                    return False
        return True
    
    def testTorchCallableWithState(self, log_path_prefix: str, torch_class: Type, 
                                 method_name: str, *args, **kwargs) -> bool:
        """
        Test a PyTorch method using state-based testing.
        
        Args:
            log_path_prefix: Path prefix for state log files
            torch_class: PyTorch class to test
            method_name: Name of the method to test
            *args: Additional arguments to pass to the method
            **kwargs: Additional keyword arguments to pass to the method
            
        Returns:
            True if test passes, False otherwise
        """
        # Find before/after state log files
        before_log = f"{log_path_prefix}._state_before_{method_name}.log"
        after_log = f"{log_path_prefix}._state_after_{method_name}.log"
        
        try:
            # Load before state
            before_state = self.logger.loadStateLog(before_log)
            if not before_state:
                print(f"Before state log not found or empty: {before_log}")
                return False
            
            # Load expected after state
            expected_after_state = self.logger.loadStateLog(after_log)
            if not expected_after_state:
                print(f"After state log not found or empty: {after_log}")
                return False
            
            # Initialize object from before state
            obj = self.initializeFromState(torch_class, before_state)
            
            # Get the method to call
            method = getattr(obj, method_name)
            
            # Call the method
            method(*args, **kwargs)
            
            # Capture current state
            current_state = {}
            for key, value in obj.__dict__.items():
                current_state[key] = value
            
            # Compare resulting state with expected after state
            result = self.compareStates(expected_after_state, current_state)
            
            if not result:
                print(f"State mismatch after calling {method_name}")
                return False
                
            return True
        except Exception as e:
            print(f"Error in state-based testing: {e}")
            return False
    
    def initializeFromState(self, torch_class: Type, state_data: Dict[str, Any], 
                          device: Optional[torch.device] = None) -> Any:
        """
        Initialize a PyTorch object from state data.
        
        Args:
            torch_class: PyTorch class to initialize
            state_data: State data dictionary
            device: PyTorch device to place tensors on
            
        Returns:
            Initialized PyTorch object
        """
        if device is None:
            device = self.default_device
        
        # Create empty instance
        obj = torch_class.__new__(torch_class)
        
        # Initialize each attribute
        for key, value in state_data.items():
            # First deserialize if the value is bytes
            if isinstance(value, bytes):
                try:
                    value = self.logger.serializer.deserialize(value)
                except Exception as e:
                    print(f"Warning: Could not deserialize {key}: {e}")
                    continue
                    
            if isinstance(value, np.ndarray):
                # Convert NumPy arrays to PyTorch tensors
                tensor = torch.tensor(value, device=device)
                if tensor.dtype.is_floating_point:
                    tensor.requires_grad_(True)
                setattr(obj, key, tensor)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                if any(isinstance(v, np.ndarray) for v in value.values()):
                    # Convert NumPy arrays in dict to tensors
                    tensor_dict = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            tensor = torch.tensor(v, device=device)
                            if tensor.dtype.is_floating_point:
                                tensor.requires_grad_(True)
                            tensor_dict[k] = tensor
                        else:
                            tensor_dict[k] = v
                    setattr(obj, key, tensor_dict)
                else:
                    setattr(obj, key, value)
            else:
                setattr(obj, key, value)
        
        # If the class has an __init__ method, check if we need to call it
        if hasattr(torch_class, '__init__') and callable(getattr(torch_class, '__init__')):
            # Check if __init__ has required parameters beyond self
            init_params = inspect.signature(torch_class.__init__).parameters
            if len(init_params) > 1:
                # Check if any required parameters are missing from state_data
                required_params = [p for p in list(init_params.keys())[1:] 
                                 if init_params[p].default == inspect.Parameter.empty]
                if not all(p in state_data for p in required_params):
                    # Call a minimal initialization if needed
                    try:
                        obj.__init__()
                    except Exception as e:
                        print(f"Warning: Could not call __init__: {e}")
        
        return obj
    
    def compareStates(self, expected_state: Dict[str, Any], actual_state: Dict[str, Any], 
                    attr_tolerances: Optional[Dict[str, Dict[str, float]]] = None) -> bool:
        """
        Compare expected and actual states with appropriate tolerances.
        
        Args:
            expected_state: Expected state dictionary
            actual_state: Actual state dictionary
            attr_tolerances: Dictionary mapping attribute names to tolerance dictionaries
                             with 'rtol' and 'atol' keys
            
        Returns:
            True if states match within tolerances, False otherwise
        """
        # Set default tolerances
        default_tolerance = {'rtol': self.rtol, 'atol': self.atol}
        attr_tolerances = attr_tolerances or {}
        
        # Check for missing attributes
        missing_attrs = set(expected_state.keys()) - set(actual_state.keys())
        if missing_attrs:
            print(f"Missing attributes in actual state: {missing_attrs}")
            return False
        
        # Compare each attribute
        for key in expected_state:
            expected = expected_state[key]
            actual = actual_state[key]
            
            # Deserialize if values are bytes
            if isinstance(expected, bytes):
                try:
                    expected = self.logger.serializer.deserialize(expected)
                except Exception as e:
                    print(f"Warning: Could not deserialize expected {key}: {e}")
                    return False
                    
            if isinstance(actual, bytes):
                try:
                    actual = self.logger.serializer.deserialize(actual)
                except Exception as e:
                    print(f"Warning: Could not deserialize actual {key}: {e}")
                    return False
            
            # Get tolerance for this attribute
            tolerance = attr_tolerances.get(key, default_tolerance)
            rtol = tolerance.get('rtol', self.rtol)
            atol = tolerance.get('atol', self.atol)
            
            # Convert PyTorch tensors to NumPy for comparison
            if hasattr(actual, 'detach') and callable(getattr(actual, 'detach')):
                actual = actual.detach().cpu().numpy()
                
            if hasattr(expected, 'detach') and callable(getattr(expected, 'detach')):
                expected = expected.detach().cpu().numpy()
            
            # Compare based on type
            if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
                # Compare shapes
                if expected.shape != actual.shape:
                    print(f"Shape mismatch for {key}: {expected.shape} vs {actual.shape}")
                    return False
                
                # Handle NaN values
                nan_mask_expected = np.isnan(expected)
                nan_mask_actual = np.isnan(actual)
                if not np.array_equal(nan_mask_expected, nan_mask_actual):
                    print(f"NaN pattern mismatch for {key}")
                    return False
                
                # Compare non-NaN values
                non_nan_mask = ~nan_mask_expected
                if np.any(non_nan_mask):
                    if not np.allclose(expected[non_nan_mask], actual[non_nan_mask], 
                                     rtol=rtol, atol=atol):
                        print(f"Value mismatch for {key}")
                        return False
            
            elif isinstance(expected, dict) and isinstance(actual, dict):
                # Compare dictionaries recursively
                if set(expected.keys()) != set(actual.keys()):
                    print(f"Key mismatch for dict {key}: {set(expected.keys())} vs {set(actual.keys())}")
                    return False
                
                # Create nested tolerances for dict attributes
                nested_tolerances = {}
                for k in expected.keys():
                    nested_key = f"{key}.{k}"
                    if nested_key in attr_tolerances:
                        nested_tolerances[k] = attr_tolerances[nested_key]
                
                if not self.compareStates(expected, actual, nested_tolerances):
                    print(f"Dict value mismatch for {key}")
                    return False
            
            elif isinstance(expected, list) and isinstance(actual, list):
                # Compare lists
                if len(expected) != len(actual):
                    print(f"Length mismatch for list {key}: {len(expected)} vs {len(actual)}")
                    return False
                
                # Convert lists to numpy arrays if they contain numeric values
                try:
                    if all(isinstance(x, (int, float)) for x in expected) and \
                       all(isinstance(x, (int, float)) for x in actual):
                        if not np.allclose(np.array(expected), np.array(actual), 
                                         rtol=rtol, atol=atol):
                            print(f"Value mismatch for list {key}")
                            return False
                    else:
                        # Compare elements individually
                        for i, (e, a) in enumerate(zip(expected, actual)):
                            if isinstance(e, np.ndarray) and isinstance(a, np.ndarray):
                                if not np.allclose(e, a, rtol=rtol, atol=atol):
                                    print(f"Value mismatch for {key}[{i}]")
                                    return False
                            elif e != a:
                                print(f"Value mismatch for {key}[{i}]")
                                return False
                except Exception as e:
                    print(f"Error comparing lists for {key}: {e}")
                    return False
            
            elif type(expected) != type(actual):
                print(f"Type mismatch for {key}: {type(expected)} vs {type(actual)}")
                return False
            
            elif expected != actual:
                # Direct comparison for other types
                print(f"Value mismatch for {key}: {expected} vs {actual}")
                return False
        
        return True
    
    def check_state_gradients(self, obj: Any, attr_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Check gradient flow through object state attributes.
        
        Args:
            obj: Object to check gradients for
            attr_names: List of attribute names to check, or None to check all tensor attributes
            
        Returns:
            Dictionary mapping attribute names to gradient status
        """
        result = {}
        
        # If no specific attributes provided, check all tensor attributes
        if attr_names is None:
            attr_names = [attr for attr in dir(obj) 
                        if not attr.startswith('_') and 
                        hasattr(getattr(obj, attr), 'requires_grad')]
        
        for attr_name in attr_names:
            if hasattr(obj, attr_name):
                attr = getattr(obj, attr_name)
                
                # Check if attribute is a tensor with requires_grad
                if hasattr(attr, 'requires_grad'):
                    result[attr_name] = attr.requires_grad
                
                # Check if attribute is a dictionary of tensors
                elif isinstance(attr, dict):
                    for k, v in attr.items():
                        if hasattr(v, 'requires_grad'):
                            result[f"{attr_name}.{k}"] = v.requires_grad
                
                # Check if attribute is a list of tensors
                elif isinstance(attr, list):
                    for i, v in enumerate(attr):
                        if hasattr(v, 'requires_grad'):
                            result[f"{attr_name}[{i}]"] = v.requires_grad
        
        return result
    
    def _numpy_to_torch(self, obj: Any) -> Any:
        """
        Convert NumPy arrays to PyTorch tensors recursively.
        
        Args:
            obj: Input object potentially containing NumPy arrays
            
        Returns:
            Object with NumPy arrays converted to PyTorch tensors
        """
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj.copy())
        elif isinstance(obj, list):
            return [self._numpy_to_torch(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_torch(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._numpy_to_torch(v) for k, v in obj.items()}
        else:
            return obj
    
    def _torch_to_numpy(self, obj: Any) -> Any:
        """
        Convert PyTorch tensors to NumPy arrays recursively.
        
        Args:
            obj: Input object potentially containing PyTorch tensors
            
        Returns:
            Object with PyTorch tensors converted to NumPy arrays
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, list):
            return [self._torch_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._torch_to_numpy(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._torch_to_numpy(v) for k, v in obj.items()}
        else:
            return obj
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """
        Compare outputs with tolerance for numerical differences.
        
        Args:
            actual: Actual output
            expected: Expected output
            
        Returns:
            True if outputs match within tolerance, False otherwise
        """
        if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            # Compare arrays with tolerance
            if actual.shape != expected.shape:
                print(f"Shape mismatch: {actual.shape} vs {expected.shape}")
                return False
            
            # Handle NaN values
            nan_mask_actual = np.isnan(actual)
            nan_mask_expected = np.isnan(expected)
            if not np.array_equal(nan_mask_actual, nan_mask_expected):
                print("NaN pattern mismatch")
                return False
            
            # Compare non-NaN values
            non_nan_mask = ~nan_mask_actual
            if np.any(non_nan_mask):
                return np.allclose(actual[non_nan_mask], expected[non_nan_mask], 
                                 rtol=self.rtol, atol=self.atol)
            return True
        
        elif isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                print(f"Length mismatch: {len(actual)} vs {len(expected)}")
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, tuple) and isinstance(expected, tuple):
            if len(actual) != len(expected):
                print(f"Length mismatch: {len(actual)} vs {len(expected)}")
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        elif isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                print(f"Key mismatch: {set(actual.keys())} vs {set(expected.keys())}")
                return False
            return all(self._compare_outputs(actual[k], expected[k]) for k in actual)
        
        else:
            # Direct comparison for other types
            return actual == expected
    
    def check_gradients(self, torch_func: Callable, inputs: List[torch.Tensor], 
                       eps: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
        """
        Check gradients of PyTorch function using finite differences.
        
        Args:
            torch_func: PyTorch function to check
            inputs: List of input tensors
            eps: Step size for finite differences
            
        Returns:
            Tuple containing:
                - True if gradients match, False otherwise
                - Dictionary with gradient statistics
        """
        # Clone inputs and set requires_grad=True
        grad_inputs = []
        for inp in inputs:
            grad_input = inp.clone().detach()
            grad_input.requires_grad_(True)
            grad_inputs.append(grad_input)
        
        # Compute forward pass
        output = torch_func(*grad_inputs)
        
        # If output is not a scalar, sum it to get a scalar
        if output.numel() > 1:
            output = output.sum()
        
        # Compute backward pass
        output.backward()
        
        # Get analytical gradients
        analytical_grads = [inp.grad.clone() for inp in grad_inputs]
        
        # Compute numerical gradients with finite differences
        numerical_grads = []
        for i, inp in enumerate(grad_inputs):
            numerical_grad = torch.zeros_like(inp)
            
            # Flatten the input for easier iteration
            flat_input = inp.flatten()
            flat_grad = numerical_grad.flatten()
            
            # Compute gradient for each element
            for j in range(flat_input.numel()):
                # Forward difference
                flat_input[j] += eps
                forward_output = torch_func(*grad_inputs).sum().item()
                
                # Backward difference
                flat_input[j] -= 2 * eps
                backward_output = torch_func(*grad_inputs).sum().item()
                
                # Central difference
                flat_grad[j] = (forward_output - backward_output) / (2 * eps)
                
                # Restore original value
                flat_input[j] += eps
            
            numerical_grads.append(numerical_grad)
        
        # Compare analytical and numerical gradients
        stats = {}
        all_match = True
        
        for i, (analytical, numerical) in enumerate(zip(analytical_grads, numerical_grads)):
            # Compute relative error
            abs_diff = torch.abs(analytical - numerical)
            abs_analytical = torch.abs(analytical)
            abs_numerical = torch.abs(numerical)
            
            # Avoid division by zero
            max_abs = torch.max(abs_analytical, abs_numerical)
            max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
            
            rel_error = abs_diff / max_abs
            
            # Compute statistics
            max_rel_error = rel_error.max().item()
            mean_rel_error = rel_error.mean().item()
            
            # Check if gradients match within tolerance
            match = (rel_error < self.rtol).all().item()
            all_match = all_match and match
            
            # Store statistics
            stats[f"input_{i}_max_rel_error"] = max_rel_error
            stats[f"input_{i}_mean_rel_error"] = mean_rel_error
            stats[f"input_{i}_match"] = match
        
        stats["all_match"] = all_match
        
        return all_match, stats
    
    def create_tensor_test_case(self, log_path_prefix: str, 
                              torch_func: Callable, numpy_func: Callable) -> Callable:
        """
        Create a test case for comparing PyTorch and NumPy implementations.
        
        Args:
            log_path_prefix: Path prefix for log files
            torch_func: PyTorch function to test
            numpy_func: NumPy function to compare against
            
        Returns:
            Test function that can be called directly
        """
        def test_case():
            # Find log files
            log_files = self.logger.searchLogDirectory(log_path_prefix)
            if not log_files:
                print(f"No log files found for {log_path_prefix}")
                return False
            
            # Process each log file
            for log_file in log_files:
                logs = self.logger.loadLog(log_file)
                for i in range(len(logs) // 2):
                    # Get inputs and expected output
                    args_bytes = logs[2 * i]['args']
                    kwargs_bytes = logs[2 * i]['kwargs']
                    
                    # Deserialize inputs
                    args = self.logger.serializer.deserialize(args_bytes)
                    kwargs = self.logger.serializer.deserialize(kwargs_bytes)
                    
                    # Run NumPy function to get expected output
                    expected_output = numpy_func(*args, **kwargs)
                    
                    # Convert inputs to PyTorch tensors
                    torch_args = self._numpy_to_torch(args)
                    torch_kwargs = self._numpy_to_torch(kwargs)
                    
                    # Run PyTorch function
                    torch_output = torch_func(*torch_args, **torch_kwargs)
                    
                    # Convert PyTorch output to NumPy
                    numpy_output = self._torch_to_numpy(torch_output)
                    
                    # Compare outputs
                    if not self._compare_outputs(numpy_output, expected_output):
                        print(f"Test failed for {log_file}")
                        return False
            
            return True
        
        return test_case

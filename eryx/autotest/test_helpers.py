import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Type, Tuple

def load_test_state(logger, module_name: str, class_name: str, method_name: str, before: bool = True) -> Dict[str, Any]:
    """
    Load state data from the correct log file.
    
    Args:
        logger: Logger instance from eryx.autotest.logger
        module_name: Module name (e.g. 'eryx.models')
        class_name: Class name (e.g. 'OnePhonon')
        method_name: Method name (e.g. '_build_kvec_Brillouin')
        before: If True, load before state, otherwise load after state
        
    Returns:
        State dictionary from the log file
        
    Example:
        state = load_test_state(self.logger, 'eryx.models', 'OnePhonon', '_build_kvec_Brillouin')
    """
    state_type = 'before' if before else 'after'
    log_path = f"logs/{module_name}.{class_name}._state_{state_type}_{method_name}.log"
    
    # Check if log exists
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    # Load state data
    return logger.loadStateLog(log_path)

def build_test_object(torch_class: Type, state_data: Dict[str, Any], device: Optional[torch.device] = None) -> Any:
    """
    Build a properly initialized test object from state data.
    
    Args:
        torch_class: PyTorch class to instantiate
        state_data: State dictionary from log file
        device: Optional device override
        
    Returns:
        Initialized instance with proper structure
        
    Example:
        model = build_test_object(OnePhonon, before_state, device=torch.device('cpu'))
    """
    from eryx.autotest.state_builder import StateBuilder
    builder = StateBuilder(device=device)
    return builder.build(torch_class, state_data)

def verify_gradient_flow(tensor: torch.Tensor, source_tensor: torch.Tensor, rtol: float = 1e-5) -> bool:
    """
    Verify gradient flow from tensor to source_tensor.
    
    Args:
        tensor: Output tensor
        source_tensor: Input tensor where gradients should flow to
        rtol: Relative tolerance for gradient verification
        
    Returns:
        True if gradients flow correctly, False otherwise
        
    Example:
        loss = torch.sum(model.kvec)
        loss.backward()
        assert verify_gradient_flow(model.kvec, model.model.A_inv)
    """
    # Check tensor requires grad
    if not tensor.requires_grad:
        print("Output tensor doesn't require gradients")
        return False
    
    # Check source tensor requires grad
    if not source_tensor.requires_grad:
        print("Source tensor doesn't require gradients")
        return False
    
    # Create loss and backpropagate
    loss = torch.sum(tensor)
    loss.backward()
    
    # Check if source_tensor has gradients
    if source_tensor.grad is None:
        print("No gradients flowed to source tensor")
        return False
    
    # Check if gradients are non-zero
    grad_sum = torch.sum(torch.abs(source_tensor.grad))
    if grad_sum < rtol:
        print(f"Gradients too small: {grad_sum.item()}")
        return False
    
    return True

def verify_tensor_matches(tensor: torch.Tensor, expected: np.ndarray, 
                         rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Verify tensor values match expected values within tolerance.
    
    Args:
        tensor: PyTorch tensor to verify
        expected: NumPy array with expected values
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if tensor matches expected values, False otherwise
        
    Example:
        assert verify_tensor_matches(model.kvec, expected_kvec)
    """
    # Convert tensor to numpy
    tensor_np = tensor.detach().cpu().numpy()
    
    # Check shapes match
    if tensor_np.shape != expected.shape:
        print(f"Shape mismatch: {tensor_np.shape} vs {expected.shape}")
        return False
    
    # Compare values with tolerance
    return np.allclose(tensor_np, expected, rtol=rtol, atol=atol)

def run_state_based_test(test_obj, torch_class: Type, module_name: str, 
                        class_name: str, method_name: str) -> Tuple[bool, str]:
    """
    Run a standard state-based test.
    
    Args:
        test_obj: Test case instance (with logger, torch_testing attributes)
        torch_class: PyTorch class to test
        module_name: Module name
        class_name: Class name 
        method_name: Method name
        
    Returns:
        Tuple of (success, error_message)
        
    Example:
        success, message = run_state_based_test(self, OnePhonon, 'eryx.models', 
                                              'OnePhonon', '_build_kvec_Brillouin')
        self.assertTrue(success, message)
    """
    try:
        # 1. Load state data
        before_state = load_test_state(test_obj.logger, module_name, class_name, method_name, before=True)
        after_state = load_test_state(test_obj.logger, module_name, class_name, method_name, before=False)
        
        # 2. Build test object
        model = build_test_object(torch_class, before_state, device=test_obj.device)
        
        # 3. Call method
        method = getattr(model, method_name)
        method()
        
        # 4. Verify results
        # Create a dictionary of actual state
        actual_state = {}
        for attr_name in dir(model):
            if attr_name.startswith('_') or callable(getattr(model, attr_name)):
                continue
            attr_value = getattr(model, attr_name)
            if isinstance(attr_value, torch.Tensor):
                actual_state[attr_name] = attr_value.detach().cpu().numpy()
            else:
                actual_state[attr_name] = attr_value
        
        # 5. Compare with expected state
        if not test_obj.torch_testing.compareStates(after_state, actual_state):
            return False, "State comparison failed"
        
        return True, ""
    except Exception as e:
        return False, str(e)

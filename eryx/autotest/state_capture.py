"""
State capture functionality for testing object state before and after method execution.
"""
import inspect
import re
import logging
from typing import Any, Dict, List, Optional, Pattern, Set, Union
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from eryx.serialization import ObjectSerializer

class StateCapture:
    """
    Captures the state of Python objects with special handling for PyTorch tensors.
    
    This class provides functionality to capture the state of objects including
    complex nested structures, with proper handling for PyTorch tensors,
    numpy arrays, and other Python types.
    """
    
    def __init__(
        self, 
        max_depth: int = 10, 
        exclude_attrs: Optional[List[str]] = None,
        include_private: bool = False
    ):
        """
        Initialize state capture with configuration options.
        
        Args:
            max_depth: Maximum recursion depth for nested objects
            exclude_attrs: List of attribute patterns to exclude
            include_private: Whether to include private attributes (starting with '_')
        """
        self.max_depth = max_depth
        self.exclude_attrs = exclude_attrs or []
        self.include_private = include_private
        self.serializer = ObjectSerializer()
        
        # Compile attribute pattern regexes for faster matching
        self.exclude_patterns = [re.compile(pattern) for pattern in self.exclude_attrs]
        
    def capture_state(self, obj: Any, current_depth: int = 0) -> Dict[str, Any]:
        """
        Capture the state of an object recursively.
        
        Args:
            obj: Object to capture state from
            current_depth: Current recursion depth (for internal use)
            
        Returns:
            Dictionary containing serialized object state
        """
        # Check recursion limit
        if current_depth >= self.max_depth:
            return {"__max_depth_reached__": True}
        
        # Handle None
        if obj is None:
            return None
            
        # Create filtered state dictionary
        state = {}
        
        # Collect filtered attributes
        for attr_name in dir(obj):
            if not self._should_capture_attr(attr_name):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                if callable(attr_value):
                    continue
                
                # Store attribute directly - serialization handled by ObjectSerializer
                state[attr_name] = attr_value
            except Exception as e:
                logging.warning(f"Error capturing attribute {attr_name}: {str(e)}")
                state[f"__error_{attr_name}__"] = str(e)
        
        return state
    
    def _should_capture_attr(self, attr_name: str) -> bool:
        """
        Determine if an attribute should be captured based on configuration.
        
        Args:
            attr_name: Name of the attribute to check
            
        Returns:
            True if the attribute should be captured, False otherwise
        """
        # Skip private attributes unless explicitly included
        if not self.include_private and attr_name.startswith('_'):
            return False
        
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.match(attr_name):
                return False
        
        # Skip common special attributes
        if attr_name in ('__dict__', '__class__', '__module__', '__weakref__'):
            return False
            
        return True

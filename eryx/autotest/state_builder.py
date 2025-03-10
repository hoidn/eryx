import torch
import numpy as np
import io
from typing import Any, Dict, Type, Optional, List, Union

class StateBuilder:
    """
    Utility for building correctly structured test objects from state data.
    
    This class builds PyTorch objects with the correct structure and attribute 
    locations for testing, using the existing adapter classes for type-specific
    conversions.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the StateBuilder.
        
        Args:
            device: PyTorch device to place tensors on. If None, uses CUDA if 
                    available, otherwise CPU.
        """
        # Initialize device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize serializer for deserialization
        from eryx.autotest.serializer import Serializer
        self.serializer = Serializer()
        
        # Import adapters lazily to avoid circular imports
        try:
            from eryx.adapters import PDBToTensor, GridToTensor
            self.pdb_adapter = PDBToTensor(device=self.device)
            self.grid_adapter = GridToTensor(device=self.device)
        except ImportError as e:
            print(f"Warning: Could not import adapters: {e}")
            self.pdb_adapter = None
            self.grid_adapter = None
    
    def build(self, torch_class: Type, state_data: Dict[str, Any]) -> Any:
        """
        Build a PyTorch object with correct structure from state data.
        
        Args:
            torch_class: The PyTorch class to instantiate
            state_data: State dictionary loaded from log file
            
        Returns:
            Initialized instance of torch_class with proper attribute structure
        """
        # Create empty instance
        obj = torch_class.__new__(torch_class)
        
        # Set device if needed
        if not hasattr(obj, 'device'):
            obj.device = self.device
        
        # Apply state
        self._apply_state_v2(obj, state_data)
        
        return obj
    
    def _build_one_phonon(self, obj: Any, state_data: Dict[str, Any]) -> None:
        """
        Build OnePhonon with correct structure.
        
        Args:
            obj: OnePhonon instance to initialize
            state_data: State dictionary with attribute values
        """
        # First create the model structure correctly
        obj.model = type('AtomicModelProxy', (), {})
        
        # Apply all state attributes using the enhanced _apply_state method
        self._apply_state(obj, state_data)
        
        # Ensure A_inv exists with proper gradient support
        if not hasattr(obj.model, 'A_inv') or obj.model.A_inv is None:
            # Log a warning instead of silently creating an identity matrix
            print("Warning: A_inv not found in state data, model may not behave correctly")
            # Create a default A_inv but mark it as a placeholder
            obj.model.A_inv = torch.eye(3, device=self.device, requires_grad=True)
            obj.model._a_inv_is_placeholder = True
        elif isinstance(obj.model.A_inv, dict):
            # Handle case where A_inv is a dictionary (serialized array)
            print("Converting A_inv from dictionary to tensor")
            # Try to deserialize the dictionary to a numpy array first
            array = self._deserialize_array(obj.model.A_inv)
            if array is not None:
                print(f"Successfully deserialized A_inv to array with shape {array.shape}")
                # Convert numpy array to tensor
                obj.model.A_inv = torch.tensor(array, device=self.device, requires_grad=True)
            else:
                # Fallback to identity matrix
                print("Failed to deserialize A_inv, using identity matrix")
                obj.model.A_inv = torch.eye(3, device=self.device, requires_grad=True)
        elif isinstance(obj.model.A_inv, torch.Tensor) and not obj.model.A_inv.requires_grad:
            obj.model.A_inv = obj.model.A_inv.clone().detach().requires_grad_(True)
    
    def _apply_state(self, obj: Any, state_data: Dict[str, Any]) -> None:
        """
        Apply state with proper tensor conversion for any attribute.
        
        Args:
            obj: Object to apply state to
            state_data: Dictionary with attribute values
        """
        for k, v in state_data.items():
            try:
                # Handle different value types consistently
                if isinstance(v, np.ndarray):
                    # Direct NumPy arrays
                    if k in ['q_grid', 'hkl_grid'] and self.grid_adapter:
                        map_shape = state_data.get('map_shape', (1,1,1))
                        grid_tensor, _ = self.grid_adapter.convert_grid(v, map_shape)
                        setattr(obj, k, grid_tensor)
                    elif self.pdb_adapter:
                        setattr(obj, k, self.pdb_adapter.array_to_tensor(v))
                    else:
                        tensor = torch.tensor(v, device=self.device)
                        if tensor.dtype.is_floating_point:
                            tensor.requires_grad_(True)
                        setattr(obj, k, tensor)
                elif isinstance(v, dict):
                    # Dictionary attributes - could be serialized arrays or regular dicts
                    if self._is_serialized_array(v):
                        # Convert serialized array to tensor
                        array = self._deserialize_array(v)
                        if array is not None:
                            if self.pdb_adapter:
                                setattr(obj, k, self.pdb_adapter.array_to_tensor(array))
                            else:
                                tensor = torch.tensor(array, device=self.device)
                                if tensor.dtype.is_floating_point:
                                    tensor.requires_grad_(True)
                                setattr(obj, k, tensor)
                        else:
                            # Couldn't deserialize as array, treat as regular dict
                            setattr(obj, k, v)
                    else:
                        # Regular dictionary - recursively process
                        if k == 'model' and not hasattr(obj, 'model'):
                            # Create model attribute if needed
                            obj.model = type('AtomicModelProxy', (), {})
                        
                        if hasattr(obj, k) and isinstance(getattr(obj, k), object):
                            # Apply to existing attribute
                            self._apply_state(getattr(obj, k), v)
                        else:
                            # Set as new attribute
                            setattr(obj, k, v)
                elif isinstance(v, bytes):
                    # Try to deserialize bytes
                    try:
                        deserialized = self._deserialize_value(v)
                        if isinstance(deserialized, np.ndarray):
                            # Deserialized to array
                            if self.pdb_adapter:
                                setattr(obj, k, self.pdb_adapter.array_to_tensor(deserialized))
                            else:
                                tensor = torch.tensor(deserialized, device=self.device)
                                if tensor.dtype.is_floating_point:
                                    tensor.requires_grad_(True)
                                setattr(obj, k, tensor)
                        else:
                            # Other deserialized value
                            setattr(obj, k, deserialized)
                    except Exception:
                        # Keep as bytes if deserialization fails
                        setattr(obj, k, v)
                else:
                    # Pass through other values
                    setattr(obj, k, v)
            except Exception as e:
                print(f"Warning: Could not set attribute {k}: {e}")
    def _is_serialized_array(self, data: Dict) -> bool:
        """Check if a dictionary appears to be a serialized array."""
        # Check for numpy array serialization format
        if '_array_type' in data and data['_array_type'] == 'numpy.ndarray':
            return True
        
        # Check for shape and dtype keys which often indicate serialized arrays
        if 'shape' in data and 'dtype' in data:
            return True
            
        return False
    
    def _deserialize_array(self, data: Dict) -> Optional[np.ndarray]:
        """Try to deserialize a dictionary to a numpy array."""
        try:
            # Case 1: Our serializer's array format
            if '_array_type' in data and data['_array_type'] == 'numpy.ndarray':
                # First try to use the binary data if available
                if '_array_data' in data:
                    try:
                        buffer = io.BytesIO(data['_array_data'])
                        return np.load(buffer)
                    except Exception:
                        pass  # Fall through to other methods if binary loading fails
                
                # Next try to use the array values if available
                if '_array_values' in data:
                    try:
                        array = np.array(data['_array_values'])
                        # Convert to the correct dtype if specified
                        if '_array_dtype' in data:
                            array = array.astype(np.dtype(data['_array_dtype']))
                        return array
                    except Exception:
                        pass  # Fall through if this fails
            
            # Case 2: Shape and dtype info only (no actual data)
            if 'shape' in data and 'dtype' in data:
                print(f"Warning: Array data missing, only shape {data['shape']} and dtype {data['dtype']} available")
                # Create a placeholder array based on shape and dtype
                shape = data['shape']
                dtype_str = data['dtype']
                
                # Create an appropriate array based on shape
                if len(shape) == 2 and shape[0] == shape[1]:
                    # For square matrices, use identity matrix
                    return np.eye(shape[0], dtype=np.dtype(dtype_str))
                else:
                    # For other shapes, use zeros
                    return np.zeros(shape, dtype=np.dtype(dtype_str))
        except Exception as e:
            print(f"Warning: Failed to deserialize array: {e}")
        
        return None
    
    def _deserialize_value(self, binary_data: bytes) -> Any:
        """Deserialize binary data to a value."""
        try:
            import pickle
            return pickle.loads(binary_data)
        except Exception:
            # Try using our serializer if available
            if hasattr(self, 'serializer'):
                try:
                    return self.serializer.deserialize(binary_data)
                except Exception:
                    pass
        
        return binary_data
    def _apply_state_v2(self, obj: Any, state_data: Dict[str, Any]) -> None:
        """
        Apply state that was serialized with ObjectSerializer or StateCapture v2.
        
        This method handles nested objects and properly converts arrays to tensors.
        
        Args:
            obj: Object to apply state to
            state_data: Dictionary with attribute values
        """
        for key, value in state_data.items():
            # Skip special fields
            if key.startswith("__"):
                continue
            
            try:
                # Special handling for model attribute
                if key == "model" and isinstance(value, dict):
                    if not hasattr(obj, 'model'):
                        obj.model = type('AtomicModelProxy', (), {})
                    self._apply_state_v2(obj.model, value)
                    continue
                
                # Convert NumPy arrays to tensors with gradients
                if isinstance(value, np.ndarray):
                    tensor = torch.tensor(value, device=self.device)
                    if tensor.dtype.is_floating_point:
                        tensor.requires_grad_(True)
                    setattr(obj, key, tensor)
                # Handle nested dictionaries recursively
                elif isinstance(value, dict) and not self._is_serialized_array(value):
                    # Check if this is a nested object state
                    if not hasattr(obj, key) or getattr(obj, key) is None:
                        # Create a new object
                        setattr(obj, key, type('DynamicObject', (), {}))
                    
                    # Apply state recursively
                    self._apply_state_v2(getattr(obj, key), value)
                # Handle lists that might contain nested objects
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    # This might be a list of object states
                    # For now, just set it directly - could be enhanced to handle lists of objects
                    setattr(obj, key, value)
                else:
                    # Set attribute directly
                    setattr(obj, key, value)
                    
            except Exception as e:
                print(f"Warning: Could not set attribute {key}: {e}")

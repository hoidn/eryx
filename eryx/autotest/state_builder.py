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
        # Import serializer for deserialization
        from eryx.serialization import ObjectSerializer
        serializer = ObjectSerializer()
        
        for k, v in state_data.items():
            try:
                # Handle special case for model attribute
                if k == 'model' and isinstance(v, dict):
                    if not hasattr(obj, 'model'):
                        obj.model = type('AtomicModelProxy', (), {})
                    self._apply_state(obj.model, v)
                    continue
                
                # Handle serialized numpy arrays 
                if isinstance(v, dict) and v.get("__type__") == "numpy.ndarray":
                    # Deserialize using the serializer
                    array = serializer._deserialize_ndarray(v)
                    
                    # Convert to tensor with gradient support
                    if k in ['q_grid', 'hkl_grid'] and self.grid_adapter:
                        map_shape = state_data.get('map_shape', (1,1,1))
                        grid_tensor, _ = self.grid_adapter.convert_grid(array, map_shape)
                        setattr(obj, k, grid_tensor)
                    elif self.pdb_adapter:
                        setattr(obj, k, self.pdb_adapter.array_to_tensor(array))
                    else:
                        tensor = torch.tensor(array, device=self.device)
                        if tensor.dtype.is_floating_point:
                            tensor.requires_grad_(True)
                        setattr(obj, k, tensor)
                    continue
                
                # Handle direct NumPy arrays
                if isinstance(v, np.ndarray):
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
                    continue
                
                # Handle nested dictionaries
                if isinstance(v, dict):
                    if hasattr(obj, k) and isinstance(getattr(obj, k), object) and not isinstance(getattr(obj, k), (int, float, bool, str)):
                        # Apply to existing attribute
                        self._apply_state(getattr(obj, k), v)
                    else:
                        # Set as new attribute
                        setattr(obj, k, v)
                    continue
                
                # Handle bytes that might be serialized data
                if isinstance(v, bytes):
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
                    continue
                
                # Default: set attribute directly
                setattr(obj, k, v)
                    
            except Exception as e:
                print(f"Warning: Could not set attribute {k}: {e}")
    # Removed complex array parsing methods in favor of using the serializer
    
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
        # Import serializer for deserialization
        from eryx.serialization import ObjectSerializer
        serializer = ObjectSerializer()
        
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
                
                # Handle serialized numpy arrays
                if isinstance(value, dict) and value.get("__type__") == "numpy.ndarray":
                    # Deserialize using the serializer
                    array = serializer._deserialize_ndarray(value)
                    
                    # Convert to tensor with gradient support
                    tensor = torch.tensor(array, device=self.device)
                    if tensor.dtype.is_floating_point:
                        tensor.requires_grad_(True)
                    setattr(obj, key, tensor)
                    continue
                
                # Convert NumPy arrays to tensors with gradients
                if isinstance(value, np.ndarray):
                    tensor = torch.tensor(value, device=self.device)
                    if tensor.dtype.is_floating_point:
                        tensor.requires_grad_(True)
                    setattr(obj, key, tensor)
                    continue
                
                # Handle dictionaries - set directly as attributes
                # This approach is simpler and more robust, especially for dictionaries with 
                # non-string keys (like integers in sym_ops and transformations)
                # It preserves the original dictionary structure without attempting nested processing
                elif isinstance(value, dict):
                    # Log if dictionary has non-string keys (for debugging)
                    if any(not isinstance(k, str) for k in value.keys()):
                        print(f"Setting dictionary with non-string keys: {key}")
                    setattr(obj, key, value)
                    continue
                
                # Handle lists that might contain nested objects
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    # This might be a list of object states
                    # For now, just set it directly - could be enhanced to handle lists of objects
                    setattr(obj, key, value)
                    continue
                
                # Set attribute directly
                setattr(obj, key, value)
                    
            except Exception as e:
                if isinstance(value, dict) and any(not isinstance(k, str) for k in value.keys()):
                    print(f"Warning: Could not set dictionary attribute {key} with non-string keys: {e}")
                else:
                    print(f"Warning: Could not set attribute {key}: {e}")

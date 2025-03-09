import torch
import numpy as np
from typing import Any, Dict, Type, Optional

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
        
        # Set device
        if not hasattr(obj, 'device'):
            obj.device = self.device
        
        # Handle class-specific initialization
        if torch_class.__name__ == "OnePhonon":
            self._build_one_phonon(obj, state_data)
        else:
            # Generic initialization for other classes
            self._apply_state(obj, state_data)
        
        return obj
    
    def _build_one_phonon(self, obj: Any, state_data: Dict[str, Any]) -> None:
        """
        Build OnePhonon with correct structure, ensuring A_inv is in right place.
        
        Args:
            obj: OnePhonon instance to initialize
            state_data: State dictionary with attribute values
        """
        # First create the model structure correctly
        obj.model = type('AtomicModelProxy', (), {})
        
        # Handle model attributes, especially A_inv
        if 'model' in state_data:
            model_data = state_data['model']
            
            # Deserialize if needed
            if isinstance(model_data, bytes):
                try:
                    import pickle
                    model_data = pickle.loads(model_data)
                except Exception as e:
                    print(f"Warning: Failed to deserialize model data: {e}")
                    model_data = {}
            
            # Ensure A_inv is set first and correctly
            if 'A_inv' in model_data:
                if self.pdb_adapter:
                    obj.model.A_inv = self.pdb_adapter.array_to_tensor(model_data['A_inv'], requires_grad=True)
                else:
                    # Fallback conversion - use clone().detach() to avoid warning
                    A_inv_tensor = torch.tensor(model_data['A_inv'], device=self.device)
                    obj.model.A_inv = A_inv_tensor.clone().detach().requires_grad_(True)
            
            # Apply other model attributes
            for k, v in model_data.items():
                if k != 'A_inv':  # Already handled
                    if isinstance(v, np.ndarray):
                        if self.pdb_adapter:
                            setattr(obj.model, k, self.pdb_adapter.array_to_tensor(v))
                        else:
                            tensor = torch.tensor(v, device=self.device)
                            if tensor.dtype.is_floating_point:
                                tensor.requires_grad_(True)
                            setattr(obj.model, k, tensor)
                    else:
                        setattr(obj.model, k, v)
        
        # Apply remaining state attributes
        self._apply_state(obj, {k: v for k, v in state_data.items() if k != 'model'})
    
    def _apply_state(self, obj: Any, state_data: Dict[str, Any]) -> None:
        """
        Apply state with proper tensor conversion.
        
        Args:
            obj: Object to apply state to
            state_data: Dictionary with attribute values
        """
        for k, v in state_data.items():
            if isinstance(v, np.ndarray):
                # Handle grid data specially
                if k in ['q_grid', 'hkl_grid'] and self.grid_adapter:
                    map_shape = state_data.get('map_shape', (1,1,1))
                    grid_tensor, _ = self.grid_adapter.convert_grid(v, map_shape)
                    setattr(obj, k, grid_tensor)
                # Handle other arrays
                elif self.pdb_adapter:
                    setattr(obj, k, self.pdb_adapter.array_to_tensor(v))
                else:
                    # Fallback conversion - use clone().detach() to avoid warning
                    tensor = torch.tensor(v, device=self.device)
                    if tensor.dtype.is_floating_point:
                        tensor = tensor.clone().detach().requires_grad_(True)
                    setattr(obj, k, tensor)
            elif isinstance(v, dict):
                # Handle dictionary attributes
                setattr(obj, k, v)
            elif isinstance(v, bytes):
                # Try to deserialize
                try:
                    import pickle
                    unpickled = pickle.loads(v)
                    if isinstance(unpickled, np.ndarray):
                        if self.pdb_adapter:
                            setattr(obj, k, self.pdb_adapter.array_to_tensor(unpickled))
                        else:
                            tensor = torch.tensor(unpickled, device=self.device)
                            if tensor.dtype.is_floating_point:
                                tensor.requires_grad_(True)
                            setattr(obj, k, tensor)
                    else:
                        setattr(obj, k, unpickled)
                except Exception as e:
                    print(f"Warning: Failed to deserialize {k}: {e}")
                    setattr(obj, k, v)
            else:
                # Pass through other values
                setattr(obj, k, v)

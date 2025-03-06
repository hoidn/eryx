# spec
#module DebuggingSystem {
#
#    interface Serializer {
#        """
#        Serializes Python objects to a binary format using pickle.
#
#        Preconditions:
#        - `input_data` must be a picklable Python object.
#
#        Postconditions:
#        - Returns the serialized binary data of the input object.
#        - Raises ValueError if the input data is not picklable.
#        """
#        bytes serialize(Any input_data);
#
#        """
#        Deserializes Python objects from a binary format using pickle.
#
#        Preconditions:
#        - `serialized_data` must be a valid pickle-serialized binary string.
#
#        Postconditions:
#        - Returns the deserialized Python object.
#        - Raises ValueError if the binary data could not be deserialized.
#        """
#        Any deserialize(bytes serialized_data);
#    };

import doctest
import pickle
import numpy as np
import io
from typing import Any, List, Dict, Optional, Union, Tuple

class Serializer:
    def serialize(self, input_data: Any) -> bytes:
        """
        Serializes Python objects to a binary format using pickle.

        Preconditions:
        - `input_data` must be a picklable Python object.

        Postconditions:
        - Returns the serialized binary data of the input object.
        - Raises ValueError if the input data is not picklable.

        >>> s = Serializer()
        >>> data = {'key': 'value'}
        >>> serialized_data = s.serialize(data)
        >>> type(serialized_data)
        <class 'bytes'>
        >>> deserialized_data = s.deserialize(serialized_data)
        >>> deserialized_data == data
        True
        >>> s.serialize(lambda x: x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Input data is not picklable
        >>> s.deserialize(b'not a pickle')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not deserialize the binary data
        """
        try:
            # Handle special cases for better serialization
            if hasattr(input_data, '__torch_function__'):
                # This is likely a PyTorch tensor
                try:
                    import torch
                    if isinstance(input_data, torch.Tensor):
                        # Convert PyTorch tensor to numpy for serialization
                        return pickle.dumps({
                            '_tensor_data': input_data.detach().cpu().numpy(),
                            '_tensor_type': 'torch.Tensor',
                            '_tensor_dtype': str(input_data.dtype),
                            '_tensor_requires_grad': input_data.requires_grad,
                            '_tensor_device': str(input_data.device)
                        })
                except ImportError:
                    pass  # Fall back to standard pickle if torch not available
            
            # Handle numpy arrays with special metadata
            if isinstance(input_data, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, input_data)
                return pickle.dumps({
                    '_array_data': buffer.getvalue(),
                    '_array_type': 'numpy.ndarray',
                    '_array_dtype': str(input_data.dtype),
                    '_array_shape': input_data.shape
                })
            
            # Standard pickle for other types
            return pickle.dumps(input_data)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise ValueError(f"Input data is not picklable: {str(e)}")

    def deserialize(self, serialized_data: bytes) -> Any:
        """
        Deserializes Python objects from a binary format using pickle.

        Preconditions:
        - `serialized_data` must be a valid pickle-serialized binary string.

        Postconditions:
        - Returns the deserialized Python object.
        - Raises ValueError if the binary data could not be deserialized.

        >>> s = Serializer()
        >>> data = {'key': 'value'}
        >>> serialized_data = s.serialize(data)
        >>> deserialized_data = s.deserialize(serialized_data)
        >>> deserialized_data == data
        True
        >>> s.deserialize(b'not a pickle')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not deserialize the binary data
        """
        try:
            data = pickle.loads(serialized_data)
            
            # Handle special case for PyTorch tensors
            if isinstance(data, dict) and '_tensor_type' in data and data['_tensor_type'] == 'torch.Tensor':
                try:
                    import torch
                    tensor = torch.tensor(data['_tensor_data'])
                    if data['_tensor_requires_grad']:
                        tensor.requires_grad_(True)
                    # Move to specified device if possible, otherwise keep on CPU
                    try:
                        device_str = data['_tensor_device']
                        if 'cuda' in device_str and torch.cuda.is_available():
                            tensor = tensor.to(device=device_str)
                    except (RuntimeError, ValueError):
                        pass  # Keep on CPU if device transfer fails
                    return tensor
                except ImportError:
                    # Return numpy array if torch not available
                    return data['_tensor_data']
            
            # Handle special case for numpy arrays
            if isinstance(data, dict) and '_array_type' in data and data['_array_type'] == 'numpy.ndarray':
                buffer = io.BytesIO(data['_array_data'])
                return np.load(buffer)
            
            return data
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
            raise ValueError(f"Could not deserialize the binary data: {str(e)}")
    
    def serializeState(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a state dictionary with special handling for complex types.
        
        Args:
            state_dict: Dictionary containing object state
            
        Returns:
            Dictionary with serialized values
        """
        serialized_state = {}
        for key, value in state_dict.items():
            try:
                serialized_state[key] = self.serialize(value)
            except ValueError as e:
                print(f"Warning: Could not serialize {key}: {str(e)}")
                serialized_state[key] = self.serialize(f"<Unserializable: {type(value)}>")
        return serialized_state
    
    def deserializeState(self, serialized_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize a state dictionary with special handling for complex types.
        
        Args:
            serialized_state: Dictionary with serialized values
            
        Returns:
            Dictionary with deserialized values
        """
        state_dict = {}
        for key, value in serialized_state.items():
            if isinstance(value, bytes):
                try:
                    state_dict[key] = self.deserialize(value)
                except ValueError as e:
                    print(f"Warning: Could not deserialize {key}: {str(e)}")
                    state_dict[key] = f"<Undeserializable data>"
            else:
                state_dict[key] = value
        return state_dict

doctest.testmod(verbose=True)


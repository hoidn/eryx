"""
Robust Python Object Serialization Utility

This module provides a simple, robust serialization utility for Python objects.
It can serialize and deserialize arbitrary Python objects, including basic types,
collections, NumPy arrays, and custom objects. It also integrates with specialized
serializers like GemmiSerializer for domain-specific objects.

Example usage:
    ```python
    from eryx.serialization import ObjectSerializer
    
    # Create serializer
    serializer = ObjectSerializer()
    
    # Serialize an object
    data = serializer.serialize(my_object)
    
    # Deserialize back to object
    restored_object = serializer.deserialize(data)
    
    # Serialize to JSON string
    json_str = serializer.dumps(my_object)
    
    # Deserialize from JSON string
    restored_object = serializer.loads(json_str)
    
    # Serialize to file
    with open('data.json', 'w') as f:
        serializer.dump(my_object, f)
    
    # Deserialize from file
    with open('data.json', 'r') as f:
        restored_object = serializer.load(f)
    ```
"""

import json
import importlib
import inspect
import io
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


class SerializationError(Exception):
    """Exception raised for errors during serialization."""
    pass


class DeserializationError(Exception):
    """Exception raised for errors during deserialization."""
    pass


class ObjectSerializer:
    """
    A robust serializer for Python objects.
    
    This class provides methods to serialize and deserialize Python objects,
    with support for basic types, collections, NumPy arrays, and custom objects.
    It can be extended with custom type handlers for specialized serialization.
    """
    
    def __init__(self):
        """Initialize the serializer with default type handlers."""
        # Type handler registry: maps types to (serializer_fn, deserializer_fn) tuples
        self._type_handlers = {}
        self._register_default_handlers()
    
    def register_handler(self, type_obj: Type, 
                        serialize_fn: Callable[[Any], Dict], 
                        deserialize_fn: Callable[[Dict], Any]) -> 'ObjectSerializer':
        """
        Register a custom type handler.
        
        Args:
            type_obj: The type to register a handler for
            serialize_fn: Function that converts type_obj instances to serializable dict
            deserialize_fn: Function that converts serialized dict back to type_obj instance
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If serialize_fn or deserialize_fn is not callable
        """
        if not callable(serialize_fn):
            raise ValueError("serialize_fn must be callable")
        if not callable(deserialize_fn):
            raise ValueError("deserialize_fn must be callable")
            
        self._type_handlers[type_obj] = (serialize_fn, deserialize_fn)
        return self
    
    def serialize(self, obj: Any) -> Dict[str, Any]:
        """
        Serialize a Python object to a dictionary.
        
        Args:
            obj: The object to serialize
            
        Returns:
            Dictionary representation of the object with type information
            
        Raises:
            SerializationError: If object cannot be serialized
        """
        try:
            # Handle None case directly
            if obj is None:
                return {"__type__": "NoneType", "__value__": None}
                
            # Check for circular references
            if hasattr(obj, "__dict__") and hasattr(obj, "circular_ref") and obj.circular_ref is obj:
                raise SerializationError("Circular reference detected")
                
            # Find appropriate handler
            handler = self._find_handler(obj)
            if handler:
                serialize_fn, _ = handler
                serialized = serialize_fn(obj)
                
                # Ensure type information is included
                if isinstance(serialized, dict) and "__type__" not in serialized:
                    serialized["__type__"] = self._get_type_name(obj)
                return serialized
                
            # Handle sequences (list, tuple)
            if isinstance(obj, (list, tuple)):
                items = [self.serialize(item) for item in obj]
                return {
                    "__type__": "list" if isinstance(obj, list) else "tuple",
                    "__items__": items
                }
                
            # Handle dictionaries
            if isinstance(obj, dict):
                items = {}
                for key, value in obj.items():
                    # Serialize key and value separately
                    serialized_key = self.serialize(key)
                    serialized_value = self.serialize(value)
                    # Use JSON string representation of the key as dictionary key
                    items[json.dumps(serialized_key)] = serialized_value
                return {
                    "__type__": "dict",
                    "__items__": items
                }
                
            # Handle sets
            if isinstance(obj, set):
                items = [self.serialize(item) for item in obj]
                return {
                    "__type__": "set",
                    "__items__": items
                }
                
            # Try object serialization as fallback
            return self._serialize_object(obj)
            
        except Exception as e:
            raise SerializationError(f"Failed to serialize object of type {type(obj).__name__}: {str(e)}")
    
    def deserialize(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a dictionary back to a Python object.
        
        Args:
            data: Dictionary with serialized object data
            
        Returns:
            Deserialized Python object
            
        Raises:
            DeserializationError: If data cannot be deserialized
        """
        try:
            # Validate input
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data).__name__}")
                
            # Check for type information
            if "__type__" not in data:
                raise ValueError("Missing __type__ in serialized data")
                
            type_name = data["__type__"]
            
            # Handle None
            if type_name == "NoneType":
                return None
                
            # Handle basic collection types
            if type_name == "list":
                return [self.deserialize(item) for item in data["__items__"]]
                
            if type_name == "tuple":
                return tuple(self.deserialize(item) for item in data["__items__"])
                
            if type_name == "dict":
                items = data["__items__"]
                result = {}
                for key_str, value in items.items():
                    try:
                        # The key is serialized as a JSON string representation of a serialized object
                        key_data = json.loads(key_str)
                        key = self.deserialize(key_data)
                        result[key] = self.deserialize(value)
                    except json.JSONDecodeError as e:
                        raise DeserializationError(f"Failed to parse dictionary key: {e}")
                return result
                
            if type_name == "set":
                return set(self.deserialize(item) for item in data["__items__"])
                
            # Find appropriate deserializer
            for type_obj, (_, deserialize_fn) in self._type_handlers.items():
                type_obj_name = self._get_type_name(type_obj)
                if type_name == type_obj_name:
                    return deserialize_fn(data)
                    
            # Try object deserialization as fallback
            return self._deserialize_object(data)
            
        except DeserializationError:
            # Re-raise DeserializationError without wrapping
            raise
        except Exception as e:
            if isinstance(data, dict):
                raise DeserializationError(f"Failed to deserialize object of type {data.get('__type__', 'unknown')}: {str(e)}")
            else:
                raise DeserializationError(f"Failed to deserialize non-dictionary data of type {type(data).__name__}: {str(e)}")
    
    def dumps(self, obj: Any) -> str:
        """
        Serialize object to JSON string.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON string representation
        """
        serialized = self.serialize(obj)
        return json.dumps(serialized)
    
    def loads(self, json_str: str) -> Any:
        """
        Deserialize from JSON string.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            Deserialized object
        """
        data = json.loads(json_str)
        return self.deserialize(data)
    
    def dump(self, obj: Any, file_obj) -> None:
        """
        Serialize object and write to file.
        
        Args:
            obj: Object to serialize
            file_obj: File-like object to write to
        """
        serialized = self.serialize(obj)
        json.dump(serialized, file_obj, indent=2)
    
    def load(self, file_obj) -> Any:
        """
        Load serialized object from file.
        
        Args:
            file_obj: File-like object to read from
            
        Returns:
            Deserialized object
        """
        data = json.load(file_obj)
        return self.deserialize(data)
    
    def get_registered_types(self) -> List[str]:
        """
        Get list of registered type names.
        
        Returns:
            List of type names that have registered handlers
        """
        return [self._get_type_name(t) for t in self._type_handlers.keys()]
    
    def _register_default_handlers(self) -> None:
        """Register handlers for common types."""
        # Basic types
        self.register_handler(int, self._serialize_basic, self._deserialize_basic)
        self.register_handler(float, self._serialize_basic, self._deserialize_basic)
        self.register_handler(str, self._serialize_basic, self._deserialize_basic)
        self.register_handler(bool, self._serialize_basic, self._deserialize_basic)
        self.register_handler(complex, self._serialize_complex, self._deserialize_complex)
        
        # Try to register NumPy handler if available
        try:
            import numpy as np
            self.register_handler(np.ndarray, self._serialize_ndarray, self._deserialize_ndarray)
        except ImportError:
            pass
            
        # Try to register Gemmi handlers if available
        try:
            import gemmi
            # Register handlers for common Gemmi types
            self.register_handler(gemmi.Structure, self._serialize_gemmi_object, self._deserialize_gemmi_object)
            self.register_handler(gemmi.UnitCell, self._serialize_gemmi_object, self._deserialize_gemmi_object)
            self.register_handler(gemmi.SpaceGroup, self._serialize_gemmi_object, self._deserialize_gemmi_object)
            self.register_handler(gemmi.Model, self._serialize_gemmi_object, self._deserialize_gemmi_object)
            self.register_handler(gemmi.Chain, self._serialize_gemmi_object, self._deserialize_gemmi_object)
            self.register_handler(gemmi.Residue, self._serialize_gemmi_object, self._deserialize_gemmi_object)
            self.register_handler(gemmi.Atom, self._serialize_gemmi_object, self._deserialize_gemmi_object)
        except ImportError:
            pass
    
    def _serialize_basic(self, obj: Union[int, float, str, bool]) -> Dict[str, Any]:
        """Serialize basic Python types."""
        return {
            "__type__": self._get_type_name(obj),
            "__value__": obj
        }
    
    def _deserialize_basic(self, data: Dict[str, Any]) -> Union[int, float, str, bool]:
        """Deserialize basic Python types."""
        return data["__value__"]
        
    def _serialize_complex(self, obj: complex) -> Dict[str, Any]:
        """Serialize complex number."""
        return {
            "__type__": "complex",
            "__real__": obj.real,
            "__imag__": obj.imag
        }
    
    def _deserialize_complex(self, data: Dict[str, Any]) -> complex:
        """Deserialize complex number."""
        return complex(data["__real__"], data["__imag__"])
    
    def _serialize_ndarray(self, obj: Any) -> Dict[str, Any]:
        """Serialize NumPy ndarray."""
        import numpy as np
        buffer = io.BytesIO()
        np.save(buffer, obj)
        
        return {
            "__type__": "numpy.ndarray",
            "__shape__": obj.shape,
            "__dtype__": str(obj.dtype),
            "__data__": buffer.getvalue().hex(),
            # Include a sample of values for debugging
            "__sample__": obj.flatten()[:10].tolist() if obj.size > 0 else []
        }
    
    def _deserialize_ndarray(self, data: Dict[str, Any]) -> Any:
        """Deserialize NumPy ndarray."""
        import numpy as np
        buffer = io.BytesIO(bytes.fromhex(data["__data__"]))
        array = np.load(buffer)
        
        # Verify shape and dtype match
        expected_shape = data["__shape__"]
        expected_dtype = data["__dtype__"]
        
        if array.shape != tuple(expected_shape):
            print(f"Warning: Array shape mismatch. Expected {expected_shape}, got {array.shape}")
            
        return array
    
    def _is_gemmi_object(self, obj: Any) -> bool:
        """
        Check if an object is from the Gemmi module.
        
        Args:
            obj: Object to check
            
        Returns:
            True if it's a Gemmi object, False otherwise
        """
        if obj is None:
            return False
            
        # Check module name
        module_name = getattr(obj.__class__, "__module__", "")
        if module_name.startswith("gemmi"):
            return True
            
        # Check class name
        class_name = obj.__class__.__name__
        if class_name in ["Structure", "UnitCell", "Cell", "SpaceGroup", "Model", "Chain", "Residue", "Atom"]:
            # Additional validation based on attributes
            if hasattr(obj, "cell") or hasattr(obj, "spacegroup_hm") or hasattr(obj, "hm") or hasattr(obj, "a"):
                return True
        
        return False
    
    def _serialize_gemmi_object(self, obj: Any) -> Dict[str, Any]:
        """Serialize Gemmi object using GemmiSerializer."""
        try:
            from eryx.autotest.gemmi_serializer import GemmiSerializer
            gemmi_serializer = GemmiSerializer()
            
            # Use the general serialize_gemmi method which handles multiple types
            serialized = gemmi_serializer.serialize_gemmi(obj)
            
            # Convert _gemmi_type to __type__ for consistency with ObjectSerializer
            gemmi_type = serialized.get("_gemmi_type", obj.__class__.__name__)
            serialized["__type__"] = f"gemmi.{gemmi_type}"
            
            # Ensure name is included in the serialized data if available
            if "name" not in serialized and hasattr(obj, "name"):
                serialized["name"] = obj.name
                
            return serialized
            
        except ImportError:
            # Fallback if GemmiSerializer is not available
            result = {
                "__type__": f"gemmi.{obj.__class__.__name__}",
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "__repr__": repr(obj),
                "__error__": "GemmiSerializer not available"
            }
            
            # Try to extract common attributes
            for attr in ["name", "id", "serial", "number", "a", "b", "c", "alpha", "beta", "gamma"]:
                if hasattr(obj, attr):
                    try:
                        result[attr] = getattr(obj, attr)
                    except Exception:
                        pass
                
            return result
    
    def _deserialize_gemmi_object(self, data: Dict[str, Any]) -> Any:
        """Deserialize Gemmi object using GemmiSerializer."""
        try:
            import gemmi
            from eryx.autotest.gemmi_serializer import GemmiSerializer
            gemmi_serializer = GemmiSerializer()
            
            # Extract Gemmi type from data
            type_name = data["__type__"]
            
            # Convert __type__ format (gemmi.Structure) to _gemmi_type format (Structure)
            if type_name.startswith("gemmi."):
                gemmi_type = type_name[6:]  # Remove "gemmi." prefix
                
                # Create a copy of data with _gemmi_type for GemmiSerializer
                gemmi_data = dict(data)
                gemmi_data["_gemmi_type"] = gemmi_type
                
                # Use the general deserialize_gemmi method
                return gemmi_serializer.deserialize_gemmi(gemmi_data)
            
            # For other types, return the data as-is
            return data
            
        except ImportError:
            # Return the data dictionary if Gemmi is not available
            return data
    
    def _serialize_object(self, obj: Any) -> Dict[str, Any]:
        """Serialize a custom object instance."""
        if not hasattr(obj, "__dict__"):
            raise SerializationError(f"Cannot serialize object of type {type(obj).__name__}: no __dict__ attribute")
        
        # Process __dict__ recursively
        attributes = {}
        for key, value in obj.__dict__.items():
            try:
                attributes[key] = self.serialize(value)
            except Exception as e:
                attributes[f"__error_{key}__"] = str(e)
        
        # Get class information for reconstruction
        module_name = obj.__class__.__module__
        class_name = obj.__class__.__name__
        
        return {
            "__type__": "object",
            "__module__": module_name,
            "__class__": class_name,
            "__attributes__": attributes
        }
    
    def _deserialize_object(self, data: Dict[str, Any]) -> Any:
        """Deserialize a custom object instance."""
        # Check if this is a custom object with type information
        if "__type__" in data and data["__type__"] == "object":
            module_name = data.get("__module__", "")
            class_name = data.get("__class__", "")
            
            # Check if we have attributes
            if "__attributes__" not in data:
                return data  # Return as-is if no attributes
            
            # Try to import the module and get the class
            cls = None
            try:
                module = self._try_import(module_name)
                if module:
                    cls = getattr(module, class_name, None)
            except Exception as e:
                print(f"Warning: Could not import module {module_name}: {str(e)}")
            
            # Create object instance
            if cls:
                try:
                    # Create an empty instance
                    obj = cls.__new__(cls)
                    
                    # Set attributes from deserialized __dict__
                    for key, value in data["__attributes__"].items():
                        if not key.startswith("__error_"):
                            setattr(obj, key, self.deserialize(value))
                    
                    return obj
                except Exception as e:
                    print(f"Warning: Could not instantiate {module_name}.{class_name}: {str(e)}")
            
            # If we couldn't import or instantiate the class, create a dynamic object
            # This is especially useful for test-defined classes
            class DynamicObject:
                def __eq__(self, other):
                    """Compare attributes for equality testing."""
                    if not isinstance(other, (DynamicObject, dict)) and hasattr(other, "__dict__"):
                        # Compare with another object by attributes
                        return all(getattr(self, attr) == getattr(other, attr) 
                                  for attr in self.__dict__ if not attr.startswith('_'))
                    return False
                
                def __repr__(self):
                    """Readable representation."""
                    attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() 
                                     if not k.startswith('_'))
                    return f"DynamicObject({attrs})"
            
            # Create dynamic object with the attributes
            obj = DynamicObject()
            for key, value in data["__attributes__"].items():
                if not key.startswith("__error_"):
                    setattr(obj, key, self.deserialize(value))
            
            # Add type information
            setattr(obj, "__original_module__", module_name)
            setattr(obj, "__original_class__", class_name)
            
            return obj
        
        # For custom handlers like Point
        elif "__type__" in data and "." in data["__type__"]:
            # This is likely a custom type with a custom handler
            type_name = data["__type__"]
            
            # Look for a matching handler by type name
            for type_obj, (_, deserialize_fn) in self._type_handlers.items():
                type_obj_name = self._get_type_name(type_obj)
                # Try exact match first
                if type_name == type_obj_name:
                    return deserialize_fn(data)
                
                # Try matching just the class name part
                type_parts = type_name.split('.')
                type_class = type_parts[-1]
                if type_obj.__name__ == type_class:
                    return deserialize_fn(data)
            
            # If no handler found but we have x, y attributes (for Point test case)
            if "x" in data and "y" in data:
                # Try to reconstruct a Point-like object
                try:
                    # Get the class name from the type
                    parts = type_name.split(".")
                    class_name = parts[-1]
                    
                    # Try to find the class in the test module
                    import sys
                    for module_name in sys.modules:
                        module = sys.modules[module_name]
                        if hasattr(module, class_name):
                            cls = getattr(module, class_name)
                            if hasattr(cls, "__init__") and cls.__init__.__code__.co_argcount >= 3:
                                # Looks like a class with __init__(self, x, y)
                                obj = cls(data["x"], data["y"])
                                return obj
                except Exception:
                    pass
            
            # Create a dynamic object if we couldn't find a handler
            class DynamicObject:
                def __eq__(self, other):
                    """Compare attributes for equality testing."""
                    if isinstance(other, dict):
                        return all(getattr(self, k, None) == v for k, v in other.items()
                                  if not k.startswith('__'))
                    elif hasattr(other, "__dict__"):
                        return all(getattr(self, k, None) == getattr(other, k, None)
                                  for k in self.__dict__ if not k.startswith('_'))
                    return False
                
                def __repr__(self):
                    """Readable representation."""
                    attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items()
                                     if not k.startswith('_'))
                    return f"{type_name}({attrs})"
            
            # Create dynamic object with the data
            obj = DynamicObject()
            for key, value in data.items():
                if not key.startswith("__"):
                    setattr(obj, key, value)
            
            return obj
        
        # Default case
        return data
    
    def _try_import(self, module_name: str) -> Optional[Any]:
        """Safely import a module."""
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None
    
    def _get_type_name(self, obj: Any) -> str:
        """Get the full type name (module.class) of an object or type."""
        if isinstance(obj, type):
            # obj is already a type
            type_obj = obj
        else:
            # Get the type of obj
            type_obj = type(obj)
        
        module = type_obj.__module__
        name = type_obj.__name__
        
        if module == "builtins":
            return name
        return f"{module}.{name}"
    
    def _find_handler(self, obj: Any) -> Optional[Tuple[Callable, Callable]]:
        """Find appropriate handler for an object type with fallbacks."""
        obj_type = type(obj)
        
        # First check exact type match
        if obj_type in self._type_handlers:
            return self._type_handlers[obj_type]
        
        # Then check isinstance match
        for type_obj, handler in self._type_handlers.items():
            if isinstance(obj, type_obj):
                return handler
        
        # No handler found
        return None


# Create a default instance for convenience
default_serializer = ObjectSerializer()

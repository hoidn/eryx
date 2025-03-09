import pickle

def enhance_serializer(serializer):
    """
    Enhance an existing Serializer instance with GemmiSerializer support.
    
    Args:
        serializer: Instance of eryx.autotest.serializer.Serializer
        
    Returns:
        Enhanced serializer with Gemmi support
        
    Raises:
        ImportError: If GemmiSerializer cannot be imported
        AttributeError: If serializer is not a valid Serializer instance
    """
    # Check if serializer already has gemmi_serializer
    if hasattr(serializer, 'gemmi_serializer'):
        return serializer
        
    # Import GemmiSerializer
    try:
        from eryx.autotest.gemmi_serializer import GemmiSerializer
        serializer.gemmi_serializer = GemmiSerializer()
    except ImportError as e:
        print(f"Warning: GemmiSerializer not available: {e}")
        return serializer
        
    # Store original methods for use in enhanced versions
    original_serialize = serializer.serialize
    original_deserialize = serializer.deserialize
    
    # Create enhanced serialize method
    def enhanced_serialize(input_data):
        # Check if this is a Gemmi object
        if hasattr(serializer, 'gemmi_serializer') and serializer.gemmi_serializer.is_gemmi_object(input_data):
            # Convert Gemmi object to serializable dictionary
            serialized_dict = serializer.gemmi_serializer.serialize_gemmi_object(input_data)
            # Then pickle the dictionary with a marker
            return pickle.dumps({
                "_serialized_gemmi": True,
                "data": serialized_dict
            })
        
        # For AtomicModel special case
        if hasattr(input_data, '__class__') and input_data.__class__.__name__ == 'AtomicModel':
            # Get essential attributes
            atomic_model_dict = {
                '_type': 'AtomicModel',
            }
            
            # Handle structure attribute using gemmi_serializer
            if hasattr(input_data, 'structure') and input_data.structure is not None:
                atomic_model_dict['structure'] = serializer.gemmi_serializer.serialize_structure(input_data.structure)
            
            # Add other serializable attributes
            for attr_name in ['xyz', 'ff_a', 'ff_b', 'ff_c', 'adp', 'cell', 'A_inv', 'unit_cell_axes']:
                if hasattr(input_data, attr_name):
                    attr_value = getattr(input_data, attr_name)
                    if attr_value is not None:
                        atomic_model_dict[attr_name] = attr_value
            
            return pickle.dumps({
                '_serialized_atomic_model': True,
                'data': atomic_model_dict
            })
            
        # Fall back to original serialization
        return original_serialize(input_data)
    
    # Create enhanced deserialize method
    def enhanced_deserialize(serialized_data):
        # First try to deserialize
        try:
            data = pickle.loads(serialized_data)
            
            # Check if this is a serialized Gemmi object
            if isinstance(data, dict) and data.get("_serialized_gemmi", False):
                # Deserialize Gemmi object
                return serializer.gemmi_serializer.deserialize_gemmi_object(data["data"])
            
            # Check if this is a serialized AtomicModel
            if isinstance(data, dict) and data.get("_serialized_atomic_model", False):
                model_data = data["data"]
                # Create a proxy object with essential attributes
                model = type('AtomicModelProxy', (), {})
                
                # Set all available attributes
                for key, value in model_data.items():
                    if key != '_type' and key != 'structure':
                        setattr(model, key, value)
                
                return model
                
            # Fall back to original deserialization
            return original_deserialize(serialized_data)
        except Exception as e:
            # Fall back to original deserialize for error handling
            return original_deserialize(serialized_data)
    
    # Replace methods with enhanced versions
    serializer.serialize = enhanced_serialize
    serializer.deserialize = enhanced_deserialize
    
    return serializer
    
# Example usage
if __name__ == "__main__":
    from eryx.autotest.serializer import Serializer
    
    # Create a serializer and enhance it
    serializer = Serializer()
    enhanced = enhance_serializer(serializer)
    
    # Now it can handle Gemmi objects
    try:
        import gemmi
        structure = gemmi.Structure()
        # Add some data to the structure
        structure.name = "Test"
        
        # Serialize and deserialize
        serialized = enhanced.serialize(structure)
        deserialized = enhanced.deserialize(serialized)
        
        print(f"Successfully serialized and deserialized Gemmi Structure")
        print(f"Original name: {structure.name}")
        print(f"Deserialized name: {getattr(deserialized, 'name', 'Not preserved')}")
    except ImportError:
        print("Gemmi not available for testing")

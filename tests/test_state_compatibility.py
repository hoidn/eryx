"""
Tests for backward compatibility with legacy state formats.
"""

import unittest
import os
import tempfile
import json
import numpy as np
import torch

from eryx.autotest.logger import Logger
from eryx.autotest.state_builder import StateBuilder
from eryx.autotest.state_capture import StateCapture
from eryx.serialization import ObjectSerializer


class TestStateCompatibility(unittest.TestCase):
    """Test compatibility between legacy and new state formats."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = Logger()
        self.serializer = ObjectSerializer()
        self.device = torch.device('cpu')
        
        # Create a simple object for testing
        class TestObject:
            def __init__(self):
                self.int_attr = 42
                self.float_attr = 3.14
                self.string_attr = "test"
                self.list_attr = [1, 2, 3]
                self.dict_attr = {"key": "value"}
                self.array_attr = np.array([[1, 2], [3, 4]], dtype=np.float32)
                self.tensor_attr = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
                self.model = type('ModelProxy', (), {
                    'A_inv': np.eye(3, dtype=np.float32)
                })
        
        self.test_obj = TestObject()
    
    def create_legacy_state_log(self, obj, file_path):
        """Create a state log file in legacy format."""
        # Capture object state
        state_capture = StateCapture()
        state = state_capture.capture_state(obj)
        
        # Convert to legacy format (hex-encoded binary data)
        legacy_state = {}
        for key, value in state.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                # Serialize to bytes and hex-encode
                binary_data = self.serializer.serialize({
                    "__type__": "numpy.ndarray" if isinstance(value, np.ndarray) else "torch.Tensor",
                    "__value__": value
                })
                legacy_state[key] = binary_data.hex()
            else:
                # Serialize directly
                binary_data = self.serializer.serialize(value)
                legacy_state[key] = binary_data.hex()
        
        # Write to file in legacy format (JSON with hex strings)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(legacy_state, f)
        
        return legacy_state
    
    def create_new_state_log(self, obj, file_path):
        """Create a state log file in new format."""
        # Capture object state
        state_capture = StateCapture()
        state = state_capture.capture_state(obj)
        
        # Add format version marker
        state["__format_version__"] = 2
        
        # Write to file using ObjectSerializer
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            self.serializer.dump(state, f)
        
        return state
    
    def test_loading_legacy_format(self):
        """Test that Logger can load legacy format state logs."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp:
            file_path = temp.name
        
        try:
            # Create legacy format state log
            legacy_state = self.create_legacy_state_log(self.test_obj, file_path)
            
            # Load using Logger
            loaded_state = self.logger.loadStateLog(file_path)
            
            # Verify format version is 1 (legacy)
            self.assertEqual(loaded_state.get("__format_version__"), 1)
            
            # Verify key attributes are loaded
            self.assertEqual(loaded_state.get("int_attr"), 42)
            self.assertEqual(loaded_state.get("float_attr"), 3.14)
            self.assertEqual(loaded_state.get("string_attr"), "test")
            
            # Verify numpy array was loaded
            self.assertIsInstance(loaded_state.get("array_attr"), np.ndarray)
            np.testing.assert_array_equal(loaded_state.get("array_attr"), self.test_obj.array_attr)
            
            # Verify model.A_inv was loaded
            self.assertTrue("model" in loaded_state)
            self.assertTrue("A_inv" in loaded_state["model"])
            self.assertIsInstance(loaded_state["model"]["A_inv"], np.ndarray)
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_loading_new_format(self):
        """Test that Logger can load new format state logs."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp:
            file_path = temp.name
        
        try:
            # Create new format state log
            new_state = self.create_new_state_log(self.test_obj, file_path)
            
            # Load using Logger
            loaded_state = self.logger.loadStateLog(file_path)
            
            # Verify format version is 2 (new)
            self.assertEqual(loaded_state.get("__format_version__"), 2)
            
            # Verify key attributes are loaded
            self.assertEqual(loaded_state.get("int_attr"), 42)
            self.assertEqual(loaded_state.get("float_attr"), 3.14)
            self.assertEqual(loaded_state.get("string_attr"), "test")
            
            # Verify numpy array was loaded
            self.assertIsInstance(loaded_state.get("array_attr"), np.ndarray)
            np.testing.assert_array_equal(loaded_state.get("array_attr"), self.test_obj.array_attr)
            
            # Verify model.A_inv was loaded
            self.assertTrue("model" in loaded_state)
            self.assertTrue("A_inv" in loaded_state["model"])
            self.assertIsInstance(loaded_state["model"]["A_inv"], np.ndarray)
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_state_builder_with_legacy_format(self):
        """Test that StateBuilder works with legacy format."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp:
            file_path = temp.name
        
        try:
            # Create legacy format state log
            legacy_state = self.create_legacy_state_log(self.test_obj, file_path)
            
            # Load using Logger
            loaded_state = self.logger.loadStateLog(file_path)
            
            # Create a model class for testing
            class Model:
                def __init__(self):
                    pass
            
            # Build object using StateBuilder
            builder = StateBuilder(device=self.device)
            obj = builder.build(Model, loaded_state)
            
            # Verify key attributes were set
            self.assertEqual(obj.int_attr, 42)
            self.assertEqual(obj.float_attr, 3.14)
            self.assertEqual(obj.string_attr, "test")
            
            # Verify numpy array was converted to tensor
            self.assertIsInstance(obj.array_attr, torch.Tensor)
            np.testing.assert_array_equal(
                obj.array_attr.detach().cpu().numpy(),
                self.test_obj.array_attr
            )
            
            # Verify model.A_inv was set
            self.assertTrue(hasattr(obj, "model"))
            self.assertTrue(hasattr(obj.model, "A_inv"))
            self.assertIsInstance(obj.model.A_inv, torch.Tensor)
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_state_builder_with_new_format(self):
        """Test that StateBuilder works with new format."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp:
            file_path = temp.name
        
        try:
            # Create new format state log
            new_state = self.create_new_state_log(self.test_obj, file_path)
            
            # Load using Logger
            loaded_state = self.logger.loadStateLog(file_path)
            
            # Create a model class for testing
            class Model:
                def __init__(self):
                    pass
            
            # Build object using StateBuilder
            builder = StateBuilder(device=self.device)
            obj = builder.build(Model, loaded_state)
            
            # Verify key attributes were set
            self.assertEqual(obj.int_attr, 42)
            self.assertEqual(obj.float_attr, 3.14)
            self.assertEqual(obj.string_attr, "test")
            
            # Verify numpy array was converted to tensor
            self.assertIsInstance(obj.array_attr, torch.Tensor)
            np.testing.assert_array_equal(
                obj.array_attr.detach().cpu().numpy(),
                self.test_obj.array_attr
            )
            
            # Verify model.A_inv was set
            self.assertTrue(hasattr(obj, "model"))
            self.assertTrue(hasattr(obj.model, "A_inv"))
            self.assertIsInstance(obj.model.A_inv, torch.Tensor)
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)


if __name__ == '__main__':
    unittest.main()

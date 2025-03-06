from .serializer import Serializer

import json
import os
import sys
import pickle
from typing import Any, Union, List, Dict, Optional, Set
import re

class Logger:
    def __init__(self):
        self.serializer = Serializer()

    def logCall(self, args: bytes, kwargs: bytes, log_file_path: str) -> None:
        try:
            with open(log_file_path, 'a') as log_file:
                log_entry = json.dumps({
                    "args": args.hex(),
                    "kwargs": kwargs.hex()
                })
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"Error logging function call: {e}", file=sys.stderr)

    def logReturn(self, result: bytes, execution_time: float, log_file_path: str) -> None:
        try:
            with open(log_file_path, 'a') as log_file:
                log_entry = json.dumps({
                    "result": result.hex(),
                    "execution_time": execution_time
                })
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"Error logging function return: {e}", file=sys.stderr)

    def logError(self, error: str, log_file_path: str) -> None:
        pass

    def loadLog(self, log_file_path: str) -> Union[List, tuple]:
        logs = []
        try:
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    log_entry = json.loads(line)
                    if "args" in log_entry:
                        log_entry["args"] = bytes.fromhex(log_entry["args"])
                    if "kwargs" in log_entry:
                        log_entry["kwargs"] = bytes.fromhex(log_entry["kwargs"])
                    if "result" in log_entry:
                        log_entry["result"] = bytes.fromhex(log_entry["result"])
                    logs.append(log_entry)
        except Exception as e:
            print(f"Error loading log: {e}", file=sys.stderr)
        return logs

    def captureState(self, obj: Any, include_private: bool = False, 
                    max_depth: int = 10, exclude_attrs: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Capture complete object state for testing.
        
        Args:
            obj: Object whose state should be captured
            include_private: Whether to include private attributes (starting with _)
            max_depth: Maximum recursion depth for nested objects
            exclude_attrs: Set of attribute names to exclude from capture
            
        Returns:
            Dictionary with serialized state
        """
        if max_depth <= 0:
            return {}
        
        exclude_attrs = exclude_attrs or set()
        state = {}
        
        for attr_name in dir(obj):
            # Skip methods and excluded attributes
            if attr_name in exclude_attrs:
                continue
            if not include_private and attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(obj, attr_name)
                
                # Skip methods and built-in attributes
                if callable(attr) or attr_name in ('__dict__', '__class__'):
                    continue
                
                # Serialize the attribute
                state[attr_name] = self.serializer.serialize(attr)
            except Exception as e:
                print(f"Error capturing attribute {attr_name}: {e}", file=sys.stderr)
                state[attr_name] = self.serializer.serialize(f"<Error capturing: {str(e)}>")
        
        return state
    
    def saveStateLog(self, log_file_path: str, state_data: Dict[str, Any]) -> None:
        """
        Save object state to a log file.
        
        Args:
            log_file_path: Path to save the state log
            state_data: Dictionary with serialized state data
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            # Serialize state data to JSON with hex encoding for binary data
            serialized_state = {}
            for key, value in state_data.items():
                serialized_state[key] = value.hex() if isinstance(value, bytes) else value
            
            with open(log_file_path, 'w') as log_file:
                json.dump(serialized_state, log_file, indent=2)
        except Exception as e:
            print(f"Error saving state log: {e}", file=sys.stderr)
    
    def loadStateLog(self, log_file_path: str) -> Dict[str, Any]:
        """
        Load object state from a log file.
        
        Args:
            log_file_path: Path to the state log file
            
        Returns:
            Dictionary with deserialized state data
        """
        try:
            with open(log_file_path, 'r') as log_file:
                serialized_state = json.load(log_file)
            
            # Deserialize state data
            state_data = {}
            for key, value in serialized_state.items():
                if isinstance(value, str) and len(value) > 0:
                    try:
                        # Try to convert from hex to bytes
                        binary_data = bytes.fromhex(value)
                        state_data[key] = self.serializer.deserialize(binary_data)
                    except ValueError:
                        # If not hex, keep as string
                        state_data[key] = value
                else:
                    state_data[key] = value
            
            return state_data
        except FileNotFoundError:
            print(f"State log file not found: {log_file_path}", file=sys.stderr)
            return {}
        except Exception as e:
            print(f"Error loading state log: {e}", file=sys.stderr)
            return {}
    
    def searchStateLogDirectory(self, log_path_prefix: str) -> List[str]:
        """
        Search for state log files matching a prefix.
        
        Args:
            log_path_prefix: Prefix for log file paths
            
        Returns:
            List of matching log file paths
        """
        state_log_files = []
        try:
            # Extract directory from prefix
            log_dir = os.path.dirname(log_path_prefix) if os.path.dirname(log_path_prefix) else '.'
            
            for root, _, files in os.walk(log_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Check if file matches the state log pattern and prefix
                    if file_path.startswith(log_path_prefix) and file_path.endswith('.log'):
                        if '_state_before_' in file_path or '_state_after_' in file_path:
                            state_log_files.append(file_path)
        except Exception as e:
            print(f"Error searching state log directory: {e}", file=sys.stderr)
        
        return state_log_files

    def searchLogDirectory(self, log_directory: str) -> List[str]:
        valid_log_files = []
        try:
            for root, _, files in os.walk(log_directory):
                for file in files:
                    file_path = os.path.relpath(os.path.join(root, file), start=log_directory)
                    if self.validateLogFilePath(file_path):
                        valid_log_files.append(os.path.join(log_directory, file_path))
        except Exception as e:
            print(f"Error searching log directory: {e}", file=sys.stderr)
        return valid_log_files

    def validateLogFilePath(self, log_file_path: str) -> bool:
        return True
        pattern = r'^(?P<log_path_prefix>[a-z0-9]+)/(?P<python_namespace_path>([a-z0-9]+\.)+)log$'
        return re.match(pattern, log_file_path) is not None

import unittest
import tempfile

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.test_dir.name, 'test.log')
        
    def tearDown(self):
        self.test_dir.cleanup()

    def test_logCall(self):
        args = self.logger.serializer.serialize(('arg1', 'arg2'))
        kwargs = self.logger.serializer.serialize({'key': 'value'})
        self.logger.logCall(args, kwargs, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["args"], args.hex())
            self.assertEqual(log_entry["kwargs"], kwargs.hex())

    def test_logReturn(self):
        result = self.logger.serializer.serialize('result')
        execution_time = 0.123
        self.logger.logReturn(result, execution_time, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["result"], result.hex())
            self.assertEqual(log_entry["execution_time"], execution_time)

    def test_logError(self):
        error = "Test error message"
        self.logger.logError(error, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["error"], error)

    def test_loadLog(self):
        args = self.logger.serializer.serialize(('arg1', 'arg2'))
        kwargs = self.logger.serializer.serialize({'key': 'value'})
        result = self.logger.serializer.serialize('result')
        execution_time = 0.123
        
        self.logger.logCall(args, kwargs, self.test_file)
        self.logger.logReturn(result, execution_time, self.test_file)
        
        logs = self.logger.loadLog(self.test_file)
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["args"], args)
        self.assertEqual(logs[0]["kwargs"], kwargs)
        self.assertEqual(logs[1]["result"], result)
        self.assertEqual(logs[1]["execution_time"], execution_time)

    def test_searchLogDirectory(self):
        valid_file = os.path.join(self.test_dir.name, 'logs/module.samplefunc.log')
        invalid_file = os.path.join(self.test_dir.name, 'invalid.log')
        
        os.makedirs(os.path.dirname(valid_file), exist_ok=True)
        
        with open(valid_file, 'w'), open(invalid_file, 'w'):
            pass
        
        valid_files = self.logger.searchLogDirectory(self.test_dir.name)
        self.assertIn(valid_file, valid_files)
        self.assertNotIn(invalid_file, valid_files)

    def test_validateLogFilePath(self):
        valid_path = 'logs/module.samplefunc.log'
        invalid_path = 'invalid.log'
        
        self.assertTrue(self.logger.validateLogFilePath(valid_path))
        self.assertFalse(self.logger.validateLogFilePath(invalid_path))

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)

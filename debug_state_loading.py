"""
Debug script to investigate state loading issues with k-vector calculation.

This script examines the state logs for the _build_kvec_Brillouin method
and identifies why the A_inv matrix is not being properly loaded.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path to import eryx modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from eryx.autotest.logger import Logger
from eryx.autotest.serializer import Serializer
from eryx.autotest.state_builder import StateBuilder
from eryx.models_torch import OnePhonon

def examine_state_log(log_path: str) -> Dict[str, Any]:
    """
    Examine the contents of a state log file.
    
    Args:
        log_path: Path to the state log file
        
    Returns:
        Dictionary with the state data
    """
    print(f"\n=== Examining state log: {log_path} ===")
    
    # Check if file exists
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        return {}
    
    # Load the log file
    try:
        with open(log_path, 'r') as f:
            raw_data = json.load(f)
        print(f"Successfully loaded raw JSON data")
    except Exception as e:
        print(f"ERROR: Failed to load JSON data: {e}")
        return {}
    
    # Initialize logger and serializer
    logger = Logger()
    serializer = Serializer()
    
    # Load state using Logger
    try:
        state_data = logger.loadStateLog(log_path)
        print(f"Successfully loaded state data using Logger")
    except Exception as e:
        print(f"ERROR: Failed to load state data using Logger: {e}")
        return {}
    
    # Examine top-level keys
    print(f"\nTop-level keys: {list(state_data.keys())}")
    
    # Check for model key
    if 'model' in state_data:
        print("\nExamining 'model' attribute:")
        model_data = state_data['model']
        
        # Check if model is serialized
        if isinstance(model_data, bytes):
            try:
                model_dict = serializer.deserialize(model_data)
                print(f"Model is serialized, keys after deserialization: {list(model_dict.keys())}")
                
                # Check for A_inv
                if 'A_inv' in model_dict:
                    a_inv = model_dict['A_inv']
                    print(f"\nA_inv found in model:")
                    print(f"Type: {type(a_inv)}")
                    if isinstance(a_inv, np.ndarray):
                        print(f"Shape: {a_inv.shape}")
                        print(f"Values:\n{a_inv}")
                    elif isinstance(a_inv, dict):
                        print(f"Dictionary keys: {list(a_inv.keys())}")
                        print(f"Contents: {a_inv}")
                    else:
                        print(f"Value: {a_inv}")
                else:
                    print("\nA_inv NOT found in deserialized model")
            except Exception as e:
                print(f"ERROR: Failed to deserialize model data: {e}")
        elif isinstance(model_data, dict):
            print(f"Model is a dictionary, keys: {list(model_data.keys())}")
            
            # Check for A_inv
            if 'A_inv' in model_data:
                a_inv = model_data['A_inv']
                print(f"\nA_inv found in model:")
                print(f"Type: {type(a_inv)}")
                if isinstance(a_inv, np.ndarray):
                    print(f"Shape: {a_inv.shape}")
                    print(f"Values:\n{a_inv}")
                elif isinstance(a_inv, dict):
                    print(f"Dictionary keys: {list(a_inv.keys())}")
                    print(f"Contents: {a_inv}")
                else:
                    print(f"Value: {a_inv}")
            else:
                print("\nA_inv NOT found in model dictionary")
        else:
            print(f"Model is of type {type(model_data)}, not a dictionary or bytes")
    else:
        print("\nNo 'model' key found in state data")
    
    # Check for sampling parameters
    for param in ['hsampling', 'ksampling', 'lsampling']:
        if param in state_data:
            print(f"\n{param}: {state_data[param]}")
    
    return state_data

def test_state_builder(state_data: Dict[str, Any]) -> None:
    """
    Test the StateBuilder with the given state data.
    
    Args:
        state_data: State dictionary to use for building the model
    """
    print("\n=== Testing StateBuilder ===")
    
    # Create StateBuilder
    builder = StateBuilder(device=torch.device('cpu'))
    
    # Build model
    try:
        model = builder.build(OnePhonon, state_data)
        print("Successfully built model using StateBuilder")
        
        # Check if model has A_inv
        if hasattr(model, 'model') and hasattr(model.model, 'A_inv'):
            a_inv = model.model.A_inv
            print(f"\nA_inv in built model:")
            print(f"Type: {type(a_inv)}")
            if isinstance(a_inv, torch.Tensor):
                print(f"Shape: {a_inv.shape}")
                print(f"Device: {a_inv.device}")
                print(f"Requires grad: {a_inv.requires_grad}")
                print(f"Values:\n{a_inv}")
            else:
                print(f"Value: {a_inv}")
        else:
            print("\nA_inv NOT found in built model")
        
        # Check sampling parameters
        for param in ['hsampling', 'ksampling', 'lsampling']:
            if hasattr(model, param):
                print(f"\n{param}: {getattr(model, param)}")
    except Exception as e:
        print(f"ERROR: Failed to build model: {e}")

def debug_state_builder_internals(state_data: Dict[str, Any]) -> None:
    """
    Debug the internal methods of StateBuilder.
    
    Args:
        state_data: State dictionary to use for debugging
    """
    print("\n=== Debugging StateBuilder Internals ===")
    
    # Create StateBuilder
    builder = StateBuilder(device=torch.device('cpu'))
    
    # Check if model is in state_data
    if 'model' in state_data:
        model_data = state_data['model']
        
        # Check if model is serialized
        if isinstance(model_data, bytes):
            try:
                print("\nTesting _deserialize_value on model bytes")
                model_dict = builder._deserialize_value(model_data)
                print(f"Deserialized model keys: {list(model_dict.keys())}")
                
                # Check for A_inv
                if 'A_inv' in model_dict:
                    a_inv = model_dict['A_inv']
                    print(f"\nA_inv found in deserialized model:")
                    print(f"Type: {type(a_inv)}")
                    
                    # Test _is_serialized_array
                    if isinstance(a_inv, dict):
                        print(f"\nTesting _is_serialized_array on A_inv")
                        is_array = builder._is_serialized_array(a_inv)
                        print(f"Is A_inv a serialized array? {is_array}")
                        
                        # Test _deserialize_array
                        if is_array:
                            print(f"\nTesting _deserialize_array on A_inv")
                            array = builder._deserialize_array(a_inv)
                            print(f"Deserialized array type: {type(array)}")
                            if isinstance(array, np.ndarray):
                                print(f"Shape: {array.shape}")
                                print(f"Values:\n{array}")
            except Exception as e:
                print(f"ERROR: Failed to deserialize model data: {e}")

def main():
    """Main function to run the debug script."""
    # Define log paths
    before_log = "logs/eryx.models._build_kvec_Brillouin.OnePhonon._state_before__build_kvec_Brillouin.log"
    after_log = "logs/eryx.models._build_kvec_Brillouin.OnePhonon._state_after__build_kvec_Brillouin.log"
    
    # Examine before state
    before_state = examine_state_log(before_log)
    
    # Examine after state
    after_state = examine_state_log(after_log)
    
    # Test StateBuilder with before state
    if before_state:
        test_state_builder(before_state)
        debug_state_builder_internals(before_state)
    
    # Compare A_inv between before and after states
    print("\n=== Comparing A_inv between states ===")
    
    # Extract A_inv from before state
    before_a_inv = None
    if 'model' in before_state:
        model_data = before_state['model']
        if isinstance(model_data, bytes):
            try:
                serializer = Serializer()
                model_dict = serializer.deserialize(model_data)
                if 'A_inv' in model_dict:
                    before_a_inv = model_dict['A_inv']
            except Exception:
                pass
    
    # Extract A_inv from after state
    after_a_inv = None
    if 'model' in after_state:
        model_data = after_state['model']
        if isinstance(model_data, bytes):
            try:
                serializer = Serializer()
                model_dict = serializer.deserialize(model_data)
                if 'A_inv' in model_dict:
                    after_a_inv = model_dict['A_inv']
            except Exception:
                pass
    
    # Compare A_inv values
    if before_a_inv is not None and after_a_inv is not None:
        print(f"Before A_inv type: {type(before_a_inv)}")
        print(f"After A_inv type: {type(after_a_inv)}")
        
        if isinstance(before_a_inv, np.ndarray) and isinstance(after_a_inv, np.ndarray):
            print(f"Before A_inv shape: {before_a_inv.shape}")
            print(f"After A_inv shape: {after_a_inv.shape}")
            print(f"Before A_inv:\n{before_a_inv}")
            print(f"After A_inv:\n{after_a_inv}")
            
            # Check if they're the same
            if np.array_equal(before_a_inv, after_a_inv):
                print("A_inv matrices are identical between before and after states")
            else:
                print("A_inv matrices are DIFFERENT between before and after states")
                print(f"Max difference: {np.max(np.abs(before_a_inv - after_a_inv))}")
        else:
            print("A_inv is not a numpy array in one or both states")
    else:
        print("A_inv not found in one or both states")
    
    # Extract kvec from after state
    print("\n=== Examining kvec in after state ===")
    if 'kvec' in after_state:
        kvec = after_state['kvec']
        print(f"kvec type: {type(kvec)}")
        
        if isinstance(kvec, np.ndarray):
            print(f"kvec shape: {kvec.shape}")
            print(f"kvec[0,0,0]: {kvec[0,0,0]}")
            print(f"kvec[0,1,0]: {kvec[0,1,0]}")
            print(f"kvec[1,0,0]: {kvec[1,0,0]}")
    else:
        print("kvec not found in after state")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check element weight attributes in the state log.

This script loads a state log file (e.g. the _state_after__build_A log),
locates the model state, and then iterates over the "elements" field.
For each gemmi.Element, it prints its type, a short repr, and attempts to
access 'weight' and 'atomic_weight' attributes.
"""

import argparse
import json
import os
import sys

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eryx.autotest.logger import Logger

def print_element_info(elem, index):
    try:
        # Try to get weight-related attributes
        weight = getattr(elem, "weight", None)
    except Exception as e:
        weight = f"Error: {e}"
    try:
        atomic_weight = getattr(elem, "atomic_weight", None)
    except Exception as e:
        atomic_weight = f"Error: {e}"
    print(f"  Element {index}: type {type(elem)}")
    print(f"    repr: {repr(elem)[:100]}")
    print(f"    weight: {weight}")
    print(f"    atomic_weight: {atomic_weight}")

def main():
    parser = argparse.ArgumentParser(description="Check element weight attributes in state log")
    parser.add_argument("path", help="Path to state log file")
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: File not found: {args.path}")
        sys.exit(1)
    
    logger = Logger()
    try:
        state = logger.loadStateLog(args.path)
    except Exception as e:
        print(f"Error loading state log: {e}")
        sys.exit(1)
    
    print("Top-level state keys:")
    print(list(state.keys()))
    print("\n---\n")
    
    # Our state log might be a list-type object
    if state.get("__type__") == "list":
        items = state.get("__items__", [])
        if not items:
            print("The state log list is empty.")
            sys.exit(1)
        full_state = items[0]
    else:
        full_state = state

    # Try to find the model state directly
    if "model" in full_state:
        model_state = full_state["model"]
        print("Found 'model' key at top-level.")
    elif "crystal" in full_state and isinstance(full_state["crystal"], dict) and "model" in full_state["crystal"]:
        model_state = full_state["crystal"]["model"]
        print("Found 'model' key under 'crystal'.")
    else:
        print("No 'model' key found in the state log.")
        sys.exit(1)
    
    print("\nKeys in the model state:")
    print(list(model_state.keys()))
    print("\n---\n")
    
    if "elements" not in model_state:
        print("The model state does not contain an 'elements' key.")
        sys.exit(1)
    
    elements = model_state["elements"]
    print("Found 'elements' key in model state.")
    print(f"Number of element groups: {len(elements)}")
    print("\nInspecting each element group for weight attributes...\n")
    
    # Iterate over element groups (usually one per ASU)
    for group_idx, group in enumerate(elements):
        print(f"Element group {group_idx}:")
        if isinstance(group, list):
            for elem_idx, elem in enumerate(group):
                # Print info for a few elements per group
                if elem_idx >= 3:
                    print("  ...")
                    break
                print_element_info(elem, elem_idx)
        else:
            print(f"  Group {group_idx} is not a list (type: {type(group)})")
        print("\n" + "-"*40 + "\n")
    
if __name__ == "__main__":
    sys.exit(main())

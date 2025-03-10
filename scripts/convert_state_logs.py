#!/usr/bin/env python3
"""
Convert legacy state logs to new ObjectSerializer format.

This script scans the logs directory for state log files in the legacy format
and converts them to the new ObjectSerializer format. It creates backup files
of the original logs and replaces them with the new format.

Usage:
    python scripts/convert_state_logs.py [--dir LOGS_DIR] [--backup]
"""

import argparse
import glob
import json
import os
import shutil
import sys
from typing import List, Dict, Any

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eryx.serialization import ObjectSerializer

def convert_file(file_path: str, backup: bool = True) -> bool:
    """
    Convert a single state log file to the new format.
    
    Args:
        file_path: Path to the state log file
        backup: If True, create a backup of the original file
        
    Returns:
        True if conversion was successful, False otherwise
    """
    print(f"Converting {file_path}...")
    
    try:
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            print(f"  Created backup: {backup_path}")
        
        # Read legacy format
        with open(file_path, 'r') as f:
            legacy_data = json.load(f)
        
        # Create serializer
        serializer = ObjectSerializer()
        
        # Convert each attribute
        new_state = {}
        for key, value_hex in legacy_data.items():
            if isinstance(value_hex, str) and len(value_hex) > 0:
                try:
                    # Try to convert hex to binary and deserialize
                    binary_data = bytes.fromhex(value_hex)
                    # Create a proper dict for the serializer
                    serialized_obj = {
                        "__type__": "binary",
                        "__data__": value_hex
                    }
                    value = serializer.deserialize(serialized_obj)
                    new_state[key] = value
                except ValueError:
                    # If not hex, keep as string
                    new_state[key] = value_hex
            else:
                new_state[key] = value_hex
        
        # Add format version marker
        new_state["__format_version__"] = 2
        
        # Write new format
        with open(file_path, 'w') as f:
            serializer.dump(new_state, f)
        
        print(f"  Successfully converted to new format")
        return True
        
    except Exception as e:
        print(f"  Error converting {file_path}: {e}")
        # If we have a backup, restore it
        if backup and os.path.exists(f"{file_path}.bak"):
            shutil.copy2(f"{file_path}.bak", file_path)
            print(f"  Restored original file from backup")
        return False

def find_state_logs(logs_dir: str) -> List[str]:
    """
    Find all state log files in the given directory.
    
    Args:
        logs_dir: Directory to search for state logs
        
    Returns:
        List of paths to state log files
    """
    pattern = os.path.join(logs_dir, "*.log")
    return [f for f in glob.glob(pattern) if "_state_" in f]

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Convert legacy state logs to new format")
    parser.add_argument("--dir", default="logs", help="Directory containing logs")
    parser.add_argument("--backup", action="store_true", help="Create backups of original files")
    args = parser.parse_args()
    
    # Find all state log files
    state_logs = find_state_logs(args.dir)
    if not state_logs:
        print(f"No state logs found in {args.dir}")
        return
    
    print(f"Found {len(state_logs)} state log files")
    
    # Convert each file
    success_count = 0
    for file_path in state_logs:
        if convert_file(file_path, args.backup):
            success_count += 1
    
    print(f"Converted {success_count}/{len(state_logs)} state log files successfully")

if __name__ == "__main__":
    main()

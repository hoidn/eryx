#!/usr/bin/env python3
"""
Script to concatenate files listed in a command file.

This script reads a file containing /add commands and generates a concatenated file
containing the contents of all those files, surrounded by file-path XML tags.

Usage:
    python concat_files.py <command_file> <output_file>
"""

import sys
import os
import re

def process_command_file(command_file_path, output_file_path):
    """
    Process a file containing /add commands and concatenate the referenced files.
    
    Args:
        command_file_path: Path to the file containing /add commands
        output_file_path: Path to the output file where concatenated content will be written
    """
    if not os.path.exists(command_file_path):
        print(f"Error: Command file '{command_file_path}' not found.")
        return False
    
    # Regular expression to match /add commands
    add_pattern = r'^/add\s+(.+)$'
    
    # Read the command file
    with open(command_file_path, 'r') as cmd_file:
        commands = cmd_file.readlines()
    
    # Open the output file
    with open(output_file_path, 'w') as output_file:
        for line in commands:
            match = re.match(add_pattern, line.strip())
            if match:
                file_path = match.group(1).strip()
                
                # Check if the file exists
                if not os.path.exists(file_path):
                    print(f"Warning: File '{file_path}' not found, skipping.")
                    continue
                
                # Write file path XML tag
                output_file.write(f"<file path=\"{file_path}\">\n")
                
                # Read and write the file content
                try:
                    with open(file_path, 'r') as input_file:
                        content = input_file.read()
                        output_file.write(content)
                        
                        # Add a newline if the file doesn't end with one
                        if content and not content.endswith('\n'):
                            output_file.write('\n')
                except Exception as e:
                    print(f"Error reading file '{file_path}': {e}")
                
                # Write closing XML tag
                output_file.write(f"</file>\n\n")
    
    print(f"Concatenated file created at '{output_file_path}'")
    return True

def main():
    """Main function to handle command-line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python concat_files.py <command_file> <output_file>")
        return 1
    
    command_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if process_command_file(command_file, output_file):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

import os
import re
import glob

def clean_up_deprecated_code():
    """
    Identify and remove deprecated code from test files.
    
    This searches test files for common workarounds and removes them,
    replacing them with the new StateBuilder approach.
    
    Specifically identifies:
    1. Custom state loading/initialization methods
    2. Attribute patching workarounds 
    3. Manual tensor conversion logic
    
    Files to check:
    - tests/test_models_torch_*.py
    - Any other test files using state-based testing
    """
    # Helper to identify deprecated patterns
    def has_deprecated_patterns(content):
        patterns = [
            # Custom state methods
            r'def _load_state\(',
            r'def _init_from_state\(',
            r'def _compare_states\(',
            
            # Attribute patching 
            r'if not hasattr\(model, [\'"]A_inv[\'"]\)',
            r'model\.model\.A_inv = model\.A_inv',
            r'wrapper\.A_inv = model\.A_inv',
            
            # Manual tensor conversion
            r'torch\.tensor\(.*\.numpy\(\)\)',
            r'tensor\.requires_grad_\(True\)'
        ]
        
        return any(re.search(pattern, content) for pattern in patterns)
    
    # Find test files
    test_files = glob.glob('tests/test_models_torch_*.py')
    
    # Additional test files that might have state-based tests
    other_files = glob.glob('tests/test_*torch*.py')
    test_files.extend([f for f in other_files if f not in test_files])
    
    # Track statistics
    files_with_deprecated_code = []
    
    # Check files
    for file_path in test_files:
        with open(file_path, 'r') as f:
            content = f.read()
            
        if has_deprecated_patterns(content):
            files_with_deprecated_code.append(file_path)
            print(f"Found deprecated code in {file_path}")
    
    print(f"\nFound {len(files_with_deprecated_code)} files with deprecated code:")
    for file in files_with_deprecated_code:
        print(f"  - {file}")
        
    print("\nTo clean up, update these files to use the StateBuilder pattern:")
    print("  1. Remove custom state methods (_load_state, _init_from_state, _compare_states)")
    print("  2. Remove attribute patching workarounds")
    print("  3. Replace with imports from eryx.autotest.test_helpers")
    print("  4. Follow the standard test pattern from docs/state_based_testing.md")
    
    return files_with_deprecated_code

if __name__ == "__main__":
    clean_up_deprecated_code()

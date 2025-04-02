# to make them accessible as eryx.run_debug
import sys
import os
import importlib.util

# Get the path to the root run_debug.py file
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_debug_path = os.path.join(root_dir, 'run_debug.py')

# Import the module dynamically
spec = importlib.util.spec_from_file_location('run_debug_root', run_debug_path)
run_debug = importlib.util.module_from_spec(spec)
sys.modules['run_debug_root'] = run_debug
spec.loader.exec_module(run_debug)

# Re-export the functions
run_np = run_debug.run_np
run_torch = run_debug.run_torch
run_torch_with_explicit_q = run_debug.run_torch_with_explicit_q
extract_q_vectors = run_debug.extract_q_vectors
validate_q_vector_consistency = run_debug.validate_q_vector_consistency
setup_logging = run_debug.setup_logging

# Add a main function to allow direct execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run diffuse scattering simulations')
    parser.add_argument('--run-mode', choices=['all', 'np', 'torch', 'torch-explicit-q'], default='all',
                       help='Specify which implementation to run (default: all)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate q-vector consistency across modes')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Run the specified implementation(s)
    if args.run_mode in ['all', 'np']:
        run_np()
    
    if args.run_mode in ['all', 'torch']:
        run_torch()
    
    if args.run_mode in ['all', 'torch-explicit-q']:
        run_torch_with_explicit_q()
    
    # Run validation if requested
    if args.validate and args.run_mode == 'all':
        validate_q_vector_consistency()

# This is a symlink to the root run_debug.py file
# to make it accessible as eryx.run_debug
from .. import run_debug

# Re-export the functions
run_np = run_debug.run_np
run_torch = run_debug.run_torch
run_torch_with_explicit_q = run_debug.run_torch_with_explicit_q
extract_q_vectors = run_debug.extract_q_vectors
validate_q_vector_consistency = run_debug.validate_q_vector_consistency
setup_logging = run_debug.setup_logging

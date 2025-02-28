# Specification: Ground Truth Generation for PyTorch Port
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Generate comprehensive ground truth data from the original NumPy implementation to enable accurate testing and validation of the PyTorch port

## Mid-Level Objectives

- Add `@debug` decorators to all designated functions in the original NumPy implementation
- Configure the autotest framework for capturing function inputs and outputs
- Implement ground truth data generation with multiple parameter sets
- Validate the captured data for completeness

## Implementation Notes

- Directly decorate original functions with `@debug` as specified in project rules
- Do NOT create wrapper functions or duplicate implementations
- Follow exactly the function list in `to_convert.json`
- Reuse the existing `run_np()` function in `run_debug.py` with minimal modifications
- Generate data with at least 3 parameter sets (small, medium, and default)
- Add comprehensive logging to verify data capture
- Ensure `DEBUG_MODE` environment variable is set to "1"

## Context

### Beginning Context

- Original implementation files (`scatter.py`, `map_utils.py`, `models.py`, etc.)
- `run_debug.py` with `run_np()` function
- Existing autotest framework in `eryx/autotest/`
- `to_convert.json` listing functions to be ported

### Ending Context

- Modified source files with `@debug` decorators
- Configuration file for autotest at `eryx/autotest_config.py`
- Modified `run_np()` function supporting parameter variations
- Ground truth generation script at `eryx/scripts/generate_ground_truth.py`
- Generated ground truth data in configured log directory
- Verification of captured data

## Low-Level Tasks
> Ordered from start to finish

1. Add Debug Decorators to Scatter Module Functions
```aider
UPDATE eryx/scatter.py:
    ADD import statement: from eryx.autotest.debug import debug
    
    ADD @debug decorator to the following functions:
    - compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    - structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, U=None, ...)
    - structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None, ...)
    
    DO NOT modify the function implementations, only add the decorators
```

2. Add Debug Decorators to Map Utilities Module Functions
```aider
UPDATE eryx/map_utils.py:
    ADD import statement: from eryx.autotest.debug import debug
    
    ADD @debug decorator to the following functions:
    - generate_grid(A_inv, hsampling, ksampling, lsampling, return_hkl=False)
    - get_symmetry_equivalents(hkl_grid, sym_ops)
    - get_ravel_indices(hkl_grid_sym, sampling)
    - compute_resolution(cell, hkl)
    - get_resolution_mask(cell, hkl_grid, res_limit)
    - get_dq_map(A_inv, hkl_grid)
    - get_centered_sampling(map_shape, sampling)
    - resize_map(new_map, old_sampling, new_sampling)
    
    DO NOT modify the function implementations, only add the decorators
```

3. Add Debug Decorators to OnePhonon Class Methods
```aider
UPDATE eryx/models.py:
    ADD import statement: from eryx.autotest.debug import debug
    
    ADD @debug decorator to the following methods of the OnePhonon class:
    - __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, group_by='asu', res_limit=0., model='gnm', gnm_cutoff=4., gamma_intra=1., gamma_inter=1., batch_size=10000, n_processes=8)
    - _setup(self, pdb_path, expand_p1, res_limit, group_by)
    - _setup_phonons(self, pdb_path, model, gnm_cutoff, gamma_intra, gamma_inter)
    - _build_A(self)
    - _build_M(self)
    - _build_M_allatoms(self)
    - _project_M(self, M_allatoms)
    - _build_kvec_Brillouin(self)
    - _center_kvec(self, x, L)
    - _at_kvec_from_miller_points(self, hkl_kvec)
    - compute_gnm_phonons(self)
    - compute_hessian(self)
    - compute_covariance_matrix(self)
    - apply_disorder(self, rank=-1, outdir=None, use_data_adp=False)
    
    DO NOT modify the method implementations, only add the decorators
```

4. Add Debug Decorators to Base Module Functions
```aider
UPDATE eryx/base.py:
    ADD import statement: from eryx.autotest.debug import debug
    
    ADD @debug decorator to the following functions:
    - compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling, U=None, expand_p1=True, expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8)
    - compute_crystal_transform(pdb_path, hsampling, ksampling, lsampling, U=None, expand_p1=True, res_limit=0, batch_size=5000, n_processes=8)
    - incoherent_sum_real(model, hkl_grid, sampling, U=None, mask=None, batch_size=10000, n_processes=8)
    - incoherent_sum_reciprocal(model, hkl_grid, sampling, U=None, batch_size=10000, n_processes=8)
    
    DO NOT modify the function implementations, only add the decorators
```

5. Add Debug Decorators to Additional Model Classes
```aider
UPDATE eryx/models.py:
    ADD @debug decorator to the following methods in RigidBodyTranslations class:
    - __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_friedel=True, res_limit=0, batch_size=10000, n_processes=8)
    - _setup(self, pdb_path, expand_friedel, res_limit, batch_size, n_processes)
    - apply_disorder(self, sigmas)
    - optimize(self, target, sigmas_min, sigmas_max, n_search=20)
    
    ADD @debug decorator to the following methods in LiquidLikeMotions class:
    - __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, border=1, res_limit=0, batch_size=5000, n_processes=8, asu_confined=False)
    - _setup(self, pdb_path, expand_p1, border, res_limit, batch_size, n_processes, asu_confined)
    - fft_convolve(self, transform, kernel)
    - apply_disorder(self, sigmas, gammas)
    - optimize(self, target, sigmas_min, sigmas_max, gammas_min, gammas_max, ns_search=20, ng_search=10)
    
    ADD @debug decorator to the following methods in RigidBodyRotations class:
    - __init__(self, pdb_path, hsampling, ksampling, lsampling, expand_p1=True, res_limit=0, batch_size=10000, n_processes=8)
    - _setup(self, pdb_path, expand_p1, res_limit)
    - generate_rotations_around_axis(sigma, num_rot, axis=np.array([0,0,1.0]))
    - apply_disorder(self, sigmas, num_rot=100, ensemble_dir=None)
    - optimize(self, target, sigma_min, sigma_max, n_search=20, num_rot=100)
    
    DO NOT modify the method implementations, only add the decorators
```

6. Add Debug Decorators to PDB Module Functions
```aider
UPDATE eryx/pdb.py:
    ADD import statement: from eryx.autotest.debug import debug
    
    ADD @debug decorator to the following functions and methods:
    - sym_str_as_matrix(sym_str)
    
    In AtomicModel class:
    - _get_xyz_asus(self, xyz)
    - flatten_model(self)
    
    In Crystal class:
    - get_asu_xyz(self, asu_id=0, unit_cell=None)
    
    In GaussianNetworkModel class:
    - __init__(self, pdb_path, enm_cutoff, gamma_intra, gamma_inter)
    - _setup_atomic_model(self, pdb_path)
    - _setup_gaussian_network_model(self)
    - build_gamma(self)
    - build_neighbor_list(self)
    - compute_hessian(self)
    - compute_K(self, hessian, kvec=None)
    - compute_Kinv(self, hessian, kvec=None, reshape=True)
    
    DO NOT modify the function implementations, only add the decorators
```

7. Add Debug Decorators to Reference Module Functions
```aider
UPDATE eryx/reference.py:
    ADD import statement: from eryx.autotest.debug import debug
    
    ADD @debug decorator to the following functions:
    - structure_factors(q_grid, xyz, elements, U=None)
    - diffuse_covmat(q_grid, xyz, elements, V)
    
    DO NOT modify the function implementations, only add the decorators
```

8. Create Autotest Configuration
```aider
CREATE eryx/autotest_config.py:
    IMPLEMENT a configuration file that:
    - Imports Configuration from eryx.autotest.configuration
    - Sets debug mode to True
    - Sets log directory to "ground_truth_data"
    - Ensures the log directory exists
    
    ADD code to query the DEBUG_MODE environment variable and warn if not set
```

9. Modify Run Debug Function for Parameter Variations
```aider
UPDATE eryx/run_debug.py:
    MODIFY run_np function to:
    - Accept an optional 'variant' parameter for different test cases
    - Support 'small', 'medium', and default parameter sets
    - Log parameter choices
    - Save output with variant-specific filename
    
    DO NOT change the core computation logic
    ENSURE the function remains compatible with existing code
```

10. Create Ground Truth Generation Script
```aider
CREATE eryx/scripts/generate_ground_truth.py:
    IMPLEMENT a script that:
    - Configures logging appropriately
    - Imports the run_np function from eryx.run_debug
    - Sets the DEBUG_MODE environment variable to "1" if not already set
    - Creates the output and log directories if they don't exist
    - Runs the function with different parameter sets:
      * Default parameters
      * Small grid parameters
      * Medium grid parameters
    - Logs when each run starts and completes
    - Verifies that log files were created
```

11. Verify Ground Truth Data Capture
```aider
CREATE eryx/scripts/verify_ground_truth.py:
    IMPLEMENT a script that:
    - Searches the ground truth data directory for log files
    - Groups log files by function
    - Counts the number of captured function calls for each function
    - Verifies that important functions have been captured
    - Reports any functions from to_convert.json that are missing logs
    - Prints a summary of the ground truth data capture
```

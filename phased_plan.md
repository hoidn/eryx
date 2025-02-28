# Final Implementation Plan for PyTorch Port


// TODO: the plan is unclear about which components can be tested using 
// the ground truth values generated from running run_np() with @debug outputs,
// vs. which components aren't 1 to 1 with a np / reference implementation 
// component and therefore have to be validated in a different way

## Current Status Overview

- **Completed**: Most PyTorch stub files have been created with placeholders, docstrings, and type hints.
- **Not Started**: Ground truth data generation, adapter implementation, and core function implementation.

## System Boundaries and Differentiability Constraints

Based on plan.md, we must clearly distinguish between components that should remain in NumPy and those that should be converted to PyTorch with differentiability:

### Non-Differentiable Components (Keep in NumPy)
- **Data Loading**: PDB loading, cell parameter extraction, symmetry operations
- **Preprocessing**: Frame extraction, form factor extraction, neighbor list construction
- **Postprocessing**: Result saving, visualization, statistical analysis

### Differentiable Components (Convert to PyTorch)
- **Core Physics Simulation**: Matrix construction, Hessian computation, phonon calculations
- **Structure Factor Calculation**: Form factors, structure factors, complex operations
- **Model Application**: Disorder models (OnePhonon, RigidBodyTranslations, etc.)
- **Grid Operations**: Grid generation, symmetry operations, Fourier transforms

## Revised Implementation Plan

### Phase 1: Ground Truth Generation (1 week)

#### Task 1.1: Add Debug Decorators to Source Files
- Directly add `@debug` decorators to all functions identified in `to_convert.json`
- Modify the original source files in place to ensure direct decoration
- Add import statements: `from eryx.autotest.debug import debug`

**Key Files to Modify (based on to_convert.json prioritization):**

1. `eryx/scatter.py`:
   ```python
   @debug
   def compute_form_factors(q_grid, ff_a, ff_b, ff_c): ...

   @debug
   def structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, U=None, ...): ...

   @debug
   def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None, ...): ...
   ```

2. `eryx/map_utils.py`:
   ```python
   @debug
   def generate_grid(A_inv, hsampling, ksampling, lsampling, return_hkl=False): ...

   @debug
   def get_symmetry_equivalents(hkl_grid, sym_ops): ...

   # Continue with other functions from to_convert.json
   ```

3. `eryx/models.py` - Add decorators to OnePhonon class methods:
   ```python
   class OnePhonon:
       @debug
       def __init__(self, pdb_path, hsampling, ksampling, lsampling, ...): ...
       
       @debug
       def _setup(self, pdb_path, expand_p1, res_limit, group_by): ...
       
       # Continue with other methods from to_convert.json
   ```

4. Additional files as specified in `to_convert.json`

#### Task 1.2: Configure Autotest Framework
- Create configuration file for autotest at `eryx/autotest_config.py`
- Ensure debug mode is enabled and log directory is properly set

#### Task 1.3: Generate Ground Truth Data

- Leverage the existing `run_np()` function in `run_debug.py` instead of creating a new script
- Modify the function slightly to ensure comprehensive ground truth data generation:

```python
# In run_debug.py, update run_np() to ensure comprehensive ground truth:

def run_np(variant=None):
    """
    Run NumPy version of diffuse scattering simulation to generate ground truth data.
    The @debug decorators will automatically capture inputs/outputs for testing.
    
    Args:
        variant: Optional string to specify parameter variations for comprehensive testing
    """
    # Base configuration
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    if variant == "small":
        # Small grid for quick testing
        hsampling, ksampling, lsampling = [-2, 2, 2], [-2, 2, 2], [-2, 2, 2]
    elif variant == "medium":
        # Medium grid with different parameters
        hsampling, ksampling, lsampling = [-4, 4, 3], [-8, 8, 3], [-8, 8, 3]
        gnm_cutoff, gamma_intra, gamma_inter = 5.0, 0.8, 1.2
    else:
        # Default/full configuration
        hsampling, ksampling, lsampling = [-4, 4, 3], [-17, 17, 3], [-29, 29, 3]
        gnm_cutoff, gamma_intra, gamma_inter = 4.0, 1.0, 1.0
    
    logging.info(f"Starting NP branch computation with variant: {variant}")
    
    # Create model with parameters
    onephonon_np = OnePhonon(
        pdb_path,
        hsampling, ksampling, lsampling,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=gnm_cutoff if 'gnm_cutoff' in locals() else 4.0,
        gamma_intra=gamma_intra if 'gamma_intra' in locals() else 1.0,
        gamma_inter=gamma_inter if 'gamma_inter' in locals() else 1.0
    )
    
    # Apply disorder to generate diffuse intensity
    Id_np = onephonon_np.apply_disorder(use_data_adp=True)
    
    # Log statistics for verification
    logging.info(f"NP branch diffuse intensity stats: min={np.nanmin(Id_np)}, max={np.nanmax(Id_np)}")
    
    # Save results
    output_filename = f"np_diffuse_intensity{'_'+variant if variant else ''}.npy"
    np.save(output_filename, Id_np)
    
    return Id_np
```

- Create a small driver script to run different variants:

```python
# Create scripts/generate_ground_truth.py:

import os
import logging
from eryx.autotest_config import config
from eryx.run_debug import run_np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Autotest debug mode: {config.getDebugFlag()}")
logger.info(f"Log directory: {config.getLogFilePrefix()}")

# Create output directory if needed
os.makedirs(config.getLogFilePrefix(), exist_ok=True)

# Run simulation with different parameter sets to ensure comprehensive coverage
logger.info("Running simulation to generate ground truth data")
run_np()  # Default parameters
run_np("small")  # Small grid for quick testing
run_np("medium")  # Different parameters
logger.info("Ground truth data generation complete")
```

- Verify ground truth data was captured properly:
  - Check that log files were created in the configured log directory
  - Confirm all instrumented functions generated logs
  - Validate that logs contain required input/output data

### Phase 2: Core Utilities Implementation (2 weeks)

#### Task 2.1: Implement Differentiable Tensor Operations
- Complete `ComplexTensorOps` in `eryx/torch_utils.py`:
  - Implement differentiable complex exponential for phase calculations
  - Ensure complex multiplication preserves gradients
  - Create differentiable Debye-Waller factor calculations
- Implement `EigenOps` with special focus on differentiable eigendecomposition:
  - Create SVD-based approach for eigenvalue decomposition
  - Ensure proper backpropagation through eigen operations
  - Document gradient flow and stability considerations
- Document all differentiability decisions

#### Task 2.2: Implement FFT Operations
- Complete `FFTOps` in `eryx/torch_utils.py`:
  - Implement differentiable FFT convolution for LiquidLikeMotions model
  - Create helpers for complex FFT operations
  - Ensure proper normalization that preserves gradients
- Document FFT implementation choices for differentiability

#### Task 2.3: Implement Gradient Utilities
- Complete `GradientUtils` in `eryx/torch_utils.py`:
  - Create tools for finite difference validation
  - Implement gradient norm calculations
  - Add gradient validation helpers
- Document validation methodology

### Phase 3: Adapter Implementation (2 weeks)

#### Task 3.1: Implement PDBToTensor Adapter
- Complete `convert_atomic_model(model)` method for AtomicModel conversion:
  - Convert only differentiable components to tensors
  - Keep non-differentiable components (like neighbor lists) in NumPy
- Implement `convert_crystal(crystal)` and `convert_gnm(gnm)` methods:
  - Document which components should be differentiable
- Support explicit device placement with proper defaults
- Add tests to verify bidirectional conversion

#### Task 3.2: Implement GridToTensor Adapter
- Complete `convert_grid(q_grid, map_shape)` method:
  - Ensure grid generation is differentiable
- Implement `convert_mask(mask)` and `convert_symmetry_ops(sym_ops)` methods:
  - Keep masks non-differentiable (boolean)
  - Ensure symmetry operations preserve gradients
- Add tests for grid conversion

#### Task 3.3: Implement TensorToNumpy and ModelAdapters
- Complete conversions with gradient preservation
- Implement model-specific adapters based on differentiability needs
- Add comprehensive tests for each adapter

### Phase 4: Core Function Implementation (3 weeks)

#### Task 4.1: Implement Map Utilities
- Complete `generate_grid` in `eryx/map_utils_torch.py`:
  - Use differentiable PyTorch grid operations
  - Preserve shape information for gradient flow
- Implement symmetry operations with gradient preservation
- Document grid generation differentiability

#### Task 4.2: Implement Structure Factor Calculations
- Complete `compute_form_factors` in `eryx/scatter_torch.py`:
  - Ensure differentiable form factor calculation
  - Preserve gradients through exponential operations
- Implement `structure_factors_batch` with proper gradient flow:
  - Use complex number operations that preserve gradients
  - Implement differentiable Debye-Waller factor application
- Add device management with proper defaults
- Document structure factor gradient flow

#### Task 4.3: Implement Base Transforms
- Complete `compute_molecular_transform` in `eryx/base_torch.py`:
  - Preserve gradients through transform calculations
- Implement `compute_crystal_transform` with gradient preservation
- Document transform differentiability considerations

### Phase 5: Model Implementation - OnePhonon (3 weeks)

#### Task 5.1: Implement OnePhonon Matrix Construction
- Complete `_build_A` method:
  - Ensure differentiable projection matrix construction
  - Document gradient flow through matrix operations
- Implement `_build_M` method:
  - Create differentiable mass matrix construction
  - Use PyTorch's matrix operations for gradient preservation
- Complete `_build_kvec_Brillouin` with differentiable operations
- Document matrix construction differentiability

#### Task 5.2: Implement OnePhonon Physics Calculations
- Complete `compute_hessian` method:
  - Ensure differentiable Hessian construction
  - Preserve gradients through matrix operations
- Implement `compute_gnm_phonons` with special attention to eigendecomposition:
  - Use differentiable eigendecomposition from `EigenOps`
  - Ensure proper handling of numerical instabilities
  - Document eigendecomposition differentiability approach
- Complete `compute_covariance_matrix` with gradient preservation
- Document physics calculation differentiability

#### Task 5.3: Implement OnePhonon Disorder Application
- Complete `apply_disorder` method:
  - Ensure end-to-end gradient flow from inputs to diffuse intensity
  - Optimize memory usage while preserving computation graph
  - Implement differentiable structure factor operations
- Add tests comparing to ground truth output
- Document differentiability through the complete model

### Phase 6: Alternative Models Implementation (3 weeks)

#### Task 6.1: Implement RigidBodyTranslations Model
- Complete `apply_disorder` method:
  - Implement differentiable Debye-Waller factor calculation
  - Ensure gradient flow through Wilson parameters
- Implement `optimize` method:
  - Add gradient-based optimization option
- Document model-specific differentiability

#### Task 6.2: Implement LiquidLikeMotions Model
- Complete `fft_convolve` method:
  - Use differentiable FFT operations from `FFTOps`
  - Ensure gradient preservation through convolution
- Implement `apply_disorder` method:
  - Preserve gradients through kernel operations
  - Ensure differentiable scaling with displacement parameters
- Document FFT-based differentiability approach

#### Task 6.3: Implement RigidBodyRotations Model
- Complete `generate_rotations_around_axis` method:
  - Ensure differentiable rotation matrix generation
  - Use PyTorch's rotation functions with gradient support
- Implement `apply_disorder` method:
  - Preserve gradients through rotational disorder application
- Document rotational differentiability considerations

### Phase 7: Integration and Optimization (2 weeks)

#### Task 7.1: Complete Run Script
- Finish implementation of `eryx/run_torch.py`:
  - Add device management and error handling
  - Create end-to-end gradient flow demonstration
- Add end-to-end tests with gradient validation

#### Task 7.2: Optimize Performance
- Implement batching for large datasets with gradient preservation
- Optimize memory usage with careful tensor management
- Document optimization decisions and tradeoffs

#### Task 7.3: Validation and Documentation
- Add comprehensive gradient validation
- Complete documentation of all differentiability considerations
- Create example notebooks demonstrating gradient-based optimization

## Implementation Priorities (based on to_convert.json and differentiability requirements)

1. Differentiable tensor operations (especially complex numbers and eigendecomposition)
2. Structure factor calculations with gradient preservation
3. Map utilities with differentiable grid operations
4. OnePhonon model with focus on eigendecomposition and matrix operations
5. Alternative models with model-specific differentiability requirements

## Timeline and Dependencies

- **Phase 1** (1 week): No dependencies, can start immediately
- **Phase 2** (2 weeks): Depends on Phase 1 for ground truth data
- **Phase 3** (2 weeks): Depends on Phase 2 for tensor operations
- **Phase 4** (3 weeks): Depends on Phase 3 for adapter components
- **Phase 5** (3 weeks): Depends on Phase 4 for core functions
- **Phase 6** (3 weeks): Depends on Phase 5 for tensor operations and physics implementations
- **Phase 7** (2 weeks): Depends on all previous phases

**Total Timeline: 16 weeks**

## Model-Specific Differentiability Requirements

### OnePhonon Model
- Matrix operations (`_build_A`, `_build_M`) must preserve gradients
- Eigendecomposition in `compute_gnm_phonons` requires special handling for differentiability
- Covariance matrix calculation must maintain the computational graph
- Complex number operations in structure factors must preserve gradients

### LiquidLikeMotions Model
- FFT convolution must be implemented with gradient preservation
- Kernel operations need to maintain the computational graph
- Complex exponential operations must preserve gradients

### RigidBodyTranslations Model
- Debye-Waller factor calculations must be differentiable
- Wilson parameter scaling must preserve gradients

### RigidBodyRotations Model
- Rotation matrix generation must be differentiable
- Ensemble averaging must preserve gradients when needed

## Documentation Requirements

Throughout all phases:
- Document all architectural decisions with focus on differentiability
- Include mathematical explanations of gradient flow through complex operations
- Document shape information and gradient handling in all operations
- Explain model-specific differentiability considerations
- Provide examples of gradient calculation and propagation

## Device Management Strategy

- All PyTorch implementations should support explicit device placement
- Default to CUDA if available, CPU otherwise
- Implement device context managers for consistent device handling
- Document device placement strategies in all components
- Ensure consistent device handling across model components

## Testing Strategy

- Use ground truth data generated in Phase 1 for all component tests
- Implement tests for each component using `torch_testing.py`
- Add gradient validation tests for all differentiable operations
- Create end-to-end tests that verify gradient flow
- Document testing methodology and tolerance settings

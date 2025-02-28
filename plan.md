# Enhanced Implementation Plan for PyTorch Port

## 1. Call Structure Analysis and System Boundaries

### Core Call Sequence
The simulation follows this primary call structure:

1. **`run_np()`** in `run_debug.py`
   - Sets up logging, parameters
   - Creates OnePhonon instance
   - Calls apply_disorder() to generate diffuse intensity
   - Saves results

2. **`OnePhonon.__init__`** in `models.py`
   - Stores sampling parameters
   - Calls `_setup()` to initialize data structures
   - Calls `_setup_phonons()` to prepare phonon calculations

3. **`OnePhonon._setup()`**
   - Creates AtomicModel from PDB data
   - Generates reciprocal space grid
   - Creates Crystal object with supercell

4. **`AtomicModel.__init__`** in `pdb.py`
   - Loads PDB structure using Gemmi
   - Extracts cell parameters and symmetry operations
   - Processes atomic data (coordinates, form factors)

5. **`OnePhonon._setup_phonons()`**
   - Initializes tensor arrays for calculations
   - Builds displacement projection matrix (A)
   - Builds mass matrix (M)
   - Computes k-vectors in Brillouin zone
   - Sets up Gaussian Network Model
   - Computes phonon modes and covariance matrix

6. **`GaussianNetworkModel.__init__`**
   - Sets up atomic model
   - Builds gamma matrix (spring constants)
   - Builds neighbor lists
   - Provides methods to compute Hessian and dynamical matrices

7. **`OnePhonon.compute_gnm_phonons()`**
   - Gets Hessian matrix
   - For each k-vector, computes dynamical matrix
   - Performs eigendecomposition to get frequencies and modes

8. **`OnePhonon.compute_covariance_matrix()`**
   - Computes atomic displacement covariances from phonon modes
   - Scales to match experimental ADPs

9. **`OnePhonon.apply_disorder()`**
   - Core computation that produces diffuse intensity
   - For each k-vector, computes structure factors
   - Combines structure factors with phonon modes
   - Returns diffuse intensity map

10. **`structure_factors()`** in `scatter.py`
    - Batches calculations for efficiency
    - Computes atomic form factors
    - Applies Debye-Waller factors
    - Computes complex structure factors

### System Boundaries

#### Non-Differentiable Components (Keep in NumPy)
- **Data Loading**:
  - `AtomicModel._get_gemmi_structure()` - PDB loading
  - `AtomicModel._extract_cell()` - Cell parameter extraction
  - `AtomicModel._get_sym_ops()` - Symmetry operation extraction
  - Reading configuration files and parameters

- **Preprocessing**:
  - `AtomicModel.extract_frame()` - Frame extraction
  - `AtomicModel._extract_ff_coefs()` - Form factor coefficient extraction
  - `GaussianNetworkModel.build_neighbor_list()` - Neighbor list construction
  - Initial grid setup and mask creation

- **Postprocessing**:
  - Saving results to disk
  - Visualization operations in `visuals.py`
  - Statistical analysis in `stats.py` (when not in training loop)

#### Differentiable Components (Convert to PyTorch)
- **Core Physics Simulation**:
  - `OnePhonon._build_A()`, `_build_M()` - Matrix construction
  - `OnePhonon.compute_hessian()` - Hessian matrix computation
  - `OnePhonon.compute_gnm_phonons()` - Phonon mode calculation
  - `OnePhonon.compute_covariance_matrix()` - Covariance computation

- **Structure Factor Calculation**:
  - `compute_form_factors()` - Form factor calculation
  - `structure_factors_batch()` - Structure factor computation
  - Complex exponential operations and phase calculations

- **Model Application**:
  - `OnePhonon.apply_disorder()` - Main calculation combining all components
  - Other disorder models (`RigidBodyTranslations.apply_disorder()`, etc.)

- **Grid Operations**:
  - `generate_grid()` - Grid generation
  - `get_symmetry_equivalents()` - Symmetry operations
  - Fourier transforms and convolutions

## 2. Project Directory Structure

```
eryx/
├── __init__.py
├── pdb.py                      # Original NumPy PDB handling
├── pdb_torch.py                # PyTorch adaptation (partial)
├── map_utils.py                # Original NumPy map utilities
├── map_utils_torch.py          # PyTorch map utilities
├── scatter.py                  # Original structure factor calculations
├── scatter_torch.py            # PyTorch structure factor calculations
├── models.py                   # Original disorder models
├── models_torch.py             # PyTorch disorder models
├── base.py                     # Original transform calculations
├── base_torch.py               # PyTorch transform calculations
├── stats.py                    # Original statistical utilities
├── stats_torch.py              # PyTorch statistical utilities
├── reference.py                # Original reference implementations
├── adapters.py                 # NEW: NumPy to PyTorch adapters
│   ├── PDBToTensor             # Convert PDB data to tensors
│   ├── GridToTensor            # Convert grid data to tensors
│   ├── TensorToNumpy           # Convert results back to NumPy
│   └── ModelAdapters           # Adapt between model representations
├── torch_utils.py              # NEW: PyTorch-specific utilities
│   ├── Complex operations      # Handling complex numbers
│   ├── FFT utilities           # Fourier transform operations
│   ├── Gradient utilities      # Gradient calculation helpers
│   └── Eigendecomposition      # Differentiable eigendecomposition
├── logging_utils.py            # Original logging utilities
├── visuals.py                  # Visualization (no port needed)
├── run_debug.py                # Original debugging script
├── run_torch.py                # NEW: PyTorch equivalent of run_debug.py
├── autotest/                   # Testing framework
│   ├── __init__.py
│   ├── debug.py
│   ├── logger.py
│   ├── serializer.py
│   ├── configuration.py
│   ├── functionmapping.py
│   ├── testing.py
│   └── torch_testing.py        # NEW: PyTorch testing extensions
└── tests/                      # Test implementations
    ├── __init__.py
    ├── test_pdb.py
    ├── test_pdb_torch.py       # NEW: PyTorch version of tests
    ├── test_scatter.py
    ├── test_scatter_torch.py   # NEW: PyTorch version of tests
    ├── test_models.py
    ├── test_models_torch.py    # NEW: PyTorch version of tests
    ├── test_base.py
    ├── test_base_torch.py      # NEW: PyTorch version of tests
    ├── test_map_utils.py
    ├── test_map_utils_torch.py # NEW: PyTorch version of tests
    ├── test_adapters.py        # NEW: Tests for adapters
    ├── test_torch_utils.py     # NEW: Tests for PyTorch utilities
    ├── test_integration.py     # NEW: End-to-end integration tests
    └── test_gradients.py       # NEW: Tests for gradient calculation
```

## 3. Phased Implementation Plan

### Phase 1: Create Parallel Project Structure (2 weeks)

#### Tasks
1. **Create Directory Structure**
   - Create parallel `_torch.py` files for all computational components
   - Set up adapter and utility modules

2. **Generate Function Stubs**
   - Create placeholder implementations with docstrings and TODOs
   - Add type hints for PyTorch tensors
   - Define tensor shape specifications in comments

3. **Build Adapter Components**
   - Implement `PDBToTensor` for atomic model conversion
   - Create `GridToTensor` for reciprocal space grid conversion
   - Implement bidirectional converters for all key data structures

4. **Define Testing Configuration**
   - Configure autotest framework for parallel testing
   - Set up data capture for regression testing

#### Deliverables
- Complete directory structure with stub files
- Adapter components with implementation skeletons
- Testing configuration for all components

### Phase 2: Test Generation and Framework (3 weeks)

#### Tasks
1. **Create Test Data Generation Pipeline**
   - Instrument NumPy implementation to capture inputs/outputs
   - Generate test data across a range of parameters
   - Implement serialization for complex data structures

2. **Generate Unit Tests**
   - Create parallel test implementations for each component
   - Implement tolerance-based comparisons for floating-point operations
   - Add gradient checking for differentiable operations

3. **Automated Regression Testing System**
   - Build test runner for parallel NumPy and PyTorch tests
   - Implement detailed reporting with precision analytics
   - Create visualization for test results

4. **End-to-End Test**
   - Create full simulation test harness
   - Implement visualization for comparing outputs
   - Add performance benchmarking

#### Deliverables
- Complete test suite covering all core functions
- Test data generation pipeline
- Regression testing system
- End-to-end test harness

### Phase 3: Implement PyTorch Functionality (6-8 weeks)

#### Tasks
1. **Core Mathematical Components**
   - Implement tensor operations for matrices and vectors
   - Convert eigendecomposition to differentiable operations
   - Ensure complex number handling is differentiable

2. **Structure Factor Calculations**
   - Implement form factor calculations with PyTorch
   - Convert structure factor computation to use tensors
   - Optimize batching for parallel computation

3. **Network Model Components**
   - Implement GaussianNetworkModel in PyTorch
   - Convert Hessian and dynamical matrix calculations
   - Implement covariance calculations

4. **Phonon Calculations**
   - Implement phonon mode calculations
   - Convert frequency and mode computations
   - Ensure eigenvalue decomposition is differentiable

5. **Disorder Models**
   - Implement OnePhonon model in PyTorch
   - Convert other disorder models (RigidBodyTranslations, etc.)
   - Ensure proper gradient flow through all operations

6. **Integration**
   - Connect all components into full simulation
   - Implement PyTorch version of `run_debug.py`
   - Create demonstration notebooks

#### Implementation Order
1. Low-level tensor operations and utilities
2. Form factor and structure factor calculations
3. GaussianNetworkModel implementation
4. Phonon calculations
5. OnePhonon and other disorder models
6. Integration and optimization

#### Deliverables
- Complete PyTorch implementation of all required functions
- Passing test suite demonstrating parity with NumPy
- Performance optimizations for critical operations
- End-to-end differentiable simulation

### Phase 4: Validation and Optimization (2-3 weeks)

#### Tasks
1. **Comprehensive Validation**
   - Run full test suite with diverse inputs
   - Validate numerical accuracy across precision ranges
   - Perform gradient validation with finite differences

2. **Performance Profiling**
   - Profile execution time of critical operations
   - Measure memory usage patterns
   - Identify performance bottlenecks

3. **Optimization**
   - Implement memory efficiency improvements
   - Optimize for GPU execution
   - Add batch processing for parallel computation

4. **Documentation and Examples**
   - Create comprehensive documentation
   - Provide usage examples for training and inference
   - Create gradient visualization tools

#### Deliverables
- Performance analysis report
- Optimized implementation
- Comprehensive documentation
- Example notebooks

## Total Estimated Timeline: 13-17 weeks

https://claude.ai/chat/28eae72e-1cd0-42d2-8012-06a2f1075e02


# Detailed impl Plan 
# Implementation Plan for PyTorch Port

## 1. Reorganized Spec Prompts and Implementation Order

Based on dependencies and logical grouping, I recommend the following reorganized order for the spec prompts:

1. **Test Framework Specification**
   - This needs to be implemented first to establish testing infrastructure
   - Will define how we capture and validate ground truth data

2. **Adapter Component Specification**
   - Essential bridge between NumPy and PyTorch implementations
   - Required by all subsequent components

3. **Grid and Transform Operations Specification**
   - Low-level grid operations used by all physics components
   - Provides foundation for higher-level physics calculations

4. **Core Physics - Structure Factor Calculation Specification**
   - Most fundamental computational component
   - Required by all disorder models

5. **Core Physics - Gaussian Network Model Specification**
   - Network model required for phonon calculations
   - More complex than structure factors but less than full phonon calculations

6. **Core Physics - Phonon Calculations Specification**
   - Central physical calculation for diffuse scattering
   - Builds on GNM and structure factors

7. **Alternative Disorder Models Specification**
   - Additional models that can be implemented after core components
   - Less critical for initial validation

8. **Integration and Execution Specification**
   - Ties all components together for end-to-end execution
   - Depends on all previous components

9. **Optimization and Validation Specification**
   - Performance optimizations after functional implementation
   - Final validation and benchmarking

## 2. Ground Truth Generation Strategy

### Approach
1. **Automatic Instrumentation**
   - Use autotest's Debug decorator to automatically instrument NumPy functions
   - Capture inputs and outputs during normal execution
   - Store serialized data for later testing

2. **Test Data Generation**
   - Create a script to run simplified simulations with various parameters
   - Ensure coverage of edge cases and typical usage patterns
   - Store the generated test data in a structured format

3. **Granularity Levels**
   - Function level: Capture inputs/outputs of individual functions
   - Component level: Capture inputs/outputs of major components
   - System level: Capture full simulation results for end-to-end testing

### Implementation
```python
# Pseudocode for ground truth generation
from eryx.autotest.debug import Debug
import eryx.scatter as scatter
import eryx.models as models

# Configure Debug decorators
debug = Debug().decorate

# Instrument functions
scatter.compute_form_factors = debug(scatter.compute_form_factors)
scatter.structure_factors = debug(scatter.structure_factors)
models.OnePhonon.apply_disorder = debug(models.OnePhonon.apply_disorder)
# ...instrument other functions as needed

# Run test cases with various parameters
def generate_ground_truth():
    # Test case 1: Basic simulation
    model = models.OnePhonon("tests/pdbs/5zck_p1.pdb", 
                            [-4, 4, 3], [-17, 17, 3], [-29, 29, 3],
                            expand_p1=True, res_limit=0.0)
    model.apply_disorder(use_data_adp=True)
    
    # Test case 2: Different parameters
    model = models.OnePhonon("tests/pdbs/5zck_p1.pdb", 
                            [-8, 8, 2], [-8, 8, 2], [-8, 8, 2],
                            expand_p1=False, res_limit=2.0)
    model.apply_disorder(use_data_adp=False)
    
    # Add more test cases as needed
```

## 3. Testing Strategy

### Test Levels
1. **Unit Tests**
   - Test individual PyTorch functions against NumPy implementations
   - Verify numerical accuracy within tolerance
   - Check gradient computation for differentiable operations

2. **Component Tests**
   - Test larger components (e.g., full GNM calculation)
   - Verify interactions between functions
   - Check end-to-end component behavior

3. **Integration Tests**
   - Test full diffuse scattering calculation
   - Compare full maps against NumPy implementation
   - Verify gradient flow through entire computation

### Test Implementation
```python
# Pseudocode for testing PyTorch implementation
from eryx.autotest.torch_testing import TorchTesting
from eryx.autotest.logger import Logger
from eryx.autotest.functionmapping import FunctionMapping

# Set up testing framework
logger = Logger()
function_mapping = FunctionMapping(log_directory="ground_truth")
torch_testing = TorchTesting(logger, function_mapping)

# Test individual functions
def test_compute_form_factors():
    from eryx.scatter_torch import compute_form_factors
    assert torch_testing.testTorchCallable("ground_truth/eryx.scatter.compute_form_factors", 
                                         compute_form_factors)

# Test gradient computation
def test_compute_form_factors_grad():
    from eryx.scatter_torch import compute_form_factors
    import torch
    
    # Create test inputs
    q_grid = torch.randn(10, 3, requires_grad=True)
    ff_a = torch.randn(5, 4, requires_grad=True)
    ff_b = torch.randn(5, 4, requires_grad=True)
    ff_c = torch.randn(5, requires_grad=True)
    
    # Forward pass
    output = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    
    # Check gradients
    grad_ok, stats = torch_testing.check_gradients(
        lambda q, a, b, c: compute_form_factors(q, a, b, c),
        [q_grid, ff_a, ff_b, ff_c]
    )
    assert grad_ok
```

## 4. Spec Prompts Draft

### 1. Test Framework Specification

#### High-Level Objective
- Create a comprehensive test framework for validating PyTorch implementations against NumPy ground truth

#### Mid-Level Objectives
- Extend autotest framework with PyTorch-specific functionality
- Implement automated ground truth data generation
- Create numerical comparison utilities with appropriate tolerances
- Implement gradient validation for differentiable functions

#### Implementation Notes
- Use autotest for serialization and storage of function inputs/outputs
- Implement custom tensor comparison with tolerance settings
- Add gradient checking with finite difference method
- Create structured test data storage

#### Low-Level Tasks
1. Extend TorchTesting class
```
CREATE eryx/autotest/torch_testing.py
    ADD class TorchTesting(Testing):
        ADD methods for tensor comparison
        ADD methods for gradient checking
        ADD methods for NumPy-PyTorch conversion
```

2. Create ground truth generation script
```
CREATE scripts/generate_ground_truth.py
    ADD function to instrument NumPy functions
    ADD function to run test cases with various parameters
    ADD function to verify ground truth data coverage
```

### 2. Adapter Component Specification

#### High-Level Objective
- Create adapter components to bridge between NumPy and PyTorch implementations

#### Mid-Level Objectives
- Design conversion utilities for all data structures
- Implement gradient-preserving tensor conversion
- Create domain-specific adapters for model classes
- Implement robust error handling

#### Implementation Notes
- Ensure all conversions preserve computational graph for gradients
- Handle complex data structures (e.g., nested dictionaries and lists)
- Implement device management for GPU acceleration
- Design clear interfaces for all adapter components

#### Low-Level Tasks
1. Implement core adapter components
```
CREATE eryx/adapters.py
    ADD PDBToTensor class for atomic model conversion
    ADD GridToTensor class for grid data conversion
    ADD TensorToNumpy class for result conversion
    ADD ModelAdapters class for model-specific conversions
```

### 3. Grid and Transform Operations Specification

#### High-Level Objective
- Implement PyTorch versions of grid and transform operations for diffuse scattering

#### Mid-Level Objectives
- Create differentiable grid generation functions
- Implement tensor-based symmetry operations
- Port resolution and masking calculations to PyTorch
- Implement differentiable transformation operations

#### Implementation Notes
- Use torch.meshgrid for grid generation
- Ensure proper gradient flow through all operations
- Implement device-agnostic operations
- Optimize memory usage for large grids

#### Low-Level Tasks
1. Implement grid operations
```
CREATE eryx/map_utils_torch.py
    ADD generate_grid function using PyTorch operations
    ADD get_symmetry_equivalents function for tensor operations
    ADD resolution and masking functions
    ADD helper functions for grid manipulation
```

### Remaining Specs (Abbreviated)

4. **Core Physics - Structure Factor Calculation Specification**
   - Implement PyTorch versions of structure factor calculations
   - Ensure proper handling of complex numbers
   - Optimize batch operations for performance
   - Implement gradient flow through all calculations

5. **Core Physics - Gaussian Network Model Specification**
   - Port GNM calculations to PyTorch
   - Implement tensor-based spring constant matrices
   - Adapt neighbor list handling for PyTorch
   - Ensure differentiable Hessian calculations

6. **Core Physics - Phonon Calculations Specification**
   - Implement differentiable eigendecomposition
   - Port covariance matrix calculations to PyTorch
   - Ensure gradient flow through phonon mode calculations
   - Preserve simulation physics in differentiable form

7. **Alternative Disorder Models Specification**
   - Implement PyTorch versions of all disorder models
   - Ensure consistent interfaces with NumPy versions
   - Adapt optimization routines for gradient-based optimization
   - Handle state management for PyTorch models

8. **Integration and Execution Specification**
   - Implement run_torch.py script
   - Create end-to-end simulation workflow
   - Implement result comparison and visualization
   - Add performance benchmarking

9. **Optimization and Validation Specification**
   - Add performance profiling utilities
   - Implement memory optimization techniques
   - Add GPU acceleration for key operations
   - Create numerical validation methodology

## 5. Implementation Timeline and Dependencies

```
Week 1-2: Test Framework & Adapters
  - Test Framework Specification
  - Adapter Component Specification

Week 3-4: Core Grid Operations
  - Grid and Transform Operations Specification

Week 5-7: Core Physics - Foundation
  - Structure Factor Calculation Specification
  - Gaussian Network Model Specification

Week 8-10: Core Physics - Advanced
  - Phonon Calculations Specification
  - Alternative Disorder Models Specification

Week 11-12: Integration & Optimization
  - Integration and Execution Specification
  - Optimization and Validation Specification

Week 13: Final Testing & Documentation
  - End-to-end validation
  - Performance benchmarking
  - Documentation and examples
```

This implementation plan provides a structured approach to creating a PyTorch port of the diffuse scattering simulation, with a focus on establishing a robust testing framework, creating essential adapter components, and implementing physics calculations in a logical order based on dependencies.

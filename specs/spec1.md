# PyTorch Port Implementation Specification

## High-Level Objective
- Create a parallel PyTorch project structure with comprehensive stub implementations for all components that need to be ported from NumPy to PyTorch

## Mid-Level Objectives
- Create the directory structure for the PyTorch implementation
- Generate stub files for all modules that need PyTorch equivalents
- Implement adapter components to bridge NumPy and PyTorch
- Create comprehensive function stubs with detailed TODOs, docstrings, and type hints
- Set up scaffolding for testing components

## Implementation Notes

### Code Organization
- Create `_torch.py` files parallel to existing NumPy implementations
- Maintain consistent function and class names
- Include imports for both NumPy and PyTorch in stub files

### Documentation Standards
- Each module should have a detailed module docstring
- Each function/method should have a detailed docstring with parameter types, return types, and references to original implementation
- Use Google style docstrings for consistency

### Type Hints
- Use full type hints throughout with `torch.Tensor` instead of `np.ndarray`
- Include shape information in comments
- Use `Optional`, `Union`, etc. as appropriate

### Implementation Considerations
- Clearly mark functions that operate on tensors vs. arrays
- Include placeholder error handling with NotImplementedError
- Consider gradient flow in all operations
- Support explicit device placement (GPU/CPU)

## System Boundaries

### Non-Differentiable Components (Keep in NumPy)
- **Data Loading**: PDB loading, cell parameter extraction, symmetry operations
- **Preprocessing**: Frame extraction, form factor extraction, neighbor list construction
- **Postprocessing**: Result saving, visualization, statistical analysis

### Differentiable Components (Convert to PyTorch)
- **Core Physics Simulation**: Matrix construction, Hessian computation, phonon calculations
- **Structure Factor Calculation**: Form factors, structure factors, complex operations
- **Model Application**: Disorder models (OnePhonon, RigidBodyTranslations, etc.)
- **Grid Operations**: Grid generation, symmetry operations, Fourier transforms

## Directory Structure
```
eryx/
├── models_torch.py              # PyTorch disorder models
├── scatter_torch.py             # PyTorch structure factors
├── map_utils_torch.py           # PyTorch map utilities
├── base_torch.py                # PyTorch transforms
├── stats_torch.py               # PyTorch statistics
├── adapters.py                  # NumPy-PyTorch adapters
├── torch_utils.py               # PyTorch-specific utilities
├── run_torch.py                 # PyTorch run script
├── autotest/
│   └── torch_testing.py         # PyTorch testing extensions
└── tests/
    ├── test_*_torch.py          # PyTorch component tests
    ├── test_adapters.py         # Adapter tests
    ├── test_integration.py      # Integration tests
    └── test_gradients.py        # Gradient tests
```

## Low-Level Tasks

1. **Create Models Module Stubs (`eryx/models_torch.py`)**
   - Create PyTorch versions of OnePhonon, RigidBodyTranslations, LiquidLikeMotions, and RigidBodyRotations classes
   - Include all methods from original classes with tensor-based implementations
   - Ensure all matrix operations use PyTorch functions for differentiability
   - Add detailed TODOs explaining conversion approach for each method
   - Reference original implementation file and line numbers

2. **Create Scatter Module Stub (`eryx/scatter_torch.py`)**
   - Implement PyTorch versions of compute_form_factors, structure_factors_batch, and structure_factors
   - Convert array operations to tensor operations
   - Ensure proper handling of complex numbers for differentiability
   - Replace multiprocessing with PyTorch's parallelization mechanisms
   - Reference original NumPy implementations

3. **Create Map Utilities Stub (`eryx/map_utils_torch.py`)**
   - Implement PyTorch versions of grid generation, symmetry operations, and resolution calculations
   - Convert mesh grid and raveling operations to PyTorch equivalents
   - Ensure proper tensor shape handling
   - Maintain the same function signatures as NumPy versions

4. **Create Base Module Stub (`eryx/base_torch.py`)**
   - Implement PyTorch versions of transform calculations
   - Convert incoherent sum operations to use tensor operations
   - Ensure symmetry operations are differentiable
   - Maintain the same interfaces as NumPy versions

5. **Create Adapter Components (`eryx/adapters.py`)**
   - Implement PDBToTensor, GridToTensor, TensorToNumpy, and ModelAdapters classes
   - Create conversion functions for domain objects and arrays/tensors
   - Ensure gradient preservation during conversion
   - Handle complex data structures like symmetry operations

6. **Create PyTorch Utilities (`eryx/torch_utils.py`)**
   - Implement ComplexTensorOps, FFTOps, EigenOps, and GradientUtils classes
   - Ensure all operations support gradient flow
   - Create helpers for common operations with complex numbers
   - Implement differentiable versions of eigendecomposition

7. **Create Run Script (`eryx/run_torch.py`)**
   - Create PyTorch equivalent of run_debug.py
   - Support device placement (CPU/GPU)
   - Include functions to run and compare NumPy and PyTorch implementations
   - Provide logging similar to NumPy version

8. **Create PyTorch Testing Utilities (`eryx/autotest/torch_testing.py`)**
   - Extend autotest framework with PyTorch-specific functionality
   - Implement tensor comparison with tolerances
   - Create utilities for gradient checking
   - Support NumPy-PyTorch conversion for testing

9. **Create Example Tests (`tests/test_scatter_torch.py`)**
   - Create test stubs for PyTorch scatter functions
   - Include ground truth data generation
   - Implement tensor comparison and gradient validation
   - Set up disabled tests that can be enabled as implementation progresses

## Testing Strategy

### Ground Truth Generation
- Instrument NumPy functions using autotest.debug to capture inputs/outputs
- Run standard test cases to generate ground truth data
- Store serialized data for each function call
- Use captured data to validate PyTorch implementations

### Test Types
- Unit tests for each component
- Tensor comparison tests (tolerance-based)
- Gradient validation tests
- Integration tests for end-to-end simulation

## Final Deliverables

1. **Stub Files**
   - Complete `_torch.py` files with detailed stubs
   - Comprehensive TODOs in all functions
   - Type hints and docstrings that match NumPy versions

2. **Adapter Components**
   - NumPy-to-PyTorch conversion utilities
   - Domain-specific adapters for model classes

3. **Test Scaffolding**
   - Unit test stubs for all components
   - Test data generation utilities
   - PyTorch-specific test utilities

4. **Documentation**
   - Module-level docstrings explaining PyTorch adaptations
   - Function-level docstrings referencing NumPy equivalents
   - Implementation notes on differentiability

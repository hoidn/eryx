# Updated TODOS: PyTorch Port Implementation Tasks

> This file contains a detailed list of tasks derived from the architecture document and phased implementation plan. For detailed component interactions and data flows, see [architecture.md](./architecture.md).

## 1. Core Utilities Implementation (torch_utils.py)

### ComplexTensorOps Class
- [ ] Implement `complex_exp(phase)` to compute e^(i*phase) returning (real, imaginary) parts
- [ ] Implement `complex_mul(a_real, a_imag, b_real, b_imag)` for complex multiplication
- [ ] Implement `complex_abs_squared(real, imag)` to compute |z|²
- [ ] Implement `complex_exp_dwf(q_vec, u_vec)` for Debye-Waller factor calculation
- [ ] Implement `complex_add(a_real, a_imag, b_real, b_imag)` for complex addition
- [ ] Implement `complex_div(a_real, a_imag, b_real, b_imag)` for complex division
- [ ] Add comprehensive tests verifying gradient flow through all operations
- [ ] Document tensor shapes and gradient requirements for all methods

**Component Interactions:**
- Output used by: `scatter_torch.structure_factors_batch`
- Critical for: Phase calculations in structure factors
- Must support backpropagation through complex operations

### EigenOps Class
- [ ] Implement `svd_decomposition(matrix)` for SVD with gradient support
- [ ] Implement `eigen_decomposition(matrix)` for eigendecomposition with gradient support
- [ ] Implement `solve_linear_system(A, b)` to solve Ax=b with gradient support
- [ ] Implement `pseudo_inverse(matrix)` for Moore-Penrose pseudoinverse
- [ ] Implement `stabilized_eigen_decomposition(matrix)` for better gradient stability
- [ ] Implement `hermitian_to_real(matrix)` for handling complex matrices
- [ ] Handle degenerate eigenvalues and numerical stability issues
- [ ] Document limitations and trade-offs in implementation approach

**Component Interactions:**
- Output used by: `OnePhonon.compute_gnm_phonons()`
- Critical for: Phonon mode calculations
- Must support backpropagation through eigendecomposition

### FFTOps Class
- [ ] Implement `fft_convolve(signal, kernel)` for FFT-based convolution
- [ ] Implement `fft_3d(input_tensor)` and `ifft_3d(input_tensor)` for 3D FFT operations
- [ ] Ensure proper normalization that preserves gradients
- [ ] Test with various input sizes and boundary conditions

**Component Interactions:**
- Output used by: `LiquidLikeMotions.fft_convolve()`
- Critical for: Liquid-like motion disorder calculations
- Must preserve gradients through FFT operations

### GradientUtils Class
- [ ] Implement `finite_differences(func, input_tensor)` for numerical gradient calculation
- [ ] Implement `validate_gradients(analytical_grad, numerical_grad)` to compare gradients
- [ ] Implement `gradient_norm(gradient)` to compute gradient L2 norm
- [ ] Implement `compute_jacobian(func, input_tensor)` for Jacobian matrix computation
- [ ] Document appropriate tolerance selection for different use cases

**Component Interactions:**
- Used for: Validating gradients in all model components
- Critical for: Development and testing of gradient implementations
- Ensures gradient calculation correctness

## 2. Adapter Components Implementation (adapters.py)

### PDBToTensor Class
- [ ] Implement `convert_atomic_model(model)` to convert AtomicModel to tensor dictionary
- [ ] Implement `convert_crystal(crystal)` to convert Crystal to tensor dictionary
- [ ] Implement `convert_gnm(gnm)` to convert GaussianNetworkModel to tensor dictionary
- [ ] Implement `array_to_tensor(array, requires_grad)` helper method
- [ ] Implement `convert_dict_of_arrays(dict_arrays)` helper method
- [ ] Support explicit device placement with proper defaults
- [ ] Document tensor shapes, dtypes, and gradient requirements

**Component Interactions:**
- Input from: AtomicModel, Crystal, GaussianNetworkModel
- Output to: PyTorch model implementations
- Preserves structure for backpropagation

### GridToTensor Class
- [ ] Implement `convert_grid(q_grid, map_shape)` to convert q_grid to tensor
- [ ] Implement `convert_mask(mask)` to convert boolean mask to tensor
- [ ] Implement `convert_symmetry_ops(sym_ops)` to convert symmetry operations to tensors
- [ ] Document which components should be differentiable vs. non-differentiable

**Component Interactions:**
- Input from: Grid parameters, resolution masks
- Output to: map_utils_torch functions
- Grid points need gradients, masks typically don't

### TensorToNumpy Class
- [ ] Implement `tensor_to_array(tensor)` to convert tensor to array
- [ ] Implement `convert_dict_of_tensors(dict_tensors)` to convert dictionary of tensors
- [ ] Implement `convert_intensity_map(intensity, map_shape)` for intensity map conversion
- [ ] Document handling of requires_grad and device placement

**Component Interactions:**
- Input from: PyTorch model outputs
- Output to: NumPy arrays for visualization
- One-way conversion (no gradient preservation needed)

### ModelAdapters Class
- [ ] Implement `adapt_one_phonon_inputs(np_model)` for OnePhonon model
- [ ] Implement `adapt_one_phonon_outputs(torch_outputs)` for OnePhonon results
- [ ] Implement similar methods for RigidBodyTranslations, LiquidLikeMotions, RigidBodyRotations
- [ ] Document model-specific conversion requirements

**Component Interactions:**
- Bidirectional flow with: OnePhonon, RigidBodyTranslations, etc.
- Preserves model structure for optimization

## 3. Map Utilities Implementation (map_utils_torch.py)

- [ ] Implement `generate_grid(A_inv, hsampling, ksampling, lsampling)` for q-grid generation
- [ ] Implement `get_symmetry_equivalents(hkl_grid, sym_ops)` for symmetry operations
- [ ] Implement `get_ravel_indices(hkl_grid_sym, sampling)` for index raveling
- [ ] Implement `compute_resolution(cell, hkl)` for resolution calculation
- [ ] Implement `get_resolution_mask(cell, hkl_grid, res_limit)` for masking
- [ ] Implement `get_dq_map(A_inv, hkl_grid)` for distance calculation
- [ ] Implement `get_centered_sampling(map_shape, sampling)` for sampling parameters
- [ ] Implement `resize_map(new_map, old_sampling, new_sampling)` for map resizing
- [ ] Add comprehensive documentation on tensor shapes and gradient flow

**Component Interactions:**
- Input from: GridToTensor adapter
- Output to: scatter_torch, disorder models
- Grid generation must preserve gradients

## 4. Structure Factor Calculation Implementation (scatter_torch.py)

- [ ] Implement `compute_form_factors(q_grid, ff_a, ff_b, ff_c)` using tensor operations
- [ ] Implement `structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c)` with complex operations
- [ ] Implement `structure_factors(q_grid, xyz, ff_a, ff_b, ff_c)` with batch processing
- [ ] Use ComplexTensorOps for all complex number operations
- [ ] Add device management with proper defaults
- [ ] Implement efficient batching strategy for large datasets
- [ ] Document tensor shapes, grad requirements, and potential numerical issues

**Component Interactions:**
- Uses: ComplexTensorOps for complex operations
- Input from: PDBToTensor (atomic data), map_utils_torch (q-grid)
- Output to: OnePhonon, RigidBodyTranslations, etc.
- Complex operations must preserve gradients

## 5. Transform Implementation (base_torch.py)

- [ ] Implement `compute_molecular_transform(pdb_path, hsampling, ksampling, lsampling)` 
- [ ] Implement `compute_crystal_transform(pdb_path, hsampling, ksampling, lsampling)`
- [ ] Implement `incoherent_sum_real(model, hkl_grid, sampling)` with PyTorch operations
- [ ] Implement `incoherent_sum_reciprocal(model, hkl_grid, sampling)` with PyTorch operations
- [ ] Ensure all functions preserve gradient information
- [ ] Document transform calculations and their differentiability

**Component Interactions:**
- Input from: PDBToTensor, map_utils_torch
- Uses: scatter_torch for structure factors
- Output to: Disorder models
- Transform calculations must preserve gradients

## 6. OnePhonon Model Implementation (models_torch.py)

- [ ] Implement `__init__` method with tensor initialization
- [ ] Implement `_setup` method for atomic model and grid setup
- [ ] Implement `_setup_phonons` for phonon calculation initialization
- [ ] Implement `_build_A()` for projection matrix construction
- [ ] Implement `_build_M()` for mass matrix construction
- [ ] Implement `_build_kvec_Brillouin()` for k-vector computation
- [ ] Implement `compute_gnm_phonons()` using EigenOps for eigendecomposition
- [ ] Implement `compute_hessian()` for Hessian matrix construction
- [ ] Implement `compute_covariance_matrix()` for displacement covariances
- [ ] Implement `apply_disorder()` with end-to-end gradient flow
- [ ] Document tensor shapes, grad requirements, and numerical considerations

**Component Interactions:**
- Input from: PDBToTensor (atomic data), GridToTensor (grid data)
- Uses: scatter_torch, EigenOps, ComplexTensorOps
- Output to: TensorToNumpy (diffuse intensity)
- End-to-end gradient flow from parameters to intensity

## 7. RigidBodyTranslations Model Implementation (models_torch.py)

- [ ] Implement `__init__` method with tensor initialization
- [ ] Implement `_setup` method for transform calculation
- [ ] Implement `apply_disorder(sigmas)` with gradient flow to sigmas
- [ ] Implement `optimize(target, sigmas_min, sigmas_max)` with gradient-based option
- [ ] Document tensor operations and parameter gradients

**Component Interactions:**
- Input from: PDBToTensor, GridToTensor
- Uses: scatter_torch, ComplexTensorOps
- Output to: TensorToNumpy
- Gradient flow from sigma parameters to intensity

## 8. LiquidLikeMotions Model Implementation (models_torch.py)

- [ ] Implement `__init__` method with tensor initialization
- [ ] Implement `_setup` method for transform and kernel setup
- [ ] Implement `fft_convolve(transform, kernel)` using FFTOps
- [ ] Implement `apply_disorder(sigmas, gammas)` with gradient preservation
- [ ] Implement `optimize(target, sigmas_min, sigmas_max, gammas_min, gammas_max)` with gradient-based option
- [ ] Document FFT-based calculations and gradient flow

**Component Interactions:**
- Input from: PDBToTensor, GridToTensor
- Uses: scatter_torch, FFTOps, ComplexTensorOps
- Output to: TensorToNumpy
- Gradient flow through FFT convolutions

## 9. RigidBodyRotations Model Implementation (models_torch.py)

- [ ] Implement `__init__` method with tensor initialization
- [ ] Implement `_setup` method for transform setup
- [ ] Implement `generate_rotations_around_axis(sigma, num_rot)` with gradient support
- [ ] Implement `apply_disorder(sigmas, num_rot)` with proper ensemble handling
- [ ] Implement `optimize(target, sigma_min, sigma_max)` with gradient-based option
- [ ] Document rotational gradient considerations

**Component Interactions:**
- Input from: PDBToTensor, GridToTensor
- Uses: scatter_torch, ComplexTensorOps
- Output to: TensorToNumpy
- Gradient flow through rotation generation

## 10. Integration and Testing

- [ ] Implement `run_torch.py` for end-to-end simulation
- [ ] Create comprehensive test suite with component and integration tests
- [ ] Implement performance benchmarking and optimization
- [ ] Create examples of gradient-based parameter optimization
- [ ] Document end-to-end workflows

## Testing Tasks

For each component, implement the following testing tasks:

### Core Utilities Testing
- [ ] Implement unit tests for ComplexTensorOps using known values and gradient validation
- [ ] Implement unit tests for EigenOps using known values and gradient validation
- [ ] Implement unit tests for FFTOps with validation against NumPy FFT
- [ ] Implement unit tests for GradientUtils to verify numerical gradient calculation

### Map Utilities Testing
- [ ] Load ground truth data for `map_utils.generate_grid`
- [ ] Test PyTorch implementation against ground truth with appropriate tolerances
- [ ] Add gradient validation for differentiable grid operations
- [ ] Repeat for all map utility functions with ground truth data

### Structure Factor Testing
- [ ] Load ground truth data for `scatter.compute_form_factors`
- [ ] Test PyTorch implementation against ground truth with appropriate tolerances
- [ ] Add gradient validation for form factor calculations
- [ ] Load ground truth data for `scatter.structure_factors_batch`
- [ ] Test with various input shapes and verify output consistency
- [ ] Add gradient validation for structure factor calculations

### Model Testing
- [ ] Load ground truth data for `models.OnePhonon.compute_gnm_phonons`
- [ ] Test PyTorch eigendecomposition against ground truth with appropriate tolerances
- [ ] Add gradient validation for eigendecomposition
- [ ] Load ground truth data for `models.OnePhonon.apply_disorder`
- [ ] Test end-to-end diffuse scattering calculation
- [ ] Add gradient validation for the complete model

## Test Template

For each component with ground truth data, implement a test function following this template:

```python
def test_component_function():
    """
    Test PyTorch implementation against NumPy ground truth.
    
    This template demonstrates the standard approach for testing
    PyTorch implementations using captured ground truth data.
    """
    # Setup
    logger = Logger()
    function_mapping = FunctionMapping()
    torch_testing = TorchTesting(logger, function_mapping)
    
    # 1. Find ground truth data files for the function
    log_path_prefix = "eryx.module.function"
    log_files = logger.searchLogDirectory(log_path_prefix)
    
    assert len(log_files) > 0, f"No ground truth data found for {log_path_prefix}"
    
    # 2. Create PyTorch implementation instance
    torch_function = function_torch  # The PyTorch implementation to test
    
    # 3. Run test against all ground truth data files
    for log_file in log_files:
        # Test output correctness
        is_valid = torch_testing.testTorchCallable(log_file, torch_function)
        assert is_valid, f"Function output doesn't match ground truth in {log_file}"
        
        # For differentiable functions, also test gradient computation
        if hasattr(torch_function, 'requires_grad'):
            # Load the input data
            logs = logger.loadLog(log_file)
            for i in range(len(logs) // 2):
                # Get inputs from log
                args = logger.serializer.deserialize(logs[2 * i]['args'])
                kwargs = logger.serializer.deserialize(logs[2 * i]['kwargs'])
                
                # Convert to tensors with requires_grad=True
                tensor_args = to_tensors_with_grad(args)
                tensor_kwargs = to_tensors_with_grad(kwargs)
                
                # Define a scalar output function for gradient checking
                def scalar_func(*args, **kwargs):
                    output = torch_function(*args, **kwargs)
                    if isinstance(output, torch.Tensor):
                        return output.sum()
                    else:
                        return sum(o.sum() for o in output if isinstance(o, torch.Tensor))
                
                # Check gradients for tensor inputs
                for arg in tensor_args:
                    if isinstance(arg, torch.Tensor) and arg.requires_grad:
                        # Compute analytical gradients
                        scalar_func(*tensor_args, **tensor_kwargs).backward()
                        analytical_grad = arg.grad.clone()
                        arg.grad.zero_()
                        
                        # Compute numerical gradients
                        numerical_grad = GradientUtils.finite_differences(
                            lambda x: scalar_func(x, *tensor_args[1:], **tensor_kwargs), 
                            arg
                        )
                        
                        # Validate gradients
                        is_valid, stats = GradientUtils.validate_gradients(
                            analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6
                        )
                        
                        assert is_valid, f"Gradient check failed for {log_file}: {stats}"

def to_tensors_with_grad(obj):
    """Helper function to convert NumPy arrays to tensors with requires_grad=True."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj.copy()).requires_grad_(True)
    elif isinstance(obj, list):
        return [to_tensors_with_grad(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_tensors_with_grad(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: to_tensors_with_grad(v) for k, v in obj.items()}
    else:
        return obj
```

## Implementation Priorities

1. **Core Utilities (torch_utils.py)**:
   - ComplexTensorOps - Critical for structure factors and most computations
   - EigenOps - Required for phonon mode calculations
   - FFTOps - Needed for liquid-like motion simulations
   - GradientUtils - Essential for validation during development

2. **Adapter Components (adapters.py)**:
   - PDBToTensor - Foundation for data conversion
   - GridToTensor - Required for proper grid handling
   - TensorToNumpy - Needed for result visualization
   - ModelAdapters - Essential for proper model interfacing

3. **Physics Calculations**:
   - map_utils_torch.py - Grid generation foundation
   - scatter_torch.py - Structure factor calculation core
   - base_torch.py - Transform calculation foundation
   
4. **Model Implementations**:
   - OnePhonon - Most complex model, high priority
   - Alternative disorder models

## Dependency Graph

```
ComplexTensorOps, EigenOps, FFTOps
↓
PDBToTensor, GridToTensor
↓
map_utils_torch, scatter_torch
↓
OnePhonon, RigidBodyTranslations, LiquidLikeMotions, RigidBodyRotations
↓
run_torch.py, Testing Suite
```

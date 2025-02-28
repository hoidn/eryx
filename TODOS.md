
This file contains a detailed list of tasks derived 1-to-1 from the spec prompts outlined in plan.md.

## 1. Test Framework Specification
- **Autotest Extension**: Extend the autotest framework with PyTorch-specific functionality (e.g., implement a TorchTesting class).
- **Numerical Comparison Utilities**: Develop utilities for tolerance-based numerical comparisons between PyTorch and NumPy outputs.
- **Gradient Checking**: Implement gradient checking utilities using finite differences to validate differentiable operations.
- **Ground Truth Generation**: Create a script to generate and serialize ground truth data from NumPy implementations for regression testing.

## 2. Adapter Component Specification
- **PDBToTensor**: Implement conversion of AtomicModel and related crystallographic data from NumPy arrays to PyTorch tensors, preserving the computational graph.
- **GridToTensor**: Create an adapter to convert reciprocal space grids from NumPy arrays to PyTorch tensors.
- **TensorToNumpy**: Implement conversion of PyTorch tensor outputs back to NumPy arrays for visualization and further processing.
- **ModelAdapters**: Develop adapters for model-specific conversions to seamlessly interoperate between NumPy and PyTorch implementations.
- **Error Handling and Device Management**: Ensure robust error handling and proper device (CPU/GPU) assignment during conversions.

## 3. Grid and Transform Operations Specification
- **Grid Generation**: Port grid generation functions (e.g., generate_grid) to PyTorch, ensuring differentiability.
- **Symmetry Operations**: Implement differentiable functions to compute symmetry-equivalent Miller indices and perform raveling using PyTorch.
- **Resolution and Masking Calculations**: Convert resolution computation and masking functions to PyTorch.
- **Transform Operations**: Create differentiable operations for transformations (e.g., Fourier transforms) using PyTorch.

## 4. Core Physics - Structure Factor Calculation Specification
- **compute_form_factors**: Develop a PyTorch version to compute atomic form factors.
- **structure_factors_batch and structure_factors**: Implement PyTorch versions of batch and overall structure factor calculations.
- **Complex Operations and Gradient Flow**: Ensure proper handling of complex number operations and preservation of gradient flow throughout the calculations.

## 5. Core Physics - Gaussian Network Model (GNM) Specification
- **GNM Calculations**: Port Gaussian Network Model calculations to PyTorch, including the construction of tensor-based spring constant matrices.
- **Neighbor List Adaptation**: Adapt neighbor list computation to work with PyTorch tensors.
- **Hessian Computation**: Implement differentiable Hessian computations for the GNM using PyTorch.

## 6. Core Physics - Phonon Calculations Specification
- **Eigendecomposition**: Implement differentiable eigendecomposition (or SVD) using PyTorch functions.
- **Covariance Matrix Calculation**: Port covariance matrix computations to PyTorch.
- **Phonon Mode Calculation**: Ensure that phonon mode calculations (frequency and mode extraction) are differentiable and physically consistent.

## 7. Alternative Disorder Models Specification
- **RigidBodyTranslations**: Implement a PyTorch version of the rigid body translations disorder model.
- **LiquidLikeMotions**: Develop a PyTorch variant of the liquid-like motions disorder model with FFT-based convolution.
- **RigidBodyRotations**: Create a PyTorch implementation for rigid body rotational disorder.
- **API Consistency and Optimization Routines**: Ensure consistency with the NumPy versions and adapt optimization routines for gradient-based optimization; manage internal state appropriately.

## 8. Integration and Execution Specification
- **run_torch.py Script**: Develop the PyTorch equivalent of the simulation script (run_torch.py) for full end-to-end diffuse scattering simulation.
- **Module Integration**: Integrate grid, physics, model, and adapter components into a cohesive PyTorch pipeline.
- **Result Comparison Utilities**: Implement utilities to compare PyTorch outputs with legacy NumPy outputs.
- **Demonstration Notebooks and Visualization**: Create demonstration notebooks and scripts for visualization and workflow presentation.
- **Performance Benchmarking**: Add performance and profiling utilities to measure execution time and memory usage.

## 9. Optimization and Validation Specification
- **Comprehensive Testing**: Develop unit tests, component tests, and integration tests for all PyTorch modules.
- **Performance Profiling**: Implement profiling tools to measure execution time and memory usage; identify bottlenecks.
- **GPU Optimization**: Optimize tensor operations for efficient GPU execution and improved memory usage.
- **Documentation and Examples**: Finalize documentation, update in-code comments, and create user-facing example notebooks.
- **Gradient and Numerical Accuracy Validation**: Validate gradient computations and ensure numerical accuracy throughout the pipeline.

## Additional Tasks (from Spec Prompts)
- **Ground Truth Generation Strategy**: Instrument NumPy functions to capture and serialize ground truth results.
- **Testing Strategy Implementation**: Establish testing pipelines for unit, component, and end-to-end tests as outlined in the plan.
- **Timeline and Dependencies Documentation**: Document project timelines and inter-module dependencies as specified in plan.md.

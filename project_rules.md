# PyTorch Port Project Rules

This document captures the key architectural constraints and project conventions for the PyTorch port of the diffuse scattering simulation codebase.

## Ground Truth and Testing

1. **Direct Decoration of Original Functions**
   - Use `@debug` decorator directly on original functions in the source files
   - Do NOT create wrapper functions or duplicate implementations
   - Add the import statement (`from eryx.autotest.debug import debug`) at the top of each file

2. **Test Ground Truth Generation**
   - Use autotest framework with debug mode enabled
   - Run the original `run_np()` function to generate logs
   - Only setup needed is ensuring autotest is configured for debug mode

3. **Test Verification**
   - Use the autotest framework to validate PyTorch implementations against NumPy ground truth
   - Maintain consistent tolerances for numerical comparisons

## Project Structure

1. **Parallel Implementation**
   - Create `_torch.py` files that directly mirror the original NumPy files
   - Maintain a one-to-one correspondence between NumPy and PyTorch components
   - Each function in the original implementation must have a PyTorch equivalent

2. **Component Mapping**
   - Follow the components identified in the JSON inventory exactly
   - Maintain same function signatures between NumPy and PyTorch versions
   - PyTorch versions may include additional device parameters

3. **Naming Conventions**
   - Use the same function/class names as the original implementation
   - Add `_torch` suffix only when needed to avoid import conflicts
   - Follow the same parameter naming conventions

## Implementation Guidelines

1. **Adapter Pattern**
   - Create adapter components to bridge NumPy and PyTorch
   - Ensure adapters preserve gradient flow
   - Implement bidirectional conversion (NumPy → PyTorch and PyTorch → NumPy)

2. **Differentiability**
   - Ensure all operations in PyTorch implementations are differentiable
   - Use PyTorch native operations wherever possible
   - Avoid in-place operations that might break the computational graph

3. **Phased Approach**
   - First create comprehensive stubs with detailed TODOs
   - Then implement adapters
   - Then implement core functions
   - Finally implement alternative models

4. **Device Management**
   - Support explicit device placement
   - Default to CUDA if available, CPU otherwise
   - Handle device consistently throughout the implementation

## Documentation

1. **Docstrings**
   - Include detailed docstrings for all PyTorch implementations
   - Reference the original NumPy implementation
   - Document parameter and return types with type hints
   - Include shape information in comments

2. **Type Annotations**
   - Use full type hints throughout the PyTorch implementation
   - Use `torch.Tensor` instead of `np.ndarray` for tensor arguments
   - Use `Optional` for parameters that can be None
   - Use `Union` for parameters that can have multiple types

## Performance Considerations

1. **Batching**
   - Implement batching for large datasets
   - Consider GPU memory constraints
   - Optimize batch size for different operations

2. **Memory Efficiency**
   - Minimize unnecessary tensor copying
   - Use in-place operations where appropriate (when not breaking gradient flow)
   - Release resources promptly when no longer needed

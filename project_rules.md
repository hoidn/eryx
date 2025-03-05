# PyTorch Port Project Rules

This document captures the key architectural constraints and project conventions for the PyTorch port of the diffuse scattering simulation codebase.

## Ground Truth and Testing

1. **Direct Decoration of Original Functions**
   - Use `@debug` decorator directly on original functions in the source files
   - Do NOT create wrapper functions or duplicate implementations
   - Add the import statement (`from eryx.autotest.debug import debug`) at the top of each file

2. **State Capture for Stateful Objects**
   - For methods that primarily operate by mutating object state, use state-based testing
   - The `@debug` decorator captures complete object state before and after method execution
   - State logs follow naming convention: `logs/eryx.module.classname._state_before_methodname.log` and `logs/eryx.module.classname._state_after_methodname.log`
   - State capture includes all relevant attributes needed to reconstruct object state

3. **Test Ground Truth Generation**
   - Use autotest framework with debug mode enabled
   - Run the original `run_np()` function to generate logs
   - For stateful objects, state capture is automatic when methods are decorated
   - Configure state capture depth through autotest_config.py
   - For methods that produce large state changes, use selective state capture

4. **Test Verification**
   - For functional components, use the autotest framework to validate PyTorch implementations against NumPy ground truth
   - For stateful components, use state-based testing:
     1. Initialize PyTorch object using the "before" state
     2. Execute the method under test
     3. Compare the resulting state with the expected "after" state
   - Maintain consistent tolerances for numerical comparisons
   - For mixed functional/stateful components, test both outputs and state changes

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
   - For state-based testing, adapters must support complete object state conversion

2. **Differentiability**
   - Ensure all operations in PyTorch implementations are differentiable
   - Use PyTorch native operations wherever possible
   - Avoid in-place operations that might break the computational graph
   - Maintain gradients during state restoration for state-based testing

3. **Phased Approach**
   - First create comprehensive stubs with detailed TODOs
   - Then implement adapters
   - Then implement core functions
   - Finally implement alternative models

4. **Device Management**
   - Support explicit device placement
   - Default to CUDA if available, CPU otherwise
   - Handle device consistently throughout the implementation
   - Ensure state restoration respects device placement

## Documentation

1. **Docstrings**
   - Include detailed docstrings for all PyTorch implementations
   - Reference the original NumPy implementation
   - Document parameter and return types with type hints
   - Include shape information in comments
   - For state-based components, document state dependencies

2. **Type Annotations**
   - Use full type hints throughout the PyTorch implementation
   - Use `torch.Tensor` instead of `np.ndarray` for tensor arguments
   - Use `Optional` for parameters that can be None
   - Use `Union` for parameters that can have multiple types

## State-Based Testing Guidelines

1. **When to Use State-Based Testing**
   - Use for methods that primarily operate by mutating internal state
   - Use for complex objects with interdependent attributes
   - Use when method output alone is insufficient to verify correctness

2. **State Capture Configuration**
   - Configure maximum state capture depth to prevent oversized logs
   - Define attribute filters for extremely large objects
   - Exclude certain attributes from state capture when not relevant for testing

3. **State Restoration Rules**
   - Initialize PyTorch objects with full state before executing test method
   - Convert NumPy arrays to PyTorch tensors with appropriate gradient settings
   - Handle device placement consistently during state restoration

4. **State Comparison Rules**
   - Compare states with appropriate numerical tolerances
   - Ignore attributes explicitly marked for exclusion
   - Report detailed differences when state comparison fails

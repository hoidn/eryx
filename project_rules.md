# PyTorch Port Project Rules

This document captures the key architectural constraints, project conventions, and specification standards for the PyTorch port of the diffuse scattering simulation codebase.

## Specification Standards

### Specification Format

All implementation specifications should follow this standard format:

1. **Title and Description**: Clear title followed by brief description
2. **High-Level Objective**: Single statement defining the overall goal
3. **Mid-Level Objectives**: 3-5 specific, measurable sub-goals
4. **Implementation Notes**: Technical guidance and constraints
5. **Context**: Beginning and ending state descriptions
6. **Low-Level Tasks**: Ordered, implementable steps with code examples

### Specification Principles

When creating specifications, adhere to these principles:

1. **Clarity With Completeness**: Balance clear, focused specifications with sufficient detail for implementation
2. **Lack of Ambiguity**: Ensure specifications have a single, clear interpretation with no undefined behaviors
3. **Minimal Viable Configuration**: Include only essential configuration options
4. **Leverage Existing Components**: Prioritize integrating with existing code over creating new abstractions
5. **Concrete Over Abstract**: Provide specific implementation details rather than conceptual suggestions
6. **Simple Over Complex**: Choose straightforward approaches over clever or sophisticated ones
7. **Practical Timelines**: Set realistic implementation timeframes for each task

### Implementation Task Guidelines

Low-level tasks should:

1. Be clearly ordered with explicit dependencies
2. Include implementation guidance at an appropriate level of specificity
3. Specify exactly what files to create or modify
4. Describe function signatures and key behavior
5. Focus on a single, cohesive implementation unit
6. Be implementable in 1 week maximum
7. Avoid excessive configuration or abstraction

### Low-Level Task Specificity Range

Tasks may range in specificity from structured guidance to complete implementations:

**Level 1: Structured Guidance** (minimum acceptable specificity)
```
CREATE eryx/autotest/state_capture.py:
    IMPLEMENT StateCapture class with:
        - Constructor taking max_depth and exclude_attrs parameters
        - capture_state method that recursively captures object state
        - _should_capture_attr helper method for attribute filtering
        
    ENSURE proper handling of:
        - Callable attributes (skip these)
        - Complex objects like tensors and arrays
        - Maximum recursion depth
        - Private attributes (exclude by default)
```

**Level 2: Detailed Pseudocode**
```
CREATE eryx/autotest/state_capture.py:
    IMPLEMENT StateCapture class:
        def __init__(self, max_depth: int = 10, exclude_attrs: List[str] = None):
            # Store parameters
            # Compile exclude patterns as regexes
            # Initialize serializer
            
        def capture_state(self, obj: Any, current_depth: int = 0) -> Dict[str, Any]:
            # Check max depth
            # For each attribute in dir(obj):
                # Filter attributes based on exclude patterns
                # Skip callable attributes
                # Try to serialize attribute
                # Handle exceptions gracefully
            # Return state dictionary
```

**Level 3: Full Implementation** (maximum specificity)
```
CREATE eryx/autotest/state_capture.py:
    """
    State capture functionality for testing object state before and after method execution.
    """
    import inspect
    import re
    import logging
    from typing import Any, Dict, List, Optional, Pattern, Set, Union
    import numpy as np
    import torch
    from eryx.autotest.serializer import Serializer
    
    class StateCapture:
        """
        Captures the state of Python objects with special handling for PyTorch tensors.
        """
        
        def __init__(self, max_depth: int = 10, exclude_attrs: Optional[List[str]] = None):
            """Initialize state capture with configuration options."""
            self.max_depth = max_depth
            self.exclude_attrs = exclude_attrs or []
            self.serializer = Serializer()
            self.exclude_patterns = [re.compile(pattern) for pattern in self.exclude_attrs]
            
        def capture_state(self, obj: Any, current_depth: int = 0) -> Dict[str, Any]:
            """Capture the state of an object recursively."""
            if current_depth >= self.max_depth:
                return {"__max_depth_reached__": True}
            
            state = {}
            for attr_name in dir(obj):
                if not self._should_capture_attr(attr_name):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    if callable(attr_value):
                        continue
                    state[attr_name] = self.serializer.serialize(attr_value)
                except Exception as e:
                    state[f"__error_{attr_name}__"] = str(e)
            
            return state
            
        def _should_capture_attr(self, attr_name: str) -> bool:
            """Determine if an attribute should be captured."""
            if attr_name.startswith('_'):
                return False
                
            for pattern in self.exclude_patterns:
                if pattern.match(attr_name):
                    return False
                    
            return True
```

### Choosing the Appropriate Specificity Level

* **Use Level 1** when the implementation approach is clear but details may vary
* **Use Level 2** when the implementation structure is important but exact code may differ
* **Use Level 3** when the implementation must exactly match a specific approach

Most tasks should use Level 1 or 2, reserving Level 3 for critical or complex components where implementation details matter significantly.

### Example Specification Elements

**Good High-Level Objective:**
- "Implement automatic gradient checking for all PyTorch tensor operations in the diffuse scattering calculation"

**Poor High-Level Objective:**
- "Improve testing reliability and ensure quality" (too vague)

**Good Mid-Level Objective:**
- "Create a focused StateCapture class that handles object state serialization with minimal configuration"

**Poor Mid-Level Objective:**
- "Implement comprehensive state capture system" (lacks specificity)

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
   - Configure state capture through exclude_attrs and max_depth parameters
   - For methods that produce large state changes, use selective state capture

4. **Test Verification**
   - For functional components, use the autotest framework to validate PyTorch implementations against NumPy ground truth
   - For stateful components, use state-based testing following this pattern:
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
   - Use minimal configuration focused on exclude_attrs and max_depth
   - Default to excluding private attributes
   - Only capture relevant attributes to keep logs manageable

3. **State Serialization**
   - Use ObjectSerializer for consistent serialization across components
   - Ensure proper handling of complex objects (tensors, arrays, custom classes)
   - Maintain type information for accurate reconstruction
   - Handle special cases like NumPy arrays, PyTorch tensors, and Gemmi objects

4. **State Restoration Rules**
   - Use StateBuilder to initialize PyTorch objects with proper structure
   - Convert NumPy arrays to PyTorch tensors with appropriate gradient settings
   - Use ensure_tensor() to handle different tensor formats consistently
   - Handle device placement consistently during state restoration

5. **State Comparison Rules**
   - Compare states with appropriate numerical tolerances (typically rtol=1e-5, atol=1e-8)
   - Use np.allclose() for tensor comparisons after detaching and converting to CPU
   - Report detailed differences when state comparison fails
   - Note: Gradient flow is not required for state-restored instances

6. **Verification Process**
   - Use verify_logs.py to check log completeness
   - Use inspect_state_log.py to examine log contents
   - Ensure before/after state pairs exist for all methods
   - Verify all required attributes are present in logs
   - Regenerate logs after significant code changes
   - Do not test gradient flow for state-restored test objects

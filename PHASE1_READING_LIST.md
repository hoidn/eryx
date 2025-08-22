# Phase 1 Implementation: Required Reading List

## Purpose
This document lists all documentation that MUST be read before implementing Phase 1 vectorization. Reading these documents will prevent common mistakes and ensure correct implementation.

## Critical Documents (MUST READ)

### 1. Project-Specific Documentation

#### Grid Parameter Understanding
- **`CLAUDE.md`** - Critical API information section
  - Lines 5-61: Grid parameter semantics
  - Understand oversampling factor vs. number of points
  - Review the performance impact table

- **`MEMORY_ANALYSIS.md`** - Why vectorization matters
  - Actual grid sizes and memory usage
  - Performance bottlenecks from nested loops

#### Vectorization Strategy
- **`plans/fixplan.md`** - Complete vectorization strategy
  - Specific line numbers of loops to replace
  - Proposed vectorized implementations
  - Expected performance improvements

- **`plans/vectorization_implementation.md`** - Detailed implementation plan
  - Phase 1 specific tasks and checklist
  - Code organization approach
  - Testing requirements

### 2. Architecture & Design

- **`docs/architecture.md`** - System architecture
  - PyTorch port design
  - Component boundaries
  - Gradient flow requirements
  - Sections: "Gradient Flow", "Component Boundaries"

- **`docs/qvec_conventions.md`** - Data flow patterns
  - Grid vs. arbitrary q-vector modes
  - Data transformations
  - Shape conventions

### 3. Gradient Preservation

- **`archive/svd_gradient_fix.md`** - Critical gradient techniques
  - Complex tensor gradient issues
  - SVD workarounds for gradient preservation
  - Testing gradient flow

### 4. Testing Patterns

- **`tests/test_gradient_flow.py`** - Gradient verification
  - How to test gradient preservation
  - Validation patterns
  - Common gradient issues

- **`tests/test_vectorization_equivalence.py`** - Equivalence testing
  - Numerical tolerance requirements (rtol=1e-12)
  - Comparison methods
  - State capture techniques

## Source Code to Study

### Primary Implementation Files

1. **`eryx/models_torch.py`**
   - Lines 1780-1900: `apply_disorder` method (main target)
   - Lines 1826-1828: Triple nested loops (grid mode)
   - Lines 1787-1804: Point-by-point loop (arbitrary q mode)
   - Lines 992-997: Gamma tensor construction

2. **`eryx/scatter_torch.py`**
   - `structure_factors` function
   - `compute_form_factors` function
   - Understand batch processing patterns

3. **`eryx/torch_utils.py`**
   - Tensor utility functions
   - Complex tensor operations
   - Device management patterns

### Reference Implementation

4. **`eryx/models.py`**
   - NumPy implementation for comparison
   - Same algorithms without vectorization
   - Reference for expected behavior

## Key Concepts to Understand

### Before Starting, Ensure You Understand:

1. **Grid Parameter Semantics**
   - Formula: `n_points = (max - min) * oversampling + 1`
   - Why original code has 10-100x more points than expected
   - Memory scaling implications

2. **Current Performance Bottlenecks**
   - Triple nested loops in grid mode
   - Point-by-point processing in arbitrary q mode
   - Memory allocation patterns

3. **Vectorization Strategy**
   - Replace loops with batch operations
   - Use `torch.bmm`, `torch.einsum`
   - Maintain gradient flow through operations

4. **Testing Requirements**
   - Numerical equivalence to 1e-12
   - Gradient flow preservation
   - Memory efficiency validation

## Pre-Implementation Checklist

Before writing ANY code:

- [ ] Read all documents in Critical Documents section
- [ ] Understand grid parameter calculation
- [ ] Located exact lines to modify in models_torch.py
- [ ] Understand gradient flow requirements
- [ ] Reviewed testing patterns
- [ ] Understand memory scaling issues
- [ ] Can explain why vectorization is needed
- [ ] Can describe the vectorization approach

## Common Pitfalls to Avoid

1. **Breaking Gradient Flow**
   - Don't use `.detach()` in forward pass
   - Maintain computational graph
   - Test gradients after each change

2. **Shape Mismatches**
   - Carefully track tensor dimensions
   - Use `.shape` assertions liberally
   - Test with small examples first

3. **Memory Explosions**
   - Don't create full matrices when unnecessary
   - Use batching for large problems
   - Monitor GPU memory usage

4. **Numerical Precision Loss**
   - Maintain float64/complex128 precision
   - Test equivalence with tight tolerances
   - Compare against original implementation

## Questions to Answer Before Starting

1. What is the actual grid size for `hsampling=(-4, 4, 3)`?
   - Answer: (4-(-4))*3+1 = 25 points per dimension, 15,625 total

2. Where are the main performance bottlenecks?
   - Lines 1826-1828 (grid mode loops)
   - Lines 1787-1804 (arbitrary q loops)

3. What is the tolerance for numerical equivalence?
   - rtol=1e-12, atol=1e-14

4. How do we preserve gradient flow?
   - No `.detach()` operations
   - Use differentiable operations only
   - Test with `verify_gradient_flow` helper

## Summary

Phase 1 is the most critical and complex phase. Proper preparation through reading these documents will:
- Prevent the grid parameter confusion that we experienced
- Ensure gradient flow is preserved
- Maintain numerical precision
- Achieve the expected 10-50x performance improvement

**Time Investment**: Plan to spend 1-2 days just reading and understanding before writing any code. This investment will save significant debugging time later.
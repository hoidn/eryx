# PyTorch Port Core Utilities Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Implement the foundational PyTorch utility classes that will enable gradient flow through complex operations in the diffuse scattering calculations

## Mid-Level Objectives

- Implement ComplexTensorOps to support differentiable complex number operations
- Implement EigenOps with differentiable eigendecomposition and SVD operations
- Implement GradientUtils for numerical gradient validation
- Create comprehensive unit tests for each utility class
- Ensure all operations maintain gradient flow for backpropagation

## Implementation Notes

- Use PyTorch 1.9+ tensor operations for all implementations
- Prioritize differentiability over performance for all operations
- Include detailed docstrings explaining gradient flow behavior
- Follow the tensor shape specifications in architecture.md
- Use explicit real/imaginary parts for complex number handling
- Add appropriate error handling for edge cases (e.g., singular matrices)
- Ensure numerical stability for all operations
- All functions should handle arbitrary batch dimensions seamlessly
- Add small epsilon values to denominators to avoid division by zero
- Use logarithmic space for operations prone to overflow/underflow

## Context

### Beginning Context
- `eryx/torch_utils.py` (with stubs for utility classes)
- `eryx/autotest/torch_testing.py` (with testing framework)

### Ending Context
- `eryx/torch_utils.py` (with fully implemented utility classes)
- `tests/test_torch_utils.py` (with comprehensive unit tests)

## Low-Level Tasks
> Ordered from start to finish

1. Implement ComplexTensorOps Class

```aider
UPDATE eryx/torch_utils.py:
    UPDATE class ComplexTensorOps:
        IMPLEMENT complex_exp(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Input: phase tensor of shape (...) with arbitrary batch dimensions, dtype=torch.float32/64
            Output: Tuple of (real, imaginary) tensors, each with shape identical to input
            Implementation approach:
            - Use torch.cos(phase) for real part
            - Use torch.sin(phase) for imaginary part
            - Ensure gradient flows through both components
            - Handle edge cases by ensuring phase is finite (use torch.isfinite if needed)
            
        IMPLEMENT complex_mul(a_real: torch.Tensor, a_imag: torch.Tensor, 
                            b_real: torch.Tensor, b_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Input: Four tensors with broadcastable shapes (...), dtype=torch.float32/64
            Output: Tuple of (real, imaginary) tensors broadcast to match inputs
            Implementation approach:
            - Calculate real part as (a_real * b_real - a_imag * b_imag)
            - Calculate imaginary part as (a_real * b_imag + a_imag * b_real)
            - Support broadcasting if tensor shapes differ
            - Verify all inputs have same device before computation
            - Do not modify input tensors (no in-place operations)
            
        IMPLEMENT complex_abs_squared(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            Input: Two tensors with broadcastable shapes (...), dtype=torch.float32/64
            Output: Single tensor broadcast to match inputs, containing squared magnitudes
            Implementation approach:
            - Calculate as real² + imag²
            - Do not use torch.abs() as it's unnecessary for squared magnitude
            - Handle edge case where both inputs contain very large values by using
              torch.maximum to prevent overflow if needed
            
        IMPLEMENT complex_exp_dwf(q_vec: torch.Tensor, u_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Input: 
              - q_vec: tensor of shape (..., 3) containing q-vectors, dtype=torch.float32/64
              - u_vec: tensor of shape (..., 3) containing displacement vectors, dtype=torch.float32/64
            Output: 
              - Tuple of (real, imaginary) tensors of shape (...)
            Implementation approach:
            - Calculate qUq = torch.sum(q_vec * u_vec * q_vec, dim=-1)
            - Apply dwf = torch.exp(-0.5 * qUq) for Debye-Waller factor
            - Return (dwf, torch.zeros_like(dwf)) as real and imaginary parts
            - Clip extremely large negative values in -0.5*qUq to prevent underflow
            - Ensure numerical stability with torch.clamp if needed for large values
```

2. Implement EigenOps Class

```aider
UPDATE eryx/torch_utils.py:
    UPDATE class EigenOps:
        IMPLEMENT svd_decomposition(matrix: torch.Tensor, compute_uv: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Input: 
              - matrix: tensor of shape (..., M, N), dtype=torch.float32/64
              - compute_uv: boolean flag to return U and V matrices
            Output: 
              - Tuple of (U, S, V) tensors:
                - U: tensor of shape (..., M, K) where K = min(M, N)
                - S: tensor of shape (..., K) with singular values
                - V: tensor of shape (..., N, K)
            Implementation approach:
            - Use torch.linalg.svd which supports backward() for gradients
            - Set some=False to compute full matrices
            - Handle non-full-rank matrices gracefully
            - Ensure U and V are returned as requested (e.g., U@torch.diag_embed(S)@V.transpose(-2,-1) ≈ matrix)
            - Handle edge case with zero-valued singular values with a small epsilon
            - Ensure gradients are preserved through the operation
        
        IMPLEMENT eigen_decomposition(matrix: torch.Tensor, symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
            Input: 
              - matrix: tensor of shape (..., N, N), dtype=torch.float32/64
              - symmetric: boolean flag indicating if matrix is symmetric
            Output: 
              - eigenvalues: tensor of shape (..., N)
              - eigenvectors: tensor of shape (..., N, N)
            Implementation approach:
            - For symmetric matrices, use torch.linalg.eigh which has better gradient support
            - For non-symmetric matrices, use torch.linalg.eig, but note limited gradient support
            - Sort eigenvalues in descending order (by magnitude) and reorder eigenvectors accordingly
            - Handle degenerate eigenvalues (consider regularization approach from literature)
            - Verify that eigenvectors are normalized
            - Add small regularization to possibly singular matrices
            - Consider providing warning when non-symmetric matrices might have limited gradient flow
        
        IMPLEMENT solve_linear_system(A: torch.Tensor, b: torch.Tensor, rcond: float = 1e-10) -> torch.Tensor:
            Input: 
              - A: tensor of shape (..., M, N), dtype=torch.float32/64
              - b: tensor of shape (..., M, K), dtype=torch.float32/64
              - rcond: threshold for singular values, float
            Output: 
              - x: tensor of shape (..., N, K) solving A @ x = b
            Implementation approach:
            - Use torch.linalg.lstsq for overdetermined/underdetermined systems
            - For square matrices, consider torch.linalg.solve which is more efficient
            - Handle potentially singular matrices by using rcond parameter
            - Verify input dimensions are compatible
            - Consider adding batch dimension support for better parallelization
            - Ensure gradients are preserved through the operation
```

3. Implement GradientUtils Class

```aider
UPDATE eryx/torch_utils.py:
    UPDATE class GradientUtils:
        IMPLEMENT finite_differences(func: Callable[[torch.Tensor], torch.Tensor], 
                                  input_tensor: torch.Tensor, 
                                  eps: float = 1e-6) -> torch.Tensor:
            Input: 
              - func: callable that takes a tensor and returns a tensor 
              - input_tensor: tensor of shape (...), dtype=torch.float32/64
              - eps: step size for finite difference, float
            Output: 
              - gradients: tensor of same shape as input_tensor
            Implementation approach:
            - Implement central difference scheme: (f(x+h) - f(x-h))/(2h)
            - Handle multidimensional inputs by iterating through each element
            - Ensure original tensor is not modified
            - Create copy of input_tensor with requires_grad=False
            - Consider using torch.autograd.grad for comparison if input_tensor.requires_grad
            - Optimize by vectorizing operations where possible
            - Handle edge cases where function might be undefined for certain inputs
        
        IMPLEMENT validate_gradients(analytical_grad: torch.Tensor, 
                                  numerical_grad: torch.Tensor, 
                                  rtol: float = 1e-4, 
                                  atol: float = 1e-6) -> Tuple[bool, torch.Tensor, torch.Tensor]:
            Input: 
              - analytical_grad: tensor of shape (...), dtype=torch.float32/64
              - numerical_grad: tensor of same shape as analytical_grad
              - rtol: relative tolerance for comparison
              - atol: absolute tolerance for comparison
            Output: 
              - valid: boolean indicating if gradients match within tolerance
              - rel_errors: tensor of same shape containing relative errors
              - abs_errors: tensor of same shape containing absolute errors
            Implementation approach:
            - Calculate element-wise absolute error: abs(analytical - numerical)
            - Calculate relative error: abs_error / (abs(numerical) + atol)
            - Compare errors against tolerance thresholds
            - Handle special case where numerical gradient is exactly 0
            - Return detailed error information for debugging
            - Calculate max error for quick assessment
        
        IMPLEMENT gradient_norm(gradient: torch.Tensor, ord: int = 2) -> torch.Tensor:
            Input: 
              - gradient: tensor of any shape, dtype=torch.float32/64
              - ord: order of the norm (1 for L1, 2 for L2, etc.)
            Output: 
              - norm: scalar tensor, dtype=torch.float32/64
            Implementation approach:
            - Use torch.norm with specified order
            - Handle case of empty tensor or tensor with zeros
            - Ensure output is scalar by using torch.sum if needed
            - Consider flatten gradients if they have complex shape
```

4. Create ComplexTensorOps Unit Tests

```aider
CREATE tests/test_torch_utils.py:
    IMPLEMENT test_complex_exp():
        Test cases:
        - Zero phase should return (1, 0)
        - π/2 phase should return (0, 1) within tolerance
        - π phase should return (-1, 0) within tolerance
        - 2π phase should return (1, 0) within tolerance
        - Batch of phases should work correctly
        - Gradient should flow through both real and imaginary components
        - Check with torch.autograd.gradcheck for a simple case
    
    IMPLEMENT test_complex_mul():
        Test cases:
        - Multiplication of (1,0) by (1,0) should give (1,0)
        - Multiplication of (0,1) by (0,1) should give (-1,0)
        - Multiplication of (1,1) by (1,1) should give (0,2)
        - Batch multiplication should work correctly
        - Broadcasting should work for different tensor shapes
        - Gradient should flow through both real and imaginary components
        - Check with torch.autograd.gradcheck for a simple case
    
    IMPLEMENT test_complex_abs_squared():
        Test cases:
        - abs_squared of (3,4) should be 25
        - abs_squared of (0,0) should be 0
        - Batch operation should work correctly
        - Gradient should flow through result
        - Check with torch.autograd.gradcheck for a simple case
    
    IMPLEMENT test_complex_exp_dwf():
        Test cases:
        - Zero displacement should give Debye-Waller factor of 1
        - Increasing displacement should decrease DWF exponentially
        - Batch application should work correctly
        - Gradient should flow through the result
        - Check handling of large displacement values
        - Verify imaginary part is always zero
    
    IMPLEMENT test_complex_operations_gradient_flow():
        Test cases:
        - Create a simple network using complex operations
        - Verify gradient flows through a chain of operations
        - Test composition of all operations together
        - Apply torch.autograd.functional.jacobian to verify full Jacobian
```

5. Create EigenOps Unit Tests

```aider
UPDATE tests/test_torch_utils.py:
    IMPLEMENT test_svd_decomposition():
        Test cases:
        - Verify decomposition on identity matrix returns correct singular values
        - Verify U @ diag(S) @ V.T reconstruction matches original matrix within tolerance
        - Test with rectangular matrices (both tall and wide)
        - Test with known singular values (e.g., diagonal matrix)
        - Verify orthogonality of U and V
        - Verify gradient flows through the operation
        - Test with ill-conditioned matrix (large condition number)
        - Verify decomposition works on batched matrices
    
    IMPLEMENT test_eigen_decomposition():
        Test cases:
        - Verify decomposition on identity matrix gives eigenvalues of 1
        - Verify decomposition on diagonal matrix gives eigenvalues matching diagonal
        - Verify eigenvector reconstruction: V @ diag(λ) @ V.inverse() ≈ A
        - Test with symmetric matrices
        - Test with non-symmetric matrices (if supported)
        - Verify orthogonality of eigenvectors for symmetric matrices
        - Test with matrices having degenerate eigenvalues
        - Verify gradient flows through the operation
        - Verify correct handling of batched matrices
    
    IMPLEMENT test_solve_linear_system():
        Test cases:
        - Verify solution for identity matrix (solution should match right-hand side)
        - Verify solution for diagonal matrix
        - Test with overdetermined system (more equations than unknowns)
        - Test with underdetermined system (more unknowns than equations)
        - Verify A @ x ≈ b within tolerance
        - Test with ill-conditioned matrix
        - Verify handling of singular or near-singular matrices
        - Verify gradient flows through the solution
        - Test with batched matrices and multiple right-hand sides
```

6. Create GradientUtils Unit Tests

```aider
UPDATE tests/test_torch_utils.py:
    IMPLEMENT test_finite_differences():
        Test cases:
        - Verify gradient of f(x) = x² at x=2 is approximately 4
        - Verify gradient of f(x) = sin(x) matches cos(x)
        - Test with multidimensional input (e.g., f(x,y) = x²+y²)
        - Compare with torch.autograd.grad for a differentiable function
        - Test with various step sizes to assess accuracy
        - Verify handling of non-continuous functions
        - Test batch processing capability
    
    IMPLEMENT test_validate_gradients():
        Test cases:
        - Test with identical gradients (should pass validation)
        - Test with slightly different gradients within tolerance
        - Test with gradients outside tolerance (should fail validation)
        - Test with zero gradients (edge case)
        - Test with very large and very small gradients
        - Verify proper calculation of relative and absolute errors
        - Test with different tolerance values
    
    IMPLEMENT test_gradient_norm():
        Test cases:
        - Verify L2 norm of [3,4] is 5
        - Verify L1 norm of [1,2,3] is 6
        - Test with higher-dimensional tensors
        - Test with zero tensor
        - Test with different norm orders (1, 2, inf)
        - Verify behavior with very large values
```

7. Update eryx/__init__.py to Export Utility Classes

```aider
UPDATE eryx/__init__.py:
    ADD the following import and export code within the existing try/except block for torch imports:
    
    # Only execute this block when torch is available
    if HAS_TORCH:
        # Import utility classes from torch_utils.py
        from eryx.torch_utils import ComplexTensorOps
        from eryx.torch_utils import EigenOps
        from eryx.torch_utils import GradientUtils
```

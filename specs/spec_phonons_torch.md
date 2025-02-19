# Specification: Differentiable GNM Phonon Computation in PyTorch 
> Ingest this specification and implement the required Low-Level Tasks to enable differentiable phonon mode calculation in the Torch branch of eryx.

## High-Level Objective
- Convert the Gaussian Network Model phonon computation to use PyTorch, enabling gradient-based optimization of the model parameters (spring constants, cutoffs, etc.)

## Mid-Level Objectives
1. Implement differentiable versions of all matrix operations in the phonon computation chain
2. Maintain complex number support throughout for proper phonon phase handling
3. Ensure all operations preserve the computational graph for backpropagation
4. Match the numerical output of the numpy implementation within specified tolerances

## Implementation Notes

### Dependencies and Requirements
- PyTorch >1.8.0 for complex number support
- All operations must support gradient tracking
- Need complex tensor support for phonon calculations
- Memory usage should be monitored due to computational graph storage

### Technical Details
- Use torch.cfloat dtype for complex tensors
- All numpy operations must be replaced with torch equivalents
- Careful handling of where() operations to maintain gradients
- Consider batched operations where possible for GPU efficiency
- SVD must use torch.linalg.svd() for gradient support

### Code Organization
- Place implementation in eryx/gaussian_network_torch.py
- Follow existing class structure 
- Add debug modes for validation against numpy version
- Include type hints throughout

### Edge Cases to Handle
- Small eigenvalues (<1e-6) need special handling
- Complex conjugate pairs in dynamical matrix
- Degenerate modes
- Numerical stability near k=0

## Context

### Beginning Context
Files that exist:
- eryx/gaussian_network_torch.py (partial implementation)
- eryx/models.py (contains numpy reference)
- eryx/onephonon_torch.py (uses GNM)
- tests/test_gaussian_network_model_torch.py

### Ending Context  
Files after implementation:
- Updated eryx/gaussian_network_torch.py with new functions
- New tests in test_gaussian_network_model_torch.py
- Updated requirements.txt if needed

## Low-Level Tasks
1. Implement Hessian Computation
```aider
UPDATE eryx/gaussian_network_torch.py:
    CREATE compute_hessian_torch(self) -> torch.Tensor:
    """Compute the Hessian matrix using pure torch operations.
    
    Returns:
        torch.Tensor: shape (n_asu, n_dof_per_asu, n_cell, n_asu, n_dof_per_asu)
            Complex tensor containing the Hessian.
    """
    
    Implementation:
    - Convert GNM neighbor list to torch tensors
    - Build sparse Hessian using torch operations
    - Handle periodic boundary conditions
    - Return complex tensor on specified device
```

2. Implement Dynamical Matrix (K-matrix) Computation
```aider
UPDATE eryx/gaussian_network_torch.py:
    CREATE compute_K_torch(self, 
                          hessian: torch.Tensor, 
                          kvec: torch.Tensor = None) -> torch.Tensor:
    """Compute the dynamical matrix K(k) for given k-vector.
    
    Args:
        hessian: The Hessian from compute_hessian_torch()
        kvec: Phonon wavevector, shape (3,)
        
    Returns:
        torch.Tensor: Dynamical matrix for this k-point
    """
    
    Implementation:
    - Handle phase factors using torch.exp()
    - Compute K = sum_d H(d)exp(ik·r_d)
    - Maintain complex phases throughout
    - Support batched k-vectors for efficiency
```

3. Implement Phonon Mode Computation
```aider
UPDATE eryx/gaussian_network_torch.py:
    CREATE compute_gnm_phonons_torch(self):
    """Compute phonon modes using differentiable operations.
    
    Stores:
        self.V: Eigenvectors for each k-point
        self.Winv: Inverse squared frequencies
    """
    
    Implementation:
    - Call compute_hessian_torch()
    - Loop over k-points in Brillouin zone
    - Use torch.linalg.svd() for eigendecomposition
    - Handle small eigenvalues carefully
    - Store results in torch tensors
```

4. Add Utility Functions
```aider
UPDATE eryx/gaussian_network_torch.py:
    CREATE _mass_weight_dynamical_matrix(self, Kmat: torch.Tensor) -> torch.Tensor:
    """Apply mass-weighting to dynamical matrix.
    
    Args:
        Kmat: Raw dynamical matrix
        
    Returns:
        torch.Tensor: Mass-weighted matrix D = L^(-1)·K·L^(-T)
    """
    
    CREATE _process_eigensystem(self, 
                              v: torch.Tensor, 
                              w: torch.Tensor,
                              epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process eigenvalues and vectors from SVD.
    
    Handles:
    - Small eigenvalue replacement
    - Order reversal
    - Proper reshaping
    """
```

5. Implement Tests
```aider
UPDATE tests/test_gaussian_network_model_torch.py:
    CREATE test_hessian_torch():
        "Verify Hessian matches numpy within tolerance"
    
    CREATE test_k_matrix_torch():
        "Check K-matrix computation against numpy"
    
    CREATE test_phonon_modes_torch():
        "Validate phonon frequencies and eigenvectors"
        
    CREATE test_gradient_flow():
        "Verify gradients propagate through the computation"
```

## Expected Changes
1. Add new torch-based GNM computation functions
2. Ensure all operations maintain gradient flow
3. Add comprehensive tests comparing to numpy
4. Add debug logging for validation
5. Update documentation with torch-specific details

## Notes
1. Debug logging should use "DEBUG_HYP_TORCH" prefix
2. Check memory usage with profiling tools
3. Consider adding a "validation mode" that compares against numpy
4. Document assumptions about input tensor shapes
5. May need to handle GPU out-of-memory for large systems

## Success Criteria
1. All tests pass with rtol=1e-5
2. Gradients flow through entire computation
3. Memory usage scales reasonably
4. Matches numpy results on test cases
5. Documentation clearly explains torch-specific behavior


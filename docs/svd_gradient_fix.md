# Fixing SVD Gradient Flow in Complex Tensors

## The Problem

When using PyTorch's SVD operation on complex tensors in the `compute_gnm_phonons` method, we encountered an issue with gradient backpropagation. The error occurs because PyTorch doesn't fully support backpropagation through the complex singular vectors of SVD.

The singular vectors in complex SVD have an arbitrary phase factor (e^{iφ}), making gradients with respect to that phase ill-defined. This is a fundamental mathematical property rather than a PyTorch limitation.

In our diffuse scattering calculations, this manifested as:
- Gradients not flowing back to model parameters
- NaN values appearing during backpropagation
- Inconsistent optimization behavior

## The Solution

Since diffuse scattering calculations only depend on the magnitude of eigenvalues (not their phase), we implemented a solution that bypasses the problematic phase gradients while maintaining physically meaningful gradient flow:

1. **Detach Eigenvectors from Computation Graph**: We compute the SVD operation within a `torch.no_grad()` context to prevent tracking gradients through the eigenvectors:

```python
with torch.no_grad():
    v, w, _ = torch.linalg.svd(Dmat, full_matrices=False)
    # Process eigenvalues and eigenvectors...
```

2. **Recompute Eigenvalues Differentiably**: We then recompute the eigenvalues in a differentiable way using the eigenvector-eigenvalue relationship. This is necessary because the SVD operation with `torch.no_grad()` detaches the singular values from the computation graph:

```python
eigenvalues = []
for i in range(v.shape[1]):
    v_i = v[:, i:i+1]
    # Compute λ_i = v_i† D v_i (maintains gradient flow through magnitudes)
    # This reattaches the eigenvalues to the computation graph
    lambda_i = torch.matmul(torch.matmul(v_i.conj().T, Dmat), v_i).real
    eigenvalues.append(lambda_i[0, 0])
```

3. **Add Numerical Stability**: We added stability controls for large/small values:

```python
# Compute inverses with stability controls
winv_value = 1.0 / (torch.sqrt(torch.abs(eig_values)) ** 2 + 1e-8)
```

## Why This Works

This solution works because:

1. **Physical Relevance**: In diffuse scattering, only the magnitudes of eigenvalues affect the final intensity, not their phases
2. **Gradient Flow Preservation**: We maintain gradient flow through the physically meaningful quantities
3. **Mathematical Correctness**: The eigenvalue-eigenvector relationship λv = Av is preserved
4. **Numerical Stability**: Added epsilon terms prevent division by zero

## Implementation

The fix was implemented in the `compute_gnm_phonons` method in `eryx/models_torch.py`. The key changes include:
- Using `torch.no_grad()` for the initial SVD computation
- Recomputing eigenvalues using the matrix-vector product
- Adding numerical stability terms
- Ensuring proper handling of NaN values

## Testing

The fix was verified by:
1. Running integration tests that compare NumPy and PyTorch implementations
2. Checking gradient flow to model parameters
3. Verifying that optimization procedures converge properly

The tests confirm that gradients now flow correctly through the model while maintaining the physical accuracy of the diffuse scattering calculations.

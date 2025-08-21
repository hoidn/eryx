# PyTorch Vectorization Plan for Eryx

## Executive Summary

The current PyTorch implementation in `eryx/models_torch.py` suffers from significant performance bottlenecks due to extensive use of explicit for loops instead of vectorized tensor operations. This plan outlines a comprehensive vectorization strategy to achieve GPU-efficient computation and improve performance by 10-100x.

## Current Performance Bottlenecks

### 1. Grid Mode Triple Nested Loops (Lines 1826-1828)
```python
for dh in range(h_dim_bz):
    for dk in range(k_dim_bz):
        for dl in range(l_dim_bz):
            # Processes one BZ point at a time
```
**Impact**: Processes ~1000s of Brillouin zone points sequentially instead of in parallel batches.

### 2. Arbitrary Q-Vector Mode Point-by-Point Processing (Lines 1787-1804)
```python
for i in range(valid_indices.numel()):
    F_i = F[i].to(self.complex_dtype)
    V_i = V_valid[i].to(self.complex_dtype)
    # Individual matrix operations per q-point
```
**Impact**: No batching of matrix operations, poor GPU utilization.

### 3. ASU Processing Loops (Line 1852)
```python
for i_asu in range(self.n_asu):
    # Structure factor calculations per ASU
```
**Impact**: ASUs processed individually instead of batched.

### 4. Gamma Tensor Construction (Lines 992-997)
```python
for i_asu in range(self.n_asu):
    for i_cell in range(self.n_cell):
        for j_asu in range(self.n_asu):
            # Scalar assignments
```
**Impact**: Triple nested loops for tensor initialization.

### 5. Hessian Projection with Manual Kronecker Products (Lines 1015-1027)
```python
for i_cell in range(self.n_cell):
    for i_asu in range(self.n_asu):
        for j_asu in range(self.n_asu):
            for i in range(h_block.shape[0]):
                for j in range(h_block.shape[1]):
                    # Manual Kronecker product computation
```
**Impact**: 5-level nested loops instead of efficient tensor operations.

## Vectorization Strategies

### Strategy 1: Batch Grid Mode Processing

**Current Approach**:
- Process one BZ point at a time
- Compute structure factors individually 
- Perform F·V·Winv operations sequentially

**Vectorized Approach**:
```python
# Batch all BZ points
all_bz_indices = torch.arange(total_k_points, device=device)
valid_mask = res_mask  # Applied to all points

# Batch structure factor computation
F_all = compute_structure_factors_batched(
    q_grid[valid_mask], asu_data_batched
)  # Shape: [n_valid, n_asu, n_dof]

# Batch matrix operations using einsum
V_valid = V[valid_mask]  # Shape: [n_valid, n_modes, n_modes]
Winv_valid = Winv[valid_mask]  # Shape: [n_valid, n_modes]

# Vectorized computation
FV = torch.einsum('qad,qdc->qac', F_all, V_valid)
FV_squared = torch.abs(FV)**2
intensity = torch.einsum('qac,qc->q', FV_squared, Winv_valid.real)
```

**Benefits**: 
- Eliminates triple nested loops
- Utilizes GPU tensor cores
- Better memory access patterns
- ~10-50x speedup expected

### Strategy 2: Vectorize Arbitrary Q-Vector Mode

**Current Approach**:
- Loop over each valid q-point
- Individual matrix operations per point

**Vectorized Approach**:
```python
# Batch matrix multiplication
F_valid = F[valid_indices]  # Shape: [n_valid, n_asu*n_dof]
V_valid = V[valid_indices]  # Shape: [n_valid, n_modes, n_modes] 
Winv_valid = Winv[valid_indices]  # Shape: [n_valid, n_modes]

if rank == -1:
    # Batch matrix multiplication for all modes
    FV = torch.bmm(F_valid.unsqueeze(1), V_valid)  # [n_valid, 1, n_modes]
    FV_squared = torch.abs(FV.squeeze(1))**2
    intensity = torch.sum(FV_squared * Winv_valid.real, dim=1)
else:
    # Specific mode calculation
    V_rank = V_valid[:, :, rank]  # [n_valid, n_modes]
    FV = torch.sum(F_valid * V_rank, dim=1)
    intensity = torch.abs(FV)**2 * Winv_valid[:, rank].real
```

**Benefits**:
- Eliminates point-by-point loop
- Uses batch matrix multiplication (torch.bmm)
- ~5-20x speedup expected

### Strategy 3: Vectorize ASU Processing

**Current Approach**:
- Loop over ASUs in structure factor calculation
- Individual calls to structure_factors function

**Vectorized Approach**:
```python
# Stack all ASU data into batch dimensions
xyz_all = torch.stack([asu_data[i]['xyz'] for i in range(n_asu)])  # [n_asu, n_atoms, 3]
ff_a_all = torch.stack([asu_data[i]['ff_a'] for i in range(n_asu)])  # [n_asu, n_atoms]
# ... other form factors

# Batch structure factor computation
F = structure_factors_vectorized(
    q_vectors.unsqueeze(1),  # [n_q, 1, 3] - broadcast over ASUs
    xyz_all.unsqueeze(0),    # [1, n_asu, n_atoms, 3] - broadcast over q
    ff_a_all.unsqueeze(0),   # [1, n_asu, n_atoms]
    # ... other parameters
)  # Output: [n_q, n_asu, n_dof]
```

**Benefits**:
- Eliminates ASU loops
- Leverages broadcasting
- Better memory utilization
- ~3-10x speedup expected

### Strategy 4: Vectorize Gamma Tensor Construction

**Current Approach**:
```python
for i_asu in range(self.n_asu):
    for i_cell in range(self.n_cell):
        for j_asu in range(self.n_asu):
            self.gamma_tensor[i_cell, i_asu, j_asu] = gamma_value
```

**Vectorized Approach**:
```python
# Create gamma tensor using broadcasting
gamma_tensor = torch.full((n_cell, n_asu, n_asu), gamma_inter, 
                         device=device, dtype=dtype)

# Create mask for intra-ASU interactions
ref_cell_mask = torch.arange(n_cell, device=device) == id_cell_ref
asu_diagonal_mask = torch.eye(n_asu, device=device, dtype=torch.bool)
intra_mask = ref_cell_mask[:, None, None] & asu_diagonal_mask[None, :, :]

# Apply intra values using mask
gamma_tensor[intra_mask] = gamma_intra
```

**Benefits**:
- Eliminates triple nested loops
- Uses tensor indexing and masking
- ~100x speedup for tensor construction

### Strategy 5: Vectorize Hessian Operations

**Current Approach**:
- Manual Kronecker product computation with nested loops
- Individual matrix operations

**Vectorized Approach**:
```python
# Use torch.kron for efficient Kronecker products
eye3 = torch.eye(3, device=device, dtype=complex_dtype)
h_block_complex = h_block.to(complex_dtype)
h_expanded = torch.kron(h_block_complex, eye3)

# Batch matrix operations
proj_batch = torch.bmm(
    Amat[i_asu].T.unsqueeze(0).expand(n_cell, -1, -1),
    torch.bmm(h_expanded.unsqueeze(0), Amat[j_asu].unsqueeze(0))
)
```

**Benefits**:
- Uses optimized Kronecker product implementation
- Batch matrix operations
- ~10-50x speedup expected

## Implementation Phases

### Phase 1: Core Computation Vectorization (High Impact)
1. **Vectorize arbitrary q-vector mode intensity calculation** (Strategy 2)
   - Remove point-by-point loop
   - Implement batch matrix operations
   - **Expected**: 5-20x speedup

2. **Vectorize grid mode BZ loops** (Strategy 1)
   - Eliminate triple nested loops
   - Batch all BZ point processing
   - **Expected**: 10-50x speedup

### Phase 2: Structure Factor Optimization (Medium Impact)  
3. **Vectorize ASU processing** (Strategy 3)
   - Batch structure factor calculations across ASUs
   - Optimize memory access patterns
   - **Expected**: 3-10x additional speedup

### Phase 3: Initialization Optimization (Low Impact, Easy Wins)
4. **Vectorize gamma tensor construction** (Strategy 4)
   - Replace loops with tensor operations
   - **Expected**: 100x speedup for initialization

5. **Vectorize hessian operations** (Strategy 5)
   - Use optimized Kronecker products
   - Batch matrix operations
   - **Expected**: 10-50x speedup for setup

## Memory Optimization Considerations

### Current Issues:
- Frequent small tensor allocations in loops
- Poor memory access patterns
- Redundant device transfers

### Optimizations:
- **Pre-allocate result tensors** with correct shapes
- **Use in-place operations** where possible (`torch.add_`, `torch.mul_`)
- **Minimize device transfers** by keeping computation on GPU
- **Use memory-efficient tensor views** instead of copies

## Testing Strategy

### Performance Benchmarking:
```python
# Before/after timing for each phase
def benchmark_vectorization():
    # Test cases: small/medium/large systems
    # Measure: wall time, GPU utilization, memory usage
    # Compare: original vs vectorized implementations
```

### Correctness Validation:
1. **Numerical equivalence tests** between original and vectorized versions
2. **Gradient flow verification** for differentiable parameters
3. **Cross-validation** with NumPy implementation results

### Memory Profiling:
- Monitor GPU memory usage patterns
- Identify memory leaks or excessive allocations
- Optimize tensor memory layouts

## Expected Performance Improvements

| Component | Current | Vectorized | Speedup |
|-----------|---------|------------|---------|
| Grid Mode BZ Loops | O(n³) serial | O(1) batch | 10-50x |
| Arbitrary Q Processing | O(n) serial | O(1) batch | 5-20x |
| ASU Structure Factors | O(n) serial | O(1) batch | 3-10x |
| Gamma Construction | O(n³) serial | O(1) tensor ops | ~100x |
| Hessian Operations | O(n⁵) nested | O(n²) batch | 10-50x |

**Overall Expected Improvement**: 50-200x speedup for typical workloads

## Risk Mitigation

### Numerical Precision:
- Maintain float64/complex128 precision where critical
- Add numerical stability checks for edge cases
- Validate gradient computations

### Memory Management:
- Add GPU memory monitoring and warnings
- Implement fallback to CPU for large problems
- Optimize batch sizes based on available memory

### Backward Compatibility:
- Maintain identical API interfaces
- Preserve existing functionality exactly
- Add performance flags for gradual migration

## Success Metrics

1. **Performance**: >10x speedup on GPU for typical workloads
2. **Accuracy**: Numerical differences <1e-12 compared to original
3. **Memory**: Reduced peak GPU memory usage
4. **Maintainability**: Cleaner, more readable vectorized code
5. **Gradient Flow**: Preserved differentiability for optimization

## Next Steps

1. **Implement Phase 1** (arbitrary q-vector and grid mode vectorization)
2. **Create comprehensive benchmarks** for performance measurement
3. **Validate numerical accuracy** against existing implementations
4. **Profile memory usage** and optimize allocation patterns
5. **Document API changes** and performance characteristics

This vectorization effort will transform the PyTorch implementation from a GPU-inefficient version with explicit loops to a properly vectorized, high-performance scientific computing implementation suitable for production use.
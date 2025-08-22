# Memory Usage Analysis - PyTorch Implementation

## Key Discovery

The third parameter in `hsampling`, `ksampling`, `lsampling` is an **oversampling factor**, not the number of grid points!

### Grid Size Calculation
```
n_points = (max - min) * oversampling + 1
```

### Example
- Parameters: `[-2, 2, 5]`
- Range: 2 - (-2) = 4 Miller indices
- Oversampling: 5x
- Points per dimension: 4 * 5 + 1 = **21 points**
- Total grid: 21³ = **9,261 points**

## Actual Grid Sizes in Benchmarks

| Configuration | Parameters | Actual Grid | Total Points | GPU Memory |
|--------------|------------|-------------|--------------|------------|
| Small | [-1,1,3] | 7x7x7 | 343 | 142 MB |
| Medium | [-2,2,5] | 21x21x21 | 9,261 | 529 MB |
| Large | [-3,3,8] | 49x49x49 | 117,649 | 5,129 MB |

## Memory Breakdown

For the medium grid (9,261 points):
- **q_grid**: [9261, 3] float64 = 0.2 MB
- **kvec**: [125, 3] float64 = 0.003 MB (Brillouin zone)
- **V**: [125, 24, 24] complex128 = 1.1 MB (phonon eigenvectors)
- **Winv**: [125, 24] complex128 = 0.05 MB (phonon eigenvalues)

### Where is the 529 MB coming from?

The memory is likely consumed by:

1. **Structure Factor Calculations**
   - For each q-point, calculate F for all atoms
   - Intermediate tensors during form factor evaluation
   - Batch processing creates large temporary arrays

2. **Nested Loop Inefficiencies**
   - Triple nested loops in grid mode (lines 1826-1828)
   - Creates many intermediate tensors
   - Poor memory reuse patterns

3. **Data Duplication**
   - Multiple copies of atomic data
   - Repeated tensor conversions
   - Gradient tracking overhead

## Performance Impact

The incorrect understanding of grid sizes means:
- We're processing **10-100x more points** than expected
- Memory usage scales cubically with oversampling
- Vectorization is **critical** for performance

## Recommendations

1. **Immediate**: Fix benchmark configurations to use reasonable grid sizes
2. **Phase 1**: Vectorize the triple nested loops to reduce memory overhead
3. **Phase 2**: Implement memory-efficient structure factor batching
4. **Future**: Consider on-the-fly computation vs pre-computation trade-offs
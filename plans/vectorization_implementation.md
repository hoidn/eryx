# Vectorization Implementation Plan

## Overview
This plan implements the vectorization strategies from `fixplan.md` while preserving gradient flow, numerical accuracy, and API compatibility. The implementation follows a phased approach with comprehensive validation at each step.

## Critical Success Factors
- ✅ **Gradient Flow**: End-to-end differentiability preserved
- ✅ **Numerical Accuracy**: Float64/Complex128 precision maintained  
- ✅ **API Compatibility**: No breaking changes to public interfaces
- ✅ **Performance**: >10x speedup on GPU for typical workloads
- ✅ **Validation**: All existing tests pass with vectorized implementation

## Phase 0: Setup and Benchmarking Infrastructure

### Objectives
Establish baseline performance metrics and testing infrastructure before any changes.

### Checklist

#### 0.1 Performance Benchmarking Framework
- [ ] Create `benchmarks/benchmark_base.py` with timing utilities
- [ ] Implement memory profiling utilities for GPU/CPU
- [ ] Create benchmark datasets (small/medium/large systems)
- [ ] Document baseline performance metrics for:
  - [ ] Grid mode execution time
  - [ ] Arbitrary q-vector mode execution time
  - [ ] Peak GPU memory usage
  - [ ] GPU utilization percentage

#### 0.2 Regression Testing Infrastructure
- [ ] Create `tests/test_vectorization_equivalence.py`
- [ ] Implement comparison utilities with tolerance checks (rtol=1e-12, atol=1e-14)
- [ ] Set up state capture for before/after comparisons
- [ ] Create gradient flow verification utilities
- [ ] Document test coverage requirements

#### 0.3 Feature Branch Setup
- [ ] Create feature branch: `feature/pytorch-vectorization`
- [ ] Set up CI/CD for automated testing
- [ ] Configure GPU testing environment
- [ ] Document rollback procedures

### Validation Criteria
- [ ] Benchmark suite runs successfully
- [ ] Baseline metrics documented
- [ ] All existing tests pass
- [ ] Git branch properly configured

---

## Phase 1: Core Intensity Calculation Vectorization

### Objectives
Vectorize the critical intensity calculation loops in both arbitrary q-vector and grid modes.

### 1A: Arbitrary Q-Vector Mode Vectorization

#### Implementation Checklist
- [ ] **Backup original implementation**
  - [ ] Copy `apply_disorder` method to `apply_disorder_legacy`
  - [ ] Add feature flag: `use_vectorized=True` parameter

- [ ] **Vectorize intensity calculation (lines 1787-1804)**
  - [ ] Replace point-by-point loop with batch operations
  - [ ] Implement using `torch.bmm` for batch matrix multiplication
  - [ ] Use `torch.einsum` for tensor contractions
  - [ ] Preserve gradient flow through all operations

- [ ] **Code changes**:
  ```python
  # OLD: for i in range(valid_indices.numel()):
  # NEW: Batch all operations
  ```
  - [ ] Implement batch matrix multiplication for F·V
  - [ ] Vectorize |FV|² calculation
  - [ ] Batch Winv application
  - [ ] Handle rank-specific mode selection

#### Testing Checklist
- [ ] **Numerical equivalence**
  - [ ] Compare with legacy implementation (rtol=1e-12)
  - [ ] Test small system (10 q-points)
  - [ ] Test medium system (1000 q-points)
  - [ ] Test large system (10000 q-points)

- [ ] **Gradient flow verification**
  - [ ] Verify gradients flow to gamma_intra
  - [ ] Verify gradients flow to gamma_inter
  - [ ] Check gradient magnitudes are preserved
  - [ ] Test with optimizer step

- [ ] **Performance validation**
  - [ ] Measure speedup vs legacy
  - [ ] Profile GPU utilization
  - [ ] Check memory usage patterns
  - [ ] Document improvements

### 1B: Grid Mode Vectorization

#### Implementation Checklist
- [ ] **Eliminate triple nested loops (lines 1826-1828)**
  - [ ] Pre-compute all BZ indices as batch
  - [ ] Vectorize BZ point processing
  - [ ] Batch structure factor calculations
  - [ ] Implement parallel intensity computation

- [ ] **Memory optimization**
  - [ ] Pre-allocate result tensors
  - [ ] Use tensor views instead of copies
  - [ ] Minimize intermediate allocations
  - [ ] Implement chunking for large grids

- [ ] **Code structure**:
  - [ ] Create `_compute_intensity_vectorized` method
  - [ ] Batch all BZ points with valid resolution mask
  - [ ] Use broadcasting for efficient computation
  - [ ] Maintain compatibility with resolution limits

#### Testing Checklist
- [ ] **Grid consistency**
  - [ ] Verify grid ordering preserved
  - [ ] Check BZ indexing correctness
  - [ ] Validate resolution masking
  - [ ] Test edge cases (single point, full grid)

- [ ] **Cross-mode validation**
  - [ ] Compare grid vs arbitrary mode results
  - [ ] Verify identical outputs for same q-points
  - [ ] Test mode switching behavior
  - [ ] Validate with different grid sizes

### Phase 1 Validation Gate
- [ ] All unit tests pass
- [ ] Numerical differences < 1e-12
- [ ] Gradient flow verified
- [ ] Performance improvement > 5x
- [ ] Memory usage stable or improved
- [ ] Code review completed

---

## Phase 2: Structure Factor Optimization

### Objectives
Vectorize ASU processing and structure factor calculations.

### Implementation Checklist

#### 2.1 ASU Batching
- [ ] **Stack ASU data for batch processing**
  - [ ] Create `_prepare_asu_batch` method
  - [ ] Stack xyz coordinates: [n_asu, n_atoms, 3]
  - [ ] Stack form factors: [n_asu, n_atoms]
  - [ ] Handle variable atom counts per ASU

- [ ] **Vectorize structure factor calls**
  - [ ] Modify `structure_factors` for batch ASU input
  - [ ] Use broadcasting for q-vector × ASU combinations
  - [ ] Optimize memory layout for GPU access
  - [ ] Preserve phase information

#### 2.2 Form Factor Optimization
- [ ] **Batch form factor calculations**
  - [ ] Vectorize Gaussian form factor evaluation
  - [ ] Implement efficient q² calculation
  - [ ] Use tensor operations for scaling
  - [ ] Cache repeated calculations

#### Testing Checklist
- [ ] **ASU processing validation**
  - [ ] Test single ASU (compatibility)
  - [ ] Test multiple ASUs (2, 4, 8)
  - [ ] Verify symmetry preservation
  - [ ] Check gradient flow through ASUs

- [ ] **Structure factor accuracy**
  - [ ] Compare with NumPy implementation
  - [ ] Test complex phase preservation
  - [ ] Validate form factor values
  - [ ] Check edge cases (q=0, large q)

### Phase 2 Validation Gate
- [ ] Structure factors match reference (rtol=1e-12)
- [ ] ASU gradient flow verified
- [ ] Additional 2-5x speedup achieved
- [ ] Memory footprint acceptable
- [ ] Integration tests pass

---

## Phase 3: Initialization and Setup Optimization

### Objectives
Vectorize gamma tensor construction, hessian operations, and other initialization code.

### Implementation Checklist

#### 3.1 Gamma Tensor Vectorization
- [ ] **Replace triple nested loops (lines 992-997)**
  - [ ] Use tensor initialization with broadcasting
  - [ ] Create mask for intra-ASU interactions
  - [ ] Apply values using advanced indexing
  - [ ] Verify gradient tracking preserved

#### 3.2 Hessian Operations
- [ ] **Optimize Kronecker products**
  - [ ] Replace manual loops with `torch.kron`
  - [ ] Batch matrix multiplications
  - [ ] Use einsum for complex contractions
  - [ ] Optimize memory access patterns

#### 3.3 Eigendecomposition Optimization
- [ ] **Batch eigenvalue computations**
  - [ ] Process unique k-vectors in parallel
  - [ ] Optimize hermitian matrix preparation
  - [ ] Handle edge cases (degenerate eigenvalues)
  - [ ] Preserve gradient flow (use SVD workaround if needed)

### Testing Checklist
- [ ] **Initialization verification**
  - [ ] Gamma tensor values correct
  - [ ] Hessian structure preserved
  - [ ] Eigenvalues/eigenvectors accurate
  - [ ] Gradient flow through initialization

- [ ] **Performance validation**
  - [ ] Measure initialization speedup
  - [ ] Profile memory allocation patterns
  - [ ] Check for memory leaks
  - [ ] Verify GPU utilization

### Phase 3 Validation Gate
- [ ] Initialization 10x+ faster
- [ ] All mathematical operations verified
- [ ] No memory leaks detected
- [ ] Gradient flow complete
- [ ] Code maintainability improved

---

## Phase 4: Integration and Optimization

### Objectives
Integrate all vectorized components, optimize memory usage, and finalize performance improvements.

### Implementation Checklist

#### 4.1 Memory Optimization
- [ ] **Implement intelligent batching**
  - [ ] Add dynamic batch size selection
  - [ ] Implement memory-aware chunking
  - [ ] Add OOM recovery mechanisms
  - [ ] Profile and optimize allocations

#### 4.2 Performance Tuning
- [ ] **GPU optimization**
  - [ ] Optimize tensor memory layout (contiguous)
  - [ ] Minimize CPU-GPU transfers
  - [ ] Use CUDA streams for parallelism
  - [ ] Profile and eliminate bottlenecks

#### 4.3 API Finalization
- [ ] **Clean up interfaces**
  - [ ] Remove legacy code paths
  - [ ] Document vectorized methods
  - [ ] Add performance hints to docstrings
  - [ ] Update type hints

### Testing Checklist
- [ ] **Full integration testing**
  - [ ] Run complete test suite
  - [ ] Validate against production datasets
  - [ ] Test with various system sizes
  - [ ] Verify backward compatibility

- [ ] **Performance benchmarking**
  - [ ] Document final speedup metrics
  - [ ] Create performance comparison report
  - [ ] Profile different hardware (GPU types)
  - [ ] Test scaling characteristics

### Phase 4 Validation Gate
- [ ] All tests pass (100% compatibility)
- [ ] Overall speedup > 10x achieved
- [ ] Memory usage optimized
- [ ] Documentation complete
- [ ] Code review approved

---

## Phase 5: Documentation and Release

### Objectives
Document changes, update user guides, and prepare for release.

### Documentation Checklist
- [ ] **Code documentation**
  - [ ] Update method docstrings
  - [ ] Add vectorization notes
  - [ ] Document performance characteristics
  - [ ] Include usage examples

- [ ] **User documentation**
  - [ ] Update README with performance notes
  - [ ] Add migration guide if needed
  - [ ] Document hardware requirements
  - [ ] Create troubleshooting guide

- [ ] **Developer documentation**
  - [ ] Document vectorization patterns
  - [ ] Add architecture diagrams
  - [ ] Create contribution guidelines
  - [ ] Include benchmark suite docs

### Release Checklist
- [ ] **Pre-release validation**
  - [ ] Run full regression suite
  - [ ] Test on multiple GPU types
  - [ ] Validate CPU fallback
  - [ ] Check memory requirements

- [ ] **Release preparation**
  - [ ] Update version numbers
  - [ ] Create release notes
  - [ ] Tag release in git
  - [ ] Merge to main branch

### Phase 5 Validation Gate
- [ ] Documentation reviewed and approved
- [ ] All benchmarks documented
- [ ] Release notes complete
- [ ] Final testing passed
- [ ] Stakeholder approval obtained

---

## Risk Mitigation

### Rollback Procedures
1. **Feature flag fallback**: Keep `use_vectorized=False` option
2. **Git reversion**: Maintain clean commit history for easy rollback
3. **Legacy preservation**: Keep original implementation available
4. **Gradual rollout**: Test with subset of users first

### Known Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Gradient flow breakage | High | Continuous gradient verification testing |
| Numerical precision loss | High | Strict tolerance testing (1e-12) |
| Memory overflow on GPU | Medium | Dynamic batching and chunking |
| API incompatibility | Medium | Comprehensive integration tests |
| Performance regression (edge cases) | Low | Performance profiling suite |

### Contingency Plans
- **If gradient flow breaks**: Revert to manual operations for affected paths
- **If memory issues arise**: Implement automatic CPU fallback
- **If performance targets not met**: Focus on highest-impact optimizations
- **If numerical accuracy degrades**: Increase precision or use stabilization techniques

---

## Success Metrics

### Performance Targets
- [ ] Grid mode: >10x speedup on GPU
- [ ] Arbitrary q mode: >5x speedup on GPU  
- [ ] Memory usage: <2x peak memory
- [ ] GPU utilization: >80% during computation

### Quality Metrics
- [ ] Test coverage: 100% of vectorized code
- [ ] Numerical accuracy: <1e-12 relative error
- [ ] Gradient verification: 100% of parameters
- [ ] Documentation: All public methods documented

### Timeline Estimates
- **Phase 0**: 2-3 days (setup and benchmarking)
- **Phase 1**: 5-7 days (core vectorization)
- **Phase 2**: 3-4 days (structure factors)
- **Phase 3**: 2-3 days (initialization)
- **Phase 4**: 3-4 days (integration)
- **Phase 5**: 2-3 days (documentation)
- **Total**: 17-24 days

---

## Appendix: Code Snippets

### A. Vectorized Intensity Calculation Example
```python
def apply_disorder_vectorized(self, ...):
    # Batch all valid q-points
    F_valid = F[valid_indices]  # [n_valid, n_asu*n_dof]
    V_valid = self.V[valid_indices]  # [n_valid, n_modes, n_modes]
    Winv_valid = self.Winv[valid_indices]  # [n_valid, n_modes]
    
    # Vectorized computation using einsum
    FV = torch.einsum('qi,qij->qj', F_valid, V_valid)
    intensity = torch.sum(torch.abs(FV)**2 * Winv_valid.real, dim=1)
    
    return intensity
```

### B. Gradient Verification Pattern
```python
def verify_gradient_flow(model, param_name):
    # Create test input
    q_test = torch.randn(100, 3, requires_grad=True)
    
    # Forward pass
    output = model.apply_disorder()
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradient exists
    param = getattr(model, param_name)
    assert param.grad is not None
    assert torch.any(param.grad != 0)
```

### C. Performance Benchmark Pattern
```python
def benchmark_vectorization(model, q_points, num_runs=10):
    # Warmup
    for _ in range(3):
        _ = model.apply_disorder()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_runs):
        output = model.apply_disorder()
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return elapsed / num_runs
```

---

This implementation plan provides a structured, low-risk approach to vectorizing the PyTorch implementation while maintaining all critical requirements and preventing regressions.
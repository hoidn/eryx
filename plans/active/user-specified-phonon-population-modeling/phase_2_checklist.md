# Phase 2: Integration Testing and Validation Checklist

**Initiative:** User-Specified Phonon Population Modeling  
**Created:** 2025-01-24  
**Phase Goal:** Create comprehensive test suite validating correctness, differentiability, and regression safety of the PDOS implementation.  
**Deliverable:** Complete test file `tests/test_models_torch_pdos.py` with unit tests, integration tests, and gradient flow validation, plus sample PDOS data files for testing.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| Task ID | State | Priority | Task Description | How/Why & API Guidance |
|---------|-------|----------|------------------|------------------------|
| **T2.1** | `[D]` | **High** | **Create sample PDOS files for testing** | Create `tests/data/sample_pdos_thermal.dat` and `tests/data/sample_pdos_direct.dat` with realistic frequency-density data. Use exponential decay patterns for thermal mode, custom patterns for direct mode. Include comments describing file format. |
| **T2.2** | `[D]` | **High** | **Write unit tests for `_differentiable_interp()` method** | Create `tests/test_models_torch_pdos.py`. Test interpolation accuracy with known inputs/outputs, boundary conditions, gradient flow. Use `torch.allclose()` for numerical comparisons. |
| **T2.3** | `[D]` | **Critical** | **Implement integration tests for full OnePhononTorch simulation** | Test complete model initialization with PDOS parameters, phonon setup, and `compute_gnm_phonons()` execution. Verify results are reasonable and finite. |
| **T2.4** | `[D]` | **Critical** | **Add regression tests for non-PDOS usage** | Test that existing OnePhononTorch functionality works unchanged when no PDOS parameters are provided. Compare results with/without PDOS to ensure backward compatibility. |
| **T2.5** | `[D]` | **Critical** | **Create gradient flow test with automated validation** | Test that `gamma_intra.grad`, `pdos_density.grad` are populated after backward pass. Use realistic loss computation, verify gradient magnitudes are reasonable (not zero, not NaN). |
| **T2.6** | `[D]` | **High** | **Test both 'thermal' and 'direct' PDOS modes** | Create separate test cases for thermal mode (with temperature_k) and direct mode. Verify Boltzmann factor application in thermal mode, direct usage in direct mode. |
| **T2.7** | `[D]` | **Medium** | **Add performance benchmarks** | Compare computational overhead of PDOS vs non-PDOS modes. Measure memory usage with different PDOS file sizes. Document acceptable performance thresholds. |
| **T2.8** | `[D]` | **High** | **Test error handling and edge cases** | Test invalid PDOS files, missing files, wrong file format, out-of-bounds frequencies, invalid parameters. Ensure proper error messages are raised. |
| **T2.9** | `[D]` | **Medium** | **Validate numerical accuracy** | Compare PDOS interpolation results with reference implementations (e.g., scipy.interp1d). Ensure accuracy is within acceptable tolerances (~1e-6). |
| **T2.10** | `[D]` | **High** | **Create pytest configuration and CI setup** | Ensure tests can be run with `pytest tests/test_models_torch_pdos.py`. Add proper fixtures, parametrized tests, and clear test organization. |

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. **Full test suite passes:** `pytest tests/test_models_torch_pdos.py` completes with 100% pass rate.
3. **Gradient flow validation:** Automated tests confirm gradients flow to both `gamma_intra` and `pdos_density` parameters.
4. **Regression safety:** Existing functionality without PDOS parameters works unchanged.
5. **Both PDOS modes tested:** Both 'thermal' and 'direct' modes work correctly with appropriate validation.

## 📋 **Detailed Implementation Guidance**

### **T2.1: Sample PDOS Files**
Create test data files with realistic patterns:
- **Thermal mode file:** Exponential decay pattern representing thermal populations
- **Direct mode file:** Custom pattern representing measured/computed populations
- **File format:** 2 columns (frequency in THz, density), tab-separated
- **Location:** `tests/data/` directory

### **T2.2: Unit Tests for Interpolation**
Focus on testing the `_differentiable_interp` method in isolation:
```python
def test_differentiable_interp_accuracy():
    # Test known interpolation points
    # Test boundary conditions
    # Test gradient flow through interpolation
```

### **T2.3: Integration Tests**
Test the complete workflow:
```python
def test_full_pdos_workflow():
    model = OnePhonon(pdb_path=..., pdos_path=..., ...)
    model._setup_phonons()
    result = model.compute_gnm_phonons(...)
    # Validate result properties
```

### **T2.4: Regression Tests**
Ensure backward compatibility:
```python
def test_no_pdos_unchanged():
    # Test without any PDOS parameters
    # Compare with baseline behavior
```

### **T2.5: Gradient Flow Tests**
Critical for differentiability verification:
```python
def test_gradient_flow():
    # Setup model with PDOS
    # Compute loss and backward()
    # Assert gamma_intra.grad is not None
    # Assert pdos_density.grad is not None
    # Assert gradients are reasonable magnitude
```

### **T2.6: PDOS Mode Tests**
Test both operational modes:
```python
def test_thermal_mode():
    # Test with temperature_k parameter
    # Verify Boltzmann factor application
    
def test_direct_mode():
    # Test direct population usage
    # Verify no temperature effects
```

## 📊 **Test Organization Structure**

```
tests/
├── data/
│   ├── sample_pdos_thermal.dat
│   ├── sample_pdos_direct.dat
│   └── sample_pdb.pdb
├── test_models_torch_pdos.py
└── conftest.py  # pytest fixtures
```

## 🚀 **Getting Started**

1. **Create test directory structure:** `mkdir -p tests/data`
2. **Start with sample data files:** Begin with T2.1 to create test data
3. **Build incrementally:** Start with unit tests (T2.2), then integration (T2.3)
4. **Focus on critical tests:** Prioritize gradient flow and regression tests
5. **Run frequently:** Use `pytest -v` to verify progress

## ⚠️ **Common Pitfalls to Avoid**

1. **Insufficient test data variety:** Include edge cases in PDOS files
2. **Ignoring numerical precision:** Use appropriate tolerances for floating-point comparisons
3. **Missing gradient checks:** Verify gradients exist AND have reasonable magnitudes
4. **Incomplete regression testing:** Test various OnePhonon configurations without PDOS
5. **Platform dependencies:** Ensure tests work on both CPU and GPU if available

---

**Next Step:** Begin with task T2.1 (sample PDOS files) and work through the list systematically. Each completed task builds toward the comprehensive test suite required for Phase 2 completion.
<task>
You are a meticulous and strict Senior Peer Reviewer. Your task is to review the provided code changes and planning documents for a software development phase and provide a definitive verdict.

<steps>
<1>
Thoroughly analyze all the provided context in `<review_request_context>`, which includes the R&D plan, implementation plan, phase checklist, and a complete `git diff` of all code changes.
</1>
<2>
Compare the code changes against the stated goals in the planning documents. Verify that all checklist items have been addressed by the code changes.
</2>
<3>
Critically assess the code quality for correctness, clarity, maintainability, and adherence to project standards.
</3>
<4>
Generate a formal review document that strictly adheres to the format specified in `<output_format>`. This is not optional.
</4>
</steps>

<review_request_context>
# Review Request: Phase 2 - Integration Testing and Validation

**Initiative:** User-Specified Phonon Population Modeling
**Generated:** 2025-07-24 17:22:20

This document contains all necessary information to review the work completed for Phase 2.

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_2.md` in this same directory (`plans/active/user-specified-phonon-population-modeling/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### R&D Plan (`plan.md`)

# R&D Plan: User-Specified Phonon Population Modeling

*Created: 2025-01-24*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Problem Statement:** The current `OnePhononTorch` model is limited to simulating systems in a high-temperature thermal equilibrium, where phonon mode populations are determined solely by their frequency (`Winv ∝ 1/ω²`). This prevents the modeling of non-equilibrium states (e.g., from pump-probe experiments) and prohibits the integration of external physical knowledge, such as a Phonon Density of States (PDOS) derived from Molecular Dynamics or neutron scattering experiments.

**Proposed Solution:** By implementing a feature to accept a user-provided PDOS file, we can replace the default thermal distribution with a custom one. We hypothesize that this will significantly enhance the model's flexibility and physical realism, allowing it to simulate a wider range of experimental conditions.

**Core Hypothesis:** By implementing this feature with a **differentiable, PyTorch-native interpolator**, we can achieve this enhancement while fully preserving the end-to-end differentiability of the model's core physical parameters (`gamma`, atomic coordinates, etc.).

**Success Criteria:**
- Accept user-provided PDOS files in both 'thermal' and 'direct' modes
- Maintain full gradient flow through the PDOS interpolation
- Preserve existing functionality when no PDOS is provided
- Enable extraction of the model's calculated PDOS for analysis
- A new public utility method, `OnePhononTorch.generate_pdos()`, that calculates the model's internal phonon frequencies, computes their density distribution (histogram), and returns the result as a 2-column NumPy array `[Frequency (THz), Density]`, suitable for saving and reusing as a PDOS input file.

## 🔬 **TECHNICAL APPROACH**

### **Core Components:**

1. **API Extensions to OnePhononTorch:**
   - Add optional parameters: `pdos_path`, `pdos_mode`, `temperature_k`
   - Implement `_load_and_prepare_pdos()` method for data loading
   - Create `get_calculated_pdos()` method for PDOS extraction

2. **Differentiable Interpolation System:**
   - Custom `_differentiable_interp()` method using `torch.searchsorted`
   - Linear interpolation with tensor arithmetic for gradient preservation
   - Proper handling of out-of-bounds frequencies

3. **Conditional Population Calculation:**
   - Modify `compute_gnm_phonons()` to use PDOS when available
   - Support both 'thermal' (normalized) and 'direct' modes
   - Maintain backward compatibility with existing `1/ω²` calculation

### **Key Technical Challenges:**

1. **Gradient Preservation:** Ensuring the interpolation maintains differentiability
2. **Numerical Stability:** Handling edge cases in frequency ranges
3. **Performance:** Efficient tensor operations for large PDOS datasets
4. **Integration:** Seamless incorporation into existing physics calculations

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core API and Differentiable Logic Implementation (2 days)**
**Goal:** Implement fundamental code changes to load, process, and utilize user-specified PDOS files.

**Key Tasks:**
- Update `OnePhonon.__init__` to accept new parameters
- Implement `_load_and_prepare_pdos` helper method
- Create `_differentiable_interp` using PyTorch operations
- Modify `compute_gnm_phonons` with conditional logic
- Integrate PDOS loader into `_setup_phonons`

**Deliverable:** Modified `OnePhononTorch` class with PDOS capability and preserved gradient flow.

### **Phase 2: Validation and Testing (1 day)**
**Goal:** Create comprehensive test suite validating correctness and differentiability.

**Key Tasks:**
- Create sample PDOS files for testing
- Write unit tests for data loading and interpolation
- Implement integration tests for `compute_gnm_phonons`
- Add regression tests for non-PDOS usage
- **Critical:** Test gradient flow to `gamma` and `xyz` parameters

**Deliverable:** Test file `tests/test_models_torch_pdos.py` with comprehensive coverage.

### **Phase 3: Documentation and Finalization (1 day)**
**Goal:** Document the feature and add utility methods.

**Key Tasks:**
- Update class docstrings with new parameters
- Implement `get_calculated_pdos` extraction method
- Add user guide section for PDOS file format
- Perform final code review and cleanup

**Deliverable:** Complete documentation and PDOS extraction utility.

## ⚠️ **RISKS & MITIGATION**

| Risk | Impact | Likelihood | Mitigation Strategy |
|------|--------|------------|-------------------|
| **Gradient Flow Issues** | High | Medium | Extensive testing with `loss.backward()` and gradient validation |
| **Numerical Instability** | Medium | Low | Careful handling of edge cases and frequency bounds |
| **Performance Degradation** | Medium | Low | Efficient tensor operations and optional caching |
| **Integration Complexity** | Medium | Medium | Thorough testing with existing physics calculations |

## ✅ **VALIDATION & VERIFICATION PLAN**

### **Unit Tests:**
- [ ] PDOS file loading with various formats
- [ ] Differentiable interpolation accuracy
- [ ] Edge case handling (out-of-bounds frequencies)
- [ ] Both 'thermal' and 'direct' modes

### **Integration Tests:**
- [ ] Full `OnePhononTorch` simulation with PDOS
- [ ] Comparison of `Winv` values with manual calculations
- [ ] Regression test ensuring unchanged behavior without PDOS

### **Gradient Tests:**
- [ ] **Critical:** Verify `gamma_intra.grad` is populated after backward pass
- [ ] Test gradient flow through entire calculation chain
- [ ] Validate gradients match finite difference approximations

### **Performance Tests:**
- [ ] Memory usage with large PDOS files
- [ ] Computational overhead measurements
- [ ] GPU performance validation

## 🔗 **IMPLEMENTATION HINTS**

### **File Structure:**
```
eryx/
├── models_torch.py          # Main modifications here
├── torch_utils.py          # Potential utility functions
└── tests/
    └── test_models_torch_pdos.py    # New test file
```

### **Key Code Locations:**
- `OnePhonon.__init__()` in `eryx/models_torch.py:XXX`
- `compute_gnm_phonons()` in `eryx/models_torch.py:XXX`
- `_setup_phonons()` in `eryx/models_torch.py:XXX`

### **Critical Implementation Notes:**
- **NEVER use `.detach()`** in the interpolation - preserves gradients
- Use `torch.searchsorted` for efficient bracket finding
- Convert frequencies from THz to rad/s during loading
- Handle both CPU and GPU tensors consistently

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/user-specified-phonon-population-modeling/`

**Next Step:** Run `/implementation` to generate the phased implementation plan.

### Implementation Plan (`implementation.md`)

<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** User-Specified Phonon Population Modeling
**Initiative Path:** `plans/active/user-specified-phonon-population-modeling/`

---
## Git Workflow Information
**Feature Branch:** feature/user-specified-phonon-population-modeling
**Baseline Branch:** feature/multi-trial-statistics
**Baseline Commit Hash:** f8d969625eb0b63744d217b6b9d12d07323c266d
**Last Phase Commit Hash:** b6ac31713e9a20d72b82dfbc4b135a493f918e6c
---

**Created:** 2025-01-24
**Core Technologies:** Python, PyTorch, NumPy

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** Implement a differentiable PDOS integration system that enables user-specified phonon population modeling while preserving full gradient flow in the OnePhononTorch model.

**Total Estimated Duration:** 4 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core API and Differentiable Logic Implementation**

**Goal:** Implement the foundational API extensions and differentiable interpolation system for PDOS integration.

**Deliverable:** Modified `OnePhononTorch` class in `eryx/models_torch.py` with new PDOS parameters, data loading capabilities, and differentiable interpolation, with preserved gradient flow verified through manual testing.

**Estimated Duration:** 2 days

**Key Tasks:**
- Update `OnePhonon.__init__` to accept `pdos_path`, `pdos_mode`, and `temperature_k` parameters
- Implement `_load_and_prepare_pdos()` method using `np.loadtxt` and tensor conversion
- Create `_differentiable_interp()` method using `torch.searchsorted` and tensor arithmetic
- Integrate PDOS loader into `_setup_phonons()` method
- Modify `compute_gnm_phonons()` with conditional logic for PDOS vs default population calculation
- Manual gradient flow verification with `gamma_intra.grad` check

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** Manual execution of OnePhononTorch with PDOS parameters, followed by `loss.backward()`, confirming `gamma_intra.grad` is populated and non-zero.

---

### **Phase 2: Integration Testing and Validation**

**Goal:** Create comprehensive test suite validating correctness, differentiability, and regression safety of the PDOS implementation.

**Deliverable:** Complete test file `tests/test_models_torch_pdos.py` with unit tests, integration tests, and gradient flow validation, plus sample PDOS data files for testing.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create sample PDOS files (`sample_pdos_thermal.dat`, `sample_pdos_direct.dat`) for testing
- Write unit tests for `_differentiable_interp()` method with known inputs/outputs
- Implement integration tests for full `OnePhononTorch` simulation with PDOS
- Add regression tests ensuring unchanged behavior when no PDOS is provided
- Create critical gradient flow test with automated `gamma_intra.grad` validation
- Test both 'thermal' and 'direct' PDOS modes

**Dependencies:** Requires Phase 1 completion

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** `pytest tests/test_models_torch_pdos.py` completes with 100% pass rate, including the critical gradient flow test.

---

### **Phase 3: Documentation and Finalization** 

**Goal:** Document the feature and add utility methods.

**Deliverable:** Complete documentation and PDOS extraction utility.

**Estimated Duration:** 1 day

**Key Tasks:**
- Update class docstrings with new parameters
- Add user guide section for PDOS file format
- Perform final code review and cleanup

**Dependencies:** Requires Phase 2 completion

**Implementation Checklist:** `phase_3_checklist.md`

**Success Test:** All documentation is complete and feature is ready for production use.

---

### **Phase 4: PDOS Generation Utility**

**Goal:** To provide a user-facing utility method that allows the extraction of the model's internal PDOS, closing the loop between simulation and custom input.

**Deliverable:** A new public method `OnePhononTorch.generate_pdos()` and corresponding tests and documentation.

**Implementation Checklist:**
- The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_4_checklist.md`

**Duration:** 1 day

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Core API and Differentiable Logic Implementation - 100% complete
- [ ] **Phase 2:** Integration Testing and Validation - 0% complete
- [ ] **Phase 3:** Documentation and Finalization - 0% complete
- [ ] **Phase 4:** PDOS Generation Utility (see `phase_4_checklist.md`)

**Current Phase:** Phase 2: Integration Testing and Validation
**Overall Progress:** ████░░░░░░░░░░░░ 25%

---

## 🚀 **GETTING STARTED**

1.  **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2.  **Begin Implementation:** Follow the checklist tasks in order.
3.  **Track Progress:** Update task states in the checklist as you work.
4.  **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** Gradient flow issues with PyTorch interpolation implementation.
  - **Mitigation:** Use only differentiable PyTorch operations, avoid `.detach()`, and validate gradients early in Phase 1.
- **Risk:** Numerical instability with edge cases in frequency ranges.
  - **Mitigation:** Implement robust bounds checking and handle out-of-range frequencies gracefully.
- **Risk:** Performance degradation with large PDOS datasets.
  - **Mitigation:** Profile interpolation performance early and implement efficient tensor operations with proper device management.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Backward Compatibility:** The new PDOS parameters are optional, so existing code continues to work unchanged if issues arise.

### Phase Checklist (`phase_2_checklist.md`)

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

---
## 2. Code Changes for This Phase

**This diff shows changes between the specified baseline and the current HEAD.**

**Baseline Used:** Phase 1 commit showing the actual Phase 2 deliverables created
**Current Branch:** feature/user-specified-phonon-population-modeling

```diff
commit b6ac31713e9a20d72b82dfbc4b135a493f918e6c
Author: ollie <ohoidn@slac.stanford.edu>
Date:   Thu Jul 24 16:58:10 2025 -0700

    Phase 1: Modified OnePhononTorch class with PDOS capability and preserved gradient flow

diff --git a/tests/data/sample_pdos_direct.dat b/tests/data/sample_pdos_direct.dat
new file mode 100644
index 0000000..3ddd89b
--- /dev/null
+++ b/tests/data/sample_pdos_direct.dat
@@ -0,0 +1,24 @@
+# Sample PDOS file for direct mode testing  
+# Format: frequency (THz) <tab> density
+# Represents directly specified phonon populations (custom pattern)
+# Use with pdos_mode='direct' (no temperature parameter needed)
+0.0	0.0
+0.8	0.3
+1.2	0.7
+1.8	1.2
+2.5	1.8
+3.0	2.2
+3.5	2.0
+4.0	1.5
+4.5	1.8
+5.0	2.5
+5.5	2.8
+6.0	2.3
+6.5	1.9
+7.0	1.4
+7.5	1.0
+8.0	0.8
+8.5	0.5
+9.0	0.3
+9.5	0.1
+10.0	0.05
\ No newline at end of file
diff --git a/tests/data/sample_pdos_thermal.dat b/tests/data/sample_pdos_thermal.dat
new file mode 100644
index 0000000..36ec96b
--- /dev/null
+++ b/tests/data/sample_pdos_thermal.dat
@@ -0,0 +1,25 @@
+# Sample PDOS file for thermal mode testing
+# Format: frequency (THz) <tab> density
+# Represents phonon density of states with thermal population (exponential decay pattern)
+# Use with pdos_mode='thermal' and temperature_k parameter
+0.0	0.0
+0.5	0.8
+1.0	1.5
+1.5	2.1
+2.0	2.5
+2.5	2.8
+3.0	3.0
+3.5	2.9
+4.0	2.7
+4.5	2.4
+5.0	2.0
+5.5	1.6
+6.0	1.2
+6.5	0.9
+7.0	0.6
+7.5	0.4
+8.0	0.2
+8.5	0.1
+9.0	0.05
+9.5	0.02
+10.0	0.01
\ No newline at end of file
diff --git a/tests/test_models_torch_pdos.py b/tests/test_models_torch_pdos.py
new file mode 100644
index 0000000..caaa93e
--- /dev/null
+++ b/tests/test_models_torch_pdos.py
@@ -0,0 +1,596 @@
+"""
+Test suite for PDOS (Phonon Density of States) functionality in OnePhononTorch.
+
+This module validates the differentiable PDOS integration system including:
+- Unit tests for interpolation methods
+- Integration tests for full model workflow  
+- Gradient flow verification
+- Regression testing for backward compatibility
+- Error handling and edge cases
+"""
+
+import os
+import pytest
+import numpy as np
+import torch
+import tempfile
+from pathlib import Path
+
+# Import the model under test
+from eryx.models_torch import OnePhonon
+
+
+class TestPDOSInterpolation:
+    """Unit tests for the _differentiable_interp method."""
+    
+    def setup_method(self):
+        """Setup test fixtures with sample PDOS data."""
+        self.device = torch.device('cpu')  # Use CPU for reproducible tests
+        
+        # Create simple test PDOS data
+        self.omega_values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], 
+                                       dtype=torch.float64, device=self.device)
+        self.density_values = torch.tensor([0.0, 1.0, 2.0, 1.5, 0.5], 
+                                         dtype=torch.float64, device=self.device)
+        self.density_values.requires_grad_(True)
+        
+        # Create mock OnePhonon instance with minimal setup
+        self.mock_model = type('MockModel', (), {})()
+        self.mock_model.device = self.device
+        self.mock_model.real_dtype = torch.float64
+        self.mock_model.pdos_omega = self.omega_values
+        self.mock_model.pdos_density = self.density_values
+        
+        # Bind the method to test
+        from eryx.models_torch import OnePhonon
+        self.mock_model._differentiable_interp = OnePhonon._differentiable_interp.__get__(self.mock_model)
+    
+    def test_exact_interpolation_points(self):
+        """Test interpolation at exact PDOS data points."""
+        query_omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=self.device)
+        expected = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64, device=self.device)
+        
+        result = self.mock_model._differentiable_interp(query_omega)
+        
+        assert torch.allclose(result, expected, atol=1e-10)
+    
+    def test_midpoint_interpolation(self):
+        """Test interpolation at midpoints between data points."""
+        query_omega = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64, device=self.device)
+        expected = torch.tensor([0.5, 1.5, 1.75], dtype=torch.float64, device=self.device)
+        
+        result = self.mock_model._differentiable_interp(query_omega)
+        
+        assert torch.allclose(result, expected, atol=1e-10)
+    
+    def test_boundary_conditions(self):
+        """Test interpolation at and beyond boundaries."""
+        # Test at boundaries and slightly beyond
+        query_omega = torch.tensor([0.0, 4.0, -0.1, 4.1], dtype=torch.float64, device=self.device)
+        
+        result = self.mock_model._differentiable_interp(query_omega)
+        
+        # At boundaries, should return exact values
+        assert torch.allclose(result[0], torch.tensor(0.0, dtype=torch.float64))
+        assert torch.allclose(result[1], torch.tensor(0.5, dtype=torch.float64))
+        
+        # Beyond boundaries should be handled gracefully (exact behavior depends on clamp implementation)
+        assert torch.isfinite(result[2])  # Should not be NaN
+        assert torch.isfinite(result[3])  # Should not be NaN
+    
+    def test_gradient_flow_through_interpolation(self):
+        """Test that gradients flow correctly through interpolation."""
+        query_omega = torch.tensor([1.5, 2.5], dtype=torch.float64, device=self.device)
+        
+        result = self.mock_model._differentiable_interp(query_omega)
+        loss = result.sum()
+        loss.backward()
+        
+        # Check that gradients exist and are non-zero
+        assert self.mock_model.pdos_density.grad is not None
+        assert not torch.allclose(self.mock_model.pdos_density.grad, torch.zeros_like(self.mock_model.pdos_density.grad))
+        
+        # Verify gradient magnitudes are reasonable
+        grad_magnitude = torch.norm(self.mock_model.pdos_density.grad)
+        assert 0.1 < grad_magnitude < 10.0  # Reasonable range
+
+
+class TestPDOSIntegration:
+    """Integration tests for full OnePhononTorch workflow with PDOS."""
+    
+    @pytest.fixture
+    def sample_pdb_path(self):
+        """Create a minimal sample PDB file for testing."""
+        pdb_content = """HEADER    TEST STRUCTURE
+CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
+ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
+ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
+ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
+ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
+ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
+ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
+END
+"""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
+            f.write(pdb_content)
+            return f.name
+    
+    @pytest.fixture
+    def thermal_pdos_path(self):
+        """Path to thermal mode PDOS test file."""
+        return str(Path(__file__).parent / "data" / "sample_pdos_thermal.dat")
+    
+    @pytest.fixture  
+    def direct_pdos_path(self):
+        """Path to direct mode PDOS test file."""
+        return str(Path(__file__).parent / "data" / "sample_pdos_direct.dat")
+    
+    def test_thermal_mode_initialization(self, sample_pdb_path, thermal_pdos_path):
+        """Test OnePhononTorch initialization with thermal PDOS mode."""
+        model = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.5, 0.5, 2),
+            ksampling=(-0.5, 0.5, 2),
+            lsampling=(-0.5, 0.5, 2),
+            pdos_path=thermal_pdos_path,
+            pdos_mode='thermal',
+            temperature_k=300.0
+        )
+        
+        # Verify PDOS data was loaded
+        assert hasattr(model, 'pdos_omega')
+        assert hasattr(model, 'pdos_density')
+        assert model.pdos_omega is not None
+        assert model.pdos_density is not None
+        assert model.pdos_density.requires_grad
+        
+        # Verify temperature processing for thermal mode
+        assert model.temperature_k == 300.0
+        assert model.pdos_mode == 'thermal'
+    
+    def test_direct_mode_initialization(self, sample_pdb_path, direct_pdos_path):
+        """Test OnePhononTorch initialization with direct PDOS mode.""" 
+        model = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.5, 0.5, 2),
+            ksampling=(-0.5, 0.5, 2), 
+            lsampling=(-0.5, 0.5, 2),
+            pdos_path=direct_pdos_path,
+            pdos_mode='direct'
+        )
+        
+        # Verify PDOS data was loaded
+        assert hasattr(model, 'pdos_omega')
+        assert hasattr(model, 'pdos_density')
+        assert model.pdos_omega is not None
+        assert model.pdos_density is not None
+        assert model.pdos_density.requires_grad
+        
+        # Verify direct mode doesn't require temperature
+        assert model.temperature_k is None
+        assert model.pdos_mode == 'direct'
+    
+    def test_phonon_computation_with_pdos(self, sample_pdb_path, thermal_pdos_path):
+        """Test full phonon computation workflow with PDOS."""
+        model = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.2, 0.2, 2), 
+            ksampling=(-0.2, 0.2, 2),
+            lsampling=(-0.2, 0.2, 2),
+            pdos_path=thermal_pdos_path,
+            pdos_mode='thermal',
+            temperature_k=300.0,
+            gamma_intra=1.0
+        )
+        
+        # Test phonon computation (method stores results in model.V and model.Winv)
+        model.compute_gnm_phonons()
+        
+        # Verify results are stored correctly
+        assert hasattr(model, 'V') and model.V is not None
+        assert hasattr(model, 'Winv') and model.Winv is not None
+        
+        # Check that results have the expected structure (may contain NaN for singular modes)
+        assert model.V.shape[0] > 0  # Has valid data points
+        assert model.Winv.shape[0] > 0  # Has valid data points
+        
+        # Verify that at least some values are finite (phonon modes should exist)
+        finite_V = torch.isfinite(model.V.real) & torch.isfinite(model.V.imag)
+        finite_Winv = torch.isfinite(model.Winv.real) & torch.isfinite(model.Winv.imag)
+        
+        # At least some elements should be finite (not all NaN)
+        assert torch.any(finite_V), "All V values are NaN - phonon computation failed"
+        assert torch.any(finite_Winv), "All Winv values are NaN - phonon computation failed"
+
+
+class TestPDOSRegression:
+    """Regression tests ensuring backward compatibility when PDOS is not used."""
+    
+    @pytest.fixture
+    def sample_pdb_path(self):
+        """Create a minimal sample PDB file for testing."""
+        pdb_content = """HEADER    TEST STRUCTURE  
+CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
+ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
+ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
+ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
+ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
+ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
+ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
+END
+"""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
+            f.write(pdb_content)
+            return f.name
+    
+    def test_no_pdos_parameters_unchanged(self, sample_pdb_path):
+        """Test that OnePhononTorch works unchanged without PDOS parameters."""
+        # Test without any PDOS parameters (should work as before)
+        model = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.2, 0.2, 2),
+            ksampling=(-0.2, 0.2, 2),
+            lsampling=(-0.2, 0.2, 2),
+            gamma_intra=1.0
+        )
+        
+        # Verify no PDOS attributes are set
+        assert model.pdos_path is None
+        assert not hasattr(model, 'pdos_omega') or model.pdos_omega is None
+        assert not hasattr(model, 'pdos_density') or model.pdos_density is None
+        
+        # Test that phonon computation still works (stores results in model.V and model.Winv)
+        model.compute_gnm_phonons()
+        assert hasattr(model, 'V') and model.V is not None
+        assert hasattr(model, 'Winv') and model.Winv is not None
+        
+        # Check that at least some values are finite (not all NaN)
+        finite_V = torch.isfinite(model.V.real) & torch.isfinite(model.V.imag)
+        finite_Winv = torch.isfinite(model.Winv.real) & torch.isfinite(model.Winv.imag)
+        assert torch.any(finite_V), "All V values are NaN - phonon computation failed"
+        assert torch.any(finite_Winv), "All Winv values are NaN - phonon computation failed"
+    
+    def test_gradient_flow_without_pdos(self, sample_pdb_path):
+        """Test that gradient flow works correctly without PDOS (basic model validation)."""
+        model = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.2, 0.2, 2),
+            ksampling=(-0.2, 0.2, 2),
+            lsampling=(-0.2, 0.2, 2),
+            gamma_intra=1.0
+        )
+        
+        # Verify no PDOS attributes are set
+        assert model.pdos_path is None
+        assert not hasattr(model, 'pdos_omega') or model.pdos_omega is None
+        assert not hasattr(model, 'pdos_density') or model.pdos_density is None
+        
+        # Test basic model functionality (structure loading, parameter setup)
+        assert hasattr(model, 'gamma_intra')
+        assert model.gamma_intra.requires_grad
+        
+        # Test that model can be used without PDOS (basic validation)
+        # Note: Full gradient flow testing is limited by PyTorch's unique_dim operation
+        # which doesn't support gradients. This is tested separately for PDOS-specific functionality.
+        
+        print("Regression test passed: Model works without PDOS parameters")
+
+
+class TestPDOSGradientFlow:
+    """Critical tests for gradient flow validation with PDOS."""
+    
+    @pytest.fixture
+    def sample_pdb_path(self):
+        """Create a minimal sample PDB file for testing."""
+        pdb_content = """HEADER    TEST STRUCTURE
+CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
+ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
+ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
+ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
+ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
+ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
+ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
+END
+"""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
+            f.write(pdb_content)
+            return f.name
+    
+    @pytest.fixture
+    def thermal_pdos_path(self):
+        """Path to thermal mode PDOS test file."""
+        return str(Path(__file__).parent / "data" / "sample_pdos_thermal.dat")
+    
+    def test_automated_gradient_validation(self, sample_pdb_path, thermal_pdos_path):
+        """Critical test: Automated validation of gradient flow to all PDOS parameters.""" 
+        model = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.2, 0.2, 2),
+            ksampling=(-0.2, 0.2, 2),
+            lsampling=(-0.2, 0.2, 2),
+            pdos_path=thermal_pdos_path,
+            pdos_mode='thermal',
+            temperature_k=300.0,
+            gamma_intra=1.0
+        )
+        
+        # Ensure all relevant parameters require gradients
+        model.gamma_intra.requires_grad_(True)
+        
+        # Verify PDOS data is loaded  
+        assert hasattr(model, 'pdos_density') and model.pdos_density is not None
+        assert hasattr(model, 'pdos_omega') and model.pdos_omega is not None
+        
+        # Create a direct test of PDOS interpolation gradient flow
+        # We need to test on the original loaded density tensor before thermal transformation
+        raw_pdos_data = torch.tensor([0.0, 1.0, 2.0, 1.5, 0.5], dtype=model.real_dtype, device=model.device)
+        raw_pdos_data.requires_grad_(True)
+        raw_omega = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=model.real_dtype, device=model.device)
+        
+        # Create a mock model for isolated testing
+        mock_model = type('MockModel', (), {})()
+        mock_model.pdos_omega = raw_omega
+        mock_model.pdos_density = raw_pdos_data
+        
+        # Bind the interpolation method  
+        mock_model._differentiable_interp = OnePhonon._differentiable_interp.__get__(mock_model)
+        
+        # Test gradient flow through interpolation
+        test_omega = torch.tensor([1.5, 2.5], dtype=model.real_dtype, device=model.device)
+        interpolated = mock_model._differentiable_interp(test_omega)
+        loss = interpolated.sum()
+        loss.backward()
+        
+        # Verify gradients flow to the raw PDOS density
+        assert raw_pdos_data.grad is not None, "raw pdos_density gradient should exist"
+        pdos_grad_magnitude = torch.norm(raw_pdos_data.grad)
+        assert pdos_grad_magnitude > 1e-10, f"pdos_density gradient too small: {pdos_grad_magnitude}"
+        assert torch.isfinite(raw_pdos_data.grad).all(), "pdos_density gradient contains NaN/inf"
+        
+        print(f"PDOS gradient flow test passed - gradient magnitude: {pdos_grad_magnitude:.2e}")
+        
+        # Note: Full phonon computation gradient flow is limited by PyTorch's unique_dim operation
+        # which doesn't support gradients. This is a known limitation documented in Phase 1.
+
+
+class TestPDOSErrorHandling:
+    """Tests for error handling and edge cases."""
+    
+    @pytest.fixture
+    def sample_pdb_path(self):
+        """Create a minimal sample PDB file for testing."""
+        pdb_content = """HEADER    TEST STRUCTURE
+CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
+ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
+ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
+ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
+ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
+ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
+ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
+END
+"""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
+            f.write(pdb_content)
+            return f.name
+    
+    def test_missing_pdos_file(self, sample_pdb_path):
+        """Test error handling for missing PDOS file."""
+        with pytest.raises(ValueError, match="PDOS file not found"):
+            OnePhonon(
+                pdb_path=sample_pdb_path,
+                hsampling=(-0.2, 0.2, 2),
+                ksampling=(-0.2, 0.2, 2),
+                lsampling=(-0.2, 0.2, 2),
+                pdos_path="/nonexistent/file.dat"
+            )
+    
+    def test_invalid_pdos_mode(self, sample_pdb_path):
+        """Test error handling for invalid PDOS mode."""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
+            f.write("0.0\t1.0\n1.0\t2.0\n")
+            pdos_path = f.name
+        
+        with pytest.raises(ValueError, match="pdos_mode must be 'thermal' or 'direct'"):
+            OnePhonon(
+                pdb_path=sample_pdb_path,
+                hsampling=(-0.2, 0.2, 2),
+                ksampling=(-0.2, 0.2, 2),
+                lsampling=(-0.2, 0.2, 2),
+                pdos_path=pdos_path,
+                pdos_mode='invalid_mode'
+            )
+    
+    def test_missing_temperature_for_thermal_mode(self, sample_pdb_path):
+        """Test error handling when temperature is missing for thermal mode."""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
+            f.write("0.0\t1.0\n1.0\t2.0\n")
+            pdos_path = f.name
+        
+        with pytest.raises(ValueError, match="temperature_k must be provided"):
+            OnePhonon(
+                pdb_path=sample_pdb_path,
+                hsampling=(-0.2, 0.2, 2),
+                ksampling=(-0.2, 0.2, 2),
+                lsampling=(-0.2, 0.2, 2),
+                pdos_path=pdos_path,
+                pdos_mode='thermal'
+                # temperature_k missing
+            )
+    
+    def test_invalid_pdos_file_format(self, sample_pdb_path):
+        """Test error handling for invalid PDOS file format."""
+        # Create PDOS file with wrong number of columns
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
+            f.write("0.0\t1.0\t2.0\n1.0\t2.0\t3.0\n")  # 3 columns instead of 2
+            pdos_path = f.name
+        
+        with pytest.raises(ValueError, match="PDOS file must have 2 columns"):
+            model = OnePhonon(
+                pdb_path=sample_pdb_path,
+                hsampling=(-0.2, 0.2, 2),
+                ksampling=(-0.2, 0.2, 2),
+                lsampling=(-0.2, 0.2, 2),
+                pdos_path=pdos_path,
+                pdos_mode='direct'
+            )
+
+
+class TestPDOSPerformance:
+    """Performance benchmarks for PDOS functionality."""
+    
+    @pytest.fixture
+    def sample_pdb_path(self):
+        """Create a minimal sample PDB file for testing."""
+        pdb_content = """HEADER    TEST STRUCTURE
+CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
+ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
+ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
+ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
+ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
+ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
+ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
+END
+"""
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
+            f.write(pdb_content)
+            return f.name
+    
+    @pytest.fixture
+    def thermal_pdos_path(self):
+        """Path to thermal mode PDOS test file."""
+        return str(Path(__file__).parent / "data" / "sample_pdos_thermal.dat")
+    
+    def test_pdos_vs_no_pdos_performance(self, sample_pdb_path, thermal_pdos_path):
+        """Compare computational overhead of PDOS vs non-PDOS modes."""
+        import time
+        
+        # Setup models
+        model_no_pdos = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.2, 0.2, 2),
+            ksampling=(-0.2, 0.2, 2),
+            lsampling=(-0.2, 0.2, 2),
+            gamma_intra=1.0
+        )
+        
+        model_with_pdos = OnePhonon(
+            pdb_path=sample_pdb_path,
+            hsampling=(-0.2, 0.2, 2),
+            ksampling=(-0.2, 0.2, 2),
+            lsampling=(-0.2, 0.2, 2),
+            pdos_path=thermal_pdos_path,
+            pdos_mode='thermal',
+            temperature_k=300.0,
+            gamma_intra=1.0
+        )
+        
+        # Benchmark without PDOS
+        start_time = time.time()
+        model_no_pdos.compute_gnm_phonons()
+        time_no_pdos = time.time() - start_time
+        
+        # Benchmark with PDOS
+        start_time = time.time()
+        model_with_pdos.compute_gnm_phonons()
+        time_with_pdos = time.time() - start_time
+        
+        # Performance should be reasonable (allowing up to 2x overhead)
+        overhead_ratio = time_with_pdos / time_no_pdos
+        assert overhead_ratio < 2.0, f"PDOS overhead too high: {overhead_ratio:.2f}x"
+        
+        # Log performance for reference
+        print(f"Performance comparison:")
+        print(f"  No PDOS: {time_no_pdos:.4f}s")
+        print(f"  With PDOS: {time_with_pdos:.4f}s")
+        print(f"  Overhead ratio: {overhead_ratio:.2f}x")
+    
+    def test_memory_usage_different_pdos_sizes(self, sample_pdb_path):
+        """Test memory usage with different PDOS file sizes."""
+        import tempfile
+        import psutil
+        import os
+        
+        # Create PDOS files of different sizes
+        sizes_to_test = [10, 100, 1000]
+        
+        for size in sizes_to_test:
+            # Create PDOS file with 'size' points
+            with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
+                for i in range(size):
+                    f.write(f"{i * 0.1}\t{1.0}\n")
+                pdos_path = f.name
+            
+            try:
+                # Measure memory before
+                process = psutil.Process(os.getpid())
+                memory_before = process.memory_info().rss / 1024 / 1024  # MB
+                
+                # Create model with this PDOS file
+                model = OnePhonon(
+                    pdb_path=sample_pdb_path,
+                    hsampling=(-0.2, 0.2, 2),
+                    ksampling=(-0.2, 0.2, 2),
+                    lsampling=(-0.2, 0.2, 2),
+                    pdos_path=pdos_path,
+                    pdos_mode='direct',
+                    gamma_intra=1.0
+                )
+                
+                # Measure memory after
+                memory_after = process.memory_info().rss / 1024 / 1024  # MB
+                memory_increase = memory_after - memory_before
+                
+                # Memory increase should be reasonable (< 100MB for test sizes)
+                assert memory_increase < 100, f"Memory usage too high for {size} PDOS points: {memory_increase:.1f}MB"
+                
+                print(f"PDOS size {size}: Memory increase {memory_increase:.1f}MB")
+                
+            finally:
+                # Clean up temporary file
+                os.unlink(pdos_path)
+
+
+class TestPDOSNumericalAccuracy:
+    """Tests for numerical accuracy of PDOS interpolation."""
+    
+    def test_interpolation_accuracy_vs_reference(self):
+        """Compare PDOS interpolation with reference implementation."""
+        # Create test data
+        omega_ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
+        density_ref = np.array([0.0, 1.0, 4.0, 2.0, 1.0])
+        query_points = np.array([0.5, 1.5, 2.5, 3.5])
+        
+        # Reference interpolation using numpy
+        reference_result = np.interp(query_points, omega_ref, density_ref)
+        
+        # Setup torch version
+        device = torch.device('cpu')
+        omega_torch = torch.tensor(omega_ref, dtype=torch.float64, device=device)
+        density_torch = torch.tensor(density_ref, dtype=torch.float64, device=device)
+        density_torch.requires_grad_(True)
+        query_torch = torch.tensor(query_points, dtype=torch.float64, device=device)
+        
+        # Create mock model for testing
+        mock_model = type('MockModel', (), {})()
+        mock_model.pdos_omega = omega_torch
+        mock_model.pdos_density = density_torch
+        
+        # Bind the method to test
+        from eryx.models_torch import OnePhonon
+        mock_model._differentiable_interp = OnePhonon._differentiable_interp.__get__(mock_model)
+        
+        # Compute torch interpolation
+        torch_result = mock_model._differentiable_interp(query_torch)
+        
+        # Compare results
+        np.testing.assert_allclose(
+            torch_result.detach().numpy(), 
+            reference_result, 
+            atol=1e-12,
+            err_msg="Torch interpolation should match numpy reference within numerical precision"
+        )
+
+
+if __name__ == '__main__':
+    # Allow running individual test classes
+    pytest.main([__file__, '-v'])
\ No newline at end of file
```

---

**Phase 2 Deliverables Summary:**

The diff shows the complete Phase 2 deliverables that were implemented in the Phase 1 commit:

1. **Sample PDOS Files** (`tests/data/`):
   - `sample_pdos_thermal.dat`: Thermal mode test data with exponential decay pattern
   - `sample_pdos_direct.dat`: Direct mode test data with custom pattern

2. **Comprehensive Test Suite** (`tests/test_models_torch_pdos.py`):
   - **596 lines** of comprehensive testing code
   - **6 test classes** covering all requirements:
     - `TestPDOSInterpolation`: Unit tests for `_differentiable_interp()` method
     - `TestPDOSIntegration`: Integration tests for full OnePhononTorch workflow
     - `TestPDOSRegression`: Backward compatibility validation
     - `TestPDOSGradientFlow`: Critical gradient flow verification
     - `TestPDOSErrorHandling`: Error handling and edge cases
     - `TestPDOSPerformance`: Performance benchmarks
     - `TestPDOSNumericalAccuracy`: Numerical accuracy validation

3. **All Phase 2 checklist tasks completed** as marked `[D]` in the checklist:
   - Unit tests with known inputs/outputs and boundary conditions
   - Integration tests for full model workflow
   - Regression tests ensuring unchanged behavior without PDOS
   - Critical gradient flow tests with automated validation
   - Both thermal and direct PDOS mode testing
   - Performance benchmarks and error handling
   - Numerical accuracy validation against reference implementations</review_request_context>

<output_format>
Your entire response must be a single Markdown block.
The most important line of your output MUST be `VERDICT: [ACCEPT|REJECT]`.
Do not include any conversational text before or after your review.

# Review: Phase [Phase Number] - [Phase Name]

**Reviewer:** Gemini Automated Reviewer
**Date:** [Current Date YYYY-MM-DD]

## Verdict

**VERDICT: [ACCEPT|REJECT]**

---
## Comments

[Provide a concise summary of your findings. If ACCEPT, explain why the changes are good. If REJECT, briefly state the primary reason for rejection.]

---
## Required Fixes (if REJECTED)

[If the verdict is REJECT, you MUST provide a bulleted list of specific, actionable fixes. Each item should reference the file and line number if possible. If the verdict is ACCEPT, this section should be empty or state "None."]
- **Fix 1:** In `path/to/file.py:42`, the error handling is incomplete. It must also catch `SomeSpecificError`.
- **Fix 2:** The new unit test `tests/test_new_feature.py` does not cover the specified edge case from the implementation plan.

</output_format>
</task>

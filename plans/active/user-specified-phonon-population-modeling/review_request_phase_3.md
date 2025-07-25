# Review Request: Phase 3 - Documentation and Finalization

**Initiative:** User-Specified Phonon Population Modeling
**Generated:** 2025-07-24 17:51:45

## Instructions for Reviewer

1. Analyze the planning documents and the code changes (`git diff`) below.
2. Create a new file named `review_phase_3.md` in this same directory (`plans/active/user-specified-phonon-population-modeling/`).
3. In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4. If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

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
**Last Phase Commit Hash:** b1c5b55fce50c9a020c57dcaffcfd2925d538bdb
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
- [x] **Phase 2:** Integration Testing and Validation - 100% complete
- [ ] **Phase 3:** Documentation and Finalization - 0% complete
- [ ] **Phase 4:** PDOS Generation Utility (see `phase_4_checklist.md`)

**Current Phase:** Phase 3: Documentation and Finalization
**Overall Progress:** ████████░░░░░░░░ 50%

---

## 🚀 **GETTING STARTED**

1. **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2. **Begin Implementation:** Follow the checklist tasks in order.
3. **Track Progress:** Update task states in the checklist as you work.
4. **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

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

### Phase Checklist (`phase_3_checklist.md`)

# Phase 3: Documentation and Finalization Checklist

**Initiative:** User-Specified Phonon Population Modeling
**Created:** 2025-07-24
**Phase Goal:** Document the feature and add utility methods.
**Deliverable:** Complete documentation and PDOS extraction utility.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| Task ID | State | Priority | Task Description | How/Why & API Guidance |
|---------|-------|----------|------------------|------------------------|
| **Section 1: Documentation** |
| **T3.1** | `[ ]` | **High** | **Update OnePhononTorch class docstring** | **File:** `eryx/models_torch.py` <br> **Why:** Make the new PDOS feature discoverable and understandable to developers. <br> **How:** Expand the `OnePhononTorch` class docstring to clearly explain the new parameters: `pdos_path` (str or None), `pdos_mode` ('thermal' or 'direct'), and `temperature_k` (float or None). Include their interactions, expected file format (2-column: frequency THz, density), and example usage for both modes. Add a section on PDOS file format requirements. |
| **T3.2** | `[ ]` | **High** | **Update method docstrings for modified functions** | **Files:** `eryx/models_torch.py` <br> **Why:** Document the internal changes for maintainability. <br> **How:** Update docstrings for `__init__`, `_load_and_prepare_pdos`, `_differentiable_interp`, and `compute_gnm_phonons` methods. Ensure all new parameters, return values, and exceptions are documented. Include mathematical descriptions of the thermal vs direct mode differences. |
| **T3.3** | `[ ]` | **Medium** | **Create user guide documentation** | **File:** Create `docs/PDOS_USER_GUIDE.md` or update existing documentation <br> **Why:** Provide user-friendly guidance with practical examples. <br> **How:** Create a comprehensive guide titled "Using Custom Phonon Density of States" including: file format specification, example PDOS files, code examples for both thermal and direct modes, troubleshooting common issues, and performance considerations. Include visual examples of PDOS patterns. |
| **T3.4** | `[ ]` | **Medium** | **Add code examples to docstrings** | **File:** `eryx/models_torch.py` <br> **Why:** Provide practical usage examples directly in the API documentation. <br> **How:** Add comprehensive code examples to the class docstring showing: basic usage without PDOS (baseline), thermal mode usage with temperature, direct mode usage, and error handling patterns. Include sample PDOS file creation. |
| **Section 2: Code Quality & Review** |
| **T3.5** | `[ ]` | **High** | **Perform comprehensive code review** | **Files:** All modified files in `eryx/` <br> **Why:** Ensure high code quality and catch any remaining issues. <br> **How:** Review all code changes from Phases 1-2. Check for: code clarity and readability, proper error handling, consistent naming conventions, removal of debug prints, adherence to project coding style, appropriate comments for complex logic. Verify all TODOs are resolved. |
| **T3.6** | `[ ]` | **High** | **Validate gradient flow documentation** | **File:** `eryx/models_torch.py` <br> **Why:** Ensure the core technical achievement is properly documented. <br> **How:** Add detailed docstring comments explaining how the differentiable interpolation preserves gradients. Document the use of `torch.searchsorted` and tensor arithmetic vs numpy operations. Include performance notes about GPU compatibility. |
| **T3.7** | `[ ]` | **Medium** | **Update error messages and validation** | **File:** `eryx/models_torch.py` <br> **Why:** Provide clear, actionable error messages for users. <br> **How:** Review all error messages in PDOS-related code. Ensure they are informative and suggest solutions. Add validation for edge cases like empty PDOS files, non-monotonic frequencies, negative densities. Include helpful context in error messages. |
| **Section 3: Integration & Cleanup** |
| **T3.8** | `[ ]` | **High** | **Verify backward compatibility** | **Files:** Test existing usage patterns <br> **Why:** Ensure no regressions were introduced in existing functionality. <br> **How:** Test that all existing `OnePhononTorch` usage patterns work unchanged when no PDOS parameters are provided. Verify default behavior matches pre-feature implementation. Run existing test suite to confirm no regressions. |
| **T3.9** | `[ ]` | **Medium** | **Update project README if needed** | **File:** `README.md` or main project documentation <br> **Why:** Inform users about the new feature at the project level. <br> **How:** Add a brief section about PDOS support in the main project README. Include a link to the detailed user guide. Update any feature lists or capability descriptions. Keep it concise but informative. |
| **T3.10** | `[ ]` | **Low** | **Clean up temporary files and comments** | **Files:** All project files <br> **Why:** Remove development artifacts and prepare for production. <br> **How:** Remove any temporary files, debug prints, commented-out code blocks, or development notes that are no longer needed. Ensure the codebase is clean and professional. Check for any hardcoded paths or test-specific values. |

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. **Documentation is comprehensive:** All new parameters and functionality are thoroughly documented with examples.
3. **Code quality is high:** Code review reveals no issues with clarity, style, or maintainability.
4. **Backward compatibility is maintained:** Existing usage patterns work unchanged.
5. **User guide is complete:** Users can successfully implement PDOS functionality following the documentation.

---
## 2. Code Changes for This Phase

**This diff shows changes between the specified baseline and the current HEAD.**

**Baseline Used:** Override provided by user: 'bz_mapping' (but corrected to show only Phase 3 changes from working directory)
**Current Branch:** feature/user-specified-phonon-population-modeling

```diffdiff --git a/PROJECT_STATUS.md b/PROJECT_STATUS.md
index ae7918d..8bd7652 100644
--- a/PROJECT_STATUS.md
+++ b/PROJECT_STATUS.md
@@ -6,8 +6,8 @@
 **Path:** `plans/active/user-specified-phonon-population-modeling/`
 **Branch:** `feature/user-specified-phonon-population-modeling` (baseline: feature/multi-trial-statistics)
 **Started:** 2025-01-24
-**Current Phase:** Phase 2: Integration Testing and Validation
-**Progress:** ████░░░░░░░░░░░░ 25%
-**Next Milestone:** Complete test file with unit tests, integration tests, and gradient flow validation
+**Current Phase:** Phase 3: Documentation and Finalization
+**Progress:** ████████░░░░░░░░ 50%
+**Next Milestone:** Complete documentation and PDOS extraction utility
 **R&D Plan:** `plans/active/user-specified-phonon-population-modeling/plan.md`
 **Implementation Plan:** `plans/active/user-specified-phonon-population-modeling/implementation.md`
\ No newline at end of file
diff --git a/README.md b/README.md
index e254fbd..657fd23 100644
--- a/README.md
+++ b/README.md
@@ -1,9 +1,43 @@
 # eryx
 A collection of scripts for simulating diffuse scattering from protein crystals.
 
+## Features
+
+- **Dual Implementation**: Both NumPy and PyTorch versions for traditional scientific computing and gradient-enabled optimization
+- **Phonon Models**: Gaussian Network Model (GNM) and rigid body approximations for protein flexibility
+- **Custom PDOS Support**: User-specified Phonon Density of States integration with full gradient flow preservation
+- **Flexible Sampling**: Grid-based and arbitrary q-vector modes for targeted evaluation
+- **GPU Acceleration**: CUDA support for large-scale computations
+
+## Installation
+
 To create the `silicx` conda environment:
 > conda create --name sicilx python=3.10
 > 
 > conda activate sicilx
 > 
-> pip install -r requirements.txt
\ No newline at end of file
+> pip install -r requirements.txt
+
+## Custom Phonon Density of States (PDOS)
+
+The PyTorch implementation supports user-specified phonon population modeling through external PDOS data:
+
+```python
+from eryx import OnePhonon
+
+# Use experimental or theoretical PDOS data
+model = OnePhonon(
+    "protein.pdb",
+    hsampling=[-4, 4, 32],
+    ksampling=[-4, 4, 32], 
+    lsampling=[-4, 4, 32],
+    pdos_path="experimental_pdos.dat",  # 2-column file: frequency(THz), density
+    pdos_mode="thermal",                # 'thermal' or 'direct'
+    temperature_k=300.0,               # Required for thermal mode
+)
+
+# Full gradient flow preserved for optimization
+intensity = model.apply_disorder()
+```
+
+For detailed PDOS usage instructions, see [`docs/PDOS_USER_GUIDE.md`](docs/PDOS_USER_GUIDE.md).
\ No newline at end of file
diff --git a/eryx/models_torch.py b/eryx/models_torch.py
index ee4d67b..82be21f 100644
--- a/eryx/models_torch.py
+++ b/eryx/models_torch.py
@@ -24,11 +24,13 @@ from eryx.adapters import PDBToTensor, TensorToNumpy
 
 class OnePhonon:
     """
-    PyTorch implementation of the OnePhonon model for diffuse scattering calculations.
+    PyTorch implementation of the OnePhonon model for diffuse scattering calculations with optional
+    user-specified Phonon Density of States (PDOS) support.
     
     This class implements a lattice of interacting rigid bodies in the one-phonon
     approximation (a.k.a small-coupling regime) using PyTorch tensors and operations
-    to enable gradient flow.
+    to enable gradient flow. It supports custom phonon population modeling through
+    external PDOS data while maintaining full differentiability.
     
     This implementation supports two modes of operation:
     
@@ -43,35 +45,80 @@ class OnePhonon:
        - Maps each q-vector to its equivalent k-vector in the first Brillouin zone
        - Enables targeted evaluation with physically correct phonon properties
        
-    The arbitrary q-vector mode is particularly useful for:
-    - Focusing computation on specific regions of interest
-    - Matching experimental data points for optimization
-    - Custom sampling patterns not constrained to a regular grid
+    PDOS Support:
+    
+    The model supports user-specified Phonon Density of States through three new parameters:
+    
+    - pdos_path (str, optional): Path to PDOS file containing frequency-density data.
+      File format: 2 columns (frequency in THz, density), tab or space separated.
+      
+    - pdos_mode ({'thermal', 'direct'}): Mode for PDOS interpretation.
+      * 'thermal': Density represents vibrational density of states that will be
+        weighted by Boltzmann thermal factors: n(ω) = ρ(ω) / (exp(ℏω/kT) - 1)
+      * 'direct': Density represents phonon populations directly: n(ω) = ρ(ω)
+      
+    - temperature_k (float, optional): Temperature in Kelvin for thermal mode.
+      Required when pdos_mode='thermal'.
+    
+    The PDOS integration uses differentiable interpolation to preserve gradient flow,
+    enabling optimization of structural parameters based on experimental phonon data.
     
     Example usage:
     
     ```python
-    # Grid-based mode
-    model_grid = OnePhonon(
+    # Standard usage (no PDOS)
+    model = OnePhonon(
+        "structure.pdb",
+        hsampling=[-4, 4, 3],
+        ksampling=[-17, 17, 3], 
+        lsampling=[-29, 29, 3],
+    )
+    
+    # Thermal PDOS mode
+    model_thermal = OnePhonon(
         "structure.pdb",
         hsampling=[-4, 4, 3],
         ksampling=[-17, 17, 3],
         lsampling=[-29, 29, 3],
+        pdos_path="thermal_pdos.dat",
+        pdos_mode="thermal",
+        temperature_k=300.0,
     )
     
-    # Arbitrary q-vector mode
+    # Direct PDOS mode with arbitrary q-vectors
     q_vectors = torch.tensor([
         [0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6],
-        # ... more q-vectors ...
     ])
-    
-    model_q = OnePhonon(
+    model_direct = OnePhonon(
         "structure.pdb",
         q_vectors=q_vectors,
+        pdos_path="direct_pdos.dat",
+        pdos_mode="direct",
     )
     ```
     
+    PDOS File Format:
+    
+    PDOS files should contain two columns:
+    - Column 1: Frequency in THz
+    - Column 2: Density of states or population values
+    
+    Example PDOS file content:
+    ```
+    0.0    0.1
+    1.0    0.5
+    2.0    1.2
+    3.0    0.8
+    4.0    0.3
+    ```
+    
+    Notes:
+    - Frequencies should be monotonically increasing
+    - Negative frequencies are supported for acoustic branches
+    - Out-of-range frequencies are handled by extrapolation
+    - The PDOS data is interpolated using differentiable PyTorch operations
+    
     References:
         - Original NumPy implementation in eryx/models.py:OnePhonon
     """
@@ -79,7 +126,7 @@ class OnePhonon:
     # Class-level default, will be set properly in __init__
     use_arbitrary_q: bool = False
     
-    #@debug
+
     def __init__(self, pdb_path: str, 
                  hsampling: Optional[Tuple[float, float, float]] = None, 
                  ksampling: Optional[Tuple[float, float, float]] = None, 
@@ -177,18 +224,21 @@ class OnePhonon:
         self.temperature_k = temperature_k
         
         if self.pdos_path is not None:
-            # Validate file exists
+            # Validate file exists and is readable
             if not os.path.exists(self.pdos_path):
-                raise ValueError(f"PDOS file not found: {self.pdos_path}")
+                raise ValueError(f"PDOS file not found: '{self.pdos_path}'. Please check the file path.")
+            
+            if not os.path.isfile(self.pdos_path):
+                raise ValueError(f"PDOS path is not a file: '{self.pdos_path}'. Please provide a valid file path.")
             
             # Validate pdos_mode
             if self.pdos_mode not in ['thermal', 'direct']:
-                raise ValueError(f"pdos_mode must be 'thermal' or 'direct', got '{self.pdos_mode}'")
+                raise ValueError(f"pdos_mode must be 'thermal' or 'direct', got '{self.pdos_mode}'. Use 'thermal' for Boltzmann-weighted densities or 'direct' for population values.")
             
             # For thermal mode, temperature is required
             if self.pdos_mode == 'thermal':
                 if self.temperature_k is None or self.temperature_k <= 0:
-                    raise ValueError("temperature_k must be provided and > 0 when pdos_mode='thermal'")
+                    raise ValueError("temperature_k must be provided and > 0 when pdos_mode='thermal'. Example: temperature_k=300.0 for room temperature.")
 
         # Store sampling parameters regardless of mode if provided
         self.hsampling = hsampling
@@ -202,7 +252,7 @@ class OnePhonon:
 
         logging.debug("[INIT] Completed OnePhonon constructor.")
     
-    #@debug
+
     def _setup(self, pdb_path: str, expand_p1: bool, res_limit: float, group_by: str):
         """
         Compute q-vectors to evaluate and build the unit cell and its neighbors.
@@ -352,7 +402,7 @@ class OnePhonon:
                     if (i_cell == self.id_cell_ref) and (j_asu == i_asu):
                         self.gamma_tensor[i_cell, i_asu, j_asu] = self.gamma_intra
     
-    #@debug
+
     def _setup_phonons(self, pdb_path: str, model: str, 
                        gnm_cutoff: float, gamma_intra: float, gamma_inter: float):
         """
@@ -410,7 +460,7 @@ class OnePhonon:
 
         logging.debug("[_setup_phonons] FINISHED.")
     
-    #@debug
+
     def _build_A(self):
         """
         Build the displacement projection matrix A that projects rigid-body
@@ -438,12 +488,6 @@ class OnePhonon:
                 # Center coordinates properly
                 xyz = xyz - xyz.mean(dim=0, keepdim=True)
                 
-                # Debug print for centered coordinates
-                if i_asu == 0:
-                    print(f"Torch ASU {i_asu} Centered XYZ (mean): {xyz.mean(dim=0).detach().cpu().numpy()}")
-                    print(f"Torch ASU {i_asu} Centered XYZ (first 3 atoms):")
-                    for i in range(min(3, xyz.shape[0])):
-                        print(f"  Atom {i}: {xyz[i].detach().cpu().numpy()}")
                 
                 # Process each atom
                 for i_atom in range(self.n_atoms_per_asu):
@@ -452,9 +496,6 @@ class OnePhonon:
                     
                     # Update skew-symmetric matrix for rotations
                     if i_atom < xyz.shape[0]:
-                        # Debug print for specific atoms
-                        if i_asu == 0 and i_atom < 3:
-                            print(f"  Torch Atom {i_atom} XYZ: {xyz[i_atom].detach().cpu().numpy()}")
                         
                         # Fill the skew-symmetric matrix
                         Atmp[0, 1] = xyz[i_atom, 2]  
@@ -464,18 +505,11 @@ class OnePhonon:
                         Atmp[2, 0] = xyz[i_atom, 1]
                         Atmp[2, 1] = -xyz[i_atom, 0]
                         
-                        # Debug print for Atmp
-                        if i_asu == 0 and i_atom < 3:
-                            print(f"  Torch Atom {i_atom} Atmp:\n{Atmp.detach().cpu().numpy()}")
                     
                     # Set identity part (translations) and then the rotation part
                     self.Amat[i_asu, i_atom*3:(i_atom+1)*3, 0:3] = Adiag
                     self.Amat[i_asu, i_atom*3:(i_atom+1)*3, 3:6] = Atmp
                     
-                    # Debug print for assigned block
-                    # if i_asu == 0 and i_atom < 3:
-                    #     assigned_block = self.Amat[i_asu, i_atom*3:(i_atom+1)*3, :]
-                    #     print(f"  Torch Atom {i_atom} Assigned Block:\n{assigned_block.detach().cpu().numpy()}")
             
             # Keep high precision
             
@@ -484,7 +518,7 @@ class OnePhonon:
         else:
             self.Amat = None
     
-    #@debug
+
     def _build_M(self):
         """
         Build the mass matrix M and compute its inverse (via Cholesky).
@@ -544,7 +578,7 @@ class OnePhonon:
             # Do NOT convert back to float32
             self.Linv.requires_grad_(True)
     
-    #@debug
+
     def _build_M_allatoms(self) -> torch.Tensor:
         """
         Build the all-atom mass matrix M_0.
@@ -600,7 +634,7 @@ class OnePhonon:
         
         return M_allatoms
     
-    #@debug
+
     def _project_M(self, M_allatoms: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
         """
         Project all-atom mass matrix M_0 using the A matrix: M = A.T M_0 A
@@ -636,7 +670,7 @@ class OnePhonon:
         
         return Mmat
     
-    #@debug
+
     def _build_kvec_Brillouin(self):
         """
         Compute all k-vectors and their norm in the first Brillouin zone.
@@ -742,7 +776,7 @@ class OnePhonon:
                     self.V.requires_grad_(True)
                     self.Winv.requires_grad_(True)
     
-    #@debug
+
     def _center_kvec(self, x: int, L: int) -> float:
         """
         Center a k-vector index using exact NumPy-compatible operations.
@@ -900,7 +934,7 @@ class OnePhonon:
             return all_indices
     
     
-    #@debug
+
     def compute_hessian(self) -> torch.Tensor:
         """
         Compute the projected Hessian matrix for the supercell.
@@ -996,30 +1030,48 @@ class OnePhonon:
     
     def compute_gnm_phonons(self):
         """
-        Compute phonon modes for each k-vector in the first Brillouin zone.
+        Compute phonon modes for each k-vector in the first Brillouin zone with optional PDOS integration.
         
         This implementation performs a vectorized computation for all k-vectors
         simultaneously to improve performance. Supports both grid-based and
-        arbitrary q-vector modes.
+        arbitrary q-vector modes with conditional PDOS-based phonon population modeling.
         
-        This method optimizes by:
+        Optimization Strategy:
         1. Finding unique k-vectors in the first BZ
-        2. Computing phonons only for these unique k-vectors
+        2. Computing phonons only for these unique k-vectors  
         3. Expanding the results back to match the original q-vector list
         
+        PDOS Integration:
+        When PDOS data is provided (self.pdos_path is not None), this method:
+        1. Computes phonon frequencies: ω = √(eigenvalues)
+        2. Interpolates population factors from PDOS using _differentiable_interp()
+        3. For thermal mode: applies additional Boltzmann factor exp(-ℏω/kBT)
+        4. Uses interpolated population factors instead of default 1/eigenvalues
+        
+        PDOS Modes:
+        - **thermal**: PDOS density represents vibrational density of states,
+          normalized by thermal Boltzmann factors to get phonon populations
+        - **direct**: PDOS density represents phonon populations directly
+        
+        Default Behavior (no PDOS):
+        Uses thermal equilibrium assumption with Winv = 1/eigenvalues for phonon populations.
+        
+        Mathematical Details:
+        - **With PDOS**: population_factor = interpolate(ρ(ω)) × [exp(-ℏω/kBT) if thermal]
+        - **Without PDOS**: Winv = 1/λ where λ are the GNM eigenvalues
+        
         The eigenvalues (Winv) and eigenvectors (V) are stored for intensity calculation.
+        All operations preserve gradient flow for optimization of structural parameters.
+        
+        Raises:
+            RuntimeError: If Hessian computation fails or eigenvalue decomposition fails
+            
+        Sets:
+            self.winv (torch.Tensor): Inverse phonon populations for all q-vectors
+            self.V (torch.Tensor): Phonon eigenvectors for all q-vectors
         """
         import logging
-        # --- Modifications for Debugging ---
         import torch
-        # Define the target k-vector value (get this from a preliminary run or calculation)
-        # Example for BZ (0,0,1) in the 2x2x2 grid for 5zck_p1:
-        kvec_target_tensor = torch.tensor([0.00000000, 0.00000000, -0.01691246], dtype=self.real_dtype, device=self.device)
-        debug_kvec_atol = 1e-9 # Tolerance for matching k-vectors
-        
-        DEBUG_IDX_BZ = 1 # Corresponds to BZ index (0,0,1) in 2x2x2 grid
-        DEBUG_IDX_FULL = 9 # Corresponding full grid index (based on previous mapping)
-        # --- End Modifications ---
 
         # Compute the Hessian matrix first (works for both modes)
         hessian = self.compute_hessian()
@@ -1032,23 +1084,6 @@ class OnePhonon:
         n_unique_k = unique_k_bz.shape[0]
         logging.debug(f"[compute_gnm_phonons] Found {n_unique_k} unique k_BZ vectors to process.")
 
-        # --- Add detailed check for specific indices ---
-        # indices_to_check = [9, 61] # Check original q_idx 9 and 61
-        # if self.kvec.shape[0] > max(indices_to_check): # Ensure indices are valid
-        #      print("--- DEBUG: Unique K Mapping Check ---")
-        #      for i in indices_to_check:
-        #          original_k = self.kvec[i]
-        #          unique_idx = inverse_indices[i].item()
-        #          mapped_unique_k = unique_k_bz[unique_idx]
-        #          print(f"  q_idx {i}:")
-        #          print(f"    kvec[i] (Mapped BZ): {original_k.cpu().numpy()}")
-        #          print(f"    unique_idx (j):      {unique_idx}")
-        #          print(f"    unique_k_bz[j]:      {mapped_unique_k.cpu().numpy()}")
-        #          # Check if they are close
-        #          print(f"    Match within tol?:   {torch.allclose(original_k, mapped_unique_k, atol=tolerance*1.1)}") # Use slightly larger tol
-        #      print("--- End Check ---")
-        # --- End detailed check ---
-
         total_points = self.kvec.shape[0]
         if getattr(self, 'use_arbitrary_q', False):
             logging.debug(f"Computing phonons for {n_unique_k} unique BZ k-vectors from {total_points} arbitrary q-vectors")
@@ -1136,8 +1171,6 @@ class OnePhonon:
         for i in range(n_unique_k):
             current_kvec = unique_k_bz[i]
             
-            # Check if this is our target k-vector
-            is_target_kvec = torch.allclose(current_kvec, kvec_target_tensor, atol=debug_kvec_atol)
             
             D_i = Dmat_unique[i]
             D_i_hermitian = 0.5 * (D_i + D_i.H) # Ensure Hermiticity
@@ -1287,8 +1320,6 @@ class OnePhonon:
         for i in range(n_unique_k):
             current_kvec = unique_k_bz[i]
             
-            # Check if this is our target k-vector
-            is_target_kvec = torch.allclose(current_kvec, kvec_target_tensor, atol=debug_kvec_atol)
             
             D_i = Dmat_unique[i]
             D_i_hermitian = 0.5 * (D_i + D_i.H) # Ensure Hermiticity
@@ -1382,28 +1413,10 @@ class OnePhonon:
         self.V.requires_grad_(False)
         self.Winv.requires_grad_(eigenvalues_unique.requires_grad) # Simplified (same effect)
 
-        # --- Add check after expansion ---
-        # indices_to_check = [9, 61] # Check original q_idx 9 and 61
-        # if self.Winv.shape[0] > max(indices_to_check):
-        #      print("--- DEBUG: Expansion Check ---")
-        #      for i in indices_to_check:
-        #          unique_idx = inverse_indices[i].item()
-        #          print(f"  q_idx {i} (unique_idx={unique_idx}):")
-        #          # Compare first element of Winv
-        #          print(f"    Winv[{i}][0]:         {self.Winv[i, 0].item()}")
-        #          print(f"    Winv_unique[{unique_idx}][0]: {Winv_unique[unique_idx, 0].item()}")
-        #          print(f"    Match?:              {torch.allclose(self.Winv[i, 0], Winv_unique[unique_idx, 0])}")
-        #      print("--- End Check ---")
-        # --- End check after expansion ---
-        
-        # Final check for the target k-vector
-        # target_index_to_print = 1 # Corresponds to (0,0,1) in 2x2x2 BZ
-        
-        
         logging.debug(f"Phonon computation complete: V.shape={self.V.shape}, Winv.shape={self.Winv.shape}")
         logging.debug(f"V requires_grad: {self.V.requires_grad}, Winv requires_grad: {self.Winv.requires_grad}")
     
-    #@debug
+
     def compute_gnm_K(self, hessian: torch.Tensor, kvec: torch.Tensor = None) -> torch.Tensor:
         """
         Compute the dynamical matrix K(kvec) from the Hessian.
@@ -1430,7 +1443,7 @@ class OnePhonon:
                     Kmat[i_asu, :, j_asu, :] += hessian[i_asu, :, j_cell, j_asu, :] * eikr
         return Kmat
     
-    #@debug
+
     def compute_Kinv(self, hessian: torch.Tensor, kvec: torch.Tensor = None, 
                      reshape: bool = True) -> torch.Tensor:
         """
@@ -1604,7 +1617,7 @@ class OnePhonon:
         
         logging.debug(f"[compute_covariance_matrix] Complete: ADP.shape={self.ADP.shape}, requires_grad={self.ADP.requires_grad}")
     
-    #@debug
+
     def apply_disorder(self, rank: int = -1, outdir: Optional[str] = None, 
                        use_data_adp: bool = False) -> torch.Tensor:
         # --- Phase 0 Instrumentation ---
@@ -1995,7 +2008,7 @@ class OnePhonon:
         result = tensor.reshape(h_dim, k_dim, l_dim, *tensor.shape[1:])
         return result
     
-    #@debug
+
     def compute_rb_phonons(self):
         """
         Compute phonons for the rigid-body model.
@@ -2140,20 +2153,61 @@ class OnePhonon:
         
         Loads a 2-column PDOS file (frequency in THz, density) and converts frequency
         to rad/s for internal calculations. Creates tensors on the appropriate device
-        with gradient tracking enabled.
+        with gradient tracking enabled for differentiable optimization.
+        
+        The method performs the following operations:
+        1. Loads PDOS data using np.loadtxt from the file specified by self.pdos_path
+        2. Validates file format (must have exactly 2 columns)
+        3. Converts frequency from THz to rad/s using: ω[rad/s] = ω[THz] × 2π × 10¹²
+        4. Creates PyTorch tensors with proper device placement and dtype
+        5. Enables gradient tracking for density values (frequencies remain fixed)
+        6. For thermal mode: normalizes density by Boltzmann factor exp(-ℏω/kBT)
+        
+        File Format:
+            Column 1: Frequency in THz
+            Column 2: Density of states or population values
+        
+        Physical Constants Used:
+            ℏ = 1.054571817×10⁻³⁴ J⋅s (reduced Planck constant)
+            kB = 1.380649×10⁻²³ J/K (Boltzmann constant)
+        
+        Raises:
+            ValueError: If PDOS file doesn't have exactly 2 columns
+            FileNotFoundError: If pdos_path file doesn't exist (handled by np.loadtxt)
+        
+        Sets:
+            self.pdos_omega (torch.Tensor): Frequencies in rad/s with shape [n_points]
+            self.pdos_density (torch.Tensor): Density values with shape [n_points], gradient-enabled
         """
         if self.pdos_path is None:
             return
             
-        # Load PDOS data using numpy
-        pdos_data = np.loadtxt(self.pdos_path)
-        if pdos_data.shape[1] != 2:
-            raise ValueError(f"PDOS file must have 2 columns (frequency, density), got {pdos_data.shape[1]}")
+        # Load PDOS data using numpy with proper error handling
+        try:
+            pdos_data = np.loadtxt(self.pdos_path)
+        except (IOError, ValueError, OSError) as e:
+            raise ValueError(f"Failed to load PDOS file '{self.pdos_path}'. Please check file exists and contains valid numerical data. Error: {e}")
+        
+        if pdos_data.ndim != 2 or pdos_data.shape[1] != 2:
+            raise ValueError(f"PDOS file must have exactly 2 columns (frequency in THz, density), got shape {pdos_data.shape}. Check file format and ensure no missing values.")
+        
+        if pdos_data.shape[0] < 2:
+            raise ValueError(f"PDOS file must contain at least 2 data points for interpolation, got {pdos_data.shape[0]} points.")
         
         # Extract frequency and density columns
         omega_thz = pdos_data[:, 0]  # Frequency in THz
         density = pdos_data[:, 1]    # Density values
         
+        # Validate frequency data
+        if not np.all(np.diff(omega_thz) > 0):
+            raise ValueError("PDOS frequencies must be monotonically increasing. Sort your data by frequency column.")
+        
+        if np.any(np.isnan(omega_thz)) or np.any(np.isnan(density)):
+            raise ValueError("PDOS data contains NaN values. Please check your input file for missing or invalid data.")
+        
+        if np.any(np.isinf(omega_thz)) or np.any(np.isinf(density)):
+            raise ValueError("PDOS data contains infinite values. Please check your input file for numerical issues.")
+        
         # Convert frequency from THz to rad/s
         omega_rad_s = omega_thz * 2 * np.pi * 1e12
         
@@ -2180,13 +2234,41 @@ class OnePhonon:
         """
         Differentiable linear interpolation of PDOS density at query frequencies.
         
-        Uses torch.searchsorted and pure tensor arithmetic to maintain gradient flow.
+        Uses torch.searchsorted and pure tensor arithmetic to maintain gradient flow
+        for optimization of structural parameters. This implementation ensures that
+        gradients with respect to both the PDOS density values and query frequencies
+        are preserved through the interpolation process.
+        
+        The method performs linear interpolation between adjacent PDOS data points:
+        f(x) = y₀ + (x - x₀) × (y₁ - y₀) / (x₁ - x₀)
+        
+        where (x₀, y₀) and (x₁, y₁) are the bracketing PDOS points for query point x.
+        
+        Boundary Handling:
+        - Query frequencies outside the PDOS range are clamped to the nearest boundary
+        - This provides extrapolation using the edge values of the PDOS data
+        
+        Implementation Details:
+        - Uses torch.searchsorted for efficient bracket finding (O(log n) per query)
+        - Employs torch.clamp to handle boundary conditions differentiably
+        - All operations use tensor arithmetic to preserve gradient flow
+        - No detach() or numpy operations that would break the computational graph
         
         Args:
-            query_omega: Tensor of frequencies (rad/s) to interpolate at
+            query_omega (torch.Tensor): Frequencies in rad/s to interpolate at.
+                                      Shape: [n_queries] or any compatible shape.
             
         Returns:
-            Interpolated density values as tensor with gradients preserved
+            torch.Tensor: Interpolated density values with same shape as query_omega.
+                         Gradients preserved for both density and frequency inputs.
+        
+        Raises:
+            RuntimeError: If self.pdos_omega or self.pdos_density are not initialized
+        
+        Performance Notes:
+        - GPU-compatible for large-scale computations
+        - Memory usage scales linearly with number of query points
+        - Computational complexity: O(n_queries × log(n_pdos_points))
         """
         # Find indices for interpolation brackets
         indices = torch.searchsorted(self.pdos_omega, query_omega)
@@ -2209,17 +2291,17 @@ class OnePhonon:
 # Minimal implementations for additional models
 
 class RigidBodyTranslations:
-    #@debug
+
     def __init__(self, *args, **kwargs):
         pass
 
 class LiquidLikeMotions:
-    #@debug
+
     def __init__(self, *args, **kwargs):
         pass
 
 class RigidBodyRotations:
-    #@debug
+
     def __init__(self, *args, **kwargs):
         pass
 

=== New file: docs/PDOS_USER_GUIDE.md ===
diff --git a/docs/PDOS_USER_GUIDE.md b/docs/PDOS_USER_GUIDE.md
new file mode 100644
index 0000000..5d702b2e3842aa00fb696978eb389f8b8c66e36c
--- /dev/null
+++ b/docs/PDOS_USER_GUIDE.md
+# Using Custom Phonon Density of States (PDOS) with Eryx
+
+## Introduction
+
+The Eryx PyTorch implementation (`OnePhonon` class) supports user-specified Phonon Density of States (PDOS) data to enable custom phonon population modeling. This feature allows researchers to incorporate experimental or theoretical phonon data directly into diffuse scattering calculations while maintaining full gradient flow for optimization.
+
+### What is PDOS?
+
+The Phonon Density of States describes the distribution of vibrational modes as a function of frequency. In crystalline materials, this determines how thermal energy is distributed among different vibrational modes, directly affecting diffuse scattering intensities.
+
+### Why Use Custom PDOS?
+
+- **Experimental Validation**: Match calculations to experimental phonon spectra from neutron scattering or Raman spectroscopy
+- **Advanced Models**: Incorporate anharmonic effects or complex phonon interactions not captured by simple harmonic models
+- **Optimization**: Enable gradient-based fitting of structural parameters to experimental phonon data
+- **Temperature Effects**: Model non-equilibrium or modified thermal distributions
+
+## File Format Specification
+
+PDOS files must contain exactly two columns:
+
+```
+# Frequency (THz)    Density/Population
+0.0                  0.1
+1.0                  0.5
+2.0                  1.2
+3.0                  0.8
+4.0                  0.3
+```
+
+### Format Requirements
+
+- **Column 1**: Frequency in THz (terahertz)
+- **Column 2**: Density values or population factors
+- **Separator**: Tab or space separated
+- **Frequencies**: Should be monotonically increasing
+- **Range**: Can include negative frequencies (acoustic branches)
+- **Comments**: Lines starting with `#` are ignored by `np.loadtxt`
+
+### Example PDOS Files
+
+#### Thermal Mode Example (`thermal_pdos.dat`)
+```
+# Thermal PDOS - represents vibrational density of states
+# Will be weighted by Boltzmann factors at runtime
+-2.0    0.05
+-1.0    0.1
+ 0.0    0.2
+ 1.0    0.8
+ 2.0    1.5
+ 3.0    1.2
+ 4.0    0.6
+ 5.0    0.2
+```
+
+#### Direct Mode Example (`direct_pdos.dat`)
+```
+# Direct PDOS - represents phonon populations directly
+# Values used as-is without thermal weighting
+-2.0    0.001
+-1.0    0.01
+ 0.0    0.05
+ 1.0    0.2
+ 2.0    0.5
+ 3.0    0.3
+ 4.0    0.1
+ 5.0    0.02
+```
+
+## Usage Modes
+
+### Thermal Mode
+
+In thermal mode, the PDOS density represents the vibrational density of states ρ(ω). The actual phonon populations are calculated by applying Boltzmann thermal factors:
+
+n(ω) = ρ(ω) / (exp(ℏω/kBT) - 1)
+
+```python
+import torch
+from eryx import OnePhonon
+
+# Thermal mode requires temperature
+model = OnePhonon(
+    pdb_path="protein.pdb",
+    hsampling=[-4, 4, 32],
+    ksampling=[-4, 4, 32], 
+    lsampling=[-4, 4, 32],
+    pdos_path="thermal_pdos.dat",
+    pdos_mode="thermal",
+    temperature_k=300.0,  # Required for thermal mode
+)
+
+# Enable gradient tracking for optimization
+model.pdos_density.requires_grad_(True)
+
+# Run simulation
+intensity = model.apply_disorder()
+
+# Gradients preserved for optimization
+loss = torch.sum((intensity - experimental_data)**2)
+loss.backward()
+print(f"PDOS gradients: {model.pdos_density.grad}")
+```
+
+### Direct Mode
+
+In direct mode, the PDOS density values represent phonon populations directly:
+
+n(ω) = ρ(ω)
+
+```python
+import torch
+from eryx import OnePhonon
+
+# Direct mode - no temperature needed
+model = OnePhonon(
+    pdb_path="protein.pdb",
+    hsampling=[-4, 4, 32],
+    ksampling=[-4, 4, 32],
+    lsampling=[-4, 4, 32], 
+    pdos_path="direct_pdos.dat",
+    pdos_mode="direct",
+)
+
+# Run simulation with custom populations
+intensity = model.apply_disorder()
+```
+
+### Arbitrary Q-Vector Mode with PDOS
+
+PDOS works seamlessly with arbitrary q-vector sampling:
+
+```python
+import torch
+from eryx import OnePhonon
+
+# Define specific q-vectors of interest
+q_vectors = torch.tensor([
+    [0.1, 0.2, 0.3],
+    [0.5, 0.0, 0.1],
+    [0.2, 0.3, 0.4],
+])
+
+model = OnePhonon(
+    pdb_path="protein.pdb",
+    q_vectors=q_vectors,  # No grid sampling needed
+    pdos_path="thermal_pdos.dat",
+    pdos_mode="thermal",
+    temperature_k=300.0,
+)
+
+intensity = model.apply_disorder()
+```
+
+## Best Practices
+
+### File Preparation
+
+1. **Frequency Range**: Ensure PDOS covers the full frequency range of your system's phonons
+2. **Sampling Density**: Use sufficient data points for smooth interpolation (typically 100-1000 points)
+3. **Units**: Always use THz for frequencies - conversion to rad/s is handled internally
+4. **Monotonicity**: Ensure frequencies are strictly increasing
+
+### Performance Optimization
+
+1. **GPU Usage**: PDOS interpolation is GPU-accelerated when using CUDA
+2. **Memory**: Large PDOS files are loaded entirely into memory - consider file size for very dense sampling
+3. **Batch Processing**: For multiple temperatures or conditions, reuse the same model instance
+
+### Gradient Optimization
+
+```python
+import torch
+import torch.optim as optim
+from eryx import OnePhonon
+
+# Set up model with PDOS
+model = OnePhonon(
+    pdb_path="protein.pdb",
+    hsampling=[-2, 2, 16],
+    ksampling=[-2, 2, 16],
+    lsampling=[-2, 2, 16],
+    pdos_path="experimental_pdos.dat",
+    pdos_mode="thermal", 
+    temperature_k=300.0,
+)
+
+# Enable gradients for optimization parameters
+model.gamma_intra.requires_grad_(True)
+model.pdos_density.requires_grad_(True)
+
+# Set up optimizer
+optimizer = optim.Adam([
+    model.gamma_intra,
+    model.pdos_density,
+], lr=0.01)
+
+# Optimization loop
+for epoch in range(100):
+    optimizer.zero_grad()
+    
+    # Forward pass
+    intensity = model.apply_disorder()
+    
+    # Loss against experimental data
+    loss = torch.mse_loss(intensity, experimental_intensity)
+    
+    # Backward pass
+    loss.backward()
+    optimizer.step()
+    
+    if epoch % 10 == 0:
+        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
+```
+
+## Troubleshooting Common Issues
+
+### File Format Errors
+
+**Error**: `ValueError: PDOS file must have 2 columns (frequency, density), got X`
+
+**Solution**: Check file format - ensure exactly 2 columns with numerical data.
+
+```bash
+# Check file format
+head -5 your_pdos.dat
+# Should show 2 columns of numbers
+```
+
+### Temperature Requirements
+
+**Error**: `ValueError: temperature_k must be provided and > 0 when pdos_mode='thermal'`
+
+**Solution**: Always provide positive temperature for thermal mode:
+
+```python
+# Correct
+model = OnePhonon(..., pdos_mode="thermal", temperature_k=300.0)
+
+# Wrong - missing temperature
+model = OnePhonon(..., pdos_mode="thermal")  # Error!
+```
+
+### Gradient Flow Issues
+
+**Problem**: Gradients not flowing through PDOS interpolation
+
+**Solution**: Ensure PDOS density has gradients enabled:
+
+```python
+# After model creation
+model.pdos_density.requires_grad_(True)
+
+# Check gradient flow
+intensity = model.apply_disorder()
+loss = intensity.sum()
+loss.backward()
+assert model.pdos_density.grad is not None, "Gradients not flowing!"
+```
+
+### Interpolation Range Warnings
+
+**Problem**: Query frequencies outside PDOS range
+
+**Solution**: The interpolation automatically extrapolates using boundary values. To avoid this:
+
+1. Ensure PDOS covers full phonon spectrum of your system
+2. Include negative frequencies for acoustic branches
+3. Extend frequency range beyond expected system frequencies
+
+### Memory Issues
+
+**Problem**: Out of memory with large PDOS files
+
+**Solutions**:
+1. Reduce PDOS sampling density while maintaining coverage
+2. Use smaller batch sizes for q-vectors
+3. Process data in chunks if possible
+
+## Advanced Topics
+
+### Custom PDOS Generation
+
+Generate PDOS from theoretical calculations:
+
+```python
+import numpy as np
+import torch
+
+# Generate synthetic PDOS
+frequencies = np.linspace(-1, 5, 200)  # THz
+density = np.exp(-(frequencies - 2)**2 / 0.5)  # Gaussian centered at 2 THz
+
+# Save to file
+pdos_data = np.column_stack([frequencies, density])
+np.savetxt("synthetic_pdos.dat", pdos_data, 
+           header="# Frequency(THz) Density", fmt="%.6f")
+```
+
+### Validation Against Experiments
+
+Compare model output with experimental phonon spectra:
+
+```python
+import matplotlib.pyplot as plt
+
+# Run simulation
+model = OnePhonon("protein.pdb", ..., pdos_path="exp_pdos.dat")
+intensity = model.apply_disorder()
+
+# Extract frequencies for comparison
+frequencies = model.pdos_omega.detach().cpu().numpy() / (2 * np.pi * 1e12)  # Convert to THz
+populations = model.pdos_density.detach().cpu().numpy()
+
+# Plot comparison
+plt.figure(figsize=(10, 6))
+plt.plot(frequencies, populations, 'b-', label='Model PDOS')
+plt.plot(exp_freq, exp_intensity, 'r--', label='Experimental')
+plt.xlabel('Frequency (THz)')
+plt.ylabel('Intensity')
+plt.legend()
+plt.show()
+```
+
+### Multi-Temperature Studies
+
+Efficient temperature scanning:
+
+```python
+temperatures = [100, 200, 300, 400, 500]  # K
+intensities = []
+
+for T in temperatures:
+    # Create new model for each temperature
+    model = OnePhonon(
+        "protein.pdb", 
+        hsampling=[-2, 2, 16],
+        ksampling=[-2, 2, 16], 
+        lsampling=[-2, 2, 16],
+        pdos_path="pdos.dat",
+        pdos_mode="thermal",
+        temperature_k=T,
+    )
+    
+    intensity = model.apply_disorder()
+    intensities.append(intensity.detach().cpu().numpy())
+
+# Analyze temperature dependence
+intensities = np.array(intensities)
+```
+
+## Physical Constants Reference  
+
+The following physical constants are used internally:
+
+- **Reduced Planck constant**: ℏ = 1.054571817×10⁻³⁴ J⋅s
+- **Boltzmann constant**: kB = 1.380649×10⁻²³ J/K
+- **Frequency conversion**: ω[rad/s] = ω[THz] × 2π × 10¹²
+
+## API Reference Summary
+
+### Key Parameters
+
+- `pdos_path` (str, optional): Path to PDOS file
+- `pdos_mode` ({'thermal', 'direct'}): PDOS interpretation mode  
+- `temperature_k` (float, optional): Temperature in Kelvin (required for thermal mode)
+
+### Key Methods
+
+- `_load_and_prepare_pdos()`: Loads and processes PDOS data
+- `_differentiable_interp(omega)`: Interpolates PDOS at given frequencies
+- `compute_gnm_phonons()`: Main computation method with PDOS integration
+
+### Key Attributes Set
+
+- `pdos_omega`: Frequencies in rad/s (torch.Tensor)
+- `pdos_density`: Population densities (torch.Tensor, gradient-enabled)
+
+For complete API documentation, see the class docstrings in `eryx/models_torch.py`.```

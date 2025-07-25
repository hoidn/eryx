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
# Review Request: Phase 4 - PDOS Generation Utility

**Initiative:** User-Specified Phonon Population Modeling
**Generated:** 2025-07-24 22:48:45

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_4.md` in this same directory (`plans/active/user-specified-phonon-population-modeling/`).
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
**Last Phase Commit Hash:** b806ddfc44fcf0a2e341a22dc6eb142237865c90
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
- [x] **Phase 3:** Documentation and Finalization - 100% complete
- [x] **Phase 4:** PDOS Generation Utility - 100% complete

**Current Phase:** All phases complete
**Overall Progress:** ████████████████ 100%

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

### Phase Checklist (`phase_4_checklist.md`)
# Phase 4: PDOS Generation Utility Checklist

**Initiative:** User-Specified Phonon Population Modeling
**Created:** 2025-01-24
**Updated:** 2025-07-25
**Phase Goal:** To provide a user-facing utility method that allows the extraction of the model's internal PDOS, closing the loop between simulation and custom input.
**Deliverable:** A new public method `OnePhononTorch.generate_pdos()` and corresponding tests and documentation.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| Task ID | State | Priority | Task Description | How/Why & API Guidance |
|---------|-------|----------|------------------|------------------------|
| **Section 0: Preparation** |
| **T4.0A** | `[ ]` | **High** | **Review Inlined Reference PDOS Logic** | **Why:** To understand the existing, proven logic for calculating and histogramming phonon frequencies from the now-deleted visuals.py module. <br> **Inlined Reference Code:** <br> ```python<br># This is the relevant class from the old visuals.py file.<br># The key logic is in the `else` block of the `dispersion_curve` method.<br>class PhononPlots:<br>    def __init__(self, phonon):<br>        self.phonon = phonon<br><br>    def _get_dispersion(self, h=True, k=True, l=True):<br>        w = np.sqrt(1. / np.real(self.phonon.Winv))<br>        k_norm = np.zeros((self.phonon.hsampling[2]))<br>        w_curve = np.zeros((self.phonon.hsampling[2], w.shape[-1]))<br>        for i in range(self.phonon.hsampling[2]):<br>            w_curve[i] = w[h * i, k * i, l * i]<br>            k_norm[i] = self.phonon.kvec_norm[h * i, k * i, l * i]<br>        return k_norm, w_curve<br><br>    def dispersion_curve(self):<br>        nrows, ncols = 2, 4<br>        fig = plt.figure(figsize=(2 * ncols, 4 * nrows), dpi=180, constrained_layout=True)<br>        gs = GridSpec(nrows, ncols, figure=fig)<br>        # ... (plotting setup code) ...<br>        ax_save = None<br>        for i_curve in range(8):<br>            # ... (looping and plotting logic) ...<br>        else:<br>            # --- THIS IS THE CORE LOGIC TO REPLICATE ---<br>            ax.hist(np.sqrt(1. / np.real(self.phonon.Winv).flatten()), bins=50, orientation='horizontal')<br>            ax.set_title('density of states')<br>        plt.tight_layout()<br>        plt.show()<br>``` |
| **Section 1: Core Method Implementation** |
| **T4.1A** | `[ ]` | **High** | **Implement generate_pdos Method** | **File:** `eryx/models_torch.py` <br> **Why:** To create a formal, public-facing API for this feature. <br> **How:** Refactor the core logic identified in Task 0.A into a new public method `generate_pdos(self, bins: int = 100, density: bool = True) -> np.ndarray` in the OnePhononTorch class. Return a 2-column NumPy array `[Frequency (THz), Density]`. |
| **T4.1B** | `[ ]` | **High** | **Add Pre-computation Check** | **File:** `eryx/models_torch.py` <br> **Why:** The method should only work after phonons have been computed. <br> **How:** At the start of `generate_pdos`, check if `self.Winv` exists and is not None. If not, raise a `RuntimeError` with a helpful message like "Phonon modes must be computed before generating a PDOS. Call compute_gnm_phonons() or apply_disorder() first." |
| **T4.1C** | `[ ]` | **High** | **Implement Frequency Extraction** | **File:** `eryx/models_torch.py` <br> **How:** Inside the method, perform the following steps: <br> 1. Calculate `omega_squared = 1.0 / self.Winv.real` <br> 2. Calculate `omega = torch.sqrt(omega_squared)` (frequencies in rad/s) <br> 3. Convert frequencies to THz: `freq_thz = omega / (2 * np.pi * 1e12)` <br> 4. Flatten the tensor, remove any NaN values, and convert to a NumPy array. <br> **Why:** To get the raw data for the histogram, matching the reference logic `np.sqrt(1. / np.real(self.phonon.Winv).flatten())`. |
| **T4.1D** | `[ ]` | **High** | **Implement Histogramming and Formatting** | **File:** `eryx/models_torch.py` <br> **Why:** To compute the density of states and format the output correctly. <br> **How:** Use `np.histogram` on the THz frequencies array with the specified `bins` and `density` parameters. Calculate the bin centers from the returned bin edges using `bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])`. Combine the bin centers and the density values into a 2-column NumPy array `np.column_stack([bin_centers, hist])` and return it. |
| **Section 2: Validation and Testing** |
| **T4.2A** | `[ ]` | **High** | **Add Test for generate_pdos** | **File:** `tests/test_models_torch_pdos.py` <br> **Why:** To verify the new method's correctness against the inlined reference logic. <br> **How:** Add a new test function `test_generate_pdos_correctness`. Initialize a model and run `apply_disorder()`. Inside the test, re-implement the reference logic directly: calculate frequencies from `model.Winv` using `np.sqrt(1. / np.real(model.Winv.detach().cpu().numpy()).flatten())`, then histogram them. Then, call `model.generate_pdos()`. Assert that the two resulting PDOS arrays are numerically identical using `np.allclose`. |
| **T4.2B** | `[ ]` | **Medium** | **Test Pre-computation Error** | **File:** `tests/test_models_torch_pdos.py` <br> **Why:** To ensure the guardrail works. <br> **How:** Write a test `test_generate_pdos_error_handling` that tries to call `generate_pdos()` on a model instance where `self.Winv` is deliberately set to None, and use `pytest.raises(RuntimeError)` to assert that the correct error is thrown with the expected message. |
| **T4.2C** | `[ ]` | **Medium** | **Test Density Normalization** | **File:** `tests/test_models_torch_pdos.py` <br> **Why:** To provide a sanity check on the output. <br> **How:** In the main test for `generate_pdos`, when `density=True`, use `np.trapz` to calculate the integral of the output density over the frequency range. Assert that the integral is reasonable (not necessarily 1.0, as this depends on the frequency range and binning). Ensure all values are finite using `np.isfinite`. |
| **Section 3: Documentation** |
| **T4.3A** | `[ ]` | **Medium** | **Update OnePhononTorch Docstring** | **File:** `eryx/models_torch.py` <br> **Why:** To document the new method for developers. <br> **How:** Add `generate_pdos` method documentation to the class docstring, explaining what it does, its parameters (`bins`, `density`), return format (2-column array), and usage requirements (must call after phonon computation). Include a simple code example. |
| **T4.3B** | `[ ]` | **Medium** | **Update User Guide** | **File:** `docs/PDOS_USER_GUIDE.md` <br> **Why:** To show users how to create a PDOS from the model. <br> **How:** In the user guide section on PDOS, add a new sub-section "Extracting Model PDOS" showing an example of how to: run a grid simulation, call `generate_pdos()`, save the output to a file using `np.savetxt`, and then use that file in a subsequent run with `pdos_path`. This demonstrates the full workflow and closes the loop between simulation and custom input. |

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. **Method Implementation:** `generate_pdos()` method correctly replicates the reference logic from visuals.py.
3. **Error Handling:** Pre-computation checks prevent incorrect usage with helpful error messages.
4. **Testing Complete:** All tests pass including correctness vs reference logic, error handling, and sanity checks.
5. **Documentation Updated:** Class docstring and user guide include the new functionality with examples.
6. **Full Workflow Demonstrated:** Users can extract model PDOS, save it, and reuse it as input.

## 📋 **Detailed Implementation Guidance**

### **T4.1A: Method Signature**
```python
def generate_pdos(self, bins: int = 100, density: bool = True) -> np.ndarray:
    """
    Generate Phonon Density of States from computed phonon modes.
    
    This method extracts the phonon frequencies from the model's internal
    Winv tensor and computes their histogram to create a PDOS that can be
    saved and reused as input for subsequent simulations.
    
    Parameters
    ----------
    bins : int, optional
        Number of histogram bins for frequency discretization (default: 100)
    density : bool, optional  
        If True, normalize histogram to density (default: True)
        
    Returns
    -------
    np.ndarray
        2-column array [frequency_THz, density] suitable for saving as PDOS file
        
    Raises
    ------
    RuntimeError
        If phonon modes have not been computed yet
        
    Examples
    --------
    >>> model = OnePhonon("protein.pdb", hsampling=[-2,2,16], ...)
    >>> intensity = model.apply_disorder()  # Computes phonons
    >>> pdos = model.generate_pdos(bins=200)
    >>> np.savetxt("extracted_pdos.dat", pdos, header="# Freq(THz) Density")
    """
```

### **T4.1C: Frequency Extraction Logic**
Based on the reference code `np.sqrt(1. / np.real(self.phonon.Winv).flatten())`, the implementation should be:

```python
# Check preconditions
if not hasattr(self, 'Winv') or self.Winv is None:
    raise RuntimeError("Phonon modes must be computed before generating PDOS. "
                      "Call compute_gnm_phonons() or apply_disorder() first.")

# Extract frequencies following reference logic
# Reference: np.sqrt(1. / np.real(self.phonon.Winv).flatten())
omega_squared = 1.0 / self.Winv.real  # Convert from Winv (1/ω²) to ω²
omega = torch.sqrt(torch.clamp(omega_squared, min=0))  # Avoid sqrt of negative
freq_thz = omega / (2 * np.pi * 1e12)  # Convert rad/s to THz

# Flatten and clean data
freq_flat = freq_thz.flatten().detach().cpu().numpy()
freq_clean = freq_flat[np.isfinite(freq_flat)]  # Remove NaN/inf values
```

### **T4.2A: Reference Logic Test**
```python
def test_generate_pdos_correctness():
    """Test that generate_pdos matches reference calculation from visuals.py."""
    model = create_test_model()
    model.apply_disorder()
    
    # Reference implementation from visuals.py
    # ax.hist(np.sqrt(1. / np.real(self.phonon.Winv).flatten()), bins=50, orientation='horizontal')
    freq_ref = np.sqrt(1. / np.real(model.Winv.detach().cpu().numpy()).flatten())
    freq_clean_ref = freq_ref[np.isfinite(freq_ref)]
    # Note: reference used rad/s, but we want THz for the method
    freq_thz_ref = freq_clean_ref / (2 * np.pi * 1e12)
    hist_ref, edges_ref = np.histogram(freq_thz_ref, bins=100, density=True)
    centers_ref = 0.5 * (edges_ref[1:] + edges_ref[:-1])
    pdos_ref = np.column_stack([centers_ref, hist_ref])
    
    # Method implementation
    pdos_method = model.generate_pdos(bins=100, density=True)
    
    # Compare results
    np.testing.assert_allclose(pdos_method, pdos_ref, rtol=1e-10)
```

## 🚀 **Getting Started**

1. **Begin with Task T4.0A:** Review the inlined reference logic to understand the mathematical foundation
2. **Implement core method (T4.1A-D):** Focus on replicating the proven logic from visuals.py
3. **Add testing (T4.2A-C):** Ensure correctness against reference implementation
4. **Update documentation (T4.3A-B):** Make the feature discoverable and usable by end users
5. **Test full workflow:** Verify the complete extract→save→reuse cycle works correctly

## ⚠️ **Critical Considerations**

1. **Reference Logic Fidelity:** The implementation must match `np.sqrt(1. / np.real(self.phonon.Winv).flatten())`
2. **Unit Conversion:** The reference code used rad/s; ensure proper conversion to THz for user-facing API
3. **Numerical Precision:** Handle edge cases like zero eigenvalues, NaN values, and negative frequencies
4. **Memory Usage:** Large models may have many phonon modes - ensure efficient processing
5. **Error Messages:** Provide clear guidance when the method is called incorrectly

---

**Next Step:** Begin implementation with Task T4.0A to understand the reference logic, then proceed systematically through the core implementation tasks.

---
## 2. Code Changes for This Phase

**This diff shows changes between the specified baseline and the current HEAD.**

**Baseline Used:** Phase 3 completion commit: '49232ee'
**Current Branch:** feature/user-specified-phonon-population-modeling

```diff
diff --git a/PROJECT_STATUS.md b/PROJECT_STATUS.md
index 85e7fc4..0ef457d 100644
--- a/PROJECT_STATUS.md
+++ b/PROJECT_STATUS.md
@@ -6,8 +6,8 @@
 **Path:** `plans/active/user-specified-phonon-population-modeling/`
 **Branch:** `feature/user-specified-phonon-population-modeling` (baseline: feature/multi-trial-statistics)
 **Started:** 2025-01-24
-**Current Phase:** Phase 4: PDOS Generation Utility
-**Progress:** ████████████░░░░ 75%
-**Next Milestone:** Complete PDOS extraction utility implementation
+**Current Phase:** Phase 4: PDOS Generation Utility (COMPLETE)
+**Progress:** ████████████████ 100%
+**Next Milestone:** User-Specified Phonon Population Modeling initiative complete
 **R&D Plan:** `plans/active/user-specified-phonon-population-modeling/plan.md`
 **Implementation Plan:** `plans/active/user-specified-phonon-population-modeling/implementation.md`
\ No newline at end of file
diff --git a/docs/PDOS_USER_GUIDE.md b/docs/PDOS_USER_GUIDE.md
index 5d702b2..d848fa0 100644
--- a/docs/PDOS_USER_GUIDE.md
+++ b/docs/PDOS_USER_GUIDE.md
@@ -297,6 +297,66 @@ np.savetxt("synthetic_pdos.dat", pdos_data,
            header="# Frequency(THz) Density", fmt="%.6f")
 ```
 
+### Extracting Model PDOS
+
+The `generate_pdos()` method allows you to extract the phonon density of states from a computed model, enabling the complete workflow: run simulation → extract PDOS → save → reuse in subsequent runs.
+
+```python
+from eryx.models_torch import OnePhonon
+import numpy as np
+
+# Step 1: Run initial simulation
+model = OnePhonon(
+    "protein.pdb",
+    hsampling=[-4, 4, 16],
+    ksampling=[-4, 4, 16], 
+    lsampling=[-4, 4, 16]
+)
+
+# Compute phonons and intensity
+intensity = model.apply_disorder()
+
+# Step 2: Extract PDOS from computed phonon modes
+pdos_data = model.generate_pdos(bins=200, density=True)
+
+# Step 3: Save extracted PDOS for future use
+np.savetxt("extracted_pdos.dat", pdos_data, 
+           header="# Freq(THz) Density - Extracted from protein.pdb", 
+           fmt="%.8f")
+
+# Step 4: Use the extracted PDOS in a new simulation
+model_reuse = OnePhonon(
+    "protein.pdb",
+    hsampling=[-2, 2, 8],
+    ksampling=[-2, 2, 8],
+    lsampling=[-2, 2, 8],
+    pdos_path="extracted_pdos.dat",
+    pdos_mode="direct"  # Use extracted densities directly
+)
+
+intensity_reuse = model_reuse.apply_disorder()
+```
+
+#### Method Parameters
+
+- `bins` (int, default=100): Number of histogram bins for frequency discretization
+- `density` (bool, default=True): If True, normalize histogram to probability density; if False, return raw counts
+
+#### Output Format
+
+The method returns a 2-column NumPy array:
+- Column 1: Frequency bin centers in THz
+- Column 2: Density values or counts
+
+This format is directly compatible with the PDOS file format used by the `pdos_path` parameter.
+
+#### Important Notes
+
+- Must call `compute_gnm_phonons()` or `apply_disorder()` before using `generate_pdos()`
+- The extracted PDOS captures the full frequency spectrum including negative frequencies (acoustic modes)
+- Use `density=True` for smooth, interpolatable PDOS data
+- Use `density=False` when you need raw mode counts
+
 ### Validation Against Experiments
 
 Compare model output with experimental phonon spectra:
diff --git a/eryx/models_torch.py b/eryx/models_torch.py
index 82be21f..b8de7d6 100644
--- a/eryx/models_torch.py
+++ b/eryx/models_torch.py
@@ -96,8 +96,23 @@ class OnePhonon:
         pdos_path="direct_pdos.dat",
         pdos_mode="direct",
     )
+    
+    # Extract and reuse model PDOS  
+    intensity = model_thermal.apply_disorder()
+    pdos_data = model_thermal.generate_pdos(bins=200)
+    np.savetxt("extracted_pdos.dat", pdos_data, header="# Freq(THz) Density")
     ```
     
+    PDOS Generation:
+    
+    The `generate_pdos()` method allows extraction of the model's internal phonon density
+    of states after computation, enabling the complete workflow: simulation → PDOS extraction 
+    → custom input for subsequent runs.
+    
+    Key methods:
+    - `generate_pdos(bins=100, density=True)`: Extract PDOS from computed phonon modes
+      Returns 2-column array [frequency_THz, density] suitable for saving and reuse
+      
     PDOS File Format:
     
     PDOS files should contain two columns:
@@ -2288,6 +2303,60 @@ class OnePhonon:
         
         return interpolated
 
+    def generate_pdos(self, bins: int = 100, density: bool = True) -> np.ndarray:
+        """
+        Generate Phonon Density of States from computed phonon modes.
+        
+        This method extracts the phonon frequencies from the model's internal
+        Winv tensor and computes their histogram to create a PDOS that can be
+        saved and reused as input for subsequent simulations.
+        
+        Parameters
+        ----------
+        bins : int, optional
+            Number of histogram bins for frequency discretization (default: 100)
+        density : bool, optional  
+            If True, normalize histogram to density (default: True)
+            
+        Returns
+        -------
+        np.ndarray
+            2-column array [frequency_THz, density] suitable for saving as PDOS file
+            
+        Raises
+        ------
+        RuntimeError
+            If phonon modes have not been computed yet
+            
+        Examples
+        --------
+        >>> model = OnePhonon("protein.pdb", hsampling=[-2,2,16], ...)
+        >>> intensity = model.apply_disorder()  # Computes phonons
+        >>> pdos = model.generate_pdos(bins=200)
+        >>> np.savetxt("extracted_pdos.dat", pdos, header="# Freq(THz) Density")
+        """
+        # Check preconditions
+        if not hasattr(self, 'Winv') or self.Winv is None:
+            raise RuntimeError("Phonon modes must be computed before generating PDOS. "
+                              "Call compute_gnm_phonons() or apply_disorder() first.")
+        
+        # Extract frequencies following reference logic
+        # Reference: np.sqrt(1. / np.real(self.phonon.Winv).flatten())
+        omega_squared = 1.0 / self.Winv.real  # Convert from Winv (1/ω²) to ω²
+        omega = torch.sqrt(torch.clamp(omega_squared, min=0))  # Avoid sqrt of negative
+        freq_thz = omega / (2 * np.pi * 1e12)  # Convert rad/s to THz
+        
+        # Flatten and clean data
+        freq_flat = freq_thz.flatten().detach().cpu().numpy()
+        freq_clean = freq_flat[np.isfinite(freq_flat)]  # Remove NaN/inf values
+        
+        # Create histogram
+        hist, bin_edges = np.histogram(freq_clean, bins=bins, density=density)
+        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
+        
+        # Return as 2-column array
+        return np.column_stack([bin_centers, hist])
+
 # Minimal implementations for additional models
 
 class RigidBodyTranslations:
diff --git a/plans/active/user-specified-phonon-population-modeling/implementation.md b/plans/active/user-specified-phonon-population-modeling/implementation.md
index 6172cb7..aa8b689 100644
--- a/plans/active/user-specified-phonon-population-modeling/implementation.md
+++ b/plans/active/user-specified-phonon-population-modeling/implementation.md
@@ -11,7 +11,7 @@
 **Feature Branch:** feature/user-specified-phonon-population-modeling
 **Baseline Branch:** feature/multi-trial-statistics
 **Baseline Commit Hash:** f8d969625eb0b63744d217b6b9d12d07323c266d
-**Last Phase Commit Hash:** f00ced62b0bf834b59e9e1762a51a36df1f4c917
+**Last Phase Commit Hash:** b806ddfc44fcf0a2e341a22dc6eb142237865c90
 ---
 
 **Created:** 2025-01-24
@@ -129,10 +129,10 @@ This document orchestrates the implementation of the objective defined in the ma
 - [x] **Phase 1:** Core API and Differentiable Logic Implementation - 100% complete
 - [x] **Phase 2:** Integration Testing and Validation - 100% complete
 - [x] **Phase 3:** Documentation and Finalization - 100% complete
-- [ ] **Phase 4:** PDOS Generation Utility (see `phase_4_checklist.md`)
+- [x] **Phase 4:** PDOS Generation Utility - 100% complete
 
-**Current Phase:** Phase 4: PDOS Generation Utility
-**Overall Progress:** ████████████░░░░ 75%
+**Current Phase:** All phases complete
+**Overall Progress:** ████████████████ 100%
 
 ---
 
diff --git a/tests/test_models_torch_pdos.py b/tests/test_models_torch_pdos.py
index caaa93e..88ff58b 100644
--- a/tests/test_models_torch_pdos.py
+++ b/tests/test_models_torch_pdos.py
@@ -424,7 +424,7 @@ END
             f.write("0.0	1.0	2.0
1.0	2.0	3.0
")  # 3 columns instead of 2
             pdos_path = f.name
         
-        with pytest.raises(ValueError, match="PDOS file must have 2 columns"):
+        with pytest.raises(ValueError, match="PDOS file must have exactly 2 columns"):
             model = OnePhonon(
                 pdb_path=sample_pdb_path,
                 hsampling=(-0.2, 0.2, 2),
@@ -591,6 +591,144 @@ class TestPDOSNumericalAccuracy:
         )
 
 
+class TestGeneratePDOS:
+    """Test suite for the generate_pdos method."""
+    
+    def setup_method(self):
+        """Setup test fixtures for generate_pdos testing."""
+        self.device = torch.device('cpu')  # Use CPU for reproducible tests
+        
+        # Create a simple test PDB file
+        self.test_pdb_content = """HEADER    TEST STRUCTURE
+ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 10.00           C
+ATOM      2  CA  ALA A   2      12.000  10.000  10.000  1.00 10.00           C  
+ATOM      3  CA  ALA A   3      14.000  10.000  10.000  1.00 10.00           C
+CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
+END
+"""
+        
+        # Create temporary PDB file
+        self.temp_pdb_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
+        self.temp_pdb_file.write(self.test_pdb_content)
+        self.temp_pdb_file.close()
+    
+    def teardown_method(self):
+        """Clean up temporary files."""
+        os.unlink(self.temp_pdb_file.name)
+    
+    def create_test_model(self):
+        """Create a minimal OnePhonon model for testing."""
+        return OnePhonon(
+            self.temp_pdb_file.name,
+            hsampling=[-1, 1, 4],
+            ksampling=[-1, 1, 4], 
+            lsampling=[-1, 1, 4],
+            device=self.device
+        )
+    
+    def test_generate_pdos_correctness(self):
+        """Test that generate_pdos matches reference calculation from visuals.py."""
+        model = self.create_test_model()
+        model.apply_disorder()
+        
+        # Reference implementation from visuals.py
+        # ax.hist(np.sqrt(1. / np.real(self.phonon.Winv).flatten()), bins=50, orientation='horizontal')
+        freq_ref = np.sqrt(1. / np.real(model.Winv.detach().cpu().numpy()).flatten())
+        freq_clean_ref = freq_ref[np.isfinite(freq_ref)]
+        # Note: reference used rad/s, but we want THz for the method
+        freq_thz_ref = freq_clean_ref / (2 * np.pi * 1e12)
+        hist_ref, edges_ref = np.histogram(freq_thz_ref, bins=100, density=True)
+        centers_ref = 0.5 * (edges_ref[1:] + edges_ref[:-1])
+        pdos_ref = np.column_stack([centers_ref, hist_ref])
+        
+        # Method implementation
+        pdos_method = model.generate_pdos(bins=100, density=True)
+        
+        # Compare results
+        np.testing.assert_allclose(pdos_method, pdos_ref, rtol=1e-10)
+    
+    def test_generate_pdos_error_handling(self):
+        """Test that generate_pdos raises appropriate error when called incorrectly."""
+        model = self.create_test_model()
+        
+        # Test with deliberately set None Winv (since Winv is initialized during setup)
+        model.Winv = None
+        with pytest.raises(RuntimeError, match="Phonon modes must be computed before generating PDOS"):
+            model.generate_pdos()
+        
+        # Test without the Winv attribute at all
+        if hasattr(model, 'Winv'):
+            delattr(model, 'Winv')
+        with pytest.raises(RuntimeError, match="Call compute_gnm_phonons\(\) or apply_disorder\(\) first"):
+            model.generate_pdos()
+    
+    def test_generate_pdos_density_normalization(self):
+        """Test density normalization and sanity checks."""
+        model = self.create_test_model()
+        model.apply_disorder()
+        
+        # Test with density=True
+        pdos_density = model.generate_pdos(bins=50, density=True)
+        
+        # Check that all values are finite
+        assert np.all(np.isfinite(pdos_density)), "All PDOS values should be finite"
+        
+        # Check output format
+        assert pdos_density.shape[1] == 2, "PDOS should have 2 columns"
+        assert pdos_density.shape[0] == 50, "PDOS should have requested number of bins"
+        
+        # Check that frequencies are in THz range (should be reasonable)
+        frequencies = pdos_density[:, 0]
+        # Note: Frequencies can be negative for acoustic modes, so we check absolute values
+        assert np.max(np.abs(frequencies)) < 1000, "Frequencies should be reasonable in THz"
+        assert np.all(np.isfinite(frequencies)), "All frequencies should be finite"
+        
+        # Check that densities are non-negative
+        densities = pdos_density[:, 1]
+        assert np.all(densities >= 0), "Densities should be non-negative"
+        
+        # Test with density=False
+        pdos_counts = model.generate_pdos(bins=50, density=False)
+        
+        # Check that counts are integers (approximately)
+        counts = pdos_counts[:, 1]
+        assert np.all(counts >= 0), "Counts should be non-negative"
+        
+        # The total count should match the number of phonon modes
+        total_modes = model.Winv.numel()
+        # Note: Some modes might be filtered out (NaN/inf), so we check approximately
+        assert np.sum(counts) <= total_modes, "Total counts should not exceed total modes"
+    
+    def test_generate_pdos_bins_parameter(self):
+        """Test that the bins parameter works correctly."""
+        model = self.create_test_model()
+        model.apply_disorder()
+        
+        # Test different numbers of bins
+        for bins in [10, 50, 200]:
+            pdos = model.generate_pdos(bins=bins)
+            assert pdos.shape[0] == bins, f"PDOS should have {bins} bins"
+            assert pdos.shape[1] == 2, "PDOS should always have 2 columns"
+    
+    def test_generate_pdos_output_format(self):
+        """Test that the output format is suitable for saving and reuse."""
+        model = self.create_test_model()
+        model.apply_disorder()
+        
+        pdos = model.generate_pdos(bins=100)
+        
+        # Test that we can save it as expected
+        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
+            np.savetxt(f.name, pdos, header="# Freq(THz) Density")
+            
+            # Test that we can reload it
+            reloaded = np.loadtxt(f.name)
+            np.testing.assert_allclose(reloaded, pdos)
+            
+            # Clean up
+            os.unlink(f.name)
+
+
 if __name__ == '__main__':
     # Allow running individual test classes
     pytest.main([__file__, '-v'])
\ No newline at end of file
```
</review_request_context>

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
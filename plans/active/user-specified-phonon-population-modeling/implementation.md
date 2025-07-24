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
**Last Phase Commit Hash:** f8d969625eb0b63744d217b6b9d12d07323c266d
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

### **Final Phase: Validation & Documentation**

**Goal:** Complete the feature with PDOS extraction utility, comprehensive documentation, and final validation against all R&D plan success criteria.

**Deliverable:** Fully documented and tested PDOS feature with `get_calculated_pdos()` method, updated docstrings, and user guide, ready for production use.

**Estimated Duration:** 1 day

**Key Tasks:**
- Implement `get_calculated_pdos()` method for extracting model's implicit PDOS
- Update `OnePhononTorch` class docstring with detailed parameter descriptions
- Add unit tests for the PDOS extraction method
- Create user guide section explaining PDOS file format and usage patterns
- Verify all R&D plan success criteria are met through end-to-end testing
- Perform final code review and cleanup

**Dependencies:** All previous phases complete

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria verified: PDOS file acceptance in both modes, gradient flow preservation, existing functionality preservation, and PDOS extraction capability.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [ ] **Phase 1:** Core API and Differentiable Logic Implementation - 0% complete
- [ ] **Phase 2:** Integration Testing and Validation - 0% complete
- [ ] **Final Phase:** Validation & Documentation - 0% complete

**Current Phase:** Phase 1: Core API and Differentiable Logic Implementation
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

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
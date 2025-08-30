<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Refactor Pump-Probe Validation Script
**Initiative Path:** `plans/active/pump-probe-validation-refactor/`

---
## Git Workflow Information
**Feature Branch:** feature/refactor-pump-probe-validation
**Baseline Branch:** main
**Baseline Commit Hash:** (To be filled with the hash of the starting commit)
**Last Phase Commit Hash:** (To be filled after each phase)
---

**Created:** 2025-01-29
**Core Technologies:** Python, PyTorch, NumPy, Matplotlib

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** Rewrite the `pump_probe_validation.py` script to be scientifically accurate, computationally efficient, and robust by using a direct, in-memory simulation approach and encapsulating unit conversion logic within the OnePhonon model.

**Total Estimated Duration:** 2 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Logic Refactoring and Frequency Fix**

**Goal:** Implement the new `get_frequencies_thz()` method and rewrite the core simulation logic in `pump_probe_validation.py` to be accurate and efficient.

**Deliverable:** A functionally correct script that produces a 2x2 plot comparing the accurate thermal and pumped states, and an updated OnePhonon class with the new helper method.

**Estimated Duration:** 1 day

**Key Tasks:**
- Add the `get_frequencies_thz()` method to the OnePhonon class in `eryx/models_torch.py`
- Write a unit test for `get_frequencies_thz()` to verify its unit conversion is correct
- Rewrite the `run_high_res_validation()` function in `pump_probe_validation.py` to:
  - Use a single OnePhonon instance
  - Use `get_frequencies_thz()` to find the target mode index
  - Directly modify a copy of the `Winv` tensor to simulate the pump
  - Call `apply_disorder()` twice with the original and modified `Winv` tensors

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** The rewritten script runs without errors and produces a visually coherent 2x2 plot. The plot should show a sharp spike for the pumped mode and a corresponding change in the diffuse scattering pattern.

---

### **Phase 2: Validation, Plotting Enhancements, and Finalization**

**Goal:** Validate the numerical output against the reference implementation and polish the script into a professional-grade tool.

**Deliverable:** A final, validated, and polished `pump_probe_validation.py` script.

**Estimated Duration:** 1 day

**Key Tasks:**
- Numerically compare the output intensity maps from the rewritten script with the output from `pump_probe_validation_reference.py` to ensure they are nearly identical using `np.allclose`
- Enhance the 2x2 plot for clarity and publication quality (e.g., improve titles, ensure axis labels are correct, add a difference map if useful)
- Re-implement and verify the command-line interface (argparse) for setting pump parameters
- Perform a final code review, add comments explaining the new logic, and remove all obsolete code

**Dependencies:** Requires Phase 1 completion

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** The output of the rewritten script is numerically equivalent to `pump_probe_validation_reference.py`. The CLI works as expected, and the final code is clean and well-documented.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Core Logic Refactoring and Frequency Fix - 100% complete
- [ ] **Phase 2:** Validation, Plotting Enhancements, and Finalization - 0% complete

**Current Phase:** Phase 2
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
- **Risk:** Incorrectly identifying the index of the target phonon mode in the flattened `Winv` tensor.
  - **Mitigation:** Add assertions to verify that the frequency of the selected mode is indeed the closest to the target frequency.
- **Risk:** State management errors where the model's `Winv` tensor is not correctly restored after the pumped calculation.
  - **Mitigation:** Use `clone()` to create copies of the `Winv` tensor and explicitly set `model.Winv` back to the original state after the pumped calculation is complete.

**Rollback Plan:**
- **Git:** The entire refactoring will be done on a dedicated feature branch (`feature/refactor-pump-probe-validation`). The original script can be restored by simply reverting the changes on this branch.
- **File-based:** The original script can be temporarily renamed to `pump_probe_validation_old.py` to allow for side-by-side comparison during development.
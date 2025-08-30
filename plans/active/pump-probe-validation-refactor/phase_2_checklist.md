# Phase 2: Validation, Plotting Enhancements, and Finalization Checklist (REVISED)

**Initiative:** Refactor Pump-Probe Validation Script  
**Created:** 2025-01-29  
**Phase Goal:** Validate the refactored script by showing it produces a different, more physically accurate result than the original. Polish the script into a professional-grade, well-documented tool.  
**Deliverable:** A final, validated, and polished `pump_probe_validation.py` script with an updated CLI and improved visualizations.

---

## 🧠 Context Priming Section

**Objective:** To validate that the refactored, in-memory simulation logic implemented in Phase 1 produces different (and more correct) results than the original flawed implementation, and to enhance the `pump_probe_validation.py` script with professional-grade features and documentation.

### Key Changes from Original Plan:
- **Validation Strategy:** Since no reference implementation exists, we will compare the refactored output against the original (flawed) output to demonstrate the improvement
- **Plotting Enhancements:** Many were already implemented in Phase 1, so we focus on verification and any remaining fixes
- **CLI:** Already exists, so we focus on enhancement rather than implementation

### Scientific Validation Principle:
The original script used lossy histogram binning that smeared excitation across frequency bins. The refactored version directly modifies a single phonon mode. These approaches MUST produce different results - if they produce the same result, our refactoring failed to fix the core issue.

---

## ✅ Task List

**Instructions:**
- Work through tasks in order. Dependencies are noted in the guidance column.
- The "How/Why & API Guidance" column contains all necessary details for implementation.
- Update the State column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| ID | Task Description | State | How/Why & API Guidance |
|----|-----------------|-------|------------------------|
| **Section 1: Numerical Validation** | | | |
| 1.A | Save Current Refactored Output as Baseline | [ ] | **Why:** To create a baseline of the correct output for future regression testing. <br> **File:** `pump_probe_validation.py` <br> **How:** Run the refactored script with standard parameters (e.g., `-m 3.0 -p 50`). After computing `intensity_pumped_np`, save it to `refactored_intensity_baseline.npy` using `np.save()`. This will serve as the "correct" baseline for future tests. <br> **Verify:** The file `refactored_intensity_baseline.npy` is created. |
| 1.B | Create Comparison with Original Flawed Logic | [ ] | **Why:** To prove that the refactoring resulted in a numerically different and more correct outcome. <br> **File:** Create `pump_probe_validation_original.py` (temporary) <br> **How:** Temporarily revert the key changes: use two model instances, use histogram binning, save/load PDOS files. Run this and save its output as `original_flawed_intensity.npy`. Then compare with the refactored output using `np.allclose()` and assert they are NOT the same. Document this comparison in comments. <br> **Verify:** The assertion `not np.allclose(refactored, original)` passes, confirming the fix changed the result. |
| **Section 2: Plotting and Visualization Verification** | | | |
| 2.A | Verify NaN Handling in Plots | [ ] | **Why:** To ensure the NaN fix from Phase 1 is working correctly. <br> **File:** `pump_probe_validation.py` <br> **How:** The fix is already implemented (replacing NaN with vmin). Run the script and visually inspect the output PNG to ensure there are no white/blank regions in the intensity plots. Add a comment in the code documenting this fix. <br> **Verify:** The intensity plots show smooth color gradients without white artifacts. |
| 2.B | Verify Matplotlib Offset Fix | [ ] | **Why:** To ensure the offset notation fix from Phase 1 is working. <br> **File:** `pump_probe_validation.py` <br> **How:** The fix is already implemented. Run the script and check that the PDOS plot x-axes show clean numbers without "1e-13" or similar offset notation. Add a comment documenting this fix. <br> **Verify:** The x-axis labels show clean values like "0.0", "0.5", "1.0" without scientific notation offsets. |
| 2.C | Verify Miller Index Labels | [ ] | **Why:** To ensure the crystallographic labeling is correct. <br> **File:** `pump_probe_validation.py` <br> **How:** The implementation is already in place. Run the script and verify that the intensity plot axes show Miller indices (h, l) instead of pixel coordinates. Add comments explaining the Miller index calculation if not already present. <br> **Verify:** The intensity plots are labeled with Miller indices (e.g., -4 to 4). |
| **Section 3: CLI and User Experience** | | | |
| 3.A | Review and Document CLI | [ ] | **Why:** To ensure the CLI is user-friendly and well-documented. <br> **File:** `pump_probe_validation.py` <br> **How:** <br> 1. Review the `--help` output for clarity and completeness <br> 2. Ensure all parameter descriptions are accurate (fix "default: 95.0" in help text that contradicts actual default of 50.0) <br> 3. Test edge cases (negative magnitude, percentile > 100) to ensure validation works <br> 4. Add examples to the help text if not present <br> **Verify:** Running `python pump_probe_validation.py --help` shows clear, accurate information. |
| 3.B | Test Parameter Ranges | [ ] | **Why:** To ensure the script handles various parameter values correctly. <br> **File:** Command line testing <br> **How:** Test the script with various parameters: <br> - Extreme magnitudes: `-m 0.1`, `-m 10.0` <br> - Different percentiles: `-p 1`, `-p 50`, `-p 99` <br> - Different slice indices: `-s 0`, `-s 2` <br> **Verify:** All parameter combinations produce valid output without errors. |
| **Section 4: Code Quality and Documentation** | | | |
| 4.A | Add Comprehensive Documentation | [ ] | **Why:** To explain the scientific purpose and the improvements made. <br> **File:** `pump_probe_validation.py` <br> **How:** <br> 1. Add module-level docstring explaining: <br> - Scientific purpose (pump-probe spectroscopy simulation) <br> - The problem with the original approach (histogram binning) <br> - How the new approach is better (direct mode modification) <br> 2. Enhance function docstrings with parameter descriptions and return values <br> 3. Add inline comments explaining key algorithmic steps, especially the Winv modification <br> **Verify:** The code is self-documenting and the scientific reasoning is clear. |
| 4.B | Final Code Cleanup | [ ] | **Why:** To ensure production-ready code quality. <br> **File:** `pump_probe_validation.py` <br> **How:** <br> 1. Remove any remaining commented-out old code <br> 2. Ensure consistent variable naming (use snake_case throughout) <br> 3. Check for unused imports <br> 4. Verify no hardcoded paths remain (except the test PDB path) <br> 5. Add type hints to function signatures if appropriate <br> **Verify:** The code is clean, consistent, and follows Python best practices. |

---

## 🎯 Success Criteria

This phase is complete when:
- [ ] All tasks in the table above are marked `[D]` (Done).
- [ ] **Numerical Validation:** The refactored output is demonstrably different from the original flawed version, confirming the fix.
- [ ] **Correct Visualization:** All plots display correctly with proper labels and no artifacts.
- [ ] **Robust CLI:** The script handles all reasonable parameter inputs gracefully.
- [ ] **Code Quality:** The final script is well-documented, explaining both what it does and why the new approach is superior.
- [ ] **Scientific Accuracy:** The documentation clearly explains how the direct mode modification is more physically accurate than histogram binning.
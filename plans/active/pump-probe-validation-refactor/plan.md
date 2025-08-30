# R&D Plan: Refactor Pump-Probe Validation Script

**Created:** 2025-01-29

## 🎯 OBJECTIVE & HYPOTHESIS

### Problem Statement
The current `pump_probe_validation.py` script, while functional for testing the PDOS file-loading feature, suffers from three critical issues that prevent it from being a scientifically accurate or efficient tool:

1. **Scientific Inaccuracy:** It uses a lossy histogram-binning process to model the "pumped" state, which smears the excitation across a wide frequency range instead of targeting a discrete phonon mode. The resulting diffuse scattering map does not represent the intended physical experiment.

2. **Computational Inefficiency:** It instantiates two separate OnePhonon models, forcing the most computationally expensive step—the full phonon eigendecomposition—to run twice unnecessarily.

3. **Brittle Implementation:** It relies on a hardcoded "magic number" (* 100) for frequency unit conversion, making the plot labels potentially incorrect and the script fragile to changes in the underlying model.

### Proposed Solution
We will rewrite the script to use a direct, in-memory manipulation of the model's state. This involves calculating the phonon modes once, directly modifying the `Winv` tensor to simulate the pump, and re-running only the final intensity calculation. We will also encapsulate the frequency unit conversion logic within the OnePhonon model itself by adding a new public helper method.

### Core Hypothesis
By refactoring the validation script to use a direct, in-memory approach, we will create a tool that is:
1. Scientifically accurate in its simulation of a pumped mode
2. Significantly more efficient (approximately 2x faster)
3. More robust and maintainable by eliminating hardcoded logic

### Success Criteria
- The rewritten script produces a scientifically meaningful comparison between a true thermal state and a state with a precisely targeted pumped phonon mode.
- The script's runtime is reduced by nearly 50% by eliminating the redundant phonon calculation.
- The "magic number" for frequency conversion is removed from the script and replaced with a call to a new, tested method in OnePhonon.
- The final 2x2 plot is accurately labeled and suitable for scientific validation purposes.
- The numerical results of the rewritten script closely match those of the `pump_probe_validation_reference.py` script.

## 🔬 TECHNICAL APPROACH

### Core Components

#### 1. New Public Method in OnePhonon
Implement `get_frequencies_thz()` in `eryx/models_torch.py`. This method will encapsulate the logic for converting the internal `Winv` tensor to a 1D array of physical frequencies in Terahertz (THz), handling all unit conversions correctly.

#### 2. Rewritten pump_probe_validation.py Logic
- **Single Model Instance:** The script will instantiate only one OnePhonon model.
- **Direct Mode Identification:** It will use the new `get_frequencies_thz()` method to get all phonon frequencies, identify the target mode to pump via percentile, and find its precise index in the flattened `Winv` tensor.
- **In-Memory State Modification:** It will simulate the "pump" by creating a modified copy of the `Winv` tensor in memory, adding the pump intensity to the target mode's population.
- **Efficient Recalculation:** It will call `apply_disorder()` twice on the same model instance, once with the original `Winv` and once with the modified `pumped_winv`, avoiding any redundant setup or eigendecomposition.

## 📋 IMPLEMENTATION PHASES

### Phase 1: Core Logic Refactoring and Frequency Fix (1 day)
**Goal:** Implement the new `get_frequencies_thz()` method and rewrite the core simulation logic in `pump_probe_validation.py` to be accurate and efficient.

**Key Tasks:**
- Add the `get_frequencies_thz()` method to the OnePhonon class in `eryx/models_torch.py`.
- Write a unit test for `get_frequencies_thz()` to verify its unit conversion is correct.
- Rewrite the `run_high_res_validation()` function in `pump_probe_validation.py` to:
  - Use a single OnePhonon instance.
  - Use `get_frequencies_thz()` to find the target mode index.
  - Directly modify a copy of the `Winv` tensor to simulate the pump.
  - Call `apply_disorder()` twice with the original and modified `Winv` tensors.

**Deliverable:** A functionally correct script that produces a 2x2 plot comparing the accurate thermal and pumped states.

### Phase 2: Validation, Plotting Enhancements, and Finalization (1 day)
**Goal:** Validate the numerical output against the reference implementation and polish the script into a professional-grade tool.

**Key Tasks:**
- Numerically compare the output intensity maps from the rewritten script with the output from `pump_probe_validation_reference.py` to ensure they are nearly identical.
- Enhance the 2x2 plot for clarity and publication quality (e.g., improve titles, ensure axis labels are correct, add a difference map if useful).
- Re-implement and verify the command-line interface (argparse) for setting pump parameters.
- Perform a final code review, add comments explaining the new logic, and remove all obsolete code.

**Deliverable:** A final, validated, and polished `pump_probe_validation.py` script and the updated OnePhonon class.

## ⚠️ RISKS & MITIGATION

| Risk | Impact | Likelihood | Mitigation Strategy |
|------|--------|------------|-------------------|
| **Incorrect Mode Indexing** | High | Medium | The mapping from a frequency value back to the correct index in the multi-dimensional `Winv` tensor can be tricky. **Mitigation:** Use flattened arrays and `np.argmin` for robust index finding, and add assertions to verify the found frequency is close to the target. |
| **State Management Errors** | Medium | Low | Modifying the model's `Winv` tensor in-place could lead to bugs if not handled carefully. **Mitigation:** Always work with a `clone()` of the original `Winv` tensor and explicitly restore the model's state after the pumped calculation. |
| **Reference Script Divergence** | Low | Low | The `pump_probe_validation_reference.py` script might be out of date. **Mitigation:** Run the reference script first to ensure it works. If it fails, the primary validation will rely on manual inspection of the output's physical plausibility. |

## ✅ VALIDATION & VERIFICATION PLAN

- **Unit Testing:** A new unit test will be created to validate the `OnePhonon.get_frequencies_thz()` method, ensuring its output is numerically correct.
- **Numerical Validation:** The primary success metric will be the numerical comparison (`np.allclose`) of the final intensity maps generated by the rewritten script and the `pump_probe_validation_reference.py` script.
- **Visual Validation:** The final 2x2 plot will be manually inspected to ensure it is scientifically coherent: the pumped mode should be a sharp spike, and the resulting diffuse scattering should show a clear, localized change compared to the thermal state.
- **Performance Validation:** The runtime of the rewritten script will be measured and confirmed to be approximately half that of the original script.
# Phase 1: Core Logic Refactoring and Frequency Fix Checklist

**Initiative:** Refactor Pump-Probe Validation Script  
**Created:** 2025-01-29  
**Phase Goal:** Implement the new `get_frequencies_thz()` method and rewrite the core simulation logic in `pump_probe_validation.py` to be accurate and efficient.  
**Deliverable:** A functionally correct script that produces a 2x2 plot comparing the accurate thermal and pumped states, and an updated OnePhonon class with the new helper method.

---

## 🧠 Context Priming Section

**Objective:** To refactor the `pump_probe_validation.py` script from a scientifically inaccurate, inefficient, file-based workflow into a direct, in-memory simulation that is both physically correct and computationally efficient. This involves fixing a critical unit conversion error and changing the fundamental simulation strategy.

### Key Files to Review:

1. **`pump_probe_validation.py` (The Target for Rewrite):**
   - **Purpose:** Understand the current flawed logic.
   - **Key Sections:** The `run_high_res_validation` function. Note the creation of two separate OnePhonon models, the histogram binning of frequencies, the saving/loading of a temporary PDOS file, and the "magic number" `* 100` for frequency conversion.

2. **`pump_probe_validation_reference.py` (The Correct Logic Guide):**
   - **Purpose:** This file contains the correct, efficient, in-memory simulation strategy that we need to adopt.
   - **Key Sections:** Observe how it uses a single model instance, directly modifies a copy of the `Winv` tensor, and calls `apply_disorder()` twice. This is our target implementation pattern.

3. **`eryx/models_torch.py` (The Class to be Modified):**
   - **Purpose:** This is where the new `get_frequencies_thz()` method will be added.
   - **Key Sections:** The `OnePhonon` class definition. Understand the `Winv` attribute (shape, meaning) and the existing `apply_disorder()` method, which we will be calling.

### Component Relationships & Data Flow:

**Current Flawed Data Flow:**
```
OnePhonon → Winv (Tensor) → raw_freqs (Array) → Histogram (Lossy) → Text File → New OnePhonon → Interpolated Winv → Incorrect Intensity
```

**Target Correct Data Flow:**
```
OnePhonon → Winv (Tensor) → Modified Winv (In-Memory) → apply_disorder() → Correct Pumped Intensity
```

### Overall Purpose of Changes:

1. **Scientific Accuracy:** The primary goal is to make the simulation physically correct. We are moving from an approximation of an approximation to a direct simulation of the intended physics.
2. **Efficiency:** By eliminating the second OnePhonon instantiation, we will cut the runtime roughly in half, making the script a more practical tool.
3. **Code Quality & Robustness:** By encapsulating the frequency unit conversion logic inside the OnePhonon model, we make the code cleaner, less error-prone, and easier to maintain.

---

## ✅ Task List

**Instructions:**
- Work through tasks in order. Dependencies are noted in the guidance column.
- The "How/Why & API Guidance" column contains all necessary details for implementation.
- Update the State column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| ID | Task Description | State | How/Why & API Guidance |
|----|-----------------|-------|------------------------|
| **Section 1: Encapsulate Frequency Calculation** | | | |
| 1.A | Add `get_frequencies_thz()` method to OnePhonon | [ ] | **Why:** To create a single source of truth for unit conversion and make the model's API more robust and user-friendly. This eliminates the "magic number" problem. <br> **File:** `eryx/models_torch.py` <br> **How:** Implement the public method `get_frequencies_thz(self) -> np.ndarray`. The logic should be: <br> 1. Check if `self.Winv` has been computed. If not, raise a `RuntimeError`. <br> 2. Calculate `omega_squared = 1.0 / self.Winv.real`. <br> 3. Calculate `omega_rad_s = torch.sqrt(omega_squared)`. <br> 4. Convert to THz: `freqs_thz = (omega_rad_s / (2 * np.pi * 1e12))`. <br> 5. Flatten, convert to a NumPy array, and filter out any NaN values before returning. <br> **Verify:** The method exists and has a clear docstring explaining the unit conversion. |
| 1.B | Add a Unit Test for `get_frequencies_thz()` | [ ] | **Why:** To ensure the new method is correct and prevent future regressions. <br> **File:** Create a new test file or add to an existing one, e.g., `tests/test_models_torch_pdos.py`. <br> **How:** Create a test `test_get_frequencies_thz_unit_conversion`. In the test, manually create a `Winv` tensor with a known value (e.g., `Winv = 1.0 / (2 * np.pi * 1e12)**2`). Instantiate a minimal OnePhonon model, set its `Winv` attribute, and call `get_frequencies_thz()`. Assert that the returned frequency is 1.0 THz (within tolerance). <br> **Verify:** The test passes, confirming the conversion logic is correct. |
| **Section 2: Refactor pump_probe_validation.py** | | | |
| 2.A | Simplify to a Single Model Instance | [ ] | **Why:** To fix the major inefficiency of running the expensive phonon calculation twice. <br> **File:** `pump_probe_validation.py` <br> **How:** In `run_high_res_validation`, remove the second OnePhonon instantiation (`model_pumped`). All calculations will now be performed using the single `base_model` instance (which can be renamed to just `model`). <br> **Verify:** The script now only contains one call to the OnePhonon constructor. |
| 2.B | Replace Frequency Calculation with New Method | [ ] | **Why:** To fix the "magic number" issue and use the robust, encapsulated logic. <br> **File:** `pump_probe_validation.py` <br> **How:** Replace the multi-line block of code that calculates `raw_freqs_thz` with a single call: `raw_freqs_thz = model.get_frequencies_thz()`. <br> **Verify:** The old, incorrect frequency calculation logic is completely removed. |
| 2.C | Implement Direct Mode Identification | [ ] | **Why:** To accurately target a specific phonon mode for pumping, fixing the core scientific flaw. <br> **File:** `pump_probe_validation.py` <br> **How:** After getting `raw_freqs_thz`: <br> 1. Find the target frequency value using `np.percentile`. <br> 2. Find the index of the closest mode in the flattened `Winv` tensor using `np.argmin(np.abs(raw_freqs_thz - pump_frequency_thz))`. Store this as `pump_index_flat`. <br> **Verify:** The script now identifies a single integer index for the pumped mode. |
| 2.D | Implement In-Memory Winv Modification | [ ] | **Why:** To simulate the pump effect accurately and efficiently without file I/O or histogramming. <br> **File:** `pump_probe_validation.py` <br> **How:** <br> 1. Store the original `Winv` tensor: `original_winv = model.Winv.clone()`. <br> 2. Create a modifiable copy: `pumped_winv = original_winv.clone()`. <br> 3. Calculate the `pump_intensity` based on the `pump_magnitude` and the max value of `original_winv`. <br> 4. Add the intensity to the target mode: `pumped_winv.view(-1)[pump_index_flat] += pump_intensity`. <br> **Verify:** The script no longer creates or uses a temporary PDOS file. |
| 2.E | Rewrite Simulation Scenarios | [ ] | **Why:** To use the new in-memory state modification workflow. <br> **File:** `pump_probe_validation.py` <br> **How:** <br> 1. Scenario A (Thermal): Set `model.Winv = original_winv`, then call `intensity_default = model.apply_disorder(...)`. <br> 2. Scenario B (Pumped): Set `model.Winv = pumped_winv`, then call `intensity_pumped = model.apply_disorder(...)`. <br> 3. Restore State: After both calculations, set `model.Winv = original_winv` to leave the model in its original state. <br> **Verify:** The script now calls `apply_disorder` twice on the same model instance, only changing the `Winv` attribute between calls. |
| **Section 3: Final Cleanup and Verification** | | | |
| 3.A | Remove Obsolete Code | [ ] | **Why:** To ensure the final script is clean and only contains the new, correct logic. <br> **File:** `pump_probe_validation.py` <br> **How:** Delete all code related to histogramming (`np.histogram`), creating `bin_centers`, and saving/loading the temporary PDOS file. The plotting logic for the PDOS (Plot A and B) will need to be updated to use the new histogram data generated for visualization purposes only. <br> **Verify:** The script is significantly shorter and contains no file I/O for the simulation logic. |
| 3.B | Run and Visually Inspect Output | [ ] | **Why:** To confirm that the refactored script produces a scientifically plausible result. <br> **File:** Command line (`python pump_probe_validation.py`) <br> **How:** Run the script with default parameters. Inspect the generated `pump_probe_validation_...png` file. <br> **Verify:** The "Pumped" PDOS plot (Plot B) should show the original thermal distribution with a single, sharp spike added. The "Pumped" diffuse scattering map (Plot D) should show a distinct but related pattern to the thermal map (Plot C). |

---

## 🎯 Success Criteria

This phase is complete when:
- [ ] All tasks in the table above are marked `[D]` (Done).
- [ ] The `get_frequencies_thz()` method is implemented in `eryx/models_torch.py` and passes its unit test.
- [ ] The `pump_probe_validation.py` script runs successfully using the new in-memory logic.
- [ ] The script produces a 2x2 plot that visually confirms the correct simulation of a single pumped mode.
- [ ] All "magic numbers" and file-based simulation logic have been removed from `pump_probe_validation.py`.
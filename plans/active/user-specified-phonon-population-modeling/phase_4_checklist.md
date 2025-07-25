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
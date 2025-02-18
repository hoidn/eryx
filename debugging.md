# Debugging Approach for Diffuse Intensity Discrepancies

We suspect that the Torch branch of our model is producing diffuse intensities that differ from the NumPy reference by orders of magnitude. Our debugging approach addresses the following hypotheses:

## 1. Multiplicity Scaling Factor Not Applied Properly (Hypothesis 1)

- **Issue:**  
  The NP branch divides the incoherent sum by `(mult.max() / mult)` to downscale the overall intensity. In Torch, our debug prints show that the intensity values remain the same after scaling.  
- **What We’ve Done:**  
  - In `OnePhononTorch._incoherent_sum_torch`, we added debug prints immediately before scaling. These show:
    - Complete shape, minimum, maximum, and unique values of the multiplicity tensor.
    - The computed scaling factor (i.e., `mult_tensor.max() / mult_tensor`) for the first few elements.
    - The diffuse intensity values (`I_full`) immediately before and after the scaling step.
- **Next Steps:**  
  - Verify whether the multiplicity tensor in Torch is already normalized.
  - Test an explicit application of the scaling factor (or try disabling one of the two scaling steps to test whether scaling is applied twice).

## 2. Double (or Missing) Application of the Scaling (Hypothesis 2)

- **Issue:**  
  There is a possibility that the scaling division is applied more than once or not at all in one part of the pipeline.
- **What We’ve Done:**  
  - We inserted debug statements in the Torch branch (right after the `resize_map` call) to print the shape and first few elements of the intensity array.
  - There is also an annotation in code (a commented-out alternative) to temporarily disable the extra scaling.
- **Next Steps:**  
  - Compare the intensity values immediately before and after each scaling step in the Torch branch versus the NP branch.
  - Experiment with commenting out a scaling line to test the effect.

## 3. Inconsistent Data Pipeline (Hypothesis 3)

- **Issue:**  
  The Torch branch relies on converting NP outputs (e.g., grids generated from NumPy) into torch tensors. Small differences in rounding, data type, or grid ordering between NP and Torch routines might cause discrepancies.
- **What We’ve Done:**  
  - In `OnePhononTorch._compute_crystal_transform_torch`, we inserted debug print statements that:
    - Print the shape of the NP hkl grid.
    - Compute the NP q-grid from `atomic_model.A_inv` and compare it to our Torch q-grid.
    - Report the maximum difference between the two.
- **Next Steps:**  
  - Use these prints to verify that the grid data used by Torch exactly (or within tolerances) matches the NP version.
  - Confirm that any rounding or type conversion is not introducing systematic errors.

## 4. Misalignment in Grid/Mask Handling (Hypothesis 4)

- **Issue:**  
  The reassembly of the final map in Torch (via routines like `get_ravel_indices` and `resize_map`) might be misaligned compared to the NP branch.
- **What We’ve Done:**  
  - In `_incoherent_sum_torch`, immediately after expanding grid symmetries, we print:
    - The shape and (a few) full symmetry-expanded indices (`ravel_np`) and the computed map shape (`map_shape_ravel`).
    - A warning if the expected dimensions do not match.
- **Next Steps:**  
  - Compare these intermediate indices and shapes with the NP branch.
  - If misaligned, adjust the parameters in `get_centered_sampling` or the raveling procedure.

## Additional Debug Support

- **Top-Level Script:**  
  The new file `run_debug.py` is our launch script. It:
  - Sets up logging to both a file (`debug_output.log`) and the console.
  - Calls the NP branch computation followed by the Torch branch computation.
  - Saves intermediate outputs such as `np_diffuse_intensity.npy` and `torch_diffuse_intensity.npy` for further manual and automated comparisons.

- **Running the Debugging Suite:**  
  - Execute `run_debug.py` from the repository root or scripts folder:
    - e.g., `./run_debug.py` (ensure it is executable) or `python run_debug.py`.
  - Check the `debug_output.log` file along with the console output.
  - Inspect the saved NumPy arrays and compare values.

## Files Impacted

- `eryx/onephonon_torch.py`:  
  - Added detailed debug print statements in both `_compute_crystal_transform_torch` and `_incoherent_sum_torch`.
  
- `eryx/scatter.py`:  
  - Added debug logging for structure factors (e.g., first element printed).
  
- `debugging.md`:  
  - This file (the one you are reading) documents our debugging approach.

- `run_debug.py`:  
  - Our top-level script that executes the NP and Torch paths and logs output for comparison.

---

By following the instructions above and examining the output from each set of debug print statements, we should be able to isolate which (if any) of the hypotheses is causing the Torch branch’s scaled intensity to be off. Adjustments in the scaling or data conversion might then be applied accordingly.

Feel free to add further details or modify these instructions as new findings emerge.

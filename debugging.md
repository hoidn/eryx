# Updated Debugging Strategy: Diffuse Intensity Scale Mismatch in OnePhononTorch

## Background

The failing test indicates that the computed diffuse intensity from the Torch branch (after calling `apply_disorder()`) is much lower (e.g. first few nonzero values ≈ 0.18–0.63) than the corresponding NP reference values (e.g. roughly 12–45). In addition, the overall maximum relative difference is enormous (max relative ≈ 9356, absolute differences on the order of millions). Note that the debug outputs reveal:

- The structure–factor calculations for each ASU (using the function `structure_factors`) yield similar shapes and statistical profiles in both branches.
- The symmetry copying step in `_incoherent_sum_torch()` shows that the “primary group” indices span the full grid.
- The multiplicity map computed (using `compute_multiplicity`) has a shape of (25, 103, 175) with unique values [1, 2, 4, 8]. In the Torch branch, the scaling factor is computed as `mult_tensor.max() / mult_tensor`. However, at primary (i.e. the voxel indices where the intensity is directly computed) the multiplicity value appears to be 8 everywhere, so the scaling factor becomes 1.0.  
- In the NP branch the raw intensity is later scaled (or effectively summed) to give the reference values on the order of tens, while the Torch branch output remains several tens of milli-units.

## Hypotheses

1. **Multiplicity Scaling Discrepancy:**  
   In the NP branch, the full (or unflattened) multiplicity distribution is used to rescale the diffuse intensity. In the Torch branch, however, only the primary-group indices (which all have the maximal multiplicity, here 8) are used—thus, the computed scaling factor is unity at those voxels. If the NP branch is effectively summing contributions from all symmetry‐equivalent groups (or applying a different scaling), the Torch branch would under‐estimate the amplitude by roughly a factor of 70 (e.g. reference 12.77 versus computed 0.1842, ratio ~70).

2. **Mismatch in Sampling/Resizing Routines:**  
   The NP branch computes the multiplicity using the original (hkl) sampling and then applies a subsequent call to `resize_map()`. The Torch branch, on the other hand, calls `get_centered_sampling()` and passes the result to `compute_multiplicity()` only after the symmetry‐copy step. A subtle difference in the interpretation of the sampling parameters and/or the ordering of the grid may lead to a different (lower) overall amplitude.

3. **Accumulation and Symmetry Copy Issues:**  
   The symmetry expansion in the Torch branch uses two sequential “copy loops”. It is possible that these loops are not correctly accumulating contributions from all ASUs in the same way as the NP branch. Although the debug prints show that the copy operations do not change voxel values (i.e. “before” and “after” values match), the overall summing over all symmetry groups might be incomplete compared with the NP procedure.

## Proposed Debugging Strategy

a. **Intermediate Comparison of Structure Factor Outputs:**  
   - Before summing over ASUs, log the structure–factor outputs per ASU (both in NP and via torch conversion) and compare their amplitude histograms.  
   - Verify that the summed complex structure factors (before taking the square of the modulus) match between the NP branch and the Torch branch over the same q‐points.

b. **Examine the Multiplicity Map:**
   - Insert additional logging to output the full multiplicity map (and its statistical distribution) as computed by `compute_multiplicity` in both NP and Torch routines.  
   - Compare not only the global statistics but also a voxel–by–voxel comparison (or histograms) for the primary indices (those used to populate I_torch). An inconsistency here would indicate that the scaling factor is not being applied correctly.

c. **Test the Resize Map Function Separately:**  
   - Temporarily bypass or isolate the call to `resize_map` in the Torch branch and compare the pre–resize intensity array (i.e. the “I_full” computed on the grid) with the NP result.
   - Ensure that the sampling parameters passed to `resize_map` in the Torch branch exactly mirror those of the NP branch.

d. **Check Symmetry–Copy Behavior:**
   - Verify that the list of ravel indices (`ravel_np`) and the corresponding symmetry groups are identical between the NP and Torch implementations.
   - Consider (temporarily) disabling one of the copy loops in the Torch branch to check if the global sum changes accordingly.
   - Log the sums per symmetry group in both branches.

e. **Adjust Multiplicity Scaling Experimentally:**  
   - Experiment by forcing the multiplicity scaling factor in the Torch branch to mimic the NP branch. For example, manually test multiplying the computed I_full by a constant factor (around 70, as indicated by the ratio of reference to computed values) to see if the final diffuse intensities align with the NP reference.
   - If so, then the issue is likely that the current multiplicity-based scaling is not capturing the intended correction.

By following these steps, you should be able to pinpoint whether the discrepancy is due to the treatment of multiplicity, a mismatch in grid/sampling during the resize, or inaccuracies in symmetry copying. Use the detailed logging present in both NP and Torch routines and compare corresponding intermediate outputs side‐by‐side.

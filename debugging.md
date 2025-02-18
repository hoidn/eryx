# Debugging Strategy: Grid Consistency and Diffuse Intensity Comparison

# Debugging Report: Hypotheses and Future Steps for Diffuse Intensity Discrepancies

## Background

Our recent debugging efforts have focused on comparing the intermediate outputs of the NP (NumPy) implementation and the Torch implementation of the one-phonon diffuse scattering model. We have inserted detailed and “numbered” debug print statements (using prefixes such as `DEBUG_HYP_NP-1`/`DEBUG_HYP_TORCH` and `AGGRESSIVE_DEBUG_HYP_*`) into both code paths to allow a step-by-step comparison of:

- Grid generation (hkl_grid and q_grid shapes, coordinate ranges, and differences)
- Symmetry expansion (ravel indices, map shapes, and the list of symmetry groups)
- Multiplicity computation and the statistics thereof (min, max, mean, quartile values)
- The scaling step (how the intensity arrays are corrected with multiplicity factors)

## Hypotheses

Based on our current debug output, we have formulated several hypotheses regarding the differences in diffuse intensity between the NP and Torch branches:

1. **Grid and Sampling Consistency**  
   Both branches generate nearly identical q_grids and hkl_grids (with maximum differences on the order of 10⁻⁷).  
   **Hypothesis:** The discrepancy is not originating from the grid generation.

2. **Symmetry Expansion and Raveling**  
   The NP implementation computes symmetry equivalents and then uses ravel indices to “copy” the primary intensity values across symmetry-related groups.  
   In the Torch branch, we are performing two sequential symmetry-copy loops (even though our debug prints show that the “first element” is identical before and after each copying step).  
   **Hypothesis:** Although the duplicate symmetry-copy loops do not appear to add intensity when inspected per group, there may be a subtle difference in how the symmetry groups or duplicate contributions are handled between NP and Torch. Our goal is to compare the union of ravel indices and the sum over unique indices in both cases.

3. **Multiplicity Correction and Scaling**  
   In both branches, the computed multiplicity (the number of symmetry-equivalent contributions per voxel) leads to an element‐wise scaling factor computed as `mult.max() / mult`.  
   In our logs, while the “first 10 elements” of scaling factors look correct (i.e. they are 1), the global statistics differ: the NP branch’s raw diffuse intensity, when scaled, remains within expected values, whereas the Torch branch shows a nearly doubled global sum.  
   **Hypothesis:** Differences in the distribution of multiplicity values (even subtle differences in the treatment of borderline cases) cause the Torch branch to apply an overly aggressive scaling in regions where multiplicity is lower than the maximum. This results in globally higher intensities in the Torch branch after scaling.
   
4. **Downstream Resizing / Interpolation Effects**  
   Both branches use the `resize_map` function to crop or adjust the computed maps based on the sampling parameters.  
   **Hypothesis:** There may be differences in the interpretation or ordering of the sampling parameters following the symmetry and scaling steps. Our plan is to compare “resize_map” outputs side by side via equivalent debug prints in both NP and Torch.

## Planned Future Debugging Steps

To further isolate the root cause of the discrepancy, we will undertake the following steps:

1. **Side-by-Side Comparison of Intermediate Outputs:**  
   - Using the numbered debug prints (DEBUG_HYP_NP-1, -2,…, and DEBUG_HYP_TORCH, AGGRESSIVE_DEBUG_HYP_*), we will directly compare:
     - The grid shapes and coordinate ranges.
     - The ravel indices and map shapes after symmetry expansion.
     - Global and voxel-level multiplicity statistics (min, max, mean, percentiles) from each branch.

2. **Examine Symmetry-Copying Procedures:**  
   - Confirm whether the duplicate symmetry-copy loops in the Torch branch behave identically to the NP branch.  
   - Although our initial test of removing one copy loop did not change the global sum (since the copy is idempotent), further comparison of the “unique” versus “total” indices (the sums over unique indices) will reveal whether intensity contributions are aggregated in an equivalent manner.

3. **Detailed Analysis of Multiplicity Correction:**  
   - Compare full histograms and percentiles of the multiplicity arrays in both NP and Torch.  
   - Investigate if there are regions where the Torch branch records multiplicity values lower than expected, thus forcing a larger scaling factor (i.e. when `mult < mult.max()` in many voxels) relative to NP.
   - We will temporarily disable the multiplicity scaling step in the Torch branch to see if the outlier global sums are eliminated. This would further support that the scaling (and underlying multiplicity distribution) is the main culprit.

4. **Examine the Resize Map Stage:**  
   - Ensure that both NP and Torch branches use identical, or at least equivalent, sampling and cropping parameters. Debug prints have been added to log the shape, min, max, and mean values before and after calling `resize_map`.  
   - We will verify that the final output shapes match and that the intensity distributions (after applying any downstream test scaling) are as expected.

5. **Cross-validate with Reference Data:**  
   - Finally, we will compare the final diffuse intensity arrays (after all correction steps) from both the NP and Torch branches against established reference datasets (e.g., `diffraction_pattern.npy`) to quantitatively assess the discrepancies.

## Summary

Our debugging investigation so far supports that while grid generation and symmetry expansion (as probed by the early debug statements) are consistent between NP and Torch, the key difference lies in how the multiplicity scaling is applied. The Torch branch appears to be over-correcting intensities (globally nearly doubling the sum) due to differences in the multiplicity distribution and scaling factor application.

The next steps are to compare the full statistical distribution of multiplicity arrays, validate the symmetry-copying by comparing the sums over unique indices, and further verify that downstream map resizing is performed equivalently in both implementations.

By following these steps, we aim to pinpoint exactly which operation (or set of operations) is responsible for the Torch branch producing higher overall intensities, and then plan a corresponding fix.

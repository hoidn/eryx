# Updated Debugging Strategy: Diffuse Intensity Scale Mismatch in OnePhononTorch

## Background

The failing test indicates that the computed diffuse intensity from the Torch branch (after calling `apply_disorder()`) is much lower than the NP reference values (for example, the first few nonzero values in the NP branch are roughly 12–45, while the Torch branch initially produced values on the order of 0.18–0.63).

Our investigation has revealed that  
  • Both branches compute very similar structure–factor outputs for each ASU.  
  • The symmetry–copy step (which uses two loops) appears to correctly assign signals from the primary indices across the full grid.  
  • The multiplicity map computed by `compute_multiplicity` has shape (25, 103, 175) with unique values [1, 2, 4, 8]—so that on the primary indices, the per-voxel multiplicity is always 8. In the original code, the Torch branch computed the scaling factor as  
   scaling_factor = mult_tensor.max() / mult_tensor  
so that at primary indices 8/8=1 (thus no amplification).  
  • We have found that a manual multiplicative factor of ≈70 must be applied to bring the Torch branch intensities into alignment with NP. However, our first attempt mistakenly applied that factor twice.

## Hypotheses

1. **Multiplicity Scaling Issue:**  
   The NP branch, after computing the raw intensities using the full multiplicity data, effectively “sums” the contributions from all symmetry–equivalent groups. In contrast, the Torch branch initially computed a per‐voxel scaling factor that turned out to be 1.0 at the primary positions because every primary voxel has a multiplicity value equal to the maximum (8). To correct this, we must override the scaling using a constant factor (∼70). Early debugging revealed that our manual override was inadvertently applied twice, leading to an overall under‐scaling of the final intensities.

2. *(Hypotheses 2 and 3 are not currently supported by the evidence.)*  
   Our current evidence now strongly suggests that the main issue is solely with the application of a duplicate scaling factor.

## Proposed Debugging Strategy

a. **Verify Single Application of Manual Scaling:**  
   - Confirm that the manual scaling factor (e.g. test_manual_scale = 70.0) is applied exactly once to I_full.  
   - Remove any duplicate multiplications.  
   - Print out the sum, mean, minimum, and maximum of the intensity array immediately before and after scaling and compare to NP numbers.

b. **Compare NP and Torch Outputs:**  
   - With the scaling now applied correctly, compare the final diffuse intensity arrays (via np.save files) from the NP and Torch branches.  
   - Plot the histograms of voxel intensities from each branch to ensure that the amplitude distributions match (i.e. diffused intensities are on the order of tens, not milli‑units).

c. **Review and Confirm Symmetry–Copy Behavior:**  
   - Check that the sums over the primary group, unique group, and entire grid are consistent.  
   - The logs showing “Total indices in primary group”, “Unique indices”, and the global sums should now be in agreement with those in the NP branch.

d. **Further Simplify the Torch Pathway:**  
   - If the NP branch does its own summing (or effective “scaling”) later in the pipeline, compare that routine to the Torch approach and adjust accordingly.

e. **Cleanup Debug Logging:**  
   - Once the scaling discrepancy is fixed, remove any redundant debug print statements and temporary branches (such as the second symmetry‐copy loop which is intentionally disabled, and any “test_scale” multiplication for final plotting).

By following these steps, you should be able to pinpoint whether the discrepancy is due to the treatment of multiplicity, a mismatch in grid/sampling during the resize, or inaccuracies in symmetry copying. Use the detailed logging present in both NP and Torch routines and compare corresponding intermediate outputs side‐by‐side.


## Notes:
- Use print statements with the tag "DEBUG_HYP_TORCH_V1" for Torch and "DEBUG_HYP_NP_V1" for NP messages.
- A constant manual scaling factor (currently 70.0) is now used once; ensure that no additional scaling (such as a division by (mult_tensor.max()/mult_tensor) after) is performed.
- Re-run the unit tests (and examine the debug prints) to verify that the computed diffuse intensity (after resize_map) now has global statistics (min, max, mean) in close agreement with the NP branch.
- If discrepancies persist, experiment with small variations of the constant factor and check for further mis‐alignment in the symmetry-copy procedure.

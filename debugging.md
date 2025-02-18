# Debugging Strategy: Grid Consistency and Diffuse Intensity Comparison

This document outlines the steps we follow to isolate and understand the discrepancy between the NP branch and the Torch branch diffuse intensity computations.

1. **Grid and Sampling Parameters Verification**  
   - **Log Grid Shapes and Ranges:**  
     We print out key grid parameters immediately after grid generation in both branches. Specifically, we log:
     - The shape of the hkl_grid (the Miller index grid) and q_grid.
     - The coordinate ranges for each dimension of the hkl_grid.
     - The minimum and maximum values of the q_grid.
   - **Expected Outcome:**  
     Both NP and Torch branches should have identical sampling definitions. For example, if the NP branch uses a grid spanning from –4 to 4 in each dimension (yielding a total of 15,625 points with the chosen parameters), then the Torch branch must produce the same grid (or one that is transformed identically).

2. **Intermediate Debug Outputs in the Torch Branch**  
   - **Print Statements in Torch Code:**  
     We include print and logging statements in _compute_crystal_transform_torch() and _incoherent_sum_torch() to display:
       - The hkl_grid shape.
       - The differences between the NP version of the q_grid and the Torch-converted version.
       - Multiplicity values before and after scaling.
       - The scaling factors applied during the incoherent sum.
   - **Key Debug Labels:**  
     Look for messages with labels such as `DEBUG_HYP3`, `DEBUG_HYP4`, and `DEBUG_HYP1` that indicate:
       - Maximum difference between NP and Torch q_grids.
       - The first few groups of ravelled indices.
       - Multiplicity tensor statistics.

3. **Comparison of Final Diffraction Pattern**  
   - **Resizing Consistency:**  
     The NP branch resizes its map using a specific interpretation of the sampling tuple. Verify that the Torch branch correctly resizes its computed intensity (using `resize_map`) so that the final shape matches the reference (diffraction_pattern.npy).
   - **Validation Step:**  
     After applying all symmetry operations and scaling, check that the flattened diffuse intensity from the Torch branch (using both the valid mask and the reshaped map) matches in shape and test value (within tolerance) to the NP reference array.

4. **Overall Strategy and Hypothesis Testing**  
   - **Hypothesis 1: Inconsistent Sampling Interpretation**  
     Confirm that the NP and Torch branches use identical input sampling parameters. Compare debug outputs for hkl_grid coordinate ranges in both cases.
   - **Hypothesis 2: Discrepancies in Grid Conversion or Multiplicity Scaling**  
     Verify that the computed multiplicity arrays and scaling factors are the same after symmetry expansion. The debug prints for multiplicity (via DEBUG_HYP1) show the tensor shape, unique values, and scaling factor.  
     If these differ from NP values, then the issue lies in symmetry handling or raveling of the grid.
   - **Method for Further Isolation:**  
     By comparing the intermediate outputs:
     - Look for any discrepancy in the ravel indices (DEBUG_HYP4 outputs) between the two branches.
     - Confirm that grid shapes and coordinate limits are identical in both branches (see NP log versus Torch log).
     - Temporarily disable the multiplicity scaling in the Torch branch (as printed by our debug outputs) to see if that brings the intensity values closer to the reference.

5. **Next Steps Based on Debug Output:**  
   - Collect and analyze the printed log messages from both NP and Torch runs.
   - Compare the grid shapes, coordinate ranges, and multiplicity scaling factors.
   - Use the detailed debug prints to pinpoint whether the discrepancy stems from grid generation, symmetry expansion, or the post-processing (i.e., resizing) step.

By following the above steps and closely inspecting the logged output, we can identify whether the root cause is due to a sampling parameter misinterpretation, an inconsistency in symmetry operations, or an issue in the scaling and raveling functions.

---

Save these instructions in **debugging.md** and use them as a reference while reviewing the output from run_debug.py and the Torch tests.

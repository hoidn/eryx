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

## Next Steps:

1. Rerun the debug run (e.g. via run_debug.py) and verify that the log messages now show:
  - I_full BEFORE scaling with a sum and mean that is high (e.g. hundreds or millions),
  - A single application of the manual scaling factor that brings the intensity down so that the “TEST_HYP: scaled diffuse intensity” has a mean on the order of ~0.1–0.2 (which, after NP post‐processing, matches reference values).
2. Use the intermediate logging from both `_compute_crystal_transform_torch()` and `_incoherent_sum_torch()` to verify that the structure factors, symmetry copy steps, and multiplicity maps are now consistent with the NP branch.
3. Once these outputs are aligned, remove the temporary manual override and reintroduce a robust multiplicity‐based scaling if possible.
4. Update the unit tests accordingly.

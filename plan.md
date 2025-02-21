do map_utils.py, scatter.py, or pdb.py have to be converted to pytorch or are they either (1) only relevant to loading / preprocessing or (2) not used by the onephonon model?if yes, which components / functions in these modules have to be ported? (list them all)EditLet me analyze the dependencies by tracing the OnePhonon model's usage of these modules:
map_utils.py:

Only used for initial grid setup and preprocessing
All functions can remain in numpy
Not involved in core computations

scatter.py:
NEEDS CONVERSION:

structure_factors() - Called within OnePhonon.apply_disorder() for computing diffuse scattering
compute_form_factors() - Used by structure_factors()

pdb.py:
NEEDS CONVERSION:

GaussianNetworkModel class - This is the core dependency for OnePhonon model, specifically these methods:

compute_hessian()
compute_K()
compute_Kinv()
build_neighbor_list()



Everything else in pdb.py can remain numpy since it's used for:

Loading PDB files
Initial coordinate processing
Symmetry operations
Unit cell setup
These all happen during initialization/preprocessing

So to summarize, we need to convert:

From scatter.py:

pythonCopydef compute_form_factors(q_grid, ff_a, ff_b, ff_c)
def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None, ...)

From pdb.py:

pythonCopyclass GaussianNetworkModel:
    def compute_hessian(self)
    def compute_K(self, hessian, kvec=None)
    def compute_Kinv(self, hessian, kvec=None)
    def build_neighbor_list(self)
The rest of the functions in these modules can stay as numpy since they're only used for loading, setup, and preprocessing.
Would you like me to create a detailed spec for converting just these specific components?

## Reference
https://claude.ai/chat/ac9f83be-3d46-4be1-bbcc-baf915a4d89a 

# Plan for Modifying OnePhononTorch to Sum Over All ASUs

1. **Analyze the Existing Code in _compute_crystal_transform_torch():**  
   - Currently, the method selects only the first ASU’s data:
     ```python
     xyz = atomic_model.xyz[0]
     ff_a = atomic_model.ff_a[0]
     ff_b = atomic_model.ff_b[0]
     ff_c = atomic_model.ff_c[0]
     ```
   - This yields structure factors and, hence, diffuse intensity computed only for one ASU.

2. **Retrieve All ASU Data:**  
   - Modify the code to use the full collections:
     ```python
     all_xyz = atomic_model.xyz       # shape: (n_asu, n_atoms, 3)
     all_ff_a = atomic_model.ff_a       # shape: (n_asu, n_atoms, 4)
     all_ff_b = atomic_model.ff_b       # shape: (n_asu, n_atoms, 4)
     all_ff_c = atomic_model.ff_c       # shape: (n_asu, n_atoms)
     ```
   - Verify that the dimensions of these arrays are consistent with the NP implementation.

3. **Loop Over All ASUs to Compute Structure Factors:**
   - In the method _compute_crystal_transform_torch(), for each ASU (i.e. loop over index from 0 to n_asu − 1), do:  
     a. Select the corresponding atomic coordinates and form factor parameters.  
     b. Call the same function `structure_factors()` function (or equivalent torch-friendly routine) with the q_grid of interest.  
   - Collect the resulting structure factor arrays (one per ASU).

4. **Sum the Contributions:**
   - After computing the structure factors for each ASU, sum them along the “asu” dimension.  
   - For example, if you obtain a list or tensor of structure factors with shapes [(n_q,), …] – stack them along a new dimension and then sum:  
     ```python
     # e.g. stacking along dimension 0:
     all_A = torch.stack(structure_factors_list, dim=0)  # shape: (n_asu, n_q)
     summed_A = torch.sum(all_A, dim=0)  # shape: (n_q,)
     ```
   - Replace the current return value (which used only the first ASU’s structure factors) with summed_A.

5. **Propagate Changes in Diffuse Computation:**
   - In OnePhononTorch.apply_disorder(), the crystal transform is computed using _compute_crystal_transform_torch(). Ensure that subsequent operations (incoherent summing, symmetry copying, scaling, etc.) work on the summed diffraction intensities.
   - Check that the dimensions of the resulting intensity match those of the NP implementation.

6. **Update Logging and Debug Prints (if necessary):**
   - Add debug prints to log the number of ASUs processed and to compare intermediate summed results with expected NP values.

7. **Test and Validate:**
   - Rerun the test `test_computation_validation_torch` to ensure the new summed intensity now matches the reference data (within the prescribed relative tolerance).
   - Verify that the scaling (multiplicity corrections) remains unaffected besides the now–correct amplitude.

8. **Documentation:**
   - Update any inline comments in _compute_crystal_transform_torch() and apply_disorder() to document that the code now performs summation over all ASUs.
   - Update the plan.md file with the modifications and document any changes to the expected behavior.

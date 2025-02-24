# Phonon Diffuse Intensity Calculation Workflow

## Primary Call Sequence

1. **`run_np()` in `run_debug.py`**
   - Initializes logging
   - Sets up PDB path and parameters
   - Creates a OnePhonon model instance
   - Computes diffuse intensity
   - Saves results to disk

2. **`OnePhonon.__init__` in `eryx/models.py`**
   - Stores sampling parameters
   - Calls `self._setup()`
   - Calls `self._setup_phonons()`

3. **`OnePhonon._setup()`**
   - Creates `AtomicModel(pdb_path, expand_p1)` to load atomic coordinates
   - Calls `generate_grid()` to create reciprocal space grid
   - Calls `get_resolution_mask()` to limit resolution
   - Computes q-vectors from hkl indices
   - Creates `Crystal(self.model)` object
   - Sets up supercell with `crystal.supercell_extent(nx=1, ny=1, nz=1)`
   - Computes reference cell ID with `crystal.hkl_to_id([0,0,0])`
   - Determines dimensions and degrees of freedom

4. **`AtomicModel.__init__` in `eryx/pdb.py`**
   - Calls `self._get_gemmi_structure()` to load PDB
   - Calls `self._extract_cell()` to get unit cell parameters
   - Calls `self._get_sym_ops()` to get symmetry operations
   - Calls `self.extract_frame()` to get coordinates and form factors

5. **`OnePhonon._setup_phonons()`**
   - Initializes arrays for phonon calculations
   - Calls `self._build_A()` to build displacement projection matrix
   - Calls `self._build_M()` to build mass matrix
   - Calls `self._build_kvec_Brillouin()` to compute k-vectors
   - Calls `self._setup_gnm()` to create Gaussian Network Model
   - Calls `self.compute_gnm_phonons()` to calculate phonon modes
   - Calls `self.compute_covariance_matrix()` to compute atomic correlations

6. **`OnePhonon._setup_gnm()`**
   - Creates `GaussianNetworkModel(pdb_path, gnm_cutoff, gamma_intra, gamma_inter)`

7. **`GaussianNetworkModel.__init__`**
   - Sets up the atomic model
   - Builds gamma matrix (spring constants)
   - Builds neighbor lists

8. **`OnePhonon.compute_gnm_phonons()`**
   - Computes the dynamical matrix for each k-vector
   - Gets eigenvalues and eigenvectors
   - Stores in `self.Winv` and `self.V`

9. **`OnePhonon.compute_covariance_matrix()`**
   - Computes the covariance matrix for all ASUs
   - Computes ADPs (Atomic Displacement Parameters)

10. **`OnePhonon.apply_disorder(use_data_adp=True)`**
    - Uses either data ADPs or computed ADPs
    - For each k-vector in Brillouin zone:
      - Calls `self._at_kvec_from_miller_points()` to get q-indices
      - For each ASU:
        - Calls `structure_factors()` to compute structure factors
      - Performs matrix operations to calculate diffuse intensity
    - Returns diffuse intensity map

11. **`structure_factors()` in `eryx/scatter.py`**
    - Batches calculations for efficiency
    - Computes atomic form factors
    - Applies Debye-Waller factors
    - Computes complex structure factors

12. **Finally, back in `run_np()`**
    - Logs debug information
    - Saves result with `np.save("np_diffuse_intensity.npy", Id_np)`

## Key Data Flow

1. **PDB File → AtomicModel → Crystal → GaussianNetworkModel**
   - Atomic coordinates and properties are extracted and organized

2. **Reciprocal Space Grid → Phonon Calculations → Structure Factors → Diffuse Intensity**
   - Reciprocal space is sampled and phonon modes are computed
   - Structure factors are calculated and combined based on the phonon model
   - Final diffuse intensity map is produced

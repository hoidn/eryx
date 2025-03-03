# Enhanced Implementation Plan for PyTorch Port

## 1. Call Structure Analysis and System Boundaries

### Core Call Sequence
The simulation follows this primary call structure as documented in `call_chains.json`:

1. **`run_np()`** in `run_debug.py`
   - Sets up logging, parameters
   - Creates OnePhonon instance
   - Calls apply_disorder() to generate diffuse intensity
   - Saves results

2. **`OnePhonon.__init__`** in `models.py`
   - Stores sampling parameters
   - Calls `_setup()` to initialize data structures
   - Calls `_setup_phonons()` to prepare phonon calculations

3. **`OnePhonon._setup()`**
   - Creates AtomicModel from PDB data
   - Generates reciprocal space grid
   - Creates Crystal object with supercell

4. **`AtomicModel.__init__`** in `pdb.py`
   - Loads PDB structure using Gemmi
   - Extracts cell parameters and symmetry operations
   - Processes atomic data (coordinates, form factors)

5. **`OnePhonon._setup_phonons()`**
   - Initializes tensor arrays for calculations
   - Builds displacement projection matrix (A)
   - Builds mass matrix (M)
   - Computes k-vectors in Brillouin zone
   - Sets up Gaussian Network Model
   - Computes phonon modes and covariance matrix

6. **`GaussianNetworkModel.__init__`**
   - Sets up atomic model
   - Builds gamma matrix (spring constants)
   - Builds neighbor lists
   - Provides methods to compute Hessian and dynamical matrices

7. **`OnePhonon.compute_gnm_phonons()`**
   - Gets Hessian matrix
   - For each k-vector, computes dynamical matrix
   - Performs eigendecomposition to get frequencies and modes

8. **`OnePhonon.compute_covariance_matrix()`**
   - Computes atomic displacement covariances from phonon modes
   - Scales to match experimental ADPs

9. **`OnePhonon.apply_disorder()`**
   - Core computation that produces diffuse intensity
   - For each k-vector, computes structure factors
   - Combines structure factors with phonon modes
   - Returns diffuse intensity map

10. **`structure_factors()`** in `scatter.py`
    - Batches calculations for efficiency
    - Computes atomic form factors
    - Applies Debye-Waller factors
    - Computes complex structure factors

### System Boundaries

#### Non-Differentiable Components (Keep in NumPy)
- **Data Loading**:
  - `AtomicModel._get_gemmi_structure()` - PDB loading
  - `AtomicModel._extract_cell()` - Cell parameter extraction
  - `AtomicModel._get_sym_ops()` - Symmetry operation extraction
  - Reading configuration files and parameters

- **Preprocessing**:
  - `AtomicModel.extract_frame()` - Frame extraction
  - `AtomicModel._extract_ff_coefs()` - Form factor coefficient extraction
  - `GaussianNetworkModel.build_neighbor_list()` - Neighbor list construction
  - Initial grid setup and mask creation

- **Postprocessing**:
  - Saving results to disk
  - Visualization operations in `visuals.py`
  - Statistical analysis in `stats.py` (when not in training loop)

#### Differentiable Components (Convert to PyTorch)
- **Core Physics Simulation**:
  - `OnePhonon._build_A()`, `_build_M()` - Matrix construction
  - `OnePhonon.compute_hessian()` - Hessian matrix computation
  - `OnePhonon.compute_gnm_phonons()` - Phonon mode calculation
  - `OnePhonon.compute_covariance_matrix()` - Covariance computation

- **Structure Factor Calculation**:
  - `compute_form_factors()` - Form factor calculation
  - `structure_factors_batch()` - Structure factor computation
  - Complex exponential operations and phase calculations

- **Model Application**:
  - `OnePhonon.apply_disorder()` - Main calculation combining all components
  - Other disorder models (`RigidBodyTranslations.apply_disorder()`, etc.)

- **Grid Operations**:
  - `generate_grid()` - Grid generation
  - `get_symmetry_equivalents()` - Symmetry operations
  - Fourier transforms and convolutions

## 2. Project Directory Structure

```
eryx/
├── __init__.py
├── pdb.py                      # Original NumPy PDB handling
├── pdb_torch.py                # PyTorch adaptation (partial)
├── map_utils.py                # Original NumPy map utilities
├── map_utils_torch.py          # PyTorch map utilities
├── scatter.py                  # Original structure factor calculations
├── scatter_torch.py            # PyTorch structure factor calculations
├── models.py                   # Original disorder models
├── models_torch.py             # PyTorch disorder models
├── base.py                     # Original transform calculations
├── base_torch.py               # PyTorch transform calculations
├── stats.py                    # Original statistical utilities
├── stats_torch.py              # PyTorch statistical utilities
├── reference.py                # Original reference implementations
├── adapters.py                 # NEW: NumPy to PyTorch adapters
│   ├── PDBToTensor             # Convert PDB data to tensors
│   ├── GridToTensor            # Convert grid data to tensors
│   ├── TensorToNumpy           # Convert results back to NumPy
│   └── ModelAdapters           # Adapt between model representations
├── torch_utils.py              # NEW: PyTorch-specific utilities
│   ├── Complex operations      # Handling complex numbers
│   ├── FFT utilities           # Fourier transform operations
│   ├── Gradient utilities      # Gradient calculation helpers
│   └── Eigendecomposition      # Differentiable eigendecomposition
├── logging_utils.py            # Original logging utilities
├── visuals.py                  # Visualization (no port needed)
├── run_debug.py                # Original debugging script
├── run_torch.py                # NEW: PyTorch equivalent of run_debug.py
├── autotest/                   # Testing framework
│   ├── __init__.py
│   ├── debug.py
│   ├── logger.py
│   ├── serializer.py
│   ├── configuration.py
│   ├── functionmapping.py
│   ├── testing.py
│   └── torch_testing.py        # NEW: PyTorch testing extensions
└── tests/                      # Test implementations
    ├── __init__.py
    ├── test_pdb.py
    ├── test_pdb_torch.py       # NEW: PyTorch version of tests
    ├── test_scatter.py
    ├── test_scatter_torch.py   # NEW: PyTorch version of tests
    ├── test_models.py
    ├── test_models_torch.py    # NEW: PyTorch version of tests
    ├── test_base.py
    ├── test_base_torch.py      # NEW: PyTorch version of tests
    ├── test_map_utils.py
    ├── test_map_utils_torch.py # NEW: PyTorch version of tests
    ├── test_adapters.py        # NEW: Tests for adapters
    ├── test_torch_utils.py     # NEW: Tests for PyTorch utilities
    ├── test_integration.py     # NEW: End-to-end integration tests
    └── test_gradients.py       # NEW: Tests for gradient calculation
```

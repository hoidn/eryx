# Specification Template
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Implement differentiable diffuse scattering computation in PyTorch with gradient flow support for learnable physics parameters

## Mid-Level Objectives

- Create PyTorch-based phonon mode calculations that match numpy implementation
- Ensure gradient flow through all physics computations 
- Integrate with existing data loading and preprocessing
- Support GPU acceleration where beneficial
- Maintain numerical equivalence with numpy version

## Implementation Notes

### Dependencies and Requirements
- PyTorch (primary computation framework)
- Existing numpy implementation for reference correctness
- Existing data loading and preprocessing code
- Support for both CPU and CUDA devices

### Architecture
- GaussianNetworkModelTorch should inherit from nn.Module
- OnePhononTorch should also inherit from nn.Module
- Clear separation between gradient-requiring and gradient-free operations
- Maintain device placement consistency

### Performance Considerations
- Vectorize operations where possible
- Batch structure factor computations efficiently
- Balance memory usage with computation speed
- Support checkpointing for large systems

### Code Organization
```
eryx/
├─ onephonon_torch.py
│  ├─ class OnePhononTorch(nn.Module, ModelRunner)
│  └─ associated utility functions
├─ gaussian_network_torch.py
│  ├─ class GaussianNetworkModelTorch(nn.Module)
│  └─ associated utility functions
└─ tests/
   ├─ test_onephonon_torch.py
   └─ test_gaussian_network_model_torch.py
```

## Context

[Previous sections remain the same until Context]

## Context

### Beginning context
```
./eryx/
├─ models.py
│  └─ class OnePhonon  [reference numpy implementation]
├─ onephonon_torch.py
│  └─ class OnePhononTorch  [current partial torch implementation]
├─ gaussian_network_torch.py
│  └─ class GaussianNetworkModelTorch [current implementation]
├─ pdb.py
│  ├─ class AtomicModel
│  ├─ class Crystal  
│  └─ class GaussianNetworkModel [numpy reference]
├─ scatter.py
│  └─ structure factors computation utilities
├─ map_utils.py
│  └─ grid and mapping utilities
└─ logging_utils.py
    └─ logging decorators and utilities

./tests/
├─ test_onephonon.py  [numpy reference tests]
├─ test_onephonon_torch.py  [current torch tests]
├─ test_gaussian_network_model_torch.py
└─ test_utils/
    ├─ log_analysis.py
    ├─ log_capture.py
    └─ model_runner.py
```

### Ending context
```
./eryx/
├─ models.py  [unchanged]
├─ onephonon_torch.py  [updated]
│  ├─ class OnePhononTorch(nn.Module, ModelRunner)
│  │  ├─ New vectorized implementations
│  │  └─ Gradient flow support
│  └─ Supporting utilities
├─ gaussian_network_torch.py  [updated]
│  ├─ class GaussianNetworkModelTorch(nn.Module) 
│  │  ├─ Learnable parameters
│  │  ├─ Vectorized operations
│  │  └─ Gradient support
│  └─ Supporting utilities
├─ pdb.py  [unchanged]
├─ scatter.py  [unchanged]
├─ map_utils.py  [unchanged]
└─ logging_utils.py  [unchanged]

./tests/
├─ test_onephonon.py  [unchanged]
├─ test_onephonon_torch.py  [updated]
│  ├─ Gradient flow tests
│  ├─ Numerical equivalence tests
│  ├─ Device placement tests
│  └─ Performance benchmarks
├─ test_gaussian_network_model_torch.py  [updated]
│  ├─ Parameter gradient tests
│  ├─ Phonon mode tests
│  └─ Device tests
└─ test_utils/  [unchanged]

Required Environment:
├─ Python 3.8+
├─ PyTorch 2.0+
├─ NumPy 1.20+
└─ Dependencies from existing pyproject.toml
```

## Low-Level Tasks

1. Implement GaussianNetworkModelTorch base
```aider
CREATE modules/gaussian_network_torch.py:
    class GaussianNetworkModelTorch(nn.Module):
        Initialize with:
        - Learnable parameters (gamma_intra, gamma_inter)
        - Device specification
        - Network configuration

        Methods:
        - build_gamma(): Creates learnable gamma tensor
        - build_neighbor_list(): Optimized neighbor computation
        - compute_hessian_torch(): Vectorized Hessian computation
        - compute_K_torch(): Vectorized K-matrix computation
        - compute_gnm_phonons_torch(): Eigendecomposition with gradient support
```

2. Implement OnePhononTorch base
```aider
CREATE modules/onephonon_torch.py:
    class OnePhononTorch(nn.Module, ModelRunner):
        Initialize with:
        - Device specification
        - Grid parameters
        - GNM configuration

        Methods:
        - forward(): Main computation path
        - compute_covariance_matrix_torch(): Vectorized covariance computation
        - apply_disorder(): Vectorized diffuse computation
```

3. Implement vectorized operations
```aider
UPDATE modules/onephonon_torch.py:
    ADD vectorized implementations for:
    - All k-space operations
    - Structure factor computations
    - Diffuse scattering computation
    
    Ensure:
    - No gradient-breaking operations
    - Efficient memory usage
    - Device consistency
```

4. Integrate preprocessing
```aider
UPDATE modules/onephonon_torch.py:
    ADD efficient preprocessing:
    - Grid generation
    - Coordinate system setup
    - Resolution mask computation
    - Structure factor preparation
    
    Maintain:
    - Clear separation from gradient computations
    - Memory efficiency
    - Numpy compatibility where needed
```

5. Implement testing and validation
```aider
CREATE tests/test_onephonon_torch.py:
    ADD test suite covering:
    - Gradient flow verification
    - Numerical equivalence with numpy
    - Device placement correctness
    - Memory usage monitoring
    - Performance benchmarking
```

### Class Structure Details

#### GaussianNetworkModelTorch
```python
class GaussianNetworkModelTorch(nn.Module):
    def __init__(self, pdb_path, enm_cutoff, gamma_intra, gamma_inter, device):
        self.gamma_intra = nn.Parameter(torch.tensor(gamma_intra))
        self.gamma_inter = nn.Parameter(torch.tensor(gamma_inter))
        # ... other initialization

    def forward(self, x):
        # Main computation path for backprop
        pass

    # Physics computation methods with gradient support
    def compute_hessian_torch(self):
        # Vectorized implementation
        pass

    def compute_K_torch(self, hessian, kvec):
        # Vectorized implementation
        pass

    def compute_gnm_phonons_torch(self):
        # Eigendecomposition with gradient support
        pass
```

#### OnePhononTorch
```python
class OnePhononTorch(nn.Module, ModelRunner):
    def __init__(self, pdb_path, hsampling, ksampling, lsampling, **kwargs):
        # Initialize components and parameters
        pass

    def forward(self):
        # Main computation sequence:
        # 1. Get phonon modes
        # 2. Compute covariance
        # 3. Compute diffuse scattering
        pass

    def compute_covariance_matrix_torch(self):
        # Vectorized implementation
        pass

    def apply_disorder(self, use_data_adp=False):
        # Vectorized implementation
        pass
```

### Integration Points

1. Loading and Grid Setup:
```python
# Preprocessing (no gradients)
- Load atomic model (numpy)
- Generate grid (numpy)
- Convert to torch tensors
- Move to device
```

2. Physics Computation:
```python
# With gradients
- Compute phonon modes
- Compute covariance matrix
- Compute diffuse scattering
```

3. Output Processing:
```python
# Mix of gradient and non-gradient ops
- Apply masks
- Reshape outputs
- Convert to numpy if needed
```

### Gradient Flow Path

```
Input Parameters
├─ gamma_intra ──────┐
├─ gamma_inter ──────┼─→ Hessian
├─ enm_cutoff ───────┘       │
                             ↓
                        K-matrix
                             │
                             ↓
                      Phonon Modes ←── Eigendecomposition
                             │
                             ↓
                    Diffuse Intensity
                             │
                             ↓
                          Loss
```

This specification provides a complete framework for implementing the torch-based diffuse scattering computation with proper gradient support. The implementation should maintain numerical equivalence with the numpy version while enabling parameter optimization through backpropagation.

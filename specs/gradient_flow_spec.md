# Specification Template
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Convert GaussianNetworkModelTorch into a proper torch.nn.Module with learnable physics parameters and complete gradient flow support

## Mid-Level Objectives

- Convert core physics parameters to torch.nn.Parameter
- Ensure gradient flow through matrix operations and eigendecomposition
- Maintain numerical equivalence with numpy GaussianNetworkModel
- Support backpropagation through entire physics computation chain

## Implementation Notes

### Dependencies and Requirements
- PyTorch (primary computation framework)
- Existing numpy GaussianNetworkModel for reference
- Support for both CPU and CUDA devices

### Parameter Conversion Strategy
- Convert physics parameters to nn.Parameter
- Maintain parameter constraints (e.g., positivity)
- Support parameter optimization
- Track gradient flow

### Computational Requirements
- Support batched operations
- Maintain reverse-mode autodiff support
- Handle complex tensor operations
- Support checkpointing for large systems

## Context

### Beginning context
```
./eryx/
├─ gaussian_network_torch.py
│  └─ class GaussianNetworkModelTorch
│      ├─ build_gamma()
│      ├─ build_neighbor_list()
│      ├─ compute_hessian()
│      ├─ compute_K()
│      └─ compute_Kinv()
├─ pdb.py
│  └─ class GaussianNetworkModel [numpy reference]
└─ tests/
    └─ test_gaussian_network_model_torch.py
```

### Ending context
```
./eryx/
├─ gaussian_network_torch.py
│  └─ class GaussianNetworkModelTorch(nn.Module)
│      ├─ Learnable Parameters:
│      │  ├─ gamma_intra
│      │  ├─ gamma_inter
│      │  └─ enm_cutoff (optional)
│      ├─ forward()
│      ├─ build_gamma() [updated]
│      ├─ build_neighbor_list()
│      ├─ compute_hessian() [updated]
│      ├─ compute_K() [updated]
│      ├─ compute_Kinv() [updated]
│      └─ compute_gnm_phonons_torch() [updated]
└─ tests/
    └─ test_gaussian_network_model_torch.py [updated]
        ├─ Parameter gradient tests
        ├─ Eigendecomposition gradient tests
        └─ End-to-end gradient tests
```

## Low-Level Tasks

1. Convert class to nn.Module and add parameters
```aider 
UPDATE gaussian_network_torch.py:
    CONVERT GaussianNetworkModelTorch to inherit from nn.Module
    ADD __init__ with:
        Initialize parent nn.Module
        Convert gamma_intra/inter to nn.Parameter
        Optional: Convert enm_cutoff to nn.Parameter
        Device handling setup
```

2. Update gamma computation 
```aider
UPDATE gaussian_network_torch.py:
    UPDATE build_gamma():
        Reimplement using pure torch operations
        Ensure gradient flow through parameters
        Maintain broadcasting for efficiency
```

3. Vectorize matrix computations
```aider
UPDATE gaussian_network_torch.py:
    UPDATE compute_hessian():
        Vectorize all operations
        Remove gradient-breaking operations
        Support batch dimension
    
    UPDATE compute_K():
        Vectorize k-point operations
        Maintain complex gradients
        Support batch computations
```

4. Update eigendecomposition 
```aider
UPDATE gaussian_network_torch.py:
    UPDATE compute_gnm_phonons_torch():
        Use torch.linalg.svd with gradient support
        Handle complex eigenvectors
        Support batch processing
        Add checkpointing for large systems
```

5. Add forward method
```aider
UPDATE gaussian_network_torch.py:
    ADD forward():
        Main computation path for backprop
        Handle parameter constraints
        Return values needed for loss computation
```

### Gradient Flow Details

Key areas requiring gradient support:
```
Parameters
├─ gamma_intra ────┐
├─ gamma_inter ────┼─→ Gamma Matrix ──→ Hessian
└─ enm_cutoff ─────┘                      │
                                          ↓
                                     K-matrix
                                          │
                                          ↓
                                   Eigendecomposition
                                          │
                                          ↓
                                    Phonon Modes
```

### Parameter Registration
```python
class GaussianNetworkModelTorch(nn.Module):
    def __init__(self, pdb_path, enm_cutoff, gamma_intra, gamma_inter, device):
        super().__init__()
        # Register learnable parameters
        self.gamma_intra = nn.Parameter(torch.tensor(gamma_intra))
        self.gamma_inter = nn.Parameter(torch.tensor(gamma_inter))
        self.enm_cutoff = nn.Parameter(torch.tensor(enm_cutoff)) # Optional
```

### Testing Requirements

1. Parameter gradients:
```python
def test_parameter_gradients():
    # Test gradient flow through:
    - gamma parameters
    - Hessian computation
    - K-matrix computation
    - Eigendecomposition
```

2. Backprop through complex operations:
```python
def test_complex_gradients():
    # Test gradient flow through:
    - Complex matrix operations
    - SVD decomposition
    - Phase factors
```

3. End-to-end gradients:
```python
def test_end_to_end_gradients():
    # Test gradient flow from:
    - Loss function
    - Through all computations
    - To parameters
```


Based on my review of the current codebase, I would rate the implementation at 6 out of 10. Here’s a more detailed breakdown:                                                                          

What’s working well:                                                                                                                                                                                   

 • Proper Inheritance and Parameter Registration:                                                                                                                                                      
   The class now correctly inherits from nn.Module and registers core physics parameters (gamma_intra, gamma_inter, and optionally enm_cutoff) as nn.Parameter. This meets one of the key high‑level   
   objectives.                                                                                                                                                                                         
 • Integration of Torch Operations:                                                                                                                                                                    
   Many parts of the computation (e.g., build_gamma(), parts of compute_hessian(), compute_K(), and the use of torch.linalg.svd in compute_gnm_phonons_torch()) have been rewritten to use torch       
   operations. This should in principle preserve gradient flow through the physics computations.                                                                                                       
 • Device Handling and Forward Method:                                                                                                                                                                 
   The code correctly handles device placement and provides a clear forward() method that implements the differentiable computation chain needed for backpropagation.                                  

Areas for improvement:                                                                                                                                                                                 

 • Incomplete Vectorization:                                                                                                                                                                           
   Although the specification calls for vectorization of matrix operations (for both efficiency and gradient flow), the “_torch” variants (e.g. compute_hessian_torch()) are not yet fully implemented.
   For example, compute_hessian_torch() currently returns a zero tensor, which does not match the expected behavior from compute_hessian(). This gap shows that the full conversion to a               
   batched/vectorized operation remains incomplete.                                                                                                                                                    
 • Numerical Equivalence and Test Placeholders:                                                                                                                                                        
   Tests such as test_hessian_torch() and test_k_matrix_torch() indicate that there are still inconsistencies or “TODO” notes regarding conversion between block‑structured and 2D representations.    
   Achieving numerical equivalence with the numpy version is a mid‑level objective that isn’t completely verified yet.                                                                                 
 • Checkpointing and Scalability:                                                                                                                                                                      
   Although the specification mentions support for checkpointing for large systems, this functionality is not evident in the current implementation.                                                   
 • Gradient Propagation Clarity:                                                                                                                                                                       
   While the use of torch.linalg.svd should, in theory, allow gradient flow through the eigendecomposition, the incomplete vectorized routines make it harder to guarantee that gradients will         
   propagate throughout every piece of the computation chain reliably.                                                                                                                                 

Conclusion:                                                                                                                                                                                            
The codebase is on the right path toward meeting the high‑ and mid‑level objectives. The core conversion work is done—including parameter registration and a forward pass that links key operations—but
several important components (notably the fully vectorized versions of the Hessian and K‑matrix computations and related numerical consistency) remain unfinished. Improving these aspects would be    
necessary to reach a fully robust, production‑ready implementation with complete gradient support.                                                                                                     

Thus, I rate the implementation state as 6/10.                                                                                                                                                         

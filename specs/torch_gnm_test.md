# Specification: Gaussian Network Model PyTorch Tests Implementation
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Create PyTorch version of test_gaussian_network_model.py that validates the PyTorch GNM implementation matches the numpy version exactly within specified tolerances

## Mid-Level Objectives

- Implement PyTorch versions of all GNM tests maintaining exact same tolerances
- Validate mathematical properties of matrices using torch operations
- Ensure efficient handling of large tensor operations
- Support both CPU and CUDA devices
- Match numpy test coverage exactly

## Implementation Notes

### Dependencies
```
pytest
torch
numpy
gemmi
scipy
```

### Test Data Dependencies
Required files:
```
tests/test_data/reference/gnm_neighbor_lists.npy
tests/test_data/reference/gnm_k_matrices.npy
tests/pdbs/5zck.pdb
```

### Required Test Fixtures
```python
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def gnm_model_torch(device):
    return GaussianNetworkModelTorch(
        pdb_path="tests/pdbs/5zck.pdb",
        enm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
```

## Context

### Beginning context
- tests/test_gaussian_network_model.py
- test utilities from existing numpy implementation

### Ending context  
- tests/test_gaussian_network_model_torch.py

## Low-Level Tasks
> Ordered from start to finish

1. Test Infrastructure Setup

```aider
CREATE tests/test_gaussian_network_model_torch.py:
    ADD imports and test class
    ADD fixtures for device and model

ENSURE imports include:
    - pytest
    - torch
    - numpy
    - GaussianNetworkModelTorch
```

2. Neighbor List Construction Test

```aider
ADD test_neighbor_list_construction_torch:
    IMPLEMENT with approach:
    - Load reference from file:
        ref_neighbors = np.load("tests/test_data/reference/gnm_neighbor_lists.npy", 
                              allow_pickle=True)
    - Compare lengths at each level:
        assert len(gnm_model_torch.asu_neighbors) == len(ref_neighbors)
        for idx, neighbors in enumerate(gnm_model_torch.asu_neighbors):
            ref = ref_neighbors[idx]
            neighbors_np = neighbors.cpu().numpy() if torch.is_tensor(neighbors) else neighbors
            assert len(neighbors_np) == len(ref)
    - If tensors used for neighbor lists, convert to CPU for comparison
    - Verify neighbor connectivity patterns match
```

3. Spring Constants Test

```aider
ADD test_spring_constants_torch:
    IMPLEMENT with approach:
    - Use input parameters as ground truth:
        gamma_intra = 1.0
        gamma_inter = 1.0
    - Loop through model dimensions:
        for i_asu in range(model.n_asu):
            for cell in range(model.n_cell):
                for j_asu in range(model.n_asu):
    - Check gamma values:
        - intra-ASU: gamma[cell,i_asu,j_asu] == gamma_intra 
        - inter-ASU: gamma[cell,i_asu,j_asu] == gamma_inter
    - Convert tensor values to float for comparison:
        gamma_val = gnm_model_torch.gamma[cell,i_asu,j_asu].item()
    - Use pytest.approx for comparisons
```

4. Hessian Symmetry Test

```aider
ADD test_hessian_symmetry_torch:
    IMPLEMENT with approach:
    - Compute hessian keeping on device
    - For each ASU:
        - Extract diagonal block:
            diag_block = hessian[i_asu,:,id_cell_ref,i_asu,:]
        - Verify symmetry using torch operations:
            assert torch.allclose(diag_block, 
                                diag_block.transpose(-2,-1).conj(),
                                rtol=1e-7)
    - No explicit reference values needed (mathematical property)
    - Handle complex tensors properly
```

5. K-Matrix Computation Test

```aider
ADD test_k_matrix_computation_torch:
    IMPLEMENT with approach:
    - Create test kvector on device:
        kvec = torch.tensor([1.0, 0.0, 0.0], device=device)
    - Compute hessian and K-matrix:
        hessian = gnm_model_torch.compute_hessian()
        Kmat = gnm_model_torch.compute_K(hessian, kvec=kvec)
    - Verify shape:
        expected_shape = (model.n_asu, model.n_atoms_per_asu,
                         model.n_asu, model.n_atoms_per_asu)
        assert Kmat.shape == expected_shape
    - Load reference:
        ref_K = np.load("tests/test_data/reference/gnm_k_matrices.npy")
    - Compare after CPU conversion:
        assert np.allclose(Kmat.cpu().numpy(), ref_K, rtol=1e-7)
```

6. Inversion Stability Test

```aider
ADD test_inversion_stability_torch:
    IMPLEMENT with approach:
    - Create test kvector on device
    - Compute K and Kinv:
        Kmat = model.compute_K(hessian, kvec=kvec)
        Kinv = model.compute_Kinv(hessian, kvec=kvec)
    - Contract using torch.einsum:
        identity_approx = torch.einsum('ijkl,klmn->ijmn', Kmat, Kinv)
    - Verify each block is identity:
        for i in range(model.n_asu):
            for j in range(model.n_atoms_per_asu):
                block = identity_approx[i,j,:,:]
                expected = torch.zeros((model.n_asu, 
                                      model.n_atoms_per_asu),
                                     device=device)
                expected[i,j] = 1.0
                assert torch.allclose(block, expected, rtol=1e-7)
    - No explicit reference values (mathematical property)
```

7. Device-Specific Tests

```aider
ADD test_device_placement:
    IMPLEMENT checks for:
    - Model tensors on correct device
    - Operations maintain device placement
    - Memory efficient matrix operations
    - Proper cleanup of large tensors

ADD test_cuda_vs_cpu:
    IMPLEMENT checks that:
    - Results match between devices
    - Large matrix operations handled properly
    - Error handling consistent across devices
```

8. Documentation

```aider
UPDATE all test functions:
    ADD docstrings explaining:
        - Test purpose
        - Ground truth source (file/parameter/mathematical)
        - Device considerations
        - Expected precision
    ADD comments for:
        - Complex tensor operations
        - Device movements
        - Memory management
```

Each test implementation must:
1. Match numpy version tolerances exactly
2. Use identical ground truth values/files
3. Handle device placement properly
4. Include proper error handling
5. Be memory efficient
6. Document ground truth sources

Regular validation should occur after each test implementation to ensure compatibility with numpy version.

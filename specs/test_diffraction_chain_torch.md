# Specification: Diffraction Chain PyTorch Tests Implementation
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Create PyTorch version of test_diffraction_chain.py that validates the PyTorch diffraction calculations match the numpy version exactly within specified tolerances

## Mid-Level Objectives

- Implement form factor and structure factor tests with hardcoded reference values
- Validate diffraction pattern symmetries using torch operations
- Support both CPU and CUDA devices
- Ensure efficient handling of large tensor operations
- Match numpy test coverage exactly

## Implementation Notes

### Dependencies
```
pytest
torch
numpy
gemmi
```

### Test Data Dependencies
```
tests/pdbs/5zck_p1.pdb
```

### Required Test Fixtures
```python
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def onephonon_torch(device):
    return OnePhononTorch(
        pdb_path="tests/pdbs/5zck_p1.pdb",
        hsampling=[-4,4,3],
        ksampling=[-17,17,3],
        lsampling=[-29,29,3],
        expand_p1=True,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
```

## Context

### Beginning context
- tests/test_diffraction_chain.py
- test utilities from existing numpy implementation

### Ending context  
- tests/test_diffraction_chain_torch.py

## Low-Level Tasks
> Ordered from start to finish

1. Test Infrastructure Setup

```aider
CREATE tests/test_diffraction_chain_torch.py:
    ADD imports and test class
    ADD fixtures for device and model

ENSURE imports include:
    - pytest
    - torch
    - numpy
    - onephonon_torch implementation
```

2. Form Factors Test

```aider
ADD test_form_factors_torch:
    IMPLEMENT with approach:
    - Take first 10 q points from model q_grid
        q_test = onephonon_torch.q_grid[:10]
    - Compute form factors keeping tensors on device:
        ff = compute_form_factors(
            q_test,
            onephonon_torch.model.ff_a[0],
            onephonon_torch.model.ff_b[0],
            onephonon_torch.model.ff_c[0]
        )
    - Verify shape:
        assert ff.shape == (10, onephonon_torch.model.ff_a[0].shape[0])
    - Check for NaNs using torch.isnan
    - Compare mean against hardcoded value:
        expected_ff_mean = 3.975718  # from numpy test
        torch.testing.assert_close(
            torch.mean(ff).cpu(),
            torch.tensor(expected_ff_mean),
            rtol=1e-5
        )
```

3. Structure Factors Test

```aider
ADD test_structure_factors_torch:
    IMPLEMENT with approach:
    - Take first 10 q points:
        q_test = onephonon_torch.q_grid[:10]
    - Compute structure factors on device:
        F = structure_factors(
            q_test,
            onephonon_torch.model.xyz[0],
            onephonon_torch.model.ff_a[0],
            onephonon_torch.model.ff_b[0],
            onephonon_torch.model.ff_c[0],
            compute_qF=True,
            project_on_components=onephonon_torch.Amat[0]
        )
    - Verify shape matches q_test:
        assert F.shape[0] == q_test.shape[0]
    - Check for NaNs
    - Compare first element against hardcoded value:
        expected_F_first = -941.71642  # from numpy test
        torch.testing.assert_close(
            torch.real(F[0]).cpu(),
            torch.tensor(expected_F_first),
            rtol=1e-5
        )
```

4. Symmetries and Hessian Test

```aider
ADD test_symmetries_and_hessian_torch:
    IMPLEMENT with approach:
    - Compute diffraction pattern:
        Id = onephonon_torch.apply_disorder(use_data_adp=True)
        Id = Id.reshape(onephonon_torch.map_shape)
    - Get central slice:
        central_h = Id.shape[0] // 2
        central_slice = Id[central_h, :, :]
    - Verify symmetry using torch.flip:
        torch.testing.assert_close(
            central_slice,
            torch.flip(central_slice, dims=[0, 1]),
            rtol=1e-5
        )
    - Compute and verify hessian symmetry:
        hessian = onephonon_torch.gnm.compute_hessian()
        for i in range(onephonon_torch.n_asu):
            hi = hessian[i, :, onephonon_torch.crystal.hkl_to_id([0,0,0]), i, :]
            torch.testing.assert_close(hi, hi.transpose(-2, -1), atol=1e-5)
```

5. Diffuse Intensity Test

```aider
ADD test_diffuse_intensity_torch:
    IMPLEMENT with approach:
    - Compute diffraction pattern:
        Id = onephonon_torch.apply_disorder(use_data_adp=True)
        Id = Id.reshape(onephonon_torch.map_shape)
    - Verify shape:
        assert Id.shape == onephonon_torch.map_shape
    - Clean NaN values:
        Id_clean = torch.nan_to_num(Id, nan=0.0)
    - Get central index:
        central_idx = (Id_clean.shape[0] // 2,
                      Id_clean.shape[1] // 2,
                      Id_clean.shape[2] // 2)
    - Compare center intensity against hardcoded value:
        expected_center_intensity = 0.0  # from numpy test
        torch.testing.assert_close(
            Id_clean[central_idx],
            torch.tensor(expected_center_intensity, device=device),
            rtol=1e-5
        )
    - Verify non-zero elements exist:
        assert torch.count_nonzero(Id_clean) > 0
```

6. Device-Specific Tests

```aider
ADD test_device_placement:
    IMPLEMENT checks for:
    - Form factors computed on correct device
    - Structure factors maintained on device
    - Diffraction patterns on device
    - Memory efficient for large tensors

ADD test_cuda_vs_cpu:
    IMPLEMENT checks that:
    - Results match between devices
    - Complex tensor operations work on both devices
    - Error handling consistent
```

7. Performance Tests

```aider
ADD test_performance:
    IMPLEMENT checks for:
    - Memory usage during large tensor operations
    - Performance with different batch sizes
    - Device memory cleanup
```

8. Documentation

```aider
UPDATE all test functions:
    ADD docstrings explaining:
        - Test purpose
        - Ground truth value source
        - Device handling
        - Expected precision
    ADD comments for:
        - Complex tensor operations
        - Device movement points
        - Memory handling strategies
```

Each test implementation must:
1. Match numpy version tolerances exactly:
   - Form factors: rtol=1e-5
   - Structure factors: rtol=1e-5
   - Symmetries: rtol=1e-5, atol=1e-5
2. Use identical hardcoded ground truth values:
   - ff_mean = 3.975718
   - F_first = -941.71642
   - center_intensity = 0.0
3. Handle device placement properly
4. Include proper error handling
5. Be memory efficient
6. Document ground truth sources

Regular validation should occur after each test implementation to ensure compatibility with numpy version.

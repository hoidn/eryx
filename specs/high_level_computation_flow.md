# GNM Integration Specification - OnePhononTorch
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Complete the physical model implementation in OnePhononTorch by properly integrating the existing GaussianNetworkModelTorch components to match the numpy calculations.

## Mid-Level Objectives

- Connect phonon mode computation into main execution flow
- Integrate covariance matrix calculations
- Implement validation routines to verify torch/numpy equivalence
- Add proper type hints and docstrings for all new/modified functions

## Implementation Notes

### Dependencies
- Existing modules: eryx.onephonon_torch, eryx.models
- torch, numpy for computation
- Type hints from typing module

### Coding Standards
- Follow existing codebase style with torch tensors
- Use proper device management throughout
- Add comprehensive docstrings following Google style
- Add detailed type hints
- Add logging statements for key computations
- Use proper tensor dtype management (float64/complex128 where needed)

### Technical Requirements
- Maintain device consistency (CPU/CUDA) throughout computation
- Ensure numerical stability in eigendecomposition
- Properly handle complex numbers in covariance computation
- Add validation asserts where appropriate

## Context

### Beginning Context
- `eryx/onephonon_torch.py`
- `eryx/gaussian_network_torch.py`
- `eryx/models.py` (original numpy version for reference)

### Ending Context  
- `eryx/onephonon_torch.py` (updated)
- `eryx/gaussian_network_torch.py` (updated)

## Low-Level Tasks
> Ordered from start to finish

1. Add Initialization of Phonon Modes

```aider
UPDATE eryx/onephonon_torch.py:
    UPDATE OnePhononTorch.__init__:
        ADD type hints
        ADD initialization of phonon modes after GNM creation
        ADD validation logging
```

2. Create Validation Method

```aider
UPDATE eryx/onephonon_torch.py:
    CREATE validate_physics_computation method:
        Input: None
        Output: None
        Purpose: Compare torch vs numpy calculations
        ADD comparison of:
            - Phonon modes
            - Covariance matrices
            - Structure factors
```

3. Integrate Covariance Matrix Into Apply Disorder

```aider
UPDATE eryx/onephonon_torch.py:
    UPDATE apply_disorder:
        ADD covariance matrix computation
        ADD proper integration with structure factors
        UPDATE docstring to reflect changes
        ADD validation/logging of computed values
```

4. Add Unit Tests

```aider
CREATE tests/test_onephonon_torch_physics.py:
    ADD tests comparing torch/numpy results:
        - Test phonon modes match
        - Test covariance matrices match
        - Test structure factors match
        - Test final diffuse intensities match
```

5. Add Forward Pass Integration

```aider
UPDATE eryx/onephonon_torch.py:
    UPDATE/CREATE forward method:
        Input: None
        Output: computed diffuse intensity
        Purpose: Enable full forward pass through model
        Ensure proper gradient flow
```

6. Create Helper Functions for Numerical Validation

```aider
UPDATE eryx/onephonon_torch.py:
    CREATE _compare_to_numpy helper:
        Input:
            torch_val: torch.Tensor
            numpy_val: np.ndarray
            name: str
            rtol: float = 1e-5
        Output: bool
        Purpose: Compare torch/numpy values with proper error handling
```

Note: The actual implementation should:
- Add proper error handling
- Include comprehensive logging
- Add appropriate tests
- Validate numerical precision
- Ensure backward compatibility
- Add performance metrics


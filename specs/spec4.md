# PyTorch Port Map Utils Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Implement reciprocal space grid utilities in PyTorch with gradient flow support for the diffuse scattering calculations

## Mid-Level Objectives

- Implement `generate_grid` function to create q-vector grids with gradient support
- Implement `compute_resolution` function to calculate resolution values for Miller indices
- Implement `get_resolution_mask` function to create boolean masks for resolution limits
- Create ground-truth-based tests using existing log files and `TorchTesting` framework
- Update implementation status in `progress.md` to reflect completion of CP3

## Implementation Notes

- Use PyTorch tensor operations for all calculations to enable gradient flow
- Replicate NumPy functions' API exactly, with optional device parameter added
- Ensure proper numerical stability for edge cases (near-zero divisions, etc.)
- Add comprehensive docstrings with type annotations
- Use adapter components from Phase 2 for NumPy-PyTorch conversions in tests
- Use torch.meshgrid with indexing='ij' to match NumPy's behavior
- Test using ground truth data in logs/eryx.map_utils.*.log with specified tolerances:
  - generate_grid: rtol=1e-5, atol=1e-8
  - compute_resolution: rtol=1e-5, atol=1e-8
  - get_resolution_mask: exact match

## Context

### Beginning Context
- `eryx/map_utils.py` (original NumPy implementation)
- `eryx/map_utils_torch.py` (with partial stubs)
- `eryx/adapters.py` (conversion utilities)
- `eryx/autotest/torch_testing.py` (testing framework)
- `logs/eryx.map_utils.*.log` (ground truth data)

### Ending Context
- `eryx/map_utils_torch.py` (with completed implementations)
- `tests/test_map_utils_torch.py` (with ground truth tests)
- `progress.md` (updated to reflect CP3 completion)

## Low-Level Tasks
> Ordered from start to finish

1. Complete `generate_grid` Implementation

```aider
UPDATE eryx/map_utils_torch.py:
    UPDATE generate_grid(A_inv: torch.Tensor, hsampling: Tuple[float, float, float], 
                         ksampling: Tuple[float, float, float], lsampling: Tuple[float, float, float], 
                         return_hkl: bool = False) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        
        ENSURE function signature and docstring match the function in the file
        
        REPLACE IMPLEMENTATION with:
            # Calculate steps for each dimension
            hsteps = int(hsampling[2] * (hsampling[1] - hsampling[0]) + 1)
            ksteps = int(ksampling[2] * (ksampling[1] - ksampling[0]) + 1)
            lsteps = int(lsampling[2] * (lsampling[1] - lsampling[0]) + 1)
            
            # Create linspace for each dimension
            h_grid = torch.linspace(hsampling[0], hsampling[1], hsteps, device=A_inv.device)
            k_grid = torch.linspace(ksampling[0], ksampling[1], ksteps, device=A_inv.device)
            l_grid = torch.linspace(lsampling[0], lsampling[1], lsteps, device=A_inv.device)
            
            # Create meshgrid - using indexing='ij' to match NumPy's default behavior
            h_mesh, k_mesh, l_mesh = torch.meshgrid(h_grid, k_grid, l_grid, indexing='ij')
            
            # Get map shape
            map_shape = (h_mesh.size(0), k_mesh.size(1), l_mesh.size(2))
            
            # Reshape and reorder dimensions to match NumPy version
            hkl_grid = torch.stack([h_mesh.flatten(), k_mesh.flatten(), l_mesh.flatten()], dim=1)
            
            if return_hkl:
                return hkl_grid, map_shape
            else:
                # Calculate q_grid using matrix multiplication: q_grid = 2π * A_inv^T * hkl_grid^T
                q_grid = 2 * torch.pi * torch.matmul(A_inv.T, hkl_grid.T).T
                return q_grid, map_shape
```

2. Complete `compute_resolution` Implementation

```aider
UPDATE eryx/map_utils_torch.py:
    UPDATE compute_resolution(cell: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    
        ENSURE function signature and docstring match the function in the file
        
        REPLACE IMPLEMENTATION with:
            # Extract cell parameters
            a, b, c = cell[0], cell[1], cell[2]
            alpha, beta, gamma = torch.deg2rad(cell[3]), torch.deg2rad(cell[4]), torch.deg2rad(cell[5])
            
            # Extract Miller indices
            h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
            
            # Calculate terms for the formula
            pf = 1.0 - cos_sq(alpha) - cos_sq(beta) - cos_sq(gamma) + 2.0 * torch.cos(alpha) * torch.cos(beta) * torch.cos(gamma)
            
            n1 = torch.square(h) * sin_sq(alpha) / torch.square(a) + \
                 torch.square(k) * sin_sq(beta) / torch.square(b) + \
                 torch.square(l) * sin_sq(gamma) / torch.square(c)
            
            n2a = 2.0 * k * l * (torch.cos(beta) * torch.cos(gamma) - torch.cos(alpha)) / (b * c)
            n2b = 2.0 * l * h * (torch.cos(gamma) * torch.cos(alpha) - torch.cos(beta)) / (c * a)
            n2c = 2.0 * h * k * (torch.cos(alpha) * torch.cos(beta) - torch.cos(gamma)) / (a * b)
            
            # Calculate resolution with safe division
            denominator = (n1 + n2a + n2b + n2c) / pf
            
            # Handle potential divide by zero
            safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator) * 1e-10)
            resolution = 1.0 / torch.sqrt(safe_denominator)
            
            # Set resolution to infinity where denominator is zero or negative
            resolution = torch.where(denominator > 0, resolution, float('inf') * torch.ones_like(resolution))
            
            return resolution
```

3. Complete `get_resolution_mask` Implementation

```aider
UPDATE eryx/map_utils_torch.py:
    UPDATE get_resolution_mask(cell: torch.Tensor, hkl_grid: torch.Tensor, 
                              res_limit: float) -> Tuple[torch.Tensor, torch.Tensor]:
    
        ENSURE function signature and docstring match the function in the file
        
        REPLACE IMPLEMENTATION with:
            # Compute resolution map for each grid point
            res_map = compute_resolution(cell, hkl_grid)
            
            # Create boolean mask by comparing to resolution limit
            # Points with resolution > res_limit are kept (True)
            res_mask = res_map > res_limit
            
            return res_mask, res_map
```

4. Create Ground Truth Tests for Map Utils

```aider
CREATE tests/test_map_utils_torch.py:
    IMPLEMENT the following tests:
    
    import unittest
    import os
    import torch
    import numpy as np
    from eryx.map_utils_torch import generate_grid, compute_resolution, get_resolution_mask
    from eryx.autotest.torch_testing import TorchTesting
    from eryx.autotest.logger import Logger
    from eryx.autotest.functionmapping import FunctionMapping
    
    class TestMapUtilsTorch(unittest.TestCase):
        def setUp(self):
            # Set up the testing framework
            self.logger = Logger()
            self.function_mapping = FunctionMapping()
            self.torch_testing = TorchTesting(self.logger, self.function_mapping, rtol=1e-5, atol=1e-8)
            
            # Set device to CPU for consistent testing
            self.device = torch.device('cpu')
            
            # Log file prefixes for ground truth data
            self.generate_grid_log = "logs/eryx.map_utils.generate_grid"
            self.compute_resolution_log = "logs/eryx.map_utils.compute_resolution"
            self.get_resolution_mask_log = "logs/eryx.map_utils.get_resolution_mask"
            
            # Ensure log files exist
            for log_file in [self.generate_grid_log, self.compute_resolution_log, self.get_resolution_mask_log]:
                log_file_path = f"{log_file}.log"
                self.assertTrue(os.path.exists(log_file_path), f"Log file {log_file_path} not found")
        
        def test_generate_grid(self):
            """Test generate_grid against ground truth data."""
            # Test against ground truth using TorchTesting
            self.assertTrue(
                self.torch_testing.testTorchCallable(self.generate_grid_log, generate_grid),
                "generate_grid failed ground truth test"
            )
            
            # Additional test for gradient flow
            A_inv = torch.eye(3, requires_grad=True)
            hsampling = (-3, 3, 1)
            ksampling = (-3, 3, 1)
            lsampling = (-3, 3, 1)
            
            q_grid, _ = generate_grid(A_inv, hsampling, ksampling, lsampling)
            loss = q_grid.sum()
            loss.backward()
            
            self.assertIsNotNone(A_inv.grad)
            self.assertFalse(torch.allclose(A_inv.grad, torch.zeros_like(A_inv.grad)))
        
        def test_compute_resolution(self):
            """Test compute_resolution against ground truth data."""
            # Test against ground truth using TorchTesting
            self.assertTrue(
                self.torch_testing.testTorchCallable(self.compute_resolution_log, compute_resolution),
                "compute_resolution failed ground truth test"
            )
            
            # Additional test for gradient flow
            cell = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], requires_grad=True)
            hkl = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            
            resolution = compute_resolution(cell, hkl)
            loss = resolution.sum()
            loss.backward()
            
            self.assertIsNotNone(cell.grad)
            self.assertFalse(torch.allclose(cell.grad, torch.zeros_like(cell.grad)))
        
        def test_get_resolution_mask(self):
            """Test get_resolution_mask against ground truth data."""
            # Test against ground truth using TorchTesting
            # Note: get_resolution_mask should have exact match for masks
            self.assertTrue(
                self.torch_testing.testTorchCallable(self.get_resolution_mask_log, get_resolution_mask),
                "get_resolution_mask failed ground truth test"
            )
            
            # Additional test for gradient flow through the resolution map
            cell = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], requires_grad=True)
            hkl_grid = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            
            _, res_map = get_resolution_mask(cell, hkl_grid, 1.0)
            loss = res_map.sum()
            loss.backward()
            
            self.assertIsNotNone(cell.grad)
            self.assertFalse(torch.allclose(cell.grad, torch.zeros_like(cell.grad)))
    
    if __name__ == '__main__':
        unittest.main()
```

5. Update Progress Tracking in progress.md

```aider
UPDATE progress.md:
    FIND the section titled "### Phase 3: Core Functions" with map_utils functions
    
    REPLACE rows for map_utils functions with:
    | generate_grid | map_utils.py | Complete | Complete | Ground truth tests pass |
    | compute_resolution | map_utils.py | Complete | Complete | Ground truth tests pass |
    | get_resolution_mask | map_utils.py | Complete | Complete | Ground truth tests pass |
    
    FIND the section titled "## Implementation Checkpoint Status"
    
    REPLACE the row for CP3 with:
    | CP3 | Map Utils Complete | Complete | March 04, 2025 |
    
    FIND the section titled "## Overall Progress Summary"
    
    REPLACE the row for Phase 3 with:
    | Phase 3 | Core Functions | In Progress | 50% |
    
    ALSO in that section, update the overall completion percentage appropriately
    
    FIND the section titled "## Current Focus"
    
    UPDATE to indicate that CP3 is complete and now moving to CP4 (Scatter Complete)
```

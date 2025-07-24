# Phase 1 Fix Checklist: Differentiable Interpolation Implementation

**Initiative:** User-Specified Phonon Population Modeling  
**Created:** 2025-01-24  
**Reviewer Verdict:** REJECT - Non-differentiable interpolation implementation  

## 🎯 **Fix Objective**

Replace the current numpy-based `_pdos_interp` method with a fully differentiable PyTorch-native interpolation system to maintain end-to-end gradient flow through PDOS interpolation.

## ✅ **Fix Tasks**

### Core Implementation Tasks

| Task | State | Priority | Description |
|------|-------|----------|-------------|
| **1. Remove Non-Differentiable Method** | `[ ]` | **Critical** | Delete the current `_pdos_interp` method from `eryx/models_torch.py` |
| **2. Implement Differentiable Interpolation** | `[ ]` | **Critical** | Create new `_differentiable_interp` method using pure PyTorch operations |
| **3. Update Method Call** | `[ ]` | **Critical** | Change `self._pdos_interp(omega)` to `self._differentiable_interp(omega)` in `compute_gnm_phonons` |
| **4. Boundary Condition Handling** | `[ ]` | **High** | Implement proper edge case handling for out-of-range frequencies |
| **5. Gradient Verification Test** | `[ ]` | **Critical** | Create test script to verify `pdos_density.grad` is populated after backward pass |

### Implementation Details

#### Task 1: Remove Non-Differentiable Method
- **File:** `eryx/models_torch.py`
- **Action:** Delete the entire `_pdos_interp` method (lines ~327-356)
- **Rationale:** This method uses `numpy.interp` which breaks gradient flow

#### Task 2: Implement Differentiable Interpolation
- **File:** `eryx/models_torch.py`
- **Method Signature:** `def _differentiable_interp(self, query_omega: torch.Tensor) -> torch.Tensor:`
- **Required Implementation:**
```python
def _differentiable_interp(self, query_omega: torch.Tensor) -> torch.Tensor:
    """
    Differentiable linear interpolation of PDOS density at query frequencies.
    
    Uses torch.searchsorted and pure tensor arithmetic to maintain gradient flow.
    
    Args:
        query_omega: Tensor of frequencies (rad/s) to interpolate at
        
    Returns:
        Interpolated density values as tensor with gradients preserved
    """
    # Find indices for interpolation brackets
    indices = torch.searchsorted(self.pdos_omega, query_omega)
    
    # Handle boundary conditions with torch operations
    indices = torch.clamp(indices, 1, len(self.pdos_omega) - 1)
    
    # Compute interpolation weights
    x0 = self.pdos_omega[indices - 1]
    x1 = self.pdos_omega[indices]
    y0 = self.pdos_density[indices - 1]
    y1 = self.pdos_density[indices]
    
    # Linear interpolation using tensor arithmetic
    weights = (query_omega - x0) / (x1 - x0)
    interpolated = y0 + weights * (y1 - y0)
    
    return interpolated
```

#### Task 3: Update Method Call
- **File:** `eryx/models_torch.py`
- **Location:** `compute_gnm_phonons` method (~line 253)
- **Change:** `population_factor = self._pdos_interp(omega)` → `population_factor = self._differentiable_interp(omega)`

#### Task 4: Boundary Condition Handling
- **Requirements:**
  - Handle frequencies below `pdos_omega.min()` (extrapolate to first value)
  - Handle frequencies above `pdos_omega.max()` (extrapolate to last value)
  - Ensure no `.detach()` or `.numpy()` operations
  - Use only differentiable PyTorch operations like `torch.clamp` and `torch.where`

#### Task 5: Gradient Verification Test
- **File:** Create `test_gradient_verification.py`
- **Requirements:**
```python
#!/usr/bin/env python3
"""
Test to verify that gradients flow through the differentiable PDOS interpolation.
"""

import torch
import numpy as np
from eryx.models_torch import OnePhonon

def test_pdos_gradient_flow():
    """Verify that pdos_density.grad is populated after backward pass."""
    
    # Create sample PDOS file
    frequencies_thz = np.linspace(0.1, 10.0, 20)
    densities = np.exp(-frequencies_thz / 5.0)
    pdos_data = np.column_stack([frequencies_thz, densities])
    np.savetxt('sample_pdos.dat', pdos_data)
    
    # Initialize model with PDOS
    model = OnePhonon(
        pdb_path='test.pdb',  # Dummy path
        pdos_path='sample_pdos.dat',
        pdos_mode='direct',
        hsampling=5, ksampling=5, lsampling=5
    )
    
    # Enable gradient tracking
    model.pdos_density.requires_grad_(True)
    
    # Run forward pass
    model._setup_phonons()
    
    # Create dummy omega for interpolation test
    omega_test = torch.linspace(1e12, 5e13, 10, dtype=model.real_dtype, device=model.device)
    result = model._differentiable_interp(omega_test)
    
    # Compute loss and backward pass
    loss = result.sum()
    loss.backward()
    
    # Verify gradient exists
    assert model.pdos_density.grad is not None, "pdos_density.grad should not be None"
    assert not torch.allclose(model.pdos_density.grad, torch.zeros_like(model.pdos_density.grad)), "Gradients should be non-zero"
    
    print("✅ Gradient flow verification passed")
    print(f"   pdos_density.grad shape: {model.pdos_density.grad.shape}")
    print(f"   pdos_density.grad norm: {model.pdos_density.grad.norm().item():.6f}")

if __name__ == "__main__":
    test_pdos_gradient_flow()
```

## 🔴 **Critical Requirements**

1. **No `.detach()` calls** - All tensor operations must preserve gradients
2. **No `.numpy()` conversions** - Must use pure PyTorch operations
3. **Use `torch.searchsorted`** - For efficient bracket finding
4. **Tensor arithmetic only** - Linear interpolation using tensor operations
5. **Gradient verification** - Must prove `pdos_density.grad` is populated

## 🧪 **Verification Steps**

### Step 1: Unit Test the New Method
```bash
python -c "
import torch
from eryx.models_torch import OnePhonon
# Test that _differentiable_interp method exists and works
"
```

### Step 2: Gradient Flow Test
```bash
python test_gradient_verification.py
```

### Step 3: Integration Test
```bash
python test_pdos_simple.py  # Should still work with new method
```

## 📋 **Success Criteria**

- [ ] `_pdos_interp` method completely removed
- [ ] `_differentiable_interp` method implemented with PyTorch operations only
- [ ] Method call updated in `compute_gnm_phonons`
- [ ] Gradient verification test passes
- [ ] No regressions in existing functionality
- [ ] All operations maintain gradient flow

## 📁 **Files to Modify**

1. **`eryx/models_torch.py`** - Replace interpolation method and update call site
2. **`test_gradient_verification.py`** - New file for gradient testing
3. **Update existing test files** - Ensure they work with new method name

---

**Next Step:** Complete all tasks above, then run `/complete-phase` to generate a new review request.
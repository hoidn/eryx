# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL API INFORMATION - READ FIRST

### Grid Sampling Parameters (COMMONLY MISUNDERSTOOD)

The parameters `hsampling`, `ksampling`, `lsampling` are **tuples of (min, max, oversampling_factor)**:
- ❌ **NOT** `(min, max, num_points)`
- ✅ The third value is a **multiplicative oversampling factor**
- 📐 Formula: `n_points = (max - min) * oversampling_factor + 1`

#### Examples with Calculations

```python
# ✅ CORRECT: Creates a 5×5×5 grid (125 total points)
hsampling = (-2, 2, 1)  # (2-(-2))*1+1 = 5 points, NOT 1 point!
ksampling = (-2, 2, 1)  # 5 points
lsampling = (-2, 2, 1)  # 5 points
model = OnePhonon(pdb_path, hsampling, ksampling, lsampling)

# ⚠️ WARNING: Creates 21×21×21 = 9,261 points!
hsampling = (-2, 2, 5)  # (2-(-2))*5+1 = 21 points, NOT 5 points!
# This is 125× more points than you might expect!
```

#### Performance Impact Table

| Oversampling | Points per dim | Total Points | Approx Memory | Approx Time |
|-------------|---------------|--------------|---------------|-------------|
| 1 | 5 | 125 | ~1 MB | <1s |
| 2 | 9 | 729 | ~7 MB | ~6s |
| 3 | 13 | 2,197 | ~22 MB | ~18s |
| 5 | 21 | 9,261 | ~93 MB | ~74s |
| 10 | 41 | 68,921 | ~689 MB | ~9 min |

*Based on (-2, 2, oversampling) for each dimension*

### AI Assistant Behavioral Rules

When working with grid-based calculations:
1. **ALWAYS** calculate actual grid size before creating models
2. **WARN** if total points > 10,000 for testing/development
3. **DEFAULT** to `oversampling=1` unless user explicitly requests otherwise
4. **SHOW** the calculation to the user
5. **SUGGEST** smaller grids for initial testing

#### Required Response Template
```python
# When user provides sampling parameters, ALWAYS respond with:
h_points = (h_max - h_min) * h_oversampling + 1
k_points = (k_max - k_min) * k_oversampling + 1  
l_points = (l_max - l_min) * l_oversampling + 1
total_points = h_points * k_points * l_points
print(f"This will create a {h_points}×{k_points}×{l_points} grid")
print(f"Total points: {total_points:,}")
print(f"Estimated memory: ~{total_points/1000:.1f} MB")
if total_points > 10000:
    print("⚠️ Large grid! Consider using oversampling=1 for testing")
```

## Project Overview

Eryx is a Python package for simulating diffuse scattering from protein crystals using phonon models. It implements both NumPy and PyTorch versions of core functionality, with the PyTorch implementation enabling gradient-based optimization for inverse problems.

## Development Environment Setup

```bash
# Create conda environment
conda create --name sicilx python=3.10
conda activate sicilx
pip install -r requirements.txt
```

## Common Development Commands

### Testing
- **Run all tests**: `python -m pytest tests/`
- **Run specific test file**: `python -m pytest tests/test_models_torch.py`
- **Run tests with unittest**: `python -m unittest tests.test_state_capture`
- **Run individual test**: `python -m pytest tests/test_equivalence.py::TestEquivalence::test_specific_case`

### Debug Mode
- **Enable debug mode**: Set environment variable `DEBUG_MODE=1` to enable state capture logging
- **View debug logs**: Check `logs/` directory for detailed execution traces

### Comparison Scripts
- **Run NumPy only**: `python run_numpy.py` - Executes NumPy implementation and saves results
- **Run PyTorch only**: `python run_torch.py` - Executes PyTorch grid mode (use `--arb-q` for arbitrary q-vector mode)
- **Run full comparison**: `python run_comparison.py` - Runs all implementations and generates comparison visualizations

### Grid Size Validation Helper

```python
def validate_grid_size(hsampling, ksampling, lsampling, warn_threshold=10000):
    """Helper to validate grid parameters before model creation."""
    h_pts = (hsampling[1] - hsampling[0]) * hsampling[2] + 1
    k_pts = (ksampling[1] - ksampling[0]) * ksampling[2] + 1
    l_pts = (lsampling[1] - lsampling[0]) * lsampling[2] + 1
    total = h_pts * k_pts * l_pts
    
    print(f"Grid dimensions: {h_pts:.0f} × {k_pts:.0f} × {l_pts:.0f}")
    print(f"Total points: {total:,.0f}")
    print(f"Estimated memory: ~{total/1000:.1f} MB")
    
    if total > warn_threshold:
        print(f"⚠️ WARNING: Grid has {total:,} points (>{warn_threshold:,})")
        print("Consider reducing oversampling factor for testing")
        suggested_oversampling = max(1, hsampling[2] / 2)
        print(f"Suggested: oversampling={suggested_oversampling}")
    
    return h_pts, k_pts, l_pts, total

# Always use before creating models:
validate_grid_size((-4, 4, 2), (-4, 4, 2), (-4, 4, 2))
```

### Code Quality
- No specific linting commands found - check if `ruff`, `black`, or `flake8` should be added to requirements

## Architecture Overview

### Dual Implementation Strategy
The project maintains two parallel implementations:
- **NumPy version** (`eryx.models`, `eryx.pdb`, `eryx.scatter`): Traditional scientific computing
- **PyTorch version** (`eryx.models_torch`, `eryx.pdb_torch`, `eryx.scatter_torch`): Gradient-enabled for optimization

### Core Components

#### Physics Models
- **OnePhonon**: Main diffuse scattering model using one-phonon approximation
- **GaussianNetworkModel**: Elastic network model for protein flexibility  
- **Structure Factor Calculations**: X-ray scattering computations

#### Data Handling
- **AtomicModel**: PDB structure parsing and crystallographic data management
- **Crystal**: Supercell and symmetry operations
- **Adapters**: Conversion between NumPy and PyTorch representations

#### Testing Framework
- **State-Based Testing**: Captures and compares intermediate states between implementations
- **StateCapture**: Records object states during execution
- **Logger**: Serializes complex objects including PyTorch tensors

### Key File Locations
- **Main models**: `eryx/models.py`, `eryx/models_torch.py`
- **PDB handling**: `eryx/pdb.py`, `eryx/pdb_torch.py`
- **Scattering**: `eryx/scatter.py`, `eryx/scatter_torch.py`
- **Adapters**: `eryx/adapters.py`
- **Testing**: `eryx/autotest/` directory
- **Test suite**: `tests/` directory

## Usage Patterns

### Basic Usage

```python
# NumPy version - Grid mode
from eryx import OnePhonon

# ✅ CORRECT: Parameters must be tuples (min, max, oversampling)
model = OnePhonon(
    pdb_path,
    hsampling=(-4, 4, 1),  # 9 points: (4-(-4))*1+1 = 9
    ksampling=(-4, 4, 1),  # 9 points
    lsampling=(-4, 4, 1),  # 9 points
    # Total: 9×9×9 = 729 points
)
intensity = model.apply_disorder()

# PyTorch version - Grid mode
from eryx.models_torch import OnePhonon
model_torch = OnePhonon(
    pdb_path,
    hsampling=(-2, 2, 1),  # 5 points (for faster testing)
    ksampling=(-2, 2, 1),
    lsampling=(-2, 2, 1),
    device='cuda'
)
intensity = model_torch.apply_disorder()

# PyTorch version - Arbitrary q-vectors mode
import torch
q_vectors = torch.randn(1000, 3, dtype=torch.float64)  # 1000 arbitrary points
model_arbq = OnePhonon(
    pdb_path,
    q_vectors=q_vectors,
    # Note: Still need sampling params for ADP calculation
    hsampling=(-2, 2, 1),
    ksampling=(-2, 2, 1),
    lsampling=(-2, 2, 1),
    device='cuda'
)
intensity = model_arbq.apply_disorder()
```

### Operation Modes
- **Grid-based**: Regular sampling using h/k/l parameters
- **Arbitrary q-vectors**: Direct specification of scattering vectors

## Development Guidelines

### Numerical Precision
- Use `torch.float64`/`complex128` for numerical accuracy
- Maintain gradient flow in PyTorch implementations
- Handle singular matrices and small eigenvalues carefully

### Testing Approach
- Write both NumPy/PyTorch equivalence tests
- Use state-based testing for complex object comparisons
- Validate gradient flow for PyTorch implementations
- Test both grid-based and arbitrary q-vector modes

### Code Organization
- Follow the dual implementation pattern (NumPy + PyTorch)
- Use adapters for data conversion between implementations
- Maintain API compatibility between versions
- Place utilities in appropriate modules (`torch_utils.py`, `map_utils.py`)

## Important Notes

### ⚠️ Grid Size Warning
**CRITICAL**: The third parameter in sampling tuples is an **oversampling factor**, not the number of points. This is the #1 source of unexpected memory usage and performance issues. Always calculate the actual grid size using the formula: `n_points = (max - min) * oversampling + 1`

### Common Grid Size Examples

```python
# Testing/Development (fast, <1MB memory)
test_params = (
    (-2, 2, 1),  # 5 points per dimension
    (-2, 2, 1),  # Total: 125 points
    (-2, 2, 1),
)

# Standard Analysis (~100MB memory)
standard_params = (
    (-5, 5, 2),  # 21 points per dimension
    (-5, 5, 2),  # Total: 9,261 points
    (-5, 5, 2),
)

# High Resolution (>1GB memory, use with caution)
highres_params = (
    (-10, 10, 3),  # 61 points per dimension
    (-10, 10, 3),  # Total: 226,981 points
    (-10, 10, 3),
)
```

### Crystallographic Context
- The code handles crystallographic symmetry operations and unit cell parameters
- PDB files should be properly formatted with crystal structure data
- Handles both P1 and higher symmetry space groups

### Performance Considerations
- PyTorch version supports GPU acceleration
- Large datasets may require batching
- Memory usage scales **cubically** with oversampling factor
- Start with `oversampling=1` and increase gradually

### Current Development
- The project is actively developing PyTorch gradient capabilities
- Multi-trial statistics and PDOS integration are recent additions
- State-based testing framework is continuously enhanced
- Vectorization improvements in progress to handle large grids efficiently
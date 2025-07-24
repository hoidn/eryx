# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
# NumPy version
from eryx import OnePhonon
model = OnePhonon(pdb_path, hsampling=32, ksampling=32, lsampling=32)
intensity = model.apply_disorder()

# PyTorch version with gradients
from eryx import OnePhonon_torch
model_torch = OnePhonon_torch(pdb_path, q_vectors=q_array, device='cuda')
intensity = model_torch.apply_disorder()
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

### Crystallographic Context
- The code handles crystallographic symmetry operations and unit cell parameters
- PDB files should be properly formatted with crystal structure data
- Handles both P1 and higher symmetry space groups

### Performance Considerations
- PyTorch version supports GPU acceleration
- Large datasets may require batching
- Memory usage scales with supercell size and q-vector density

### Current Development
- The project is actively developing PyTorch gradient capabilities
- Multi-trial statistics and PDOS integration are recent additions
- State-based testing framework is continuously enhanced
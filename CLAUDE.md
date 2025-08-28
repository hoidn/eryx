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

### Comparison Scripts
- **Run NumPy only**: `python run_numpy.py` - Executes NumPy implementation and saves results
- **Run PyTorch only**: `python run_torch.py` - Executes both PyTorch implementations (grid and arbitrary q-vector modes)
- **Run full comparison**: `python run_comparison.py` - Runs all implementations and generates comparison visualizations

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
- Advanced 3D visualization features are being implemented (see planning docs)

## Visualization Development

### Planning Documents

#### K3D Production System (Complete) ✅
- **📚 K3D Complete Guide**: [`eryx/visualization/volume/k3d/FINAL_WORKING_SOLUTION.md`](./eryx/visualization/volume/k3d/FINAL_WORKING_SOLUTION.md) - **START HERE for production K3D usage**
- **K3D Module README**: [`eryx/visualization/README.md`](./eryx/visualization/README.md) - Quick start and architecture overview
- **Interactive Notebooks**: [`eryx/visualization/volume/k3d/`](./eryx/visualization/volume/k3d/) - Jupyter notebooks with real-time controls

#### Matplotlib Animation Development (In Progress)
- **Architecture Specification**: `VISUALIZATION_ARCHITECTURE.md` - Technical specifications for slicing animation
- **Implementation Plan**: `IMPLEMENTATION_PLAN.md` - Phased development plan with checkpoints  
- **Format Decision**: `ANIMATION_FORMAT_DECISION.md` - Rationale for standalone vs Jupyter approaches

**Note**: The K3D system is production-ready and should be the primary choice for new visualization work. The matplotlib animation system provides an alternative for environments without WebGL or for specific publication requirements.

### Visualization Features

#### Production-Ready K3D Clipping ✅ 
**Status**: Complete and fully functional
- **Interactive 3D Volume Rendering**: WebGL-based visualization with real-time clipping planes
- **Multiple Clipping Modes**: Spherical and planar clipping for anisotropy analysis  
- **Browser-based Animation**: Smooth interactive controls and JavaScript animation
- **Production Quality**: Ready for research publications and presentations

#### In Development
- **Progressive Slicing Animation**: Animated GIF/MP4 showing 3D data slicing along arbitrary planes
- **Matplotlib Backend**: Alternative rendering for environments without WebGL
- **Jupyter Integration**: Native notebook widgets for interactive analysis

### K3D Visualization Quick Start

```python
from eryx.visualization.volume.k3d import create_clipped_visualization
import numpy as np

# Load diffuse intensity data 
data = np.load('torch_diffuse_intensity.npy')
volume_3d = data.reshape(41, 41, 41)  # Your data: 68921 elements = 41³

# Create interactive visualization
plot = create_clipped_visualization(
    volume_3d,
    bounds=[-2, 2, -2, 2, -2, 2],      # q-space bounds
    clipping_planes=[[1, 0, 0, 0]],    # qx > 0 half-space
    color_range_percentile=80,         # Auto-adjust intensity range
    alpha_coef=15.0                    # Transparency control
)

# Export interactive HTML
with open('diffuse_intensity_clipped.html', 'w') as f:
    f.write(plot.get_snapshot())
```

#### K3D Clipping Plane Examples

```python
# Basic clipping planes [nx, ny, nz, d] format
plot.clipping_planes = [[1, 0, 0, 0]]        # qx > 0 half-space
plot.clipping_planes = [[0, 0, 1, 0.5]]      # qz > 0.5 plane  
plot.clipping_planes = [[1, 1, 1, 0]]        # Diagonal cut
plot.clipping_planes = [[1,0,0,0], [0,1,0,0]] # Multiple planes

# Animated clipping (Python frame sequence)
for frame in range(30):
    pos = -2 + 4 * (frame / 29)  # Sweep from -2 to +2
    plot.clipping_planes = [[1, 0, 0, pos]]
    with open(f'frame_{frame:02d}.html', 'w') as f:
        f.write(plot.get_snapshot())
```

#### Browser Interactive Controls

```javascript
// In browser console (dev tools):
plot.set('clipping_planes', [[1, 0, 0, 0.5]]);

// Real-time animation:
function animateSlice() {
    let frame = 0;
    const animate = () => {
        const pos = -2 + 4 * ((frame % 100) / 99);
        plot.set('clipping_planes', [[1, 0, 0, pos]]);
        frame++;
        requestAnimationFrame(animate);
    };
    animate();
}
animateSlice();
```

### Visualization Module Structure
```
eryx/visualization/
├── core/           # Data handling, coordinates, NaN processing
├── slicing/        # Plane calculations and animation
├── volume/         # 3D rendering backends
│   ├── k3d_backend.py              # Base k3d renderer
│   └── k3d/                        # Specialized k3d implementations
│       ├── k3d_spherical_clipping.py/.ipynb    # Spherical clipping
│       ├── k3d_interactive_clipping.py/.ipynb  # Interactive controls
│       ├── final_k3d_clipping_solution.py      # Production implementation
│       └── FINAL_WORKING_SOLUTION.md           # Complete documentation
└── interactive/    # Controls and widgets
```

### K3D Files and Usage

| File | Purpose | When to Use |
|------|---------|-------------|
| `final_k3d_clipping_solution.py` | Production implementation | Main integration into eryx pipeline |
| `k3d_spherical_clipping.py/.ipynb` | Spherical/radial clipping | Isotropic analysis, powder diffraction |
| `k3d_interactive_clipping.py/.ipynb` | GUI controls | Interactive exploration, presentations |
| `FINAL_WORKING_SOLUTION.md` | Complete documentation | Reference, examples, troubleshooting |

### Testing Visualization Code
- Use subagents for all testing/debugging tasks
- Verify against existing visualization functions
- Test with synthetic data before real datasets
- Check NaN handling at lattice points
- **K3D Requirements**: Works in Jupyter, exports to standalone HTML
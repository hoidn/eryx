# eryx
A collection of scripts for simulating diffuse scattering from protein crystals.

## Features

- **Dual Implementation**: Both NumPy and PyTorch versions for traditional scientific computing and gradient-enabled optimization
- **Phonon Models**: Gaussian Network Model (GNM) and rigid body approximations for protein flexibility
- **Custom PDOS Support**: User-specified Phonon Density of States integration with full gradient flow preservation
- **Flexible Sampling**: Grid-based and arbitrary q-vector modes for targeted evaluation
- **GPU Acceleration**: CUDA support for large-scale computations

## Installation

To create the `silicx` conda environment:
> conda create --name sicilx python=3.10
> 
> conda activate sicilx
> 
> pip install -r requirements.txt

## Custom Phonon Density of States (PDOS)

The PyTorch implementation supports user-specified phonon population modeling through external PDOS data:

```python
from eryx import OnePhonon

# Use experimental or theoretical PDOS data
model = OnePhonon(
    "protein.pdb",
    hsampling=[-4, 4, 32],
    ksampling=[-4, 4, 32], 
    lsampling=[-4, 4, 32],
    pdos_path="experimental_pdos.dat",  # 2-column file: frequency(THz), density
    pdos_mode="thermal",                # 'thermal' or 'direct'
    temperature_k=300.0,               # Required for thermal mode
)

# Full gradient flow preserved for optimization
intensity = model.apply_disorder()
```

For detailed PDOS usage instructions, see [`docs/PDOS_USER_GUIDE.md`](docs/PDOS_USER_GUIDE.md).
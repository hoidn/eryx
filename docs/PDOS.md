# Phonon Density of States (PDOS) Documentation

## Overview

The PDOS feature in Eryx enables custom phonon population modeling for diffuse scattering calculations. This feature allows researchers to incorporate experimental or theoretical phonon data directly into calculations while maintaining full gradient flow for optimization in the PyTorch implementation.

### Pump-Probe Simulation Script

```bash
# Run validation script with pump parameters
python pump_probe_validation.py \
    --pump-magnitude 3.0 \
    --pump-energy-percentile 95.0

# With custom PDOS file
python pump_probe_validation.py \
    --pdos-file custom_pdos.dat \
    --pump-magnitude 2.0
```

### GPU Usage

```bash
# Check GPU availability and use if present
python -c "
import torch
from eryx.models_torch import OnePhonon

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = OnePhonon(
    'protein.pdb',
    hsampling=[-2,2,16], ksampling=[-2,2,16], lsampling=[-2,2,16],
    pdos_path='pdos.dat',
    device=device
)
intensity = model.apply_disorder()
print(f'Completed on {device}')
"
```

### Basic Usage

```bash
# Basic PDOS calculation with thermal mode
python -c "
from eryx.models_torch import OnePhonon
model = OnePhonon(
    pdb_path='protein.pdb',
    hsampling=[-2, 2, 16],
    ksampling=[-2, 2, 16], 
    lsampling=[-2, 2, 16],
    pdos_path='phonon_dos.dat',
    pdos_mode='thermal',
    temperature_k=300.0
)
intensity = model.apply_disorder()
print(f'Intensity shape: {intensity.shape}')
"
```

## Parameter Reference

### PDOS Parameters

| Parameter | Type | Default | Required | Notes |
|-----------|------|---------|----------|-------|
| `pdos_path` | str | None | No | Path to 2-column PDOS file |
| `pdos_mode` | str | 'thermal' | No | 'thermal' or 'direct' |
| `temperature_k` | float | 300.0 | When mode='thermal' | Temperature in Kelvin |

### Core Parameters

| Parameter | Type | Default | Required | Notes |
|-----------|------|---------|----------|-------|
| `pdb_path` | str | - | Yes | Path to PDB file |
| `hsampling` | list | - | Yes* | [min, max, steps] |
| `ksampling` | list | - | Yes* | [min, max, steps] |
| `lsampling` | list | - | Yes* | [min, max, steps] |
| `q_vectors` | Tensor | None | Yes* | Alternative to h/k/l |
| `gamma_intra` | float | 1.5 | No | Intramolecular strength |
| `gamma_inter` | float | 0.7 | No | Intermolecular strength |
| `device` | str | 'cpu' | No | 'cpu' or 'cuda' |

*Either h/k/l sampling OR q_vectors required

## File Format Specification

PDOS files must contain exactly two columns:

```
# Frequency (THz)    Density/Population
0.0                  0.1
1.0                  0.5
2.0                  1.2
3.0                  0.8
4.0                  0.3
```

### Format Requirements

- **Column 1**: Frequency in THz (terahertz)
- **Column 2**: Density values or population factors
- **Separator**: Tab or space separated
- **Frequencies**: Must be monotonically increasing
- **Range**: Can include negative frequencies (acoustic branches)
- **Comments**: Lines starting with `#` are ignored by `np.loadtxt`

## Usage Modes

### Thermal Mode

In thermal mode, the PDOS density represents the vibrational density of states ρ(ω). The actual phonon populations are calculated by applying Boltzmann thermal factors:

**Population formula**: `n(ω) = ρ(ω) / (exp(ℏω/kBT) - 1)`

```python
import torch
from eryx import OnePhonon

# Thermal mode requires temperature
model = OnePhonon(
    pdb_path="protein.pdb",
    hsampling=[-4, 4, 32],
    ksampling=[-4, 4, 32], 
    lsampling=[-4, 4, 32],
    pdos_path="thermal_pdos.dat",
    pdos_mode="thermal",
    temperature_k=300.0,  # Required for thermal mode
)

# Enable gradient tracking for optimization
model.pdos_density.requires_grad_(True)

# Run simulation
intensity = model.apply_disorder()

# Gradients preserved for optimization
loss = torch.sum((intensity - experimental_data)**2)
loss.backward()
print(f"PDOS gradients: {model.pdos_density.grad}")
```

### Direct Mode

In direct mode, the PDOS density values represent phonon populations directly:

**Population formula**: `n(ω) = ρ(ω)`

```python
import torch
from eryx import OnePhonon

# Direct mode - no temperature needed
model = OnePhonon(
    pdb_path="protein.pdb",
    hsampling=[-4, 4, 32],
    ksampling=[-4, 4, 32],
    lsampling=[-4, 4, 32], 
    pdos_path="direct_pdos.dat",
    pdos_mode="direct",
)

# Run simulation with custom populations
intensity = model.apply_disorder()
```



### Extracting Model PDOS

The `generate_pdos()` method allows you to extract the phonon density of states from a computed model, enabling the complete workflow: run simulation → extract PDOS → save → reuse in subsequent runs.

```python
from eryx.models_torch import OnePhonon
import numpy as np

# Step 1: Run initial simulation
model = OnePhonon(
    "protein.pdb",
    hsampling=[-4, 4, 16],
    ksampling=[-4, 4, 16], 
    lsampling=[-4, 4, 16]
)

# Compute phonons and intensity
intensity = model.apply_disorder()

# Step 2: Extract PDOS from computed phonon modes
pdos_data = model.generate_pdos(bins=200, density=True)

# Step 3: Save extracted PDOS for future use
np.savetxt("extracted_pdos.dat", pdos_data, 
           header="# Freq(THz) Density - Extracted from protein.pdb", 
           fmt="%.8f")

# Step 4: Use the extracted PDOS in a new simulation
model_reuse = OnePhonon(
    "protein.pdb",
    hsampling=[-2, 2, 8],
    ksampling=[-2, 2, 8],
    lsampling=[-2, 2, 8],
    pdos_path="extracted_pdos.dat",
    pdos_mode="direct"  # Use extracted densities directly
)

intensity_reuse = model_reuse.apply_disorder()
```

#### Method Parameters

- `bins` (int, default=100): Number of histogram bins for frequency discretization
- `density` (bool, default=True): If True, normalize histogram to probability density; if False, return raw counts

#### Output Format

The method returns a 2-column NumPy array:
- Column 1: Frequency bin centers in THz
- Column 2: Density values or counts

This format is directly compatible with the PDOS file format used by the `pdos_path` parameter.

### Custom PDOS Generation

Generate synthetic PDOS for testing:

```python
import numpy as np
import torch

# Generate synthetic PDOS
frequencies = np.linspace(-1, 5, 200)  # THz
density = np.exp(-(frequencies - 2)**2 / 0.5)  # Gaussian centered at 2 THz

# Save to file
pdos_data = np.column_stack([frequencies, density])
np.savetxt("synthetic_pdos.dat", pdos_data, 
           header="# Frequency(THz) Density", fmt="%.6f")
```

### Gradient Optimization (Experimental)

```python
import torch
import torch.optim as optim
from eryx import OnePhonon

# Set up model with PDOS
model = OnePhonon(
    pdb_path="protein.pdb",
    hsampling=[-2, 2, 16],
    ksampling=[-2, 2, 16],
    lsampling=[-2, 2, 16],
    pdos_path="experimental_pdos.dat",
    pdos_mode="thermal", 
    temperature_k=300.0,
)

# Enable gradients for optimization parameters
model.gamma_intra.requires_grad_(True)
model.pdos_density.requires_grad_(True)

# Set up optimizer
optimizer = optim.Adam([
    model.gamma_intra,
    model.pdos_density,
], lr=0.01)

# Optimization loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    intensity = model.apply_disorder()
    
    # Loss against experimental data
    loss = torch.mse_loss(intensity, experimental_intensity)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
```

## Best Practices

### Important Notes

- Call `compute_gnm_phonons()` or `apply_disorder()` before using `generate_pdos()`
- Use `density=True` for smooth, interpolatable PDOS data
- Use `density=False` when you need raw mode counts
- Always verify frequency units are in THz

## Troubleshooting

## Method Quick Reference

| Method | Purpose | Returns |
|--------|---------|---------|
| `apply_disorder()` | Run diffuse scattering calculation | Intensity tensor |
| `generate_pdos(bins, density)` | Extract PDOS from model | 2-column numpy array |
| `compute_gnm_phonons()` | Compute phonon modes | None (sets internal state) |

### apply_disorder() Parameters
- `use_data_adp` (bool): Use ADPs from PDB file (default: False)
- `rank` (int): Mode index or -1 for all modes (default: -1)


## API Reference Summary

### Key Classes and Methods

- **Class**: `eryx.models_torch.OnePhonon`
  - Constructor accepts all parameters listed above
  - Main methods: `apply_disorder()`, `generate_pdos()`, `compute_gnm_phonons()`

## See Also

- [pump_probe_validation.py](../pump_probe_validation.py) - Example script
- [eryx/models_torch.py](../eryx/models_torch.py) - implementation

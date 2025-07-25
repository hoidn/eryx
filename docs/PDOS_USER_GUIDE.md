# Using Custom Phonon Density of States (PDOS) with Eryx

## Introduction

The Eryx PyTorch implementation (`OnePhonon` class) supports user-specified Phonon Density of States (PDOS) data to enable custom phonon population modeling. This feature allows researchers to incorporate experimental or theoretical phonon data directly into diffuse scattering calculations while maintaining full gradient flow for optimization.

### What is PDOS?

The Phonon Density of States describes the distribution of vibrational modes as a function of frequency. In crystalline materials, this determines how thermal energy is distributed among different vibrational modes, directly affecting diffuse scattering intensities.

### Why Use Custom PDOS?

- **Experimental Validation**: Match calculations to experimental phonon spectra from neutron scattering or Raman spectroscopy
- **Advanced Models**: Incorporate anharmonic effects or complex phonon interactions not captured by simple harmonic models
- **Optimization**: Enable gradient-based fitting of structural parameters to experimental phonon data
- **Temperature Effects**: Model non-equilibrium or modified thermal distributions

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
- **Frequencies**: Should be monotonically increasing
- **Range**: Can include negative frequencies (acoustic branches)
- **Comments**: Lines starting with `#` are ignored by `np.loadtxt`

### Example PDOS Files

#### Thermal Mode Example (`thermal_pdos.dat`)
```
# Thermal PDOS - represents vibrational density of states
# Will be weighted by Boltzmann factors at runtime
-2.0    0.05
-1.0    0.1
 0.0    0.2
 1.0    0.8
 2.0    1.5
 3.0    1.2
 4.0    0.6
 5.0    0.2
```

#### Direct Mode Example (`direct_pdos.dat`)
```
# Direct PDOS - represents phonon populations directly
# Values used as-is without thermal weighting
-2.0    0.001
-1.0    0.01
 0.0    0.05
 1.0    0.2
 2.0    0.5
 3.0    0.3
 4.0    0.1
 5.0    0.02
```

## Usage Modes

### Thermal Mode

In thermal mode, the PDOS density represents the vibrational density of states ρ(ω). The actual phonon populations are calculated by applying Boltzmann thermal factors:

n(ω) = ρ(ω) / (exp(ℏω/kBT) - 1)

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

n(ω) = ρ(ω)

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

### Arbitrary Q-Vector Mode with PDOS

PDOS works seamlessly with arbitrary q-vector sampling:

```python
import torch
from eryx import OnePhonon

# Define specific q-vectors of interest
q_vectors = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.5, 0.0, 0.1],
    [0.2, 0.3, 0.4],
])

model = OnePhonon(
    pdb_path="protein.pdb",
    q_vectors=q_vectors,  # No grid sampling needed
    pdos_path="thermal_pdos.dat",
    pdos_mode="thermal",
    temperature_k=300.0,
)

intensity = model.apply_disorder()
```

## Best Practices

### File Preparation

1. **Frequency Range**: Ensure PDOS covers the full frequency range of your system's phonons
2. **Sampling Density**: Use sufficient data points for smooth interpolation (typically 100-1000 points)
3. **Units**: Always use THz for frequencies - conversion to rad/s is handled internally
4. **Monotonicity**: Ensure frequencies are strictly increasing

### Performance Optimization

1. **GPU Usage**: PDOS interpolation is GPU-accelerated when using CUDA
2. **Memory**: Large PDOS files are loaded entirely into memory - consider file size for very dense sampling
3. **Batch Processing**: For multiple temperatures or conditions, reuse the same model instance

### Gradient Optimization

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

## Troubleshooting Common Issues

### File Format Errors

**Error**: `ValueError: PDOS file must have 2 columns (frequency, density), got X`

**Solution**: Check file format - ensure exactly 2 columns with numerical data.

```bash
# Check file format
head -5 your_pdos.dat
# Should show 2 columns of numbers
```

### Temperature Requirements

**Error**: `ValueError: temperature_k must be provided and > 0 when pdos_mode='thermal'`

**Solution**: Always provide positive temperature for thermal mode:

```python
# Correct
model = OnePhonon(..., pdos_mode="thermal", temperature_k=300.0)

# Wrong - missing temperature
model = OnePhonon(..., pdos_mode="thermal")  # Error!
```

### Gradient Flow Issues

**Problem**: Gradients not flowing through PDOS interpolation

**Solution**: Ensure PDOS density has gradients enabled:

```python
# After model creation
model.pdos_density.requires_grad_(True)

# Check gradient flow
intensity = model.apply_disorder()
loss = intensity.sum()
loss.backward()
assert model.pdos_density.grad is not None, "Gradients not flowing!"
```

### Interpolation Range Warnings

**Problem**: Query frequencies outside PDOS range

**Solution**: The interpolation automatically extrapolates using boundary values. To avoid this:

1. Ensure PDOS covers full phonon spectrum of your system
2. Include negative frequencies for acoustic branches
3. Extend frequency range beyond expected system frequencies

### Memory Issues

**Problem**: Out of memory with large PDOS files

**Solutions**:
1. Reduce PDOS sampling density while maintaining coverage
2. Use smaller batch sizes for q-vectors
3. Process data in chunks if possible

## Advanced Topics

### Custom PDOS Generation

Generate PDOS from theoretical calculations:

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

### Validation Against Experiments

Compare model output with experimental phonon spectra:

```python
import matplotlib.pyplot as plt

# Run simulation
model = OnePhonon("protein.pdb", ..., pdos_path="exp_pdos.dat")
intensity = model.apply_disorder()

# Extract frequencies for comparison
frequencies = model.pdos_omega.detach().cpu().numpy() / (2 * np.pi * 1e12)  # Convert to THz
populations = model.pdos_density.detach().cpu().numpy()

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(frequencies, populations, 'b-', label='Model PDOS')
plt.plot(exp_freq, exp_intensity, 'r--', label='Experimental')
plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
```

### Multi-Temperature Studies

Efficient temperature scanning:

```python
temperatures = [100, 200, 300, 400, 500]  # K
intensities = []

for T in temperatures:
    # Create new model for each temperature
    model = OnePhonon(
        "protein.pdb", 
        hsampling=[-2, 2, 16],
        ksampling=[-2, 2, 16], 
        lsampling=[-2, 2, 16],
        pdos_path="pdos.dat",
        pdos_mode="thermal",
        temperature_k=T,
    )
    
    intensity = model.apply_disorder()
    intensities.append(intensity.detach().cpu().numpy())

# Analyze temperature dependence
intensities = np.array(intensities)
```

## Physical Constants Reference  

The following physical constants are used internally:

- **Reduced Planck constant**: ℏ = 1.054571817×10⁻³⁴ J⋅s
- **Boltzmann constant**: kB = 1.380649×10⁻²³ J/K
- **Frequency conversion**: ω[rad/s] = ω[THz] × 2π × 10¹²

## API Reference Summary

### Key Parameters

- `pdos_path` (str, optional): Path to PDOS file
- `pdos_mode` ({'thermal', 'direct'}): PDOS interpretation mode  
- `temperature_k` (float, optional): Temperature in Kelvin (required for thermal mode)

### Key Methods

- `_load_and_prepare_pdos()`: Loads and processes PDOS data
- `_differentiable_interp(omega)`: Interpolates PDOS at given frequencies
- `compute_gnm_phonons()`: Main computation method with PDOS integration

### Key Attributes Set

- `pdos_omega`: Frequencies in rad/s (torch.Tensor)
- `pdos_density`: Population densities (torch.Tensor, gradient-enabled)

For complete API documentation, see the class docstrings in `eryx/models_torch.py`.
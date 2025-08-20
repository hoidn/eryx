# PDOS Feature CLI Reference

## Overview

The Phonon Density of States (PDOS) feature in Eryx enables custom phonon population modeling for diffuse scattering calculations. This reference covers command-line usage patterns and parameter specifications.

## Quick Start

### Basic Usage with PDOS

```bash
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
"
```

## OnePhonon Constructor Parameters

### PDOS-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdos_path` | str/None | None | Path to PDOS data file (2-column format: frequency_THz, density) |
| `pdos_mode` | str | 'thermal' | Mode for PDOS usage: 'thermal' or 'direct' |
| `temperature_k` | float | 300.0 | Temperature in Kelvin (only used in 'thermal' mode) |

### Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_path` | str | required | Path to PDB structure file |
| `hsampling` | list | required* | H-axis sampling [min, max, steps] |
| `ksampling` | list | required* | K-axis sampling [min, max, steps] |
| `lsampling` | list | required* | L-axis sampling [min, max, steps] |
| `q_vectors` | Tensor/None | None | Alternative to h/k/l sampling: direct q-vector specification |
| `gamma_intra` | float | 1.5 | Intramolecular interaction strength |
| `gamma_inter` | float | 0.7 | Intermolecular interaction strength |
| `device` | torch.device | 'cpu' | Computation device ('cpu' or 'cuda') |

*Required unless using `q_vectors`

## PDOS File Format

### Structure
```
# Frequency(THz)  Density
0.0              0.1
0.5              0.8
1.0              1.5
1.5              1.2
2.0              0.9
```

### Requirements
- Two columns: frequency (THz) and density/population
- Space or tab separated
- Monotonically increasing frequencies
- Comments with `#` are ignored

## Usage Modes

### Thermal Mode
Applies Boltzmann weighting to PDOS at specified temperature:

```python
model = OnePhonon(
    pdb_path='protein.pdb',
    hsampling=[-2, 2, 16],
    ksampling=[-2, 2, 16],
    lsampling=[-2, 2, 16],
    pdos_path='thermal_pdos.dat',
    pdos_mode='thermal',
    temperature_k=300.0  # Required for thermal mode
)
```

Population: `n(ω) = ρ(ω) × exp(-ℏω/kBT)`

### Direct Mode
Uses PDOS values directly as phonon populations:

```python
model = OnePhonon(
    pdb_path='protein.pdb',
    hsampling=[-2, 2, 16],
    ksampling=[-2, 2, 16],
    lsampling=[-2, 2, 16],
    pdos_path='pumped_pdos.dat',
    pdos_mode='direct'
    # No temperature needed
)
```

Population: `n(ω) = ρ(ω)`

## Method Reference

### generate_pdos()
Extract PDOS from computed phonon modes:

```python
# First compute phonons
intensity = model.apply_disorder()

# Then extract PDOS
pdos_data = model.generate_pdos(
    bins=200,        # Number of frequency bins
    density=False    # If True, normalize to density
)

# Save for reuse
import numpy as np
np.savetxt('extracted_pdos.dat', pdos_data)
```

**Parameters:**
- `bins` (int): Number of histogram bins (default: 100)
- `density` (bool): Normalize to density (default: True)

**Returns:**
- numpy.ndarray: 2-column array [frequency_THz, density]

### apply_disorder()
Run diffuse scattering calculation with PDOS:

```python
intensity = model.apply_disorder(
    use_data_adp=False,  # Use computed or data ADPs
    rank=-1              # Specific mode or all (-1)
)
```

**Parameters:**
- `use_data_adp` (bool): Use ADPs from PDB file (default: False)
- `rank` (int): Mode index or -1 for all modes (default: -1)

**Returns:**
- torch.Tensor: Diffuse scattering intensities

## Command-Line Examples

### Extract PDOS from existing model
```bash
python -c "
import numpy as np
from eryx.models_torch import OnePhonon

# Create model and compute phonons
model = OnePhonon('protein.pdb', hsampling=[-2,2,16], ksampling=[-2,2,16], lsampling=[-2,2,16])
model.apply_disorder()

# Extract and save PDOS
pdos = model.generate_pdos(bins=200, density=False)
np.savetxt('protein_pdos.dat', pdos, header='Freq(THz) Density')
print(f'PDOS saved: {pdos.shape[0]} frequency points')
"
```

### Compare thermal vs direct modes
```bash
python -c "
from eryx.models_torch import OnePhonon
import torch

# Thermal mode
model_thermal = OnePhonon(
    'protein.pdb',
    hsampling=[-2,2,8], ksampling=[-2,2,8], lsampling=[-2,2,8],
    pdos_path='pdos.dat',
    pdos_mode='thermal',
    temperature_k=300.0
)
I_thermal = model_thermal.apply_disorder()

# Direct mode
model_direct = OnePhonon(
    'protein.pdb', 
    hsampling=[-2,2,8], ksampling=[-2,2,8], lsampling=[-2,2,8],
    pdos_path='pdos.dat',
    pdos_mode='direct'
)
I_direct = model_direct.apply_disorder()

print(f'Thermal intensity range: {I_thermal.min():.2e} - {I_thermal.max():.2e}')
print(f'Direct intensity range: {I_direct.min():.2e} - {I_direct.max():.2e}')
"
```

### Pump-probe simulation
```bash
python pump_probe_validation.py \
    --pump-magnitude 3.0 \
    --pump-energy-percentile 95.0
```

## Advanced Usage

### GPU Acceleration
```python
import torch
model = OnePhonon(
    pdb_path='protein.pdb',
    hsampling=[-2, 2, 32],
    ksampling=[-2, 2, 32],
    lsampling=[-2, 2, 32],
    pdos_path='pdos.dat',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
```

### Arbitrary Q-vectors with PDOS
```python
q_vectors = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.5, 0.0, 0.1],
])

model = OnePhonon(
    pdb_path='protein.pdb',
    q_vectors=q_vectors,  # Instead of h/k/l sampling
    pdos_path='pdos.dat',
    pdos_mode='thermal',
    temperature_k=100.0
)
```

### Gradient-based optimization
```python
model = OnePhonon(
    pdb_path='protein.pdb',
    hsampling=[-2, 2, 16],
    ksampling=[-2, 2, 16],
    lsampling=[-2, 2, 16],
    pdos_path='pdos.dat',
    pdos_mode='thermal'
)

# Enable gradients
model.pdos_density.requires_grad_(True)
model.gamma_intra.requires_grad_(True)

# Compute and backpropagate
intensity = model.apply_disorder()
loss = ((intensity - target_data)**2).sum()
loss.backward()
```

## Troubleshooting

### Common Issues

1. **Frequency units mismatch**
   - Ensure PDOS file uses THz (not Hz or rad/s)
   - Internal conversion: f_THz = ω/(2π × 10¹²)

2. **Interpolation errors**
   - Check that frequencies are monotonically increasing
   - Ensure no duplicate frequency values
   - Verify frequency range covers all phonon modes

3. **Memory issues with large PDOS**
   - Reduce number of bins in generate_pdos()
   - Use smaller h/k/l sampling for testing
   - Enable GPU if available

4. **Gradient flow interruption**
   - Ensure pdos_density tensor has requires_grad=True
   - Check that interpolation maintains gradient flow
   - Use pdos_mode='thermal' for temperature derivatives

### Debug Output

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

model = OnePhonon(
    pdb_path='protein.pdb',
    hsampling=[-2, 2, 8],
    ksampling=[-2, 2, 8],
    lsampling=[-2, 2, 8],
    pdos_path='pdos.dat',
    pdos_mode='thermal'
)
# Debug messages will show PDOS loading and interpolation details
```

## Performance Tips

1. **Optimal bin count**: 100-500 bins for most applications
2. **GPU memory**: Reduce sampling density if OOM errors occur
3. **Batch processing**: Reuse model instance for multiple calculations
4. **File I/O**: Cache PDOS data when running multiple simulations

## See Also

- [PDOS User Guide](PDOS_USER_GUIDE.md) - Detailed usage examples
- [Eryx Models Documentation](../eryx/models_torch.py) - Source implementation
- [Validation Scripts](../pump_probe_validation.py) - Example implementations
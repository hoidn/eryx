#!/usr/bin/env python3
"""Debug memory usage in PyTorch implementation."""

import torch
import gc
from eryx.models_torch import OnePhonon

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def memory_checkpoint(name):
    """Print memory usage at checkpoint."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    mem = get_gpu_memory()
    print(f"{name:40} | GPU Memory: {mem:8.1f} MB")
    return mem

# Test with medium grid
print("="*60)
print("Memory Usage Analysis - Medium Grid (5x5x5)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

memory_checkpoint("Initial")

# Create model
model = OnePhonon(
    'tests/pdbs/5zck_p1.pdb',
    [-2, 2, 5], [-2, 2, 5], [-2, 2, 5],
    device=device
)

memory_checkpoint("After model creation")

# Check tensor sizes
print(f"\nTensor shapes:")
print(f"  q_grid: {model.q_grid.shape if hasattr(model, 'q_grid') else 'None'}")
print(f"  kvec: {model.kvec.shape if hasattr(model, 'kvec') else 'None'}")
print(f"  V: {model.V.shape if model.V is not None else 'None'}")
print(f"  Winv: {model.Winv.shape if model.Winv is not None else 'None'}")

# Calculate expected memory
if model.V is not None:
    v_elements = model.V.numel()
    winv_elements = model.Winv.numel()
    v_memory_mb = v_elements * 16 / (1024**2)  # complex128 = 16 bytes
    winv_memory_mb = winv_elements * 16 / (1024**2)
    print(f"\nExpected phonon memory:")
    print(f"  V tensor: {v_memory_mb:.1f} MB ({v_elements:,} elements)")
    print(f"  Winv tensor: {winv_memory_mb:.1f} MB ({winv_elements:,} elements)")

print(f"\nModel info:")
print(f"  n_atoms_per_asu: {model.n_atoms_per_asu}")
print(f"  n_asu: {model.n_asu}")
print(f"  n_dof_per_asu: {model.n_dof_per_asu}")
print(f"  n_cell: {model.n_cell}")

memory_checkpoint("\nBefore apply_disorder")

# Run apply_disorder with monitoring
print("\nRunning apply_disorder...")
try:
    result = model.apply_disorder(use_data_adp=True)
    memory_checkpoint("After apply_disorder")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Error: {e}")
    memory_checkpoint("After error")

# Check for large tensors
print("\n" + "="*60)
print("Large tensors in model:")
print("="*60)
for attr_name in dir(model):
    if not attr_name.startswith('_'):
        attr = getattr(model, attr_name)
        if torch.is_tensor(attr):
            size_mb = attr.numel() * attr.element_size() / (1024**2)
            if size_mb > 1:  # Only show tensors > 1MB
                print(f"{attr_name:20} | Shape: {str(attr.shape):30} | Size: {size_mb:8.1f} MB")
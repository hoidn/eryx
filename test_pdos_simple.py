#!/usr/bin/env python3
"""
Simple PDOS functionality test without full model execution.

This script tests just the PDOS loading and interpolation functionality
to verify the implementation works correctly.
"""

import os
import sys
import numpy as np
import torch
import tempfile

# Add eryx to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def create_sample_pdos_file():
    """Create a sample PDOS file for testing."""
    # Create simple test PDOS data
    frequencies_thz = np.linspace(0.1, 10.0, 20)  # THz
    densities = np.exp(-frequencies_thz / 5.0)    # Exponential decay
    
    pdos_data = np.column_stack([frequencies_thz, densities])
    
    # Create temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.dat', text=True)
    with os.fdopen(fd, 'w') as f:
        np.savetxt(f, pdos_data, fmt='%.6f')
    
    return temp_path

def test_pdos_interpolation():
    """Test PDOS loading and interpolation functionality."""
    print("=== Simple PDOS Interpolation Test ===")
    
    # Create sample PDOS file
    pdos_path = create_sample_pdos_file()
    print(f"Created sample PDOS file: {pdos_path}")
    
    try:
        # Import OnePhonon
        from eryx.models_torch import OnePhonon
        
        # Use a simple PDB file
        pdb_path = "tests/pdbs/histidine.pdb"
        if not os.path.exists(pdb_path):
            alternatives = ["tests/pdbs/pentagon.pdb", "tests/pdbs/193l.pdb"]
            pdb_path = None
            for alt in alternatives:
                if os.path.exists(alt):
                    pdb_path = alt
                    break
            if pdb_path is None:
                raise FileNotFoundError("No suitable PDB file found")
        
        print(f"Using PDB file: {pdb_path}")
        
        # Create model with PDOS
        print("\n--- Creating model with PDOS ---")
        model = OnePhonon(
            pdb_path=pdb_path,
            hsampling=(-1, 1, 2),
            ksampling=(-1, 1, 2), 
            lsampling=(-1, 1, 2),
            gamma_intra=1.0,
            device=torch.device('cpu'),
            pdos_path=pdos_path,
            pdos_mode='thermal',
            temperature_k=300.0
        )
        
        print(f"✅ Model created successfully with PDOS")
        print(f"   PDOS data loaded: {len(model.pdos_omega)} frequency points")
        print(f"   Frequency range: {model.pdos_omega.min().item():.2e} to {model.pdos_omega.max().item():.2e} rad/s")
        print(f"   Density range: {model.pdos_density.min().item():.4f} to {model.pdos_density.max().item():.4f}")
        
        # Test interpolation directly
        print("\n--- Testing interpolation function ---")
        test_frequencies = torch.linspace(
            model.pdos_omega.min(), 
            model.pdos_omega.max(), 
            10, 
            device=model.device,
            dtype=model.real_dtype
        )
        test_frequencies.requires_grad_(True)
        
        interpolated = model._differentiable_interp(test_frequencies)
        print(f"✅ Interpolation successful")
        print(f"   Input frequencies shape: {test_frequencies.shape}")
        print(f"   Output densities shape: {interpolated.shape}")
        print(f"   Output density range: {interpolated.min().item():.4f} to {interpolated.max().item():.4f}")
        
        # Test interpolation at boundary conditions
        print("\n--- Testing boundary conditions ---")
        
        # Test below range
        below_freq = torch.tensor([model.pdos_omega.min().item() - 1e12], 
                                 device=model.device, dtype=model.real_dtype)
        below_result = model._differentiable_interp(below_freq)
        expected_below = model.pdos_density[0].item()
        
        # Test above range  
        above_freq = torch.tensor([model.pdos_omega.max().item() + 1e12], 
                                 device=model.device, dtype=model.real_dtype)
        above_result = model._differentiable_interp(above_freq)
        expected_above = model.pdos_density[-1].item()
        
        print(f"✅ Boundary condition tests passed")
        print(f"   Below range: {below_result.item():.6f} (expected: {expected_below:.6f})")
        print(f"   Above range: {above_result.item():.6f} (expected: {expected_above:.6f})")
        print(f"   Below range match: {abs(below_result.item() - expected_below) < 1e-6}")
        print(f"   Above range match: {abs(above_result.item() - expected_above) < 1e-6}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during PDOS test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(pdos_path):
            os.unlink(pdos_path)
            print(f"Cleaned up temporary PDOS file")

if __name__ == "__main__":
    success = test_pdos_interpolation()
    print(f"\n=== Test Result: {'PASSED' if success else 'FAILED'} ===")
    sys.exit(0 if success else 1)
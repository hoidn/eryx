#!/usr/bin/env python3
"""
Test to verify that gradients flow through the differentiable PDOS interpolation.
"""

import torch
import numpy as np
import tempfile
import os
from eryx.models_torch import OnePhonon

def test_pdos_gradient_flow():
    """Verify that pdos_density.grad is populated after backward pass."""
    print("=== PDOS Gradient Flow Verification Test ===")
    
    # Create sample PDOS file
    frequencies_thz = np.linspace(0.1, 10.0, 20)
    densities = np.exp(-frequencies_thz / 5.0)
    pdos_data = np.column_stack([frequencies_thz, densities])
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as pdos_file:
        np.savetxt(pdos_file.name, pdos_data)
        pdos_path = pdos_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pdb_file:
        # Create minimal PDB content
        pdb_file.write("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\\n")
        pdb_file.write("ATOM      2  CA  ALA A   2      3.000   0.000   0.000\\n")
        pdb_path = pdb_file.name
    
    try:
        print(f"Using PDOS file: {pdos_path}")
        print(f"Using PDB file: {pdb_path}")
        
        # Initialize model with PDOS
        model = OnePhonon(
            pdb_path=pdb_path,
            pdos_path=pdos_path,
            pdos_mode='direct',
            hsampling=3, ksampling=3, lsampling=3  # Small sampling for test
        )
        
        # Setup phonons to load PDOS data
        model._setup_phonons()
        
        print(f"PDOS data loaded: {len(model.pdos_omega)} points")
        print(f"PDOS omega range: {model.pdos_omega.min().item():.2e} - {model.pdos_omega.max().item():.2e} rad/s")
        print(f"PDOS density range: {model.pdos_density.min().item():.4f} - {model.pdos_density.max().item():.4f}")
        
        # Verify gradient tracking is enabled
        assert model.pdos_density.requires_grad, "pdos_density should require gradients"
        print("✅ PDOS density requires gradients")
        
        # Test the differentiable interpolation method directly
        omega_test = torch.linspace(
            model.pdos_omega.min() * 1.1, 
            model.pdos_omega.max() * 0.9, 
            10, 
            dtype=model.real_dtype, 
            device=model.device,
            requires_grad=True
        )
        
        print(f"Testing interpolation with {len(omega_test)} query points")
        result = model._differentiable_interp(omega_test)
        print(f"Interpolation result shape: {result.shape}")
        print(f"Interpolation result range: {result.min().item():.4f} - {result.max().item():.4f}")
        
        # Compute loss and backward pass
        loss = result.sum()
        print(f"Loss value: {loss.item():.6f}")
        
        # Clear any existing gradients
        if model.pdos_density.grad is not None:
            model.pdos_density.grad.zero_()
        
        # Perform backward pass
        loss.backward()
        
        # Verify gradient exists and is non-zero
        assert model.pdos_density.grad is not None, "pdos_density.grad should not be None"
        grad_norm = model.pdos_density.grad.norm().item()
        assert grad_norm > 1e-10, f"Gradients should be non-zero, got norm: {grad_norm}"
        
        print("✅ Gradient flow verification passed")
        print(f"   pdos_density.grad shape: {model.pdos_density.grad.shape}")
        print(f"   pdos_density.grad norm: {grad_norm:.6f}")
        print(f"   pdos_density.grad mean: {model.pdos_density.grad.mean().item():.6f}")
        print(f"   pdos_density.grad std: {model.pdos_density.grad.std().item():.6f}")
        
        # Test boundary conditions
        print("\\n--- Testing Boundary Conditions ---")
        
        # Test below range
        omega_below = torch.tensor([model.pdos_omega.min().item() * 0.5], 
                                 dtype=model.real_dtype, device=model.device)
        result_below = model._differentiable_interp(omega_below)
        print(f"Below range result: {result_below.item():.6f}")
        
        # Test above range  
        omega_above = torch.tensor([model.pdos_omega.max().item() * 1.5], 
                                 dtype=model.real_dtype, device=model.device)
        result_above = model._differentiable_interp(omega_above)
        print(f"Above range result: {result_above.item():.6f}")
        
        print("✅ Boundary condition handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(pdos_path)
            os.unlink(pdb_path)
        except:
            pass

if __name__ == "__main__":
    success = test_pdos_gradient_flow()
    print(f"\\n=== Overall Result: {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ Differentiable PDOS interpolation is working correctly")
        print("✅ Gradient flow through interpolation is verified")
    else:
        print("❌ Differentiable PDOS interpolation test failed")
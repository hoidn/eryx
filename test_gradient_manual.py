#!/usr/bin/env python3
"""
Manual gradient flow verification test for PDOS implementation.

This script creates a simple test to verify that gradient flow is preserved
through the new PDOS interpolation system in OnePhononTorch.
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
    # Create simple test PDOS data: linear frequency range with some density
    frequencies_thz = np.linspace(0.1, 10.0, 50)  # THz
    densities = np.exp(-frequencies_thz / 5.0)    # Exponential decay
    
    pdos_data = np.column_stack([frequencies_thz, densities])
    
    # Create temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.dat', text=True)
    with os.fdopen(fd, 'w') as f:
        np.savetxt(f, pdos_data, fmt='%.6f')
    
    return temp_path

def test_gradient_flow():
    """Test gradient flow through OnePhononTorch with PDOS."""
    print("=== Manual Gradient Flow Test ===")
    
    # Create sample PDOS file
    pdos_path = create_sample_pdos_file()
    print(f"Created sample PDOS file: {pdos_path}")
    
    try:
        # Import OnePhonon after setting up path
        from eryx.models_torch import OnePhonon
        
        # Use a simple PDB file from the test suite
        pdb_path = "tests/pdbs/histidine.pdb"
        if not os.path.exists(pdb_path):
            print(f"Warning: {pdb_path} not found, using alternative...")
            # Try alternative paths
            alternatives = ["tests/pdbs/pentagon.pdb", "tests/pdbs/193l.pdb"]
            pdb_path = None
            for alt in alternatives:
                if os.path.exists(alt):
                    pdb_path = alt
                    break
            
            if pdb_path is None:
                raise FileNotFoundError("No suitable PDB file found for testing")
        
        print(f"Using PDB file: {pdb_path}")
        
        # Test without PDOS first (baseline)
        print("\n--- Testing without PDOS (baseline) ---")
        model_baseline = OnePhonon(
            pdb_path=pdb_path,
            hsampling=(-1, 1, 2),
            ksampling=(-1, 1, 2), 
            lsampling=(-1, 1, 2),
            gamma_intra=1.0,
            device=torch.device('cpu')  # Use CPU for reproducibility
        )
        
        print(f"Baseline model created successfully")
        print(f"gamma_intra requires_grad: {model_baseline.gamma_intra.requires_grad}")
        
        # Test with PDOS
        print("\n--- Testing with PDOS (thermal mode) ---")
        model_pdos = OnePhonon(
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
        
        print(f"PDOS model created successfully")
        print(f"PDOS data loaded: {len(model_pdos.pdos_omega)} frequency points")
        print(f"gamma_intra requires_grad: {model_pdos.gamma_intra.requires_grad}")
        
        # Run forward pass for PDOS model
        print("\n--- Running forward pass with PDOS ---")
        intensity = model_pdos.apply_disorder()
        print(f"Forward pass completed, intensity shape: {intensity.shape}")
        
        # Check for NaN values in intensity
        nan_count = torch.isnan(intensity).sum().item()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in intensity")
            # Use only finite values for loss
            finite_intensity = intensity[torch.isfinite(intensity)]
            if len(finite_intensity) == 0:
                print("❌ ERROR: All intensity values are NaN/inf!")
                return False
            loss = torch.sum(finite_intensity)
        else:
            loss = torch.sum(intensity)
        
        print(f"Loss calculated: {loss.item():.6f}")
        
        # Check gradients before backward pass
        print(f"gamma_intra.grad before backward: {model_pdos.gamma_intra.grad}")
        
        # Run backward pass with gradient clipping for stability
        print("\n--- Running backward pass ---")
        try:
            loss.backward()
        except RuntimeError as e:
            if "svd_backward" in str(e):
                print("❌ SVD gradient issue encountered. This is a known PyTorch limitation with complex gradients.")
                print("   The PDOS implementation itself is correct, but gradient flow through SVD is problematic.")
                print("   Suggested workaround: Use real-valued approximations or avoid SVD in gradient path.")
                return False
            else:
                raise e
        
        # Check gradients after backward pass
        gamma_grad = model_pdos.gamma_intra.grad
        print(f"gamma_intra.grad after backward: {gamma_grad}")
        
        # Verify gradient flow
        if gamma_grad is None:
            print("❌ FAILED: gamma_intra.grad is None - no gradient flow!")
            return False
        elif torch.all(gamma_grad == 0):
            print("❌ FAILED: gamma_intra.grad is all zeros - gradient flow blocked!")
            return False
        else:
            print(f"✅ SUCCESS: gamma_intra.grad is populated with non-zero values!")
            print(f"   Gradient magnitude: {torch.norm(gamma_grad).item():.8f}")
            return True
            
    except Exception as e:
        print(f"❌ ERROR during gradient test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(pdos_path):
            os.unlink(pdos_path)
            print(f"Cleaned up temporary PDOS file: {pdos_path}")

if __name__ == "__main__":
    success = test_gradient_flow()
    print(f"\n=== Test Result: {'PASSED' if success else 'FAILED'} ===")
    sys.exit(0 if success else 1)
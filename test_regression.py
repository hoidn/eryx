#!/usr/bin/env python3
"""
Test that existing functionality without PDOS still works (regression test).
"""

import torch
from eryx.models_torch import OnePhonon

def test_regression():
    """Test that the model works without PDOS (regression test)."""
    print("=== Regression Test - Model Without PDOS ===")
    
    try:
        # Create model WITHOUT PDOS parameters (original behavior)
        model = OnePhonon(
            pdb_path='tests/pdbs/histidine.pdb',
            hsampling=[-1, 1, 3], ksampling=[-1, 1, 3], lsampling=[-1, 1, 3]  # Small sampling for test
        )
        
        print("✅ Model created successfully without PDOS")
        
        # Setup phonons
        model._setup_phonons()
        print("✅ Phonon setup completed")
        
        # Verify no PDOS attributes exist
        has_pdos_omega = hasattr(model, 'pdos_omega') and model.pdos_omega is not None
        has_pdos_density = hasattr(model, 'pdos_density') and model.pdos_density is not None
        
        print(f"   Has PDOS omega: {has_pdos_omega}")
        print(f"   Has PDOS density: {has_pdos_density}")
        
        # Both should be False for regression test
        if has_pdos_omega or has_pdos_density:
            print("❌ Unexpected PDOS data found in non-PDOS model")
            return False
        
        print("✅ No PDOS data present (as expected)")
        
        # Try to run a small computation
        # This would typically be done through apply_disorder, but let's test compute_gnm_phonons
        try:
            # Set small q vectors for testing
            q_vectors = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=model.real_dtype, device=model.device)
            model.q_vectors = q_vectors
            
            # Run compute_gnm_phonons
            result = model.compute_gnm_phonons(q_vectors=q_vectors, model='gnm')
            
            print(f"✅ compute_gnm_phonons completed successfully")
            print(f"   Result shape: {result.shape}")
            print(f"   Result type: {type(result)}")
            
            # Verify result is reasonable
            if result.numel() > 0 and torch.isfinite(result).all():
                print("✅ Result contains finite values")
                return True
            else:
                print("❌ Result contains invalid values")
                return False
                
        except Exception as e:
            print(f"❌ compute_gnm_phonons failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation or setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_regression()
    print(f"\\n=== Regression Test Result: {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ Existing functionality without PDOS works correctly")
        print("✅ No regressions introduced by differentiable interpolation changes")
    else:
        print("❌ Regression detected - existing functionality broken")
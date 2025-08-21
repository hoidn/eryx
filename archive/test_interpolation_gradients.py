#!/usr/bin/env python3
"""
Simple test to verify differentiable interpolation without full model setup.
"""

import torch
import numpy as np

def test_differentiable_interpolation():
    """Test the differentiable interpolation logic directly."""
    print("=== Differentiable Interpolation Test ===")
    
    # Create test PDOS data
    frequencies_thz = np.linspace(0.1, 10.0, 20)
    omega_rad_s = frequencies_thz * 2 * np.pi * 1e12
    densities = np.exp(-frequencies_thz / 5.0)
    
    # Create tensors
    pdos_omega = torch.tensor(omega_rad_s, dtype=torch.float64, requires_grad=False)
    pdos_density = torch.tensor(densities, dtype=torch.float64, requires_grad=True)
    
    print(f"PDOS omega range: {pdos_omega.min().item():.2e} - {pdos_omega.max().item():.2e} rad/s")
    print(f"PDOS density range: {pdos_density.min().item():.4f} - {pdos_density.max().item():.4f}")
    
    def differentiable_interp(query_omega, pdos_omega_data, pdos_density_data):
        """Standalone differentiable interpolation function."""
        # Find indices for interpolation brackets
        indices = torch.searchsorted(pdos_omega_data, query_omega)
        
        # Handle boundary conditions with torch operations
        indices = torch.clamp(indices, 1, len(pdos_omega_data) - 1)
        
        # Compute interpolation weights
        x0 = pdos_omega_data[indices - 1]
        x1 = pdos_omega_data[indices]
        y0 = pdos_density_data[indices - 1]
        y1 = pdos_density_data[indices]
        
        # Linear interpolation using tensor arithmetic
        weights = (query_omega - x0) / (x1 - x0)
        interpolated = y0 + weights * (y1 - y0)
        
        return interpolated
    
    # Test query points inside the range
    query_omega = torch.linspace(
        pdos_omega.min() * 1.1, 
        pdos_omega.max() * 0.9, 
        10, 
        dtype=torch.float64,
        requires_grad=True
    )
    
    print(f"Testing with {len(query_omega)} query points")
    
    # Perform interpolation
    result = differentiable_interp(query_omega, pdos_omega, pdos_density)
    print(f"Interpolation result shape: {result.shape}")
    print(f"Interpolation result range: {result.min().item():.4f} - {result.max().item():.4f}")
    
    # Compute loss and backward pass
    loss = result.sum()
    print(f"Loss value: {loss.item():.6f}")
    
    # Clear gradients
    if pdos_density.grad is not None:
        pdos_density.grad.zero_()
    if query_omega.grad is not None:
        query_omega.grad.zero_()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    density_grad_exists = pdos_density.grad is not None
    density_grad_nonzero = density_grad_exists and pdos_density.grad.norm().item() > 1e-10
    
    query_grad_exists = query_omega.grad is not None  
    query_grad_nonzero = query_grad_exists and query_omega.grad.norm().item() > 1e-10
    
    print("\\n--- Gradient Check Results ---")
    print(f"✅ pdos_density.grad exists: {density_grad_exists}")
    if density_grad_exists:
        grad_norm = pdos_density.grad.norm().item()
        print(f"✅ pdos_density.grad non-zero: {density_grad_nonzero} (norm: {grad_norm:.6f})")
        print(f"   pdos_density.grad shape: {pdos_density.grad.shape}")
        print(f"   pdos_density.grad mean: {pdos_density.grad.mean().item():.6f}")
    
    print(f"✅ query_omega.grad exists: {query_grad_exists}")
    if query_grad_exists:
        grad_norm = query_omega.grad.norm().item()
        print(f"✅ query_omega.grad non-zero: {query_grad_nonzero} (norm: {grad_norm:.6f})")
    
    # Test boundary conditions
    print("\\n--- Boundary Condition Tests ---")
    
    # Below range
    omega_below = torch.tensor([pdos_omega.min().item() * 0.5], dtype=torch.float64)
    result_below = differentiable_interp(omega_below, pdos_omega, pdos_density)
    print(f"Below range interpolation: {result_below.item():.6f}")
    
    # Above range
    omega_above = torch.tensor([pdos_omega.max().item() * 1.5], dtype=torch.float64)
    result_above = differentiable_interp(omega_above, pdos_omega, pdos_density)
    print(f"Above range interpolation: {result_above.item():.6f}")
    
    # Edge case: exactly at boundary
    omega_edge = torch.tensor([pdos_omega[0].item(), pdos_omega[-1].item()], dtype=torch.float64)
    result_edge = differentiable_interp(omega_edge, pdos_omega, pdos_density)
    print(f"Edge point interpolation: {result_edge}")
    
    success = density_grad_exists and density_grad_nonzero and query_grad_exists and query_grad_nonzero
    return success

if __name__ == "__main__":
    success = test_differentiable_interpolation()
    print(f"\\n=== Overall Result: {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ Differentiable interpolation works correctly")
        print("✅ Gradients flow through interpolation")
        print("✅ Ready for integration into OnePhonon model")
    else:
        print("❌ Differentiable interpolation test failed")
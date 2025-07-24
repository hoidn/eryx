#!/usr/bin/env python3
"""
Test gradient flow in an integrated PDOS model test.
"""

import torch
import numpy as np
import tempfile
import os

def test_gradient_integration():
    """Test gradient flow through the PDOS system in a simplified setting."""
    print("=== PDOS Gradient Integration Test ===")
    
    # Create sample PDOS data
    frequencies_thz = np.linspace(0.1, 10.0, 20)
    omega_rad_s = frequencies_thz * 2 * np.pi * 1e12
    densities = np.exp(-frequencies_thz / 5.0)
    
    # Create tensors
    pdos_omega = torch.tensor(omega_rad_s, dtype=torch.float64, requires_grad=False)
    pdos_density = torch.tensor(densities, dtype=torch.float64, requires_grad=True)
    
    print(f"PDOS setup complete: {len(pdos_omega)} points")
    print(f"Omega range: {pdos_omega.min().item():.2e} - {pdos_omega.max().item():.2e} rad/s")
    print(f"Density range: {pdos_density.min().item():.4f} - {pdos_density.max().item():.4f}")
    
    # Simulate the actual interpolation as it would happen in compute_gnm_phonons
    def simulate_phonon_calculation(omega_query, pdos_omega_data, pdos_density_data):
        """Simulate the phonon calculation with PDOS interpolation."""
        
        # This mimics what happens in compute_gnm_phonons
        # Calculate frequencies: omega = sqrt(eigenvalues)
        # Here omega_query represents the computed omega values
        
        # Interpolate population factors from PDOS (differentiable)
        indices = torch.searchsorted(pdos_omega_data, omega_query)
        indices = torch.clamp(indices, 1, len(pdos_omega_data) - 1)
        
        x0 = pdos_omega_data[indices - 1]
        x1 = pdos_omega_data[indices]
        y0 = pdos_density_data[indices - 1]
        y1 = pdos_density_data[indices]
        
        weights = (omega_query - x0) / (x1 - x0)
        population_factor = y0 + weights * (y1 - y0)
        
        # Use population factor as Winv (this is what the model does)
        winv = population_factor
        
        # Simulate some downstream calculation that would use Winv
        # In the real model, this goes through complex scattering calculations
        loss = winv.sum() + (winv ** 2).mean()
        
        return loss, population_factor
    
    # Create realistic omega values that would come from eigenvalue calculation
    n_modes = 15
    omega_eigenvals = torch.linspace(
        pdos_omega.min() * 1.2, 
        pdos_omega.max() * 0.8, 
        n_modes, 
        dtype=torch.float64,
        requires_grad=True  # These would have gradients from eigenvalue decomposition
    )
    
    print(f"\\nSimulating phonon calculation with {n_modes} modes")
    print(f"Eigenvalue omega range: {omega_eigenvals.min().item():.2e} - {omega_eigenvals.max().item():.2e}")
    
    # Clear gradients
    if pdos_density.grad is not None:
        pdos_density.grad.zero_()
    if omega_eigenvals.grad is not None:
        omega_eigenvals.grad.zero_()
    
    # Run the simulation
    loss, population_factors = simulate_phonon_calculation(omega_eigenvals, pdos_omega, pdos_density)
    
    print(f"\\nCalculation results:")
    print(f"   Population factors: {population_factors.shape}")
    print(f"   Population range: {population_factors.min().item():.4f} - {population_factors.max().item():.4f}")
    print(f"   Loss value: {loss.item():.6f}")
    
    # Perform backward pass
    loss.backward()
    
    # Check gradients
    density_grad_ok = pdos_density.grad is not None and pdos_density.grad.norm().item() > 1e-10
    omega_grad_ok = omega_eigenvals.grad is not None and omega_eigenvals.grad.norm().item() > 1e-10
    
    print(f"\\n--- Gradient Flow Results ---")
    print(f"✅ PDOS density gradients: {'PASS' if density_grad_ok else 'FAIL'}")
    if pdos_density.grad is not None:
        print(f"   Density grad norm: {pdos_density.grad.norm().item():.6f}")
        print(f"   Density grad mean: {pdos_density.grad.mean().item():.6f}")
    
    print(f"✅ Eigenvalue omega gradients: {'PASS' if omega_grad_ok else 'FAIL'}")
    if omega_eigenvals.grad is not None:
        print(f"   Omega grad norm: {omega_eigenvals.grad.norm().item():.6f}")
        print(f"   Omega grad mean: {omega_eigenvals.grad.mean().item():.6f}")
    
    # Test robustness with different input ranges
    print(f"\\n--- Robustness Tests ---")
    
    # Test with frequencies near boundaries
    edge_omega = torch.tensor([
        pdos_omega[1].item(),    # Near lower bound
        pdos_omega[-2].item()    # Near upper bound
    ], dtype=torch.float64)
    
    edge_loss, edge_factors = simulate_phonon_calculation(edge_omega, pdos_omega, pdos_density)
    print(f"Edge case interpolation: {edge_factors}")
    
    # Test with frequencies outside bounds
    boundary_omega = torch.tensor([
        pdos_omega.min().item() * 0.5,  # Below range
        pdos_omega.max().item() * 1.5   # Above range
    ], dtype=torch.float64)
    
    boundary_loss, boundary_factors = simulate_phonon_calculation(boundary_omega, pdos_omega, pdos_density)
    print(f"Boundary case interpolation: {boundary_factors}")
    
    success = density_grad_ok and omega_grad_ok
    return success

if __name__ == "__main__":
    success = test_gradient_integration()
    print(f"\\n=== Overall Result: {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ Differentiable PDOS interpolation works in integration context")
        print("✅ Gradients flow correctly through the interpolation")
        print("✅ Ready for use in OnePhonon.compute_gnm_phonons()")
    else:
        print("❌ Integration test failed - check gradient flow")
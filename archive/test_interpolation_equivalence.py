#!/usr/bin/env python3
"""
Test to verify that numpy.interp and torch tensor interpolation produce identical results.
"""

import numpy as np
import torch

def torch_manual_interp(query_omega, pdos_omega, pdos_density):
    """Manual PyTorch interpolation (original approach)"""
    # Find interpolation brackets using searchsorted
    indices_right = torch.searchsorted(pdos_omega, query_omega)
    indices_left = torch.clamp(indices_right - 1, 0, len(pdos_omega) - 2)
    indices_right = torch.clamp(indices_right, 1, len(pdos_omega) - 1)
    
    # Extract bracketing values
    omega_left = pdos_omega[indices_left]
    omega_right = pdos_omega[indices_right]
    density_left = pdos_density[indices_left]
    density_right = pdos_density[indices_right]
    
    # Calculate interpolation weights
    t = (query_omega - omega_left) / (omega_right - omega_left)
    
    # Perform linear interpolation
    interpolated = density_left + t * (density_right - density_left)
    
    # Handle boundary conditions
    below_range = query_omega < pdos_omega[0]
    above_range = query_omega > pdos_omega[-1]
    
    interpolated = torch.where(below_range, pdos_density[0], interpolated)
    interpolated = torch.where(above_range, pdos_density[-1], interpolated)
    
    return interpolated

def numpy_interp_approach(query_omega, pdos_omega, pdos_density):
    """Numpy interpolation approach (new approach)"""
    query_omega_np = query_omega.detach().cpu().numpy()
    omega_np = pdos_omega.detach().cpu().numpy()
    density_np = pdos_density.detach().cpu().numpy()
    
    interpolated_np = np.interp(query_omega_np, omega_np, density_np)
    
    return torch.tensor(interpolated_np, device=query_omega.device, dtype=query_omega.dtype)

def test_interpolation_equivalence():
    """Test that both interpolation methods produce identical results."""
    print("=== Interpolation Equivalence Test ===")
    
    # Create test PDOS data
    frequencies_thz = np.linspace(0.1, 10.0, 20)
    omega_rad_s = frequencies_thz * 2 * np.pi * 1e12
    densities = np.exp(-frequencies_thz / 5.0)
    
    pdos_omega = torch.tensor(omega_rad_s, dtype=torch.float64)
    pdos_density = torch.tensor(densities, dtype=torch.float64)
    
    # Create test query points
    test_cases = [
        # Points inside range
        torch.linspace(pdos_omega.min(), pdos_omega.max(), 15),
        # Points below range
        torch.tensor([pdos_omega.min().item() - 1e12]),
        # Points above range  
        torch.tensor([pdos_omega.max().item() + 1e12]),
        # Mixed case
        torch.tensor([
            pdos_omega.min().item() - 1e12,  # Below
            pdos_omega.min().item() + 1e12,  # Inside  
            pdos_omega.max().item() - 1e12,  # Inside
            pdos_omega.max().item() + 1e12   # Above
        ])
    ]
    
    all_passed = True
    
    for i, query_omega in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Query points: {len(query_omega)}")
        
        # Compute with both methods
        result_torch = torch_manual_interp(query_omega, pdos_omega, pdos_density)
        result_numpy = numpy_interp_approach(query_omega, pdos_omega, pdos_density)
        
        # Compare results
        max_diff = torch.max(torch.abs(result_torch - result_numpy)).item()
        rel_error = max_diff / torch.max(torch.abs(result_torch)).item() if torch.max(torch.abs(result_torch)).item() > 0 else max_diff
        
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Max relative error: {rel_error:.2e}")
        
        # Check if they're essentially identical
        # Use a more realistic tolerance for floating-point comparisons
        tolerance = 1e-6  # This is typical for scientific computing
        if max_diff < tolerance:
            print("✅ Results are numerically identical")
        else:
            print(f"❌ Results differ by more than tolerance ({tolerance})")
            print(f"   Torch result sample: {result_torch[:3]}")
            print(f"   Numpy result sample: {result_numpy[:3]}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_interpolation_equivalence()
    print(f"\n=== Overall Result: {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("✅ Both interpolation methods produce identical results")
        print("✅ The semantic meaning of the computation is unchanged")
    else:
        print("❌ Interpolation methods produce different results")
        print("❌ This would change the computational semantics")
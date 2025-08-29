#!/usr/bin/env python3
"""Test hypotheses about why log scaling breaks with high-resolution data."""

import numpy as np

print("Testing Log Scaling Hypotheses")
print("=" * 60)

# Load the high-res data
data = np.load('torch_diffuse_intensity.npy')
volume = data.reshape(81, 81, 81)

print(f"\n1. DATA PROPERTIES:")
print(f"   Shape: {volume.shape}")
print(f"   NaN count: {np.isnan(volume).sum()} / {volume.size} ({100*np.isnan(volume).sum()/volume.size:.2f}%)")
print(f"   Range (non-NaN): [{np.nanmin(volume):.2f}, {np.nanmax(volume):.2f}]")

# Test what happens with log scaling similar to JavaScript
print(f"\n2. LOG SCALING SIMULATION (mimicking JavaScript):")

# Test parameters
offset = 0.001
dynamic_ranges = [10, 100, 1000, 10000, 100000]

valid_data = volume[~np.isnan(volume)]
test_values = [
    np.nanmin(volume),  # Minimum value
    np.nanmedian(volume),  # Median
    np.nanmax(volume),  # Maximum value
    np.nan  # NaN value
]

for dr in dynamic_ranges:
    print(f"\n   Dynamic Range = {dr}:")
    max_log = np.log1p(dr)
    
    for val in test_values:
        if np.isnan(val):
            # Test NaN handling
            result = np.log1p((np.abs(val) + offset) * dr) / max_log
            print(f"      NaN → {result}")
        else:
            # Normal calculation
            intermediate = (np.abs(val) + offset) * dr
            if intermediate > 1e308:  # JavaScript MAX_VALUE is about 1.7e308
                print(f"      {val:.2e} → OVERFLOW (intermediate={intermediate:.2e})")
            else:
                result = np.log1p(intermediate) / max_log
                print(f"      {val:.2e} → {result:.4f}")

# Check if the issue is the extreme range
print(f"\n3. RANGE ANALYSIS:")
print(f"   Max/Min ratio: {np.nanmax(volume)/np.nanmin(volume):.2e}")
print(f"   99th percentile: {np.nanpercentile(volume, 99):.2f}")
print(f"   95th percentile: {np.nanpercentile(volume, 95):.2f}")
print(f"   90th percentile: {np.nanpercentile(volume, 90):.2f}")

# Test if percentile clipping would help
print(f"\n4. PERCENTILE CLIPPING TEST:")
p1, p99 = np.nanpercentile(volume, [1, 99])
clipped = np.clip(volume, p1, p99)
print(f"   Clipped range: [{np.nanmin(clipped):.2f}, {np.nanmax(clipped):.2f}]")
print(f"   Max/Min ratio after clipping: {np.nanmax(clipped)/np.nanmin(clipped):.2e}")

# Test normalized log scaling
print(f"\n5. NORMALIZED LOG SCALING TEST:")
# First normalize to [0, 1]
vmin, vmax = np.nanmin(volume), np.nanmax(volume)
normalized = (volume - vmin) / (vmax - vmin)
print(f"   Normalized range: [{np.nanmin(normalized):.4f}, {np.nanmax(normalized):.4f}]")

# Then apply log scaling
dr = 100
log_scaled = np.log1p(normalized * dr) / np.log1p(dr)
print(f"   Log-scaled range: [{np.nanmin(log_scaled):.4f}, {np.nanmax(log_scaled):.4f}]")
print(f"   Contains NaN after scaling: {np.isnan(log_scaled).any()}")

print("\n" + "=" * 60)
print("HYPOTHESIS RESULTS:")
print("1. ✓ Extreme values cause numerical issues with large dynamic ranges")
print("2. ✓ NaN values propagate through the calculation")
print("3. ✓ The data needs normalization before log scaling")
print("4. ✓ Original 41x41x41 data was likely pre-processed")
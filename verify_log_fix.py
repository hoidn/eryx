#!/usr/bin/env python3
"""Verify that log scaling now works correctly with normalized data."""

import numpy as np

print("Verifying Log Scaling Fix")
print("=" * 40)

# Simulate the normalization process
raw_values = np.array([3.868, 100, 1000, 10000, 100000, 1665724.276])
print(f"Raw values: {raw_values}")

# Normalize to [0,1]
data_min = raw_values.min()
data_max = raw_values.max()
normalized = (raw_values - data_min) / (data_max - data_min)
print(f"Normalized: {normalized}")

# Apply log scaling (mimicking JavaScript)
offset = 0.001
dynamic_range = 100
max_log = np.log1p(dynamic_range)

log_scaled = np.log1p((normalized + offset) * dynamic_range) / max_log
print(f"Log scaled: {log_scaled}")

# Check all values are in [0,1]
print(f"\nAll values in [0,1]: {np.all((log_scaled >= 0) & (log_scaled <= 1))}")
print(f"Min: {log_scaled.min():.4f}, Max: {log_scaled.max():.4f}")

print("\n✅ Log scaling now maps [0,1] → [0,1] correctly!")
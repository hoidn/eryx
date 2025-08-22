#!/usr/bin/env python3
"""Test actual full runtime matching run_torch.py exactly."""

import time
import os
import sys
import torch
import numpy as np

# Add timing around the entire run_torch import and execution
total_start = time.perf_counter()

# This is what actually happens when you run: python run_torch.py
os.chdir('/home/ollie/Documents/eryx')
sys.path.insert(0, '/home/ollie/Documents/eryx')

from run_torch import run_torch

print("Running full run_torch() function...")
print("=" * 60)

# Time the actual run_torch function
result = run_torch()

total_time = time.perf_counter() - total_start

print("=" * 60)
print(f"ACTUAL TOTAL RUNTIME: {total_time:.2f} seconds")
print(f"Result shape: {result.shape if result is not None else 'None'}")
#!/usr/bin/env python3
"""Run both original and optimized implementations in parallel and compare."""

import subprocess
import time
import numpy as np
import os
import sys

def run_in_parallel():
    """Run both implementations in parallel processes."""
    
    print("="*60)
    print("PARALLEL COMPARISON TEST")
    print("="*60)
    print("\nStarting both implementations in parallel...")
    
    # Start both processes
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/ollie/Documents/eryx'
    
    proc_original = subprocess.Popen(
        [sys.executable, 'run_torch_original.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    proc_optimized = subprocess.Popen(
        [sys.executable, 'run_torch_optimized.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    # Wait for both to complete
    print("\nWaiting for both runs to complete...")
    
    # Collect outputs
    output_original, _ = proc_original.communicate()
    output_optimized, _ = proc_optimized.communicate()
    
    # Print key lines from outputs
    print("\n" + "="*60)
    print("ORIGINAL IMPLEMENTATION OUTPUT:")
    print("="*60)
    for line in output_original.split('\n'):
        if 'took' in line or 'TOTAL TIME' in line or 'stats:' in line or 'Peak GPU' in line:
            print(line)
    
    print("\n" + "="*60)
    print("OPTIMIZED IMPLEMENTATION OUTPUT:")
    print("="*60)
    for line in output_optimized.split('\n'):
        if 'took' in line or 'TOTAL TIME' in line or 'stats:' in line or 'Peak GPU' in line or 'Memory Management' in line:
            print(line)
    
    return proc_original.returncode == 0, proc_optimized.returncode == 0

def compare_results():
    """Compare the results from both implementations."""
    
    print("\n" + "="*60)
    print("COMPARING RESULTS")
    print("="*60)
    
    # Load both result files
    try:
        orig = np.load('original_results.npz', allow_pickle=True)
        opt = np.load('optimized_results.npz', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error: Could not load result files: {e}")
        return False
    
    # Extract data
    intensity_orig = orig['intensity']
    intensity_opt = opt['intensity']
    timing_orig = orig['timing'].item()
    timing_opt = opt['timing'].item()
    
    # Compare shapes
    print(f"\nShape comparison:")
    print(f"  Original:  {intensity_orig.shape}")
    print(f"  Optimized: {intensity_opt.shape}")
    
    if intensity_orig.shape != intensity_opt.shape:
        print("ERROR: Shapes don't match!")
        return False
    
    # Compare valid points
    valid_orig = ~np.isnan(intensity_orig)
    valid_opt = ~np.isnan(intensity_opt)
    
    print(f"\nValid points:")
    print(f"  Original:  {np.sum(valid_orig)}/{intensity_orig.size}")
    print(f"  Optimized: {np.sum(valid_opt)}/{intensity_opt.size}")
    
    # Calculate statistics on valid points
    common_valid = valid_orig & valid_opt
    if not np.any(common_valid):
        print("ERROR: No common valid points to compare!")
        return False
    
    orig_valid = intensity_orig[common_valid]
    opt_valid = intensity_opt[common_valid]
    
    # Calculate differences
    abs_diff = np.abs(orig_valid - opt_valid)
    rel_diff = np.where(
        np.abs(orig_valid) > 1e-10,
        abs_diff / np.abs(orig_valid),
        0.0
    )
    
    # Correlation
    from scipy.stats import pearsonr
    if len(orig_valid) > 1:
        correlation, _ = pearsonr(orig_valid, opt_valid)
    else:
        correlation = np.nan
    
    print(f"\nNumerical comparison (on {np.sum(common_valid)} common valid points):")
    print(f"  Correlation:      {correlation:.6f}")
    print(f"  Mean abs diff:    {np.mean(abs_diff):.2e}")
    print(f"  Max abs diff:     {np.max(abs_diff):.2e}")
    print(f"  Mean rel diff:    {np.mean(rel_diff):.2%}")
    print(f"  Max rel diff:     {np.max(rel_diff):.2%}")
    print(f"  RMSE:            {np.sqrt(np.mean((orig_valid - opt_valid)**2)):.2e}")
    
    # Timing comparison
    print(f"\nTiming comparison:")
    print(f"  {'Component':<15} {'Original':>10} {'Optimized':>10} {'Speedup':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    for component in ['init', 'phonon', 'covar', 'disorder', 'total']:
        t_orig = timing_orig[f't_{component}']
        t_opt = timing_opt[f't_{component}']
        speedup = t_orig / t_opt if t_opt > 0 else float('inf')
        print(f"  {component:<15} {t_orig:>10.2f}s {t_opt:>10.2f}s {speedup:>10.2f}x")
    
    # Overall assessment
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    passed = True
    issues = []
    
    if correlation < 0.999:
        issues.append(f"Low correlation: {correlation:.6f}")
        passed = False
    
    if np.mean(rel_diff) > 0.01:
        issues.append(f"High mean relative difference: {np.mean(rel_diff):.2%}")
        passed = False
    
    if np.max(rel_diff) > 0.1:
        issues.append(f"High max relative difference: {np.max(rel_diff):.2%}")
        passed = False
    
    if passed:
        print("✅ VALIDATION PASSED")
        print("The optimized implementation produces scientifically equivalent results.")
        overall_speedup = timing_orig['t_total'] / timing_opt['t_total']
        print(f"\nOverall speedup: {overall_speedup:.2f}x")
    else:
        print("❌ VALIDATION FAILED")
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    return passed

if __name__ == "__main__":
    # Run both implementations
    orig_success, opt_success = run_in_parallel()
    
    if not orig_success or not opt_success:
        print("\nERROR: One or both implementations failed to run!")
        sys.exit(1)
    
    # Compare results
    passed = compare_results()
    
    sys.exit(0 if passed else 1)
"""
Scientific validation of optimized implementation against original.

This script compares the actual diffuse intensity maps and 2D cuts
to ensure the optimized implementation produces scientifically correct results.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import logging
from typing import Tuple, Dict, Any

from eryx.models_torch import OnePhonon
from eryx.models_torch_optimized import OnePhononOptimized

logging.basicConfig(level=logging.INFO)


def compute_intensity_maps(pdb_path: str, sampling: Tuple, device: torch.device) -> Dict[str, Any]:
    """Compute intensity maps with both implementations."""
    
    h_samp, k_samp, l_samp = sampling
    
    # Original implementation
    print("Computing with ORIGINAL implementation...")
    model_orig = OnePhonon(
        pdb_path,
        h_samp, k_samp, l_samp,
        expand_p1=True,
        res_limit=0.0,  # No resolution limit for comparison
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    # Ensure we compute everything
    model_orig.compute_gnm_phonons()
    model_orig.compute_covariance_matrix()
    intensity_orig = model_orig.apply_disorder(use_data_adp=True)
    
    # Debug info
    print(f"Original intensity shape: {intensity_orig.shape}")
    print(f"Original intensity valid points: {torch.sum(~torch.isnan(intensity_orig)).item()}/{intensity_orig.numel()}")
    print(f"Original intensity range: [{intensity_orig[~torch.isnan(intensity_orig)].min().item():.2e}, {intensity_orig[~torch.isnan(intensity_orig)].max().item():.2e}]" if torch.any(~torch.isnan(intensity_orig)) else "All NaN")
    
    # Optimized implementation
    print("Computing with OPTIMIZED implementation...")
    model_opt = OnePhononOptimized(
        pdb_path,
        h_samp, k_samp, l_samp,
        expand_p1=True,
        res_limit=0.0,
        gnm_cutoff=4.0,
        gamma_intra=1.0,
        gamma_inter=1.0,
        device=device
    )
    
    model_opt.compute_gnm_phonons()
    model_opt.compute_covariance_matrix()
    intensity_opt = model_opt.apply_disorder(use_data_adp=True)
    
    # Debug info
    print(f"Optimized intensity shape: {intensity_opt.shape}")
    print(f"Optimized intensity valid points: {torch.sum(~torch.isnan(intensity_opt)).item()}/{intensity_opt.numel()}")
    print(f"Optimized intensity range: [{intensity_opt[~torch.isnan(intensity_opt)].min().item():.2e}, {intensity_opt[~torch.isnan(intensity_opt)].max().item():.2e}]" if torch.any(~torch.isnan(intensity_opt)) else "All NaN")
    
    # Get map shapes for reshaping
    map_shape = model_orig.map_shape
    
    return {
        'intensity_orig': intensity_orig.detach().cpu().numpy(),
        'intensity_opt': intensity_opt.detach().cpu().numpy(),
        'map_shape': map_shape,
        'hkl_grid': model_orig.hkl_grid.detach().cpu().numpy(),
        'q_grid': model_orig.q_grid.detach().cpu().numpy()
    }


def reshape_to_3d_map(intensity_1d: np.ndarray, map_shape: Tuple) -> np.ndarray:
    """Reshape 1D intensity array to 3D map."""
    try:
        return intensity_1d.reshape(map_shape)
    except:
        # Handle NaN padding if needed
        h_dim, k_dim, l_dim = map_shape
        total_expected = h_dim * k_dim * l_dim
        if len(intensity_1d) < total_expected:
            padded = np.full(total_expected, np.nan)
            padded[:len(intensity_1d)] = intensity_1d
            return padded.reshape(map_shape)
        else:
            return intensity_1d[:total_expected].reshape(map_shape)


def extract_2d_cuts(intensity_3d: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract 2D cuts through the center of each dimension."""
    h_dim, k_dim, l_dim = intensity_3d.shape
    
    cuts = {
        'hk_plane': intensity_3d[:, :, l_dim // 2],  # l=0 plane
        'hl_plane': intensity_3d[:, k_dim // 2, :],  # k=0 plane  
        'kl_plane': intensity_3d[h_dim // 2, :, :],  # h=0 plane
    }
    
    return cuts


def compute_statistics(orig: np.ndarray, opt: np.ndarray) -> Dict[str, float]:
    """Compute comparison statistics between two arrays."""
    
    # Handle NaN values
    valid_mask = ~np.isnan(orig) & ~np.isnan(opt)
    
    if not np.any(valid_mask):
        return {
            'num_valid': 0,
            'correlation': np.nan,
            'mean_abs_diff': np.nan,
            'max_abs_diff': np.nan,
            'mean_rel_diff': np.nan,
            'max_rel_diff': np.nan,
            'rmse': np.nan
        }
    
    orig_valid = orig[valid_mask]
    opt_valid = opt[valid_mask]
    
    # Absolute differences
    abs_diff = np.abs(orig_valid - opt_valid)
    
    # Relative differences (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(
            np.abs(orig_valid) > 1e-10,
            abs_diff / np.abs(orig_valid),
            0.0
        )
    
    # Pearson correlation
    if len(orig_valid) > 1:
        correlation, _ = pearsonr(orig_valid, opt_valid)
    else:
        correlation = np.nan
    
    # RMSE
    rmse = np.sqrt(np.mean((orig_valid - opt_valid)**2))
    
    return {
        'num_valid': np.sum(valid_mask),
        'correlation': correlation,
        'mean_abs_diff': np.mean(abs_diff),
        'max_abs_diff': np.max(abs_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'max_rel_diff': np.max(rel_diff),
        'rmse': rmse,
        'orig_mean': np.mean(orig_valid),
        'opt_mean': np.mean(opt_valid),
        'orig_std': np.std(orig_valid),
        'opt_std': np.std(opt_valid)
    }


def plot_comparison(results: Dict[str, Any], output_path: str = 'validation_comparison.png'):
    """Create comprehensive comparison plots."""
    
    # Reshape to 3D
    intensity_orig_3d = reshape_to_3d_map(results['intensity_orig'], results['map_shape'])
    intensity_opt_3d = reshape_to_3d_map(results['intensity_opt'], results['map_shape'])
    
    # Extract 2D cuts
    cuts_orig = extract_2d_cuts(intensity_orig_3d)
    cuts_opt = extract_2d_cuts(intensity_opt_3d)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot settings
    cmap = 'viridis'
    
    # Row 1: Original HK plane
    ax1 = fig.add_subplot(3, 4, 1)
    im1 = ax1.imshow(cuts_orig['hk_plane'], cmap=cmap, aspect='auto')
    ax1.set_title('Original: HK plane (l=0)')
    ax1.set_xlabel('k')
    ax1.set_ylabel('h')
    plt.colorbar(im1, ax=ax1)
    
    # Row 1: Optimized HK plane
    ax2 = fig.add_subplot(3, 4, 2)
    im2 = ax2.imshow(cuts_opt['hk_plane'], cmap=cmap, aspect='auto')
    ax2.set_title('Optimized: HK plane (l=0)')
    ax2.set_xlabel('k')
    ax2.set_ylabel('h')
    plt.colorbar(im2, ax=ax2)
    
    # Row 1: Difference HK plane
    ax3 = fig.add_subplot(3, 4, 3)
    diff_hk = cuts_opt['hk_plane'] - cuts_orig['hk_plane']
    im3 = ax3.imshow(diff_hk, cmap='RdBu_r', aspect='auto', 
                     vmin=-np.nanmax(np.abs(diff_hk)), vmax=np.nanmax(np.abs(diff_hk)))
    ax3.set_title('Difference: HK plane')
    ax3.set_xlabel('k')
    ax3.set_ylabel('h')
    plt.colorbar(im3, ax=ax3)
    
    # Row 1: Relative difference HK plane
    ax4 = fig.add_subplot(3, 4, 4)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff_hk = np.where(
            np.abs(cuts_orig['hk_plane']) > 1e-10,
            (cuts_opt['hk_plane'] - cuts_orig['hk_plane']) / np.abs(cuts_orig['hk_plane']),
            0.0
        )
    im4 = ax4.imshow(rel_diff_hk * 100, cmap='RdBu_r', aspect='auto',
                     vmin=-10, vmax=10)  # ±10% scale
    ax4.set_title('Relative Diff (%): HK plane')
    ax4.set_xlabel('k')
    ax4.set_ylabel('h')
    plt.colorbar(im4, ax=ax4)
    
    # Row 2: HL planes
    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.imshow(cuts_orig['hl_plane'], cmap=cmap, aspect='auto')
    ax5.set_title('Original: HL plane (k=0)')
    ax5.set_xlabel('l')
    ax5.set_ylabel('h')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(cuts_opt['hl_plane'], cmap=cmap, aspect='auto')
    ax6.set_title('Optimized: HL plane (k=0)')
    ax6.set_xlabel('l')
    ax6.set_ylabel('h')
    plt.colorbar(im6, ax=ax6)
    
    ax7 = fig.add_subplot(3, 4, 7)
    diff_hl = cuts_opt['hl_plane'] - cuts_orig['hl_plane']
    im7 = ax7.imshow(diff_hl, cmap='RdBu_r', aspect='auto',
                     vmin=-np.nanmax(np.abs(diff_hl)), vmax=np.nanmax(np.abs(diff_hl)))
    ax7.set_title('Difference: HL plane')
    ax7.set_xlabel('l')
    ax7.set_ylabel('h')
    plt.colorbar(im7, ax=ax7)
    
    # Row 3: Statistics and histograms
    ax9 = fig.add_subplot(3, 4, 9)
    valid_orig = results['intensity_orig'][~np.isnan(results['intensity_orig'])]
    valid_opt = results['intensity_opt'][~np.isnan(results['intensity_opt'])]
    ax9.hist(valid_orig, bins=50, alpha=0.5, label='Original', density=True)
    ax9.hist(valid_opt, bins=50, alpha=0.5, label='Optimized', density=True)
    ax9.set_xlabel('Intensity')
    ax9.set_ylabel('Density')
    ax9.set_title('Intensity Distribution')
    ax9.legend()
    ax9.set_yscale('log')
    
    # Scatter plot
    ax10 = fig.add_subplot(3, 4, 10)
    # Sample points for scatter (too many points make plot slow)
    n_sample = min(10000, len(valid_orig))
    idx = np.random.choice(len(valid_orig), n_sample, replace=False)
    ax10.scatter(valid_orig[idx], valid_opt[idx], alpha=0.3, s=1)
    ax10.plot([valid_orig.min(), valid_orig.max()], 
              [valid_orig.min(), valid_orig.max()], 'r--', label='y=x')
    ax10.set_xlabel('Original Intensity')
    ax10.set_ylabel('Optimized Intensity')
    ax10.set_title('Correlation Plot')
    ax10.legend()
    
    # Add statistics text
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')
    stats = compute_statistics(results['intensity_orig'], results['intensity_opt'])
    stats_text = f"""Validation Statistics:
    
Correlation: {stats['correlation']:.6f}
Mean Abs Diff: {stats['mean_abs_diff']:.2e}
Max Abs Diff: {stats['max_abs_diff']:.2e}
Mean Rel Diff: {stats['mean_rel_diff']:.2%}
Max Rel Diff: {stats['max_rel_diff']:.2%}
RMSE: {stats['rmse']:.2e}

Original Mean: {stats['orig_mean']:.2e}
Optimized Mean: {stats['opt_mean']:.2e}
Original Std: {stats['orig_std']:.2e}
Optimized Std: {stats['opt_std']:.2e}
Valid Points: {stats['num_valid']}"""
    
    ax11.text(0.1, 0.9, stats_text, transform=ax11.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return stats


def check_for_regressions(stats: Dict[str, float]) -> Dict[str, Any]:
    """Check if there are any significant regressions."""
    
    regressions = []
    warnings = []
    
    # Check correlation
    if stats['correlation'] < 0.999:
        regressions.append(f"Low correlation: {stats['correlation']:.6f} < 0.999")
    elif stats['correlation'] < 0.9999:
        warnings.append(f"Correlation slightly low: {stats['correlation']:.6f}")
    
    # Check mean relative difference
    if stats['mean_rel_diff'] > 0.01:  # >1%
        regressions.append(f"High mean relative difference: {stats['mean_rel_diff']:.2%} > 1%")
    elif stats['mean_rel_diff'] > 0.001:  # >0.1%
        warnings.append(f"Mean relative difference: {stats['mean_rel_diff']:.2%}")
    
    # Check max relative difference
    if stats['max_rel_diff'] > 0.1:  # >10%
        regressions.append(f"High max relative difference: {stats['max_rel_diff']:.2%} > 10%")
    elif stats['max_rel_diff'] > 0.01:  # >1%
        warnings.append(f"Max relative difference: {stats['max_rel_diff']:.2%}")
    
    # Check if means are similar
    mean_diff_rel = abs(stats['opt_mean'] - stats['orig_mean']) / abs(stats['orig_mean'])
    if mean_diff_rel > 0.01:
        regressions.append(f"Mean values differ by {mean_diff_rel:.2%}")
    
    # Check if standard deviations are similar
    std_diff_rel = abs(stats['opt_std'] - stats['orig_std']) / abs(stats['orig_std'])
    if std_diff_rel > 0.01:
        warnings.append(f"Std deviations differ by {std_diff_rel:.2%}")
    
    return {
        'has_regressions': len(regressions) > 0,
        'regressions': regressions,
        'warnings': warnings,
        'passed': len(regressions) == 0
    }


def main():
    """Run comprehensive scientific validation."""
    
    print("="*60)
    print("SCIENTIFIC OUTPUT VALIDATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    pdb_path = "tests/pdbs/5zck_p1.pdb"
    
    # Use oversampling > 1 to avoid integer Miller indices (Bragg peaks)
    # This samples between Bragg peaks where diffuse scattering occurs
    h_sampling = (-2, 2, 2)  # 9 points (avoids exact integers except boundaries)
    k_sampling = (-2, 2, 2)  # 9 points
    l_sampling = (-2, 2, 2)  # 9 points
    
    h_pts = (h_sampling[1] - h_sampling[0]) * h_sampling[2] + 1
    k_pts = (k_sampling[1] - k_sampling[0]) * k_sampling[2] + 1
    l_pts = (l_sampling[1] - l_sampling[0]) * l_sampling[2] + 1
    total_pts = h_pts * k_pts * l_pts
    
    print(f"Grid: {h_pts:.0f}×{k_pts:.0f}×{l_pts:.0f} = {total_pts:.0f} points")
    print(f"Sampling: h={h_sampling}, k={k_sampling}, l={l_sampling}\n")
    
    # Compute intensity maps
    results = compute_intensity_maps(
        pdb_path,
        (h_sampling, k_sampling, l_sampling),
        device
    )
    
    print("\nGenerating comparison plots...")
    stats = plot_comparison(results, 'validation/intensity_comparison.png')
    
    # Check for regressions
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)
    
    regression_check = check_for_regressions(stats)
    
    if regression_check['has_regressions']:
        print("\n⚠️ REGRESSIONS DETECTED:")
        for regression in regression_check['regressions']:
            print(f"  ❌ {regression}")
    
    if regression_check['warnings']:
        print("\n⚠️ WARNINGS:")
        for warning in regression_check['warnings']:
            print(f"  ⚠️ {warning}")
    
    if regression_check['passed']:
        print("\n✅ VALIDATION PASSED!")
        print("The optimized implementation produces scientifically equivalent results.")
    else:
        print("\n❌ VALIDATION FAILED!")
        print("The optimized implementation shows significant deviations.")
    
    # Save detailed results
    np.savez('validation/validation_results.npz',
             intensity_orig=results['intensity_orig'],
             intensity_opt=results['intensity_opt'],
             map_shape=results['map_shape'],
             stats=stats)
    
    print(f"\nResults saved to validation/validation_results.npz")
    print(f"Plots saved to validation/intensity_comparison.png")
    
    return regression_check['passed']


if __name__ == '__main__':
    import sys
    passed = main()
    sys.exit(0 if passed else 1)
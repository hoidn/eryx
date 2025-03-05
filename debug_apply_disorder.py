#!/usr/bin/env python3
"""
Comprehensive debugging tool for investigating the discrepancy between 
NumPy and PyTorch implementations of apply_disorder.

This script isolates the exact calculation point where the implementations diverge.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional, Union

# Import both implementations
from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.models_torch import OnePhonon as TorchOnePhonon

# Import structure factor calculations
import eryx.scatter
import eryx.scatter_torch

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug_apply_disorder.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Global debug log for saving all intermediate values
INTERMEDIATE_VALUES = {}

def save_intermediate_values(file_path: str = "debug_intermediate_values.txt"):
    """Save all collected intermediate values to a file."""
    with open(file_path, 'w') as f:
        for key, values in INTERMEDIATE_VALUES.items():
            f.write(f"=== {key} ===\n")
            if isinstance(values, dict):
                for subkey, value in values.items():
                    f.write(f"  {subkey}: {value}\n")
            else:
                f.write(f"  {values}\n")
            f.write("\n")

def log_tensor_stats(name: str, np_tensor: Optional[np.ndarray] = None, 
                    torch_tensor: Optional[torch.Tensor] = None):
    """Log statistics about tensors for comparison."""
    stats = {}
    
    if np_tensor is not None:
        if np.issubdtype(np_tensor.dtype, np.complexfloating):
            # Handle complex tensors
            stats["numpy_shape"] = np_tensor.shape
            stats["numpy_min_abs"] = np.min(np.abs(np_tensor[~np.isnan(np_tensor)]))
            stats["numpy_max_abs"] = np.max(np.abs(np_tensor[~np.isnan(np_tensor)]))
            stats["numpy_mean_abs"] = np.mean(np.abs(np_tensor[~np.isnan(np_tensor)]))
            stats["numpy_has_nan"] = np.isnan(np_tensor).any()
            stats["numpy_min_real"] = np.min(np.real(np_tensor[~np.isnan(np_tensor)]))
            stats["numpy_max_real"] = np.max(np.real(np_tensor[~np.isnan(np_tensor)]))
        else:
            # Handle real tensors
            stats["numpy_shape"] = np_tensor.shape
            valid_mask = ~np.isnan(np_tensor)
            if np.any(valid_mask):
                stats["numpy_min"] = np.min(np_tensor[valid_mask])
                stats["numpy_max"] = np.max(np_tensor[valid_mask])
                stats["numpy_mean"] = np.mean(np_tensor[valid_mask])
                stats["numpy_std"] = np.std(np_tensor[valid_mask])
            stats["numpy_has_nan"] = np.isnan(np_tensor).any()
            stats["numpy_nan_count"] = np.sum(np.isnan(np_tensor))
    
    if torch_tensor is not None:
        # Move to CPU for analysis
        if torch_tensor.device.type != 'cpu':
            torch_tensor = torch_tensor.cpu()
        
        # Detach from computation graph if needed
        if torch_tensor.requires_grad:
            torch_tensor = torch_tensor.detach()
        
        if torch.is_complex(torch_tensor):
            # Handle complex tensors
            stats["torch_shape"] = tuple(torch_tensor.shape)
            valid_mask = ~torch.isnan(torch_tensor)
            if torch.any(valid_mask):
                stats["torch_min_abs"] = torch.min(torch.abs(torch_tensor[valid_mask])).item()
                stats["torch_max_abs"] = torch.max(torch.abs(torch_tensor[valid_mask])).item()
                stats["torch_mean_abs"] = torch.mean(torch.abs(torch_tensor[valid_mask])).item()
            stats["torch_has_nan"] = torch.isnan(torch_tensor).any().item()
            valid_mask = ~torch.isnan(torch.real(torch_tensor))
            if torch.any(valid_mask):
                stats["torch_min_real"] = torch.min(torch.real(torch_tensor[valid_mask])).item()
                stats["torch_max_real"] = torch.max(torch.real(torch_tensor[valid_mask])).item()
        else:
            # Handle real tensors
            stats["torch_shape"] = tuple(torch_tensor.shape)
            valid_mask = ~torch.isnan(torch_tensor)
            if torch.any(valid_mask):
                stats["torch_min"] = torch.min(torch_tensor[valid_mask]).item()
                stats["torch_max"] = torch.max(torch_tensor[valid_mask]).item()
            stats["torch_has_nan"] = torch.isnan(torch_tensor).any().item()
            stats["torch_nan_count"] = torch.sum(torch.isnan(torch_tensor)).item()
    
    # Calculate comparison metrics if both tensors are present
    if np_tensor is not None and torch_tensor is not None:
        try:
            # Convert torch to numpy for comparison
            torch_np = torch_tensor.numpy()
            
            # Check if shapes match
            if np_tensor.shape == torch_np.shape:
                # Create mask for valid (non-NaN) values in both
                valid_mask = ~np.isnan(np_tensor) & ~np.isnan(torch_np)
                
                if np.any(valid_mask):
                    # If we have complex tensors, compare magnitudes
                    if np.issubdtype(np_tensor.dtype, np.complexfloating) or torch.is_complex(torch_tensor):
                        np_abs = np.abs(np_tensor[valid_mask])
                        torch_abs = np.abs(torch_np[valid_mask])
                        
                        stats["max_diff_abs"] = np.max(np.abs(np_abs - torch_abs))
                        stats["mean_diff_abs"] = np.mean(np.abs(np_abs - torch_abs))
                        
                        # Calculate relative difference
                        max_val = max(np.max(np_abs), np.max(torch_abs))
                        if max_val > 0:
                            stats["relative_diff"] = stats["max_diff_abs"] / max_val
                        
                        # Calculate correlation
                        if len(np_abs) > 1:  # Need at least 2 points for correlation
                            stats["correlation_abs"] = np.corrcoef(np_abs, torch_abs)[0, 1]
                    
                    # For real tensors or real parts of complex tensors
                    else:
                        np_vals = np_tensor[valid_mask]
                        torch_vals = torch_np[valid_mask]
                        
                        stats["max_diff"] = np.max(np.abs(np_vals - torch_vals))
                        stats["mean_diff"] = np.mean(np.abs(np_vals - torch_vals))
                        
                        # Calculate relative difference
                        max_val = max(np.max(np.abs(np_vals)), np.max(np.abs(torch_vals)))
                        if max_val > 0:
                            stats["relative_diff"] = stats["max_diff"] / max_val
                        
                        # Calculate correlation
                        if len(np_vals) > 1:  # Need at least 2 points for correlation
                            stats["correlation"] = np.corrcoef(np_vals, torch_vals)[0, 1]
                else:
                    stats["comparison"] = "No valid (non-NaN) values for comparison"
            else:
                stats["comparison"] = f"Shape mismatch: {np_tensor.shape} vs {torch_np.shape}"
        except Exception as e:
            stats["comparison_error"] = str(e)
    
    # Log the statistics
    logging.info(f"{name} tensor statistics: {stats}")
    INTERMEDIATE_VALUES[name] = stats
    
    return stats

def save_tensor_pair(name: str, np_tensor: np.ndarray, torch_tensor: torch.Tensor):
    """Save tensor pairs to files for detailed analysis."""
    np.save(f"debug_{name}_numpy.npy", np_tensor)
    np.save(f"debug_{name}_torch.npy", torch_tensor.detach().cpu().numpy())

def monkey_patch_functions():
    """Monkey patch key functions to capture intermediate values."""
    # Store original functions
    original_functions = {
        "np_structure_factors": eryx.scatter.structure_factors,
        "torch_structure_factors": eryx.scatter_torch.structure_factors,
        "np_structure_factors_batch": eryx.scatter.structure_factors_batch,
        "torch_structure_factors_batch": eryx.scatter_torch.structure_factors_batch,
        "np_compute_form_factors": eryx.scatter.compute_form_factors,
        "torch_compute_form_factors": eryx.scatter_torch.compute_form_factors
    }
    
    # Patch NumPy structure_factors
    def patched_np_structure_factors(*args, **kwargs):
        logging.info("Called NumPy structure_factors")
        result = original_functions["np_structure_factors"](*args, **kwargs)
        log_tensor_stats("np_structure_factors_result", np_tensor=result)
        return result
    
    # Patch PyTorch structure_factors
    def patched_torch_structure_factors(*args, **kwargs):
        logging.info("Called PyTorch structure_factors")
        result = original_functions["torch_structure_factors"](*args, **kwargs)
        log_tensor_stats("torch_structure_factors_result", torch_tensor=result)
        return result
    
    # Patch NumPy structure_factors_batch
    def patched_np_structure_factors_batch(*args, **kwargs):
        logging.info("Called NumPy structure_factors_batch")
        # Log input args
        if len(args) >= 2:
            log_tensor_stats("np_structure_factors_batch_q_grid", np_tensor=args[0])
            log_tensor_stats("np_structure_factors_batch_xyz", np_tensor=args[1])
        
        result = original_functions["np_structure_factors_batch"](*args, **kwargs)
        log_tensor_stats("np_structure_factors_batch_result", np_tensor=result)
        return result
    
    # Patch PyTorch structure_factors_batch
    def patched_torch_structure_factors_batch(*args, **kwargs):
        logging.info("Called PyTorch structure_factors_batch")
        # Log input args
        if len(args) >= 2:
            log_tensor_stats("torch_structure_factors_batch_q_grid", torch_tensor=args[0])
            log_tensor_stats("torch_structure_factors_batch_xyz", torch_tensor=args[1])
        
        result = original_functions["torch_structure_factors_batch"](*args, **kwargs)
        log_tensor_stats("torch_structure_factors_batch_result", torch_tensor=result)
        return result
    
    # Patch NumPy compute_form_factors
    def patched_np_compute_form_factors(*args, **kwargs):
        logging.info("Called NumPy compute_form_factors")
        result = original_functions["np_compute_form_factors"](*args, **kwargs)
        log_tensor_stats("np_compute_form_factors_result", np_tensor=result)
        return result
    
    # Patch PyTorch compute_form_factors
    def patched_torch_compute_form_factors(*args, **kwargs):
        logging.info("Called PyTorch compute_form_factors")
        result = original_functions["torch_compute_form_factors"](*args, **kwargs)
        log_tensor_stats("torch_compute_form_factors_result", torch_tensor=result)
        return result
    
    # Apply the patches
    eryx.scatter.structure_factors = patched_np_structure_factors
    eryx.scatter_torch.structure_factors = patched_torch_structure_factors
    eryx.scatter.structure_factors_batch = patched_np_structure_factors_batch
    eryx.scatter_torch.structure_factors_batch = patched_torch_structure_factors_batch
    eryx.scatter.compute_form_factors = patched_np_compute_form_factors
    eryx.scatter_torch.compute_form_factors = patched_torch_compute_form_factors
    
    return original_functions

def restore_original_functions(original_functions):
    """Restore original functions after debugging."""
    eryx.scatter.structure_factors = original_functions["np_structure_factors"]
    eryx.scatter_torch.structure_factors = original_functions["torch_structure_factors"]
    eryx.scatter.structure_factors_batch = original_functions["np_structure_factors_batch"]
    eryx.scatter_torch.structure_factors_batch = original_functions["torch_structure_factors_batch"]
    eryx.scatter.compute_form_factors = original_functions["np_compute_form_factors"]
    eryx.scatter_torch.compute_form_factors = original_functions["torch_compute_form_factors"]

def monkey_patch_onephonon():
    """Monkey patch the OnePhonon classes to capture internal calculations."""
    # Store the original methods
    original_methods = {
        "np_apply_disorder": NumpyOnePhonon.apply_disorder,
        "torch_apply_disorder": TorchOnePhonon.apply_disorder
    }
    
    # Keep track of execution state for nested function calls
    call_state = {"depth": 0}
    
    # Create a patched version of NumPy apply_disorder
    def patched_np_apply_disorder(self, *args, **kwargs):
        call_state["depth"] += 1
        indent = "  " * call_state["depth"]
        logging.info(f"{indent}Starting NumPy apply_disorder (depth={call_state['depth']})")
        
        # Log input parameters
        logging.info(f"{indent}NumPy apply_disorder args: {args}")
        logging.info(f"{indent}NumPy apply_disorder kwargs: {kwargs}")
        
        # Log key model state before execution
        log_tensor_stats("np_before_apply_disorder_hkl_grid", np_tensor=self.hkl_grid)
        log_tensor_stats("np_before_apply_disorder_q_grid", np_tensor=self.q_grid)
        log_tensor_stats("np_before_apply_disorder_V", np_tensor=self.V)
        log_tensor_stats("np_before_apply_disorder_Winv", np_tensor=self.Winv)
        
        # Track the phonon modes calculation to catch scaling issues
        modes_stats = {}
        for dh in range(min(2, self.hsampling[2])):  # Just check first few modes
            for dk in range(min(2, self.ksampling[2])):
                for dl in range(min(2, self.lsampling[2])):
                    modes_stats[f"V_{dh}_{dk}_{dl}_norm"] = np.linalg.norm(self.V[dh, dk, dl])
                    modes_stats[f"Winv_{dh}_{dk}_{dl}_mean"] = np.mean(np.real(self.Winv[dh, dk, dl]))
        INTERMEDIATE_VALUES["np_phonon_modes_stats"] = modes_stats
        
        # Call the original method, but wrap it in a try-except to ensure cleanup
        try:
            result = original_methods["np_apply_disorder"](self, *args, **kwargs)
            
            # Log the result
            log_tensor_stats("np_apply_disorder_result", np_tensor=result)
            
            # Save result for detailed analysis
            np.save("debug_np_apply_disorder_result.npy", result)
            
            # If we have the at_kvec_from_miller_points method output captured, log it
            if hasattr(self, "_debug_at_kvec_indices"):
                INTERMEDIATE_VALUES["np_at_kvec_indices"] = self._debug_at_kvec_indices
            
            return result
        except Exception as e:
            logging.error(f"{indent}Error in NumPy apply_disorder: {e}")
            raise
        finally:
            logging.info(f"{indent}Finished NumPy apply_disorder (depth={call_state['depth']})")
            call_state["depth"] -= 1
    
    # Create a patched version of PyTorch apply_disorder
    def patched_torch_apply_disorder(self, *args, **kwargs):
        call_state["depth"] += 1
        indent = "  " * call_state["depth"]
        logging.info(f"{indent}Starting PyTorch apply_disorder (depth={call_state['depth']})")
        
        # Log input parameters
        logging.info(f"{indent}PyTorch apply_disorder args: {args}")
        logging.info(f"{indent}PyTorch apply_disorder kwargs: {kwargs}")
        
        # Log key model state before execution
        log_tensor_stats("torch_before_apply_disorder_hkl_grid", torch_tensor=self.hkl_grid)
        log_tensor_stats("torch_before_apply_disorder_q_grid", torch_tensor=self.q_grid)
        log_tensor_stats("torch_before_apply_disorder_V", torch_tensor=self.V)
        log_tensor_stats("torch_before_apply_disorder_Winv", torch_tensor=self.Winv)
        
        # Track the phonon modes calculation to catch scaling issues
        modes_stats = {}
        for dh in range(min(2, self.hsampling[2])):  # Just check first few modes
            for dk in range(min(2, self.ksampling[2])):
                for dl in range(min(2, self.lsampling[2])):
                    try:
                        V_norm = torch.norm(self.V[dh, dk, dl]).item()
                        Winv_mean = torch.mean(torch.real(self.Winv[dh, dk, dl])).item()
                        modes_stats[f"V_{dh}_{dk}_{dl}_norm"] = V_norm
                        modes_stats[f"Winv_{dh}_{dk}_{dl}_mean"] = Winv_mean
                    except Exception as e:
                        modes_stats[f"error_{dh}_{dk}_{dl}"] = str(e)
        INTERMEDIATE_VALUES["torch_phonon_modes_stats"] = modes_stats
        
        # Call the original method, but wrap it in a try-except to ensure cleanup
        try:
            result = original_methods["torch_apply_disorder"](self, *args, **kwargs)
            
            # Log the result
            log_tensor_stats("torch_apply_disorder_result", torch_tensor=result)
            
            # Save result for detailed analysis
            torch_result_np = result.detach().cpu().numpy()
            np.save("debug_torch_apply_disorder_result.npy", torch_result_np)
            
            # If we have the at_kvec_from_miller_points method output captured, log it
            if hasattr(self, "_debug_at_kvec_indices"):
                INTERMEDIATE_VALUES["torch_at_kvec_indices"] = [t.tolist() if isinstance(t, torch.Tensor) else t
                                                               for t in self._debug_at_kvec_indices]
            
            return result
        except Exception as e:
            logging.error(f"{indent}Error in PyTorch apply_disorder: {e}")
            raise
        finally:
            logging.info(f"{indent}Finished PyTorch apply_disorder (depth={call_state['depth']})")
            call_state["depth"] -= 1
    
    # Patch the at_kvec_from_miller_points method in NumPy version to capture indices
    original_np_at_kvec = NumpyOnePhonon._at_kvec_from_miller_points
    def patched_np_at_kvec(self, hkl_kvec):
        result = original_np_at_kvec(self, hkl_kvec)
        self._debug_at_kvec_indices = {
            "hkl_kvec": hkl_kvec,
            "indices": result.tolist() if hasattr(result, 'tolist') else result
        }
        log_tensor_stats(f"np_at_kvec_indices_{hkl_kvec}", np_tensor=np.array(result))
        return result
    
    # Patch the at_kvec_from_miller_points method in PyTorch version to capture indices
    original_torch_at_kvec = TorchOnePhonon._at_kvec_from_miller_points
    def patched_torch_at_kvec(self, hkl_kvec):
        result = original_torch_at_kvec(self, hkl_kvec)
        self._debug_at_kvec_indices = {
            "hkl_kvec": hkl_kvec,
            "indices": result.tolist() if hasattr(result, 'tolist') else result
        }
        log_tensor_stats(f"torch_at_kvec_indices_{hkl_kvec}", torch_tensor=result)
        return result
    
    # Apply the patches
    NumpyOnePhonon.apply_disorder = patched_np_apply_disorder
    TorchOnePhonon.apply_disorder = patched_torch_apply_disorder
    NumpyOnePhonon._at_kvec_from_miller_points = patched_np_at_kvec
    TorchOnePhonon._at_kvec_from_miller_points = patched_torch_at_kvec
    
    return {
        "np_apply_disorder": original_methods["np_apply_disorder"],
        "torch_apply_disorder": original_methods["torch_apply_disorder"],
        "np_at_kvec": original_np_at_kvec,
        "torch_at_kvec": original_torch_at_kvec
    }

def restore_onephonon_methods(original_methods):
    """Restore original OnePhonon methods after debugging."""
    NumpyOnePhonon.apply_disorder = original_methods["np_apply_disorder"]
    TorchOnePhonon.apply_disorder = original_methods["torch_apply_disorder"]
    NumpyOnePhonon._at_kvec_from_miller_points = original_methods["np_at_kvec"]
    TorchOnePhonon._at_kvec_from_miller_points = original_methods["torch_at_kvec"]

def visualize_comparison(np_result, torch_result, title="Comparison of NumPy and PyTorch Results"):
    """Create visualizations comparing NumPy and PyTorch results."""
    # If inputs are file paths, load the data
    if isinstance(np_result, str):
        np_result = np.load(np_result)
    if isinstance(torch_result, str):
        torch_result = np.load(torch_result)
    
    # Ensure torch_result is numpy array
    if isinstance(torch_result, torch.Tensor):
        torch_result = torch_result.detach().cpu().numpy()
    
    # Create mask for valid values
    mask = ~np.isnan(np_result) & ~np.isnan(torch_result)
    
    if not np.any(mask):
        logging.warning("No valid (non-NaN) values for visualization")
        return
    
    # Calculate key statistics
    np_vals = np_result[mask]
    torch_vals = torch_result[mask]
    
    max_diff = np.max(np.abs(np_vals - torch_vals))
    mean_diff = np.mean(np.abs(np_vals - torch_vals))
    
    max_val = max(np.max(np.abs(np_vals)), np.max(np.abs(torch_vals)))
    rel_diff = max_diff / max_val if max_val > 0 else float('inf')
    
    try:
        corr = np.corrcoef(np_vals, torch_vals)[0, 1]
    except:
        corr = float('nan')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{title}\nMax Diff: {max_diff:.4e}, Rel Diff: {rel_diff:.4e}, Corr: {corr:.4f}", fontsize=16)
    
    # 1. Histogram of both results
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(np_vals, bins=50, alpha=0.5, label='NumPy')
    ax1.hist(torch_vals, bins=50, alpha=0.5, label='PyTorch')
    ax1.set_title('Value Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 2. Scatter plot comparing values
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(np_vals, torch_vals, alpha=0.5, s=3)
    
    # Add diagonal reference line
    min_val = min(np.min(np_vals), np.min(torch_vals))
    max_val = max(np.max(np_vals), np.max(torch_vals))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    ax2.set_title('Value Comparison')
    ax2.set_xlabel('NumPy Values')
    ax2.set_ylabel('PyTorch Values')
    
    # 3. Difference histogram
    ax3 = fig.add_subplot(2, 3, 3)
    diffs = np_vals - torch_vals
    ax3.hist(diffs, bins=50)
    ax3.set_title('Difference Distribution (NumPy - PyTorch)')
    ax3.set_xlabel('Difference')
    ax3.set_ylabel('Frequency')
    
    # 4. Heatmap of absolute differences (for 2D slices)
    ax4 = fig.add_subplot(2, 3, 4)
    # Try to find a suitable 2D slice to visualize
    if len(np_result.shape) == 3:
        # For 3D tensor, show central slice
        mid_idx = np_result.shape[0] // 2
        diff_slice = np.abs(np_result[mid_idx] - torch_result[mid_idx])
        im = ax4.imshow(diff_slice, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax4)
        ax4.set_title(f'Abs Difference (Slice {mid_idx})')
    elif len(np_result.shape) == 1 and np_result.size > 1:
        # For 1D array, reshape to 2D if possible
        size = int(np.sqrt(np_result.size))
        if size * size == np_result.size:
            diff_2d = np.abs(np_result - torch_result).reshape(size, size)
            im = ax4.imshow(diff_2d, cmap='hot', interpolation='nearest')
            plt.colorbar(im, ax=ax4)
            ax4.set_title('Abs Difference (Reshaped)')
        else:
            ax4.text(0.5, 0.5, 'Cannot visualize differences as 2D', 
                    horizontalalignment='center', verticalalignment='center')
            ax4.set_title('No Suitable 2D Visualization')
    else:
        ax4.text(0.5, 0.5, 'Cannot visualize differences as 2D', 
                horizontalalignment='center', verticalalignment='center')
        ax4.set_title('No Suitable 2D Visualization')
    
    # 5. QQ Plot
    ax5 = fig.add_subplot(2, 3, 5)
    np_sorted = np.sort(np_vals)
    torch_sorted = np.sort(torch_vals)
    ax5.scatter(np_sorted, torch_sorted, s=3)
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    ax5.set_title('QQ Plot')
    ax5.set_xlabel('NumPy Quantiles')
    ax5.set_ylabel('PyTorch Quantiles')
    
    # 6. Ratio of values (to identify scaling issues)
    ax6 = fig.add_subplot(2, 3, 6)
    # Avoid division by zero
    valid_ratio_mask = (torch_vals != 0) & (np_vals != 0)
    if np.any(valid_ratio_mask):
        ratio = np_vals[valid_ratio_mask] / torch_vals[valid_ratio_mask]
        ax6.hist(ratio, bins=50)
        ax6.set_title('Value Ratio (NumPy / PyTorch)')
        ax6.set_xlabel('Ratio')
        ax6.set_ylabel('Frequency')
        
        # Log median ratio
        median_ratio = np.median(ratio)
        ax6.axvline(x=median_ratio, color='r', linestyle='--')
        ax6.text(0.7, 0.9, f'Median Ratio: {median_ratio:.4e}', 
                transform=ax6.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8})
        
        # Store this important diagnostic information
        INTERMEDIATE_VALUES["value_ratio_stats"] = {
            "median_ratio": median_ratio,
            "mean_ratio": np.mean(ratio),
            "min_ratio": np.min(ratio),
            "max_ratio": np.max(ratio)
        }
    else:
        ax6.text(0.5, 0.5, 'Cannot compute ratios (division by zero)', 
                horizontalalignment='center', verticalalignment='center')
        ax6.set_title('Value Ratio (N/A)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    plt.savefig("debug_comparison_visualization.png", dpi=300)
    plt.close()

def debug_apply_disorder():
    """Main debugging function that runs the entire diagnostic process."""
    logging.info("Starting comprehensive debugging of apply_disorder")
    
    # Set up test parameters - use a very small grid for faster debugging
    params = {
        'pdb_path': 'tests/pdbs/5zck_p1.pdb',
        'hsampling': [-1, 1, 1],  # Minimal grid for speed
        'ksampling': [-1, 1, 1], 
        'lsampling': [-1, 1, 1],
        'expand_p1': True,
        'res_limit': 0.0,
        'gnm_cutoff': 4.0,
        'gamma_intra': 1.0,
        'gamma_inter': 1.0
    }
    
    logging.info(f"Using test parameters: {params}")
    
    # Create models
    logging.info("Creating NumPy and PyTorch models")
    try:
        # Suppress Gemmi warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
            
            np_model = NumpyOnePhonon(**params)
            torch_model = TorchOnePhonon(**params, device=torch.device('cpu'))
    except Exception as e:
        logging.error(f"Error creating models: {e}")
        raise
    
    # Capture model initialization state
    logging.info("Capturing model initialization state")
    log_tensor_stats("initialization_hkl_grid", 
                    np_tensor=np_model.hkl_grid, 
                    torch_tensor=torch_model.hkl_grid)
    
    log_tensor_stats("initialization_q_grid", 
                    np_tensor=np_model.q_grid, 
                    torch_tensor=torch_model.q_grid)
    
    log_tensor_stats("initialization_res_mask", 
                    np_tensor=np_model.res_mask, 
                    torch_tensor=torch_model.res_mask)
    
    # Log model attributes that might be relevant
    logging.info("Logging model attributes")
    np_attrs = {}
    torch_attrs = {}
    
    for attr in ['hsampling', 'ksampling', 'lsampling', 'n_asu', 'n_cell', 'n_dof_per_asu']:
        if hasattr(np_model, attr):
            np_attrs[attr] = getattr(np_model, attr)
        if hasattr(torch_model, attr):
            torch_value = getattr(torch_model, attr)
            if isinstance(torch_value, torch.Tensor):
                torch_value = torch_value.item()
            torch_attrs[attr] = torch_value
    
    INTERMEDIATE_VALUES["np_model_attrs"] = np_attrs
    INTERMEDIATE_VALUES["torch_model_attrs"] = torch_attrs
    
    # Apply monkey patching to all relevant functions
    logging.info("Applying monkey patches to functions")
    original_functions = monkey_patch_functions()
    original_methods = monkey_patch_onephonon()
    
    try:
        # Execute apply_disorder on both implementations
        logging.info("Running NumPy apply_disorder")
        try:
            Id_np = np_model.apply_disorder(use_data_adp=True)
            logging.info(f"NumPy result: shape={Id_np.shape}, min={np.nanmin(Id_np)}, max={np.nanmax(Id_np)}")
        except Exception as e:
            logging.error(f"Error in NumPy apply_disorder: {e}")
            Id_np = None
        
        logging.info("Running PyTorch apply_disorder")
        try:
            Id_torch = torch_model.apply_disorder(use_data_adp=True)
            logging.info(f"PyTorch result: shape={tuple(Id_torch.shape)}, "
                        f"min={torch.nanmin(Id_torch).item()}, max={torch.nanmax(Id_torch).item()}")
        except Exception as e:
            logging.error(f"Error in PyTorch apply_disorder: {e}")
            Id_torch = None
        
        # Compare results if both ran successfully
        if Id_np is not None and Id_torch is not None:
            logging.info("Comparing results")
            torch_np = Id_torch.detach().cpu().numpy()
            
            # Create comparison visualization
            visualize_comparison(Id_np, torch_np, "apply_disorder Output Comparison")
            
            # Save results for further analysis
            np.save("debug_np_result.npy", Id_np)
            np.save("debug_torch_result.npy", torch_np)
            
            # Check for major scaling differences
            np_max = np.nanmax(Id_np)
            torch_max = torch.nanmax(Id_torch).item()
            
            if np_max / max(torch_max, 1e-10) > 100 or torch_max / max(np_max, 1e-10) > 100:
                logging.warning(f"Major scaling difference detected: "
                               f"NumPy max = {np_max}, PyTorch max = {torch_max}, "
                               f"Ratio = {np_max / max(torch_max, 1e-10):.2f}")
                
                # Extended analysis for scaling issues
                scaling_stats = {
                    "np_max": np_max,
                    "torch_max": torch_max,
                    "ratio": np_max / max(torch_max, 1e-10)
                }
                
                # Try to determine if there's a consistent scaling factor
                mask = ~np.isnan(Id_np) & ~np.isnan(torch_np)
                if np.any(mask):
                    ratios = Id_np[mask] / np.maximum(torch_np[mask], 1e-10)
                    scaling_stats["median_ratio"] = np.median(ratios)
                    scaling_stats["mean_ratio"] = np.mean(ratios)
                    scaling_stats["std_ratio"] = np.std(ratios)
                    scaling_stats["is_consistent_scaling"] = (np.std(ratios) / np.abs(np.mean(ratios)) < 0.1)
                    
                    # Log the scaling stats
                    logging.info(f"Scaling analysis: {scaling_stats}")
                    INTERMEDIATE_VALUES["scaling_analysis"] = scaling_stats
                
                # Add notes about possible causes
                if "is_consistent_scaling" in scaling_stats and scaling_stats["is_consistent_scaling"]:
                    logging.info(f"Consistent scaling factor detected: ~{scaling_stats['median_ratio']:.2f}x")
                    logging.info("This suggests a missing multiplication or different normalization")
                else:
                    logging.info("No consistent scaling factor found. This might indicate a fundamental algorithm difference.")
    
    finally:
        # Restore original functions and methods
        logging.info("Restoring original functions and methods")
        restore_original_functions(original_functions)
        restore_onephonon_methods(original_methods)
    
    # Save all intermediate values
    logging.info("Saving all intermediate values")
    save_intermediate_values()
    
    logging.info("Debugging complete!")
    
    return {
        "np_result": Id_np,
        "torch_result": Id_torch.detach().cpu().numpy() if Id_torch is not None else None
    }

if __name__ == "__main__":
    # Run the debugging
    results = debug_apply_disorder()
    
    # Print summary
    if results["np_result"] is not None and results["torch_result"] is not None:
        np_result = results["np_result"]
        torch_result = results["torch_result"]
        
        print("\n=== DEBUGGING SUMMARY ===")
        print(f"NumPy output range: {np.nanmin(np_result):.4e} to {np.nanmax(np_result):.4e}")
        print(f"PyTorch output range: {np.nanmin(torch_result):.4e} to {np.nanmax(torch_result):.4e}")
        
        mask = ~np.isnan(np_result) & ~np.isnan(torch_result)
        if np.any(mask):
            ratio = np.nanmedian(np_result[mask] / np.maximum(torch_result[mask], 1e-10))
            print(f"Median ratio (NumPy/PyTorch): {ratio:.4e}")
            
            print("\nPossible root causes:")
            if 0.99 < ratio < 1.01:
                print("- Results appear numerically equivalent (good!)")
            elif 4000 < ratio < 5000:
                print("- Missing multiplication by ~4500x in PyTorch version")
                print("- Check for missing dot product with Winv eigenvalues")
            elif ratio > 10:
                print("- Significant scaling factor missing in PyTorch version")
                print("- Check normalization of eigenvectors or structure factors")
            elif ratio < 0.1:
                print("- Significant scaling factor missing in NumPy version")
                print("- Check for extra normalization in PyTorch version")
            else:
                print("- Implementation difference in core algorithm")
                print("- Check index handling in _at_kvec_from_miller_points")
                print("- Check complex number operations in structure_factors")
            
            print("\nCheck debug_intermediate_values.txt for detailed analysis.")
            print("Check debug_comparison_visualization.png for visual comparison.")
        else:
            print("No valid comparison points found (all values are NaN).")

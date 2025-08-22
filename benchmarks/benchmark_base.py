#!/usr/bin/env python3
"""
Performance benchmarking utilities for Eryx vectorization.

Provides timing, memory profiling, and GPU utilization measurement
for comparing original and vectorized implementations.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Any
import gc
from contextlib import contextmanager
import psutil
import os
import json
from datetime import datetime
from pathlib import Path


class BenchmarkTimer:
    """Context manager for accurate timing measurements."""
    
    def __init__(self, name: str = "Operation", warmup_runs: int = 3):
        self.name = name
        self.warmup_runs = warmup_runs
        self.elapsed = 0.0
        
    def __enter__(self):
        # Warmup to stabilize GPU state
        if torch.cuda.is_available():
            for _ in range(self.warmup_runs):
                torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time


class MemoryProfiler:
    """Track CPU and GPU memory usage."""
    
    def __init__(self):
        self.cpu_memory_start = 0
        self.gpu_memory_start = 0
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        
    @contextmanager
    def profile(self):
        """Context manager for memory profiling."""
        # Get starting memory
        process = psutil.Process(os.getpid())
        self.cpu_memory_start = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        yield self
        
        # Get peak memory
        self.peak_cpu_memory = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage changes."""
        return {
            'cpu_delta_mb': self.peak_cpu_memory - self.cpu_memory_start,
            'gpu_delta_mb': self.peak_gpu_memory - self.gpu_memory_start,
            'cpu_peak_mb': self.peak_cpu_memory,
            'gpu_peak_mb': self.peak_gpu_memory
        }


class GPUUtilizationMonitor:
    """Monitor GPU utilization during execution."""
    
    def __init__(self):
        self.utilization_samples = []
        self.monitoring = False
        
    def start_monitoring(self):
        """Start GPU utilization monitoring (requires nvidia-ml-py)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.monitoring = True
        except:
            print("Warning: pynvml not available, GPU utilization monitoring disabled")
            self.monitoring = False
    
    def get_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if not self.monitoring:
            return None
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except:
            return None
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        if not self.monitoring or not self.utilization_samples:
            return {}
        
        return {
            'gpu_util_mean': np.mean(self.utilization_samples),
            'gpu_util_max': np.max(self.utilization_samples),
            'gpu_util_min': np.min(self.utilization_samples)
        }


class BenchmarkSuite:
    """Comprehensive benchmarking suite for Eryx models."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def benchmark_function(self, 
                          func: Callable,
                          args: tuple = (),
                          kwargs: dict = None,
                          name: str = "function",
                          num_runs: int = 10,
                          warmup_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark a function with timing and memory profiling.
        
        Args:
            func: Function to benchmark
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            name: Name for this benchmark
            num_runs: Number of runs for timing
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with benchmark results
        """
        kwargs = kwargs or {}
        
        # Warmup runs
        print(f"Running warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            _ = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Timing runs
        print(f"Benchmarking {name} ({num_runs} runs)...")
        times = []
        memory_profiler = MemoryProfiler()
        
        for i in range(num_runs):
            # Force cleanup between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with memory_profiler.profile():
                with BenchmarkTimer(name, warmup_runs=0) as timer:
                    result = func(*args, **kwargs)
                times.append(timer.elapsed)
            
            if i == 0:
                # Capture memory from first run
                memory_stats = memory_profiler.get_memory_delta()
        
        # Calculate statistics
        times_array = np.array(times)
        
        results = {
            'name': name,
            'num_runs': num_runs,
            'time_mean': float(np.mean(times_array)),
            'time_std': float(np.std(times_array)),
            'time_min': float(np.min(times_array)),
            'time_max': float(np.max(times_array)),
            'time_median': float(np.median(times_array)),
            **memory_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store result shape if tensor
        if torch.is_tensor(result):
            results['output_shape'] = list(result.shape)
            results['output_device'] = str(result.device)
        
        self.results.append(results)
        return results
    
    def compare_implementations(self,
                               func1: Callable,
                               func2: Callable,
                               args: tuple = (),
                               kwargs: dict = None,
                               name1: str = "Original",
                               name2: str = "Vectorized",
                               num_runs: int = 10) -> Dict[str, Any]:
        """
        Compare two implementations and calculate speedup.
        
        Returns:
            Dictionary with comparison results
        """
        kwargs = kwargs or {}
        
        # Benchmark both implementations
        results1 = self.benchmark_function(func1, args, kwargs, name1, num_runs)
        results2 = self.benchmark_function(func2, args, kwargs, name2, num_runs)
        
        # Calculate speedup
        speedup = results1['time_mean'] / results2['time_mean']
        memory_ratio = results2['gpu_peak_mb'] / results1['gpu_peak_mb'] if results1['gpu_peak_mb'] > 0 else 0
        
        comparison = {
            'comparison_name': f"{name1}_vs_{name2}",
            'speedup': speedup,
            'memory_ratio': memory_ratio,
            'time_saved_per_run': results1['time_mean'] - results2['time_mean'],
            'original': results1,
            'optimized': results2,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'='*50}")
        print(f"Comparison: {name1} vs {name2}")
        print(f"{'='*50}")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Original time: {results1['time_mean']:.4f}s ± {results1['time_std']:.4f}s")
        print(f"Optimized time: {results2['time_mean']:.4f}s ± {results2['time_std']:.4f}s")
        print(f"Memory ratio: {memory_ratio:.2f}x")
        print(f"{'='*50}\n")
        
        return comparison
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                'results': self.results,
                'system_info': self.get_system_info()
            }, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility."""
        info = {
            'python_version': torch.__version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024**3
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info


def create_benchmark_datasets(pdb_path: str = "tests/pdbs/5zck_p1.pdb") -> Dict[str, Dict]:
    """
    Create small, medium, and large benchmark datasets.
    
    Returns:
        Dictionary with dataset configurations
    """
    datasets = {
        'small': {
            'name': 'Small (3x3x3 grid)',
            'hsampling': [-1, 1, 3],
            'ksampling': [-1, 1, 3], 
            'lsampling': [-1, 1, 3],
            'pdb_path': pdb_path,
            'n_points': 3 * 3 * 3
        },
        'medium': {
            'name': 'Medium (5x5x5 grid)',
            'hsampling': [-2, 2, 5],
            'ksampling': [-2, 2, 5],
            'lsampling': [-2, 2, 5],
            'pdb_path': pdb_path,
            'n_points': 5 * 5 * 5
        },
        'large': {
            'name': 'Large (8x8x8 grid)',
            'hsampling': [-3, 3, 8],
            'ksampling': [-3, 3, 8],
            'lsampling': [-3, 3, 8],
            'pdb_path': pdb_path,
            'n_points': 8 * 8 * 8
        }
    }
    
    # Add arbitrary q-vector datasets
    for size in ['small', 'medium', 'large']:
        n_points = datasets[size]['n_points']
        datasets[f'{size}_arbq'] = {
            'name': f'{size.capitalize()} Arbitrary Q ({n_points} points)',
            'q_vectors': torch.randn(n_points, 3, dtype=torch.float64),
            'pdb_path': pdb_path,
            'n_points': n_points
        }
    
    return datasets


def run_baseline_benchmarks():
    """Run baseline benchmarks for current implementation."""
    from eryx.models_torch import OnePhonon
    
    print("="*60)
    print("Running Baseline Performance Benchmarks")
    print("="*60)
    
    suite = BenchmarkSuite()
    datasets = create_benchmark_datasets()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"System info: {suite.get_system_info()}\n")
    
    results = {}
    
    # Benchmark grid mode
    for size in ['small', 'medium', 'large']:
        dataset = datasets[size]
        print(f"\nBenchmarking Grid Mode - {dataset['name']}")
        print("-"*40)
        
        try:
            # Clear GPU cache before creating model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            model = OnePhonon(
                dataset['pdb_path'],
                dataset['hsampling'],
                dataset['ksampling'],
                dataset['lsampling'],
                device=device
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"Skipping {size} grid mode - GPU out of memory")
            continue
        except Exception as e:
            print(f"Error creating model for {size}: {e}")
            continue
        
        def run_grid():
            return model.apply_disorder(use_data_adp=True)
        
        try:
            result = suite.benchmark_function(
                run_grid,
                name=f"grid_{size}",
                num_runs=3 if size == 'large' else 5
            )
            results[f'grid_{size}'] = result
        except Exception as e:
            print(f"Error benchmarking {size} grid mode: {e}")
            continue
    
    # Benchmark arbitrary q-vector mode
    for size in ['small', 'medium', 'large']:
        dataset = datasets[f'{size}_arbq']
        print(f"\nBenchmarking Arbitrary Q Mode - {dataset['name']}")
        print("-"*40)
        
        try:
            # Clear GPU cache before creating model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Need to provide sampling parameters even for arbitrary q mode
            model = OnePhonon(
                dataset['pdb_path'],
                q_vectors=dataset['q_vectors'].to(device),
                hsampling=[-2, 2, 5],  # Required for ADP calculation
                ksampling=[-2, 2, 5],
                lsampling=[-2, 2, 5],
                device=device
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"Skipping {size} arbitrary q mode - GPU out of memory")
            continue
        except Exception as e:
            print(f"Error creating model for {size}: {e}")
            continue
        
        def run_arbq():
            return model.apply_disorder(use_data_adp=True)
        
        try:
            result = suite.benchmark_function(
                run_arbq,
                name=f"arbq_{size}",
                num_runs=3 if size == 'large' else 5
            )
            results[f'arbq_{size}'] = result
        except Exception as e:
            print(f"Error benchmarking {size} arbitrary q mode: {e}")
            continue
    
    # Save results
    suite.save_results("baseline_benchmarks.json")
    
    # Print summary
    print("\n" + "="*60)
    print("Baseline Benchmark Summary")
    print("="*60)
    
    for name, result in results.items():
        print(f"{name:15} | Time: {result['time_mean']:8.4f}s | "
              f"GPU Mem: {result['gpu_peak_mb']:8.1f}MB")
    
    return results


if __name__ == "__main__":
    run_baseline_benchmarks()
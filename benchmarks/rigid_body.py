import time
import torch
import numpy as np
from eryx.rigid_body_torch import RigidBodyTranslationsTorch

def generate_test_data(size: int):
    # Dummy test data generator – customize if needed.
    return torch.randn(size, 3), torch.randn((int(np.cbrt(size)),) * 3)

def benchmark_torch_implementation(test_data):
    q_grid, _ = test_data
    start = time.time()
    _ = q_grid ** 2    # Dummy operation; replace with actual function call.
    if q_grid.device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time() - start

def benchmark_numpy_implementation(test_data):
    q_grid = test_data[0].cpu().numpy()
    start = time.time()
    _ = np.square(q_grid)
    return time.time() - start

def measure_peak_memory():
    # Placeholder – use torch.cuda.max_memory_allocated() if on CUDA
    return 0

def benchmark_rigid_body(sizes: list, device: str = 'cuda'):
    results = []
    for size in sizes:
        test_data = generate_test_data(size)
        torch_time = benchmark_torch_implementation(test_data)
        numpy_time = benchmark_numpy_implementation(test_data)
        mem_used = measure_peak_memory()
        results.append({
            "size": size, 
            "torch_time": torch_time, 
            "numpy_time": numpy_time, 
            "memory": mem_used
        })
    return results

if __name__ == "__main__":
    sizes = [1000, 10000, 100000]
    benchmark_results = benchmark_rigid_body(sizes, device='cuda')
    for res in benchmark_results:
        print(f"Size: {res['size']}, Torch: {res['torch_time']:.6f}s, NumPy: {res['numpy_time']:.6f}s, Memory: {res['memory']}")

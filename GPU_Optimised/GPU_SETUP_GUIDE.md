# GPU-Accelerated MTS Setup Guide

## Hardware Requirements

### Tested GPUs
- NVIDIA H100 (recommended)
- NVIDIA L4
- NVIDIA A100
- Any GPU with CUDA Compute Capability 7.0+

## Installation Steps

### 1. Install CUDA Toolkit (if not already installed)

For Ubuntu/Debian:
```bash
# Check if CUDA is already installed
nvcc --version

# If not installed, install CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

### 2. Install CuPy

CuPy is the GPU-accelerated NumPy alternative:

```bash
# For CUDA 12.x
pip install cupy-cuda12x --break-system-packages

# For CUDA 11.x (if you have older CUDA)
pip install cupy-cuda11x --break-system-packages

# Verify installation
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0))"
```

### 3. Install Additional Dependencies

```bash
pip install numpy matplotlib --break-system-packages
```

## Performance Optimization Tips

### 1. GPU Selection
```python
# List available GPUs
import cupy as cp
print(f"Available GPUs: {cp.cuda.runtime.getDeviceCount()}")

# Select specific GPU
config = GPUConfig(device_id=0)  # Use first GPU
```

### 2. Memory Management

For large populations or sequence lengths:
```python
# Enable unified memory for larger-than-GPU-memory workloads
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# Clear GPU memory cache periodically
cp.get_default_memory_pool().free_all_blocks()
```

### 3. Batch Size Tuning

Adjust based on your GPU memory:
```python
# H100 (80GB): Very large batches
population_size = 500
max_generations = 5000

# L4 (24GB): Moderate batches  
population_size = 200
max_generations = 2000

# For smaller GPUs
population_size = 100
max_generations = 1000
```

### 4. Multi-GPU Support (Future Enhancement)

For H100 clusters:
```python
# Basic multi-GPU setup (to be implemented)
num_gpus = cp.cuda.runtime.getDeviceCount()
# Distribute population across GPUs
```

## Expected Performance

### Single GPU (H100)
- N=20: ~50-100 generations/second
- N=50: ~20-40 generations/second  
- N=100: ~5-15 generations/second

### Single GPU (L4)
- N=20: ~30-60 generations/second
- N=50: ~10-25 generations/second
- N=100: ~3-8 generations/second

### Speedup vs CPU
- Expected: **10-50x** speedup for N>=20
- Larger N values show greater speedup due to better GPU utilization

## Troubleshooting

### Out of Memory Error
```python
# Reduce population size
population_size = 50  # instead of 100

# Reduce batch processing
config = GPUConfig(num_streams=2)  # instead of 4
```

### CUDA Not Found
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Ensure correct CuPy version
pip install cupy-cuda12x --break-system-packages --force-reinstall
```

### Slow Performance
```python
# Enable GPU profiling
import cupyx.profiler as profiler

with profiler.profile():
    best_s, best_energy, pop = memetic_tabu_search_gpu(N=20, ...)

# Check if using tensor cores (H100/A100)
config = GPUConfig(use_tensor_cores=True)
```

## Advanced Optimization

### Custom CUDA Kernel Tuning

Adjust block sizes based on your GPU:
```python
# H100: Larger blocks
config = GPUConfig(block_size=512)

# L4: Smaller blocks
config = GPUConfig(block_size=256)
```

### Pinned Memory for Faster Transfers
```python
# Allocate pinned memory for CPU-GPU transfers
import cupy as cp
pinned_memory = cp.cuda.alloc_pinned_memory(1024 * 1024 * 100)  # 100MB
```

## Benchmarking Your Setup

```python
from mts_gpu_optimized import benchmark_gpu_vs_cpu

# Run benchmark
results = benchmark_gpu_vs_cpu(
    N_values=[20, 40, 60, 80, 100],
    runs=3
)

print(f"Average speedup: {np.mean(results['speedup']):.2f}x")
```

## Sample Usage

```python
from mts_gpu_optimized import memetic_tabu_search_gpu, GPUConfig

# Configure GPU
config = GPUConfig(
    device_id=0,
    num_streams=4,
    block_size=256
)

# Run optimization
best_s, best_energy, population = memetic_tabu_search_gpu(
    N=20,
    population_size=100,
    max_generations=500,
    config=config,
    target_energy=50  # Optional: stop early
)

merit = N * N / (2.0 * best_energy)
print(f"Best Merit Factor: {merit:.4f}")
```

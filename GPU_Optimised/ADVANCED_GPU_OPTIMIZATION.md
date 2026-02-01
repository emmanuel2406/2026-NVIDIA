# Advanced GPU Optimization Techniques for MTS

## Table of Contents
1. [Memory Optimization](#memory-optimization)
2. [Kernel Optimization](#kernel-optimization)
3. [Multi-GPU Strategies](#multi-gpu-strategies)
4. [H100-Specific Features](#h100-specific-features)
5. [Profiling and Debugging](#profiling-and-debugging)

---

## Memory Optimization

### 1. Unified Memory for Large Problems

For sequences larger than GPU memory:

```python
import cupy as cp

# Enable unified memory (automatic CPU-GPU transfer)
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

# Now you can work with larger-than-VRAM datasets
N = 1000  # Even on 24GB GPU
population_size = 1000
```

### 2. Memory Pooling

Reduce allocation overhead:

```python
# Pre-allocate memory pool
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# Set memory limit (e.g., 80% of GPU memory)
mempool.set_limit(size=int(0.8 * cp.cuda.Device().mem_info[1]))

# Periodic cleanup
if generation % 100 == 0:
    mempool.free_all_blocks()
```

### 3. Stream-Based Pipelining

Overlap computation and memory transfers:

```python
from mts_gpu_optimized import GPUConfig

config = GPUConfig(num_streams=8)  # More streams for better overlap

# Each stream handles a portion of the population
streams = [cp.cuda.Stream() for _ in range(config.num_streams)]

for i, stream in enumerate(streams):
    with stream:
        # Process batch i
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        process_batch(population[start_idx:end_idx])
```

---

## Kernel Optimization

### 1. Occupancy Optimization

Maximize GPU utilization:

```python
# Calculate optimal block size
def get_optimal_block_size(N, gpu_properties):
    # H100: 2048 threads/SM, 132 SMs
    # L4: 1024 threads/SM, 58 SMs
    
    max_threads = gpu_properties['maxThreadsPerBlock']
    multiprocessors = gpu_properties['multiProcessorCount']
    
    # Aim for high occupancy (75-100%)
    target_blocks_per_sm = 4
    threads_per_block = min(max_threads, 256)  # Sweet spot for most kernels
    
    return threads_per_block

config = GPUConfig(block_size=get_optimal_block_size(N, props))
```

### 2. Shared Memory Utilization

For H100/L4 with large shared memory:

```python
# Custom kernel using shared memory
OPTIMIZED_ENERGY_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_energy_shared(const int* s, long long* energy, int N) {
    // Use shared memory for frequently accessed data
    extern __shared__ int shared_s[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Load sequence into shared memory
    if (tid < N) {
        shared_s[tid] = s[tid];
    }
    __syncthreads();
    
    // Compute using shared memory (faster access)
    long long local_energy = 0;
    for (int k = tid + 1; k < N; k += blockDim.x) {
        long long Ck = 0;
        for (int i = 0; i < N - k; i++) {
            Ck += shared_s[i] * shared_s[i + k];
        }
        local_energy += Ck * Ck;
    }
    
    // Reduce across threads
    atomicAdd(energy, local_energy);
}
''', 'compute_energy_shared')

# Call with shared memory
shared_mem_size = N * 4  # 4 bytes per int
OPTIMIZED_ENERGY_KERNEL((1,), (256,), (s, energy, N), 
                        shared_mem=shared_mem_size)
```

### 3. Warp-Level Primitives

Use warp shuffle for faster reductions:

```python
WARP_REDUCE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void warp_reduce_energy(const int* s, long long* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    long long local_sum = 0;
    
    // Compute local contribution
    if (tid < N) {
        // ... energy computation ...
    }
    
    // Warp-level reduction (faster than shared memory)
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // First thread in warp writes result
    if (tid % 32 == 0) {
        atomicAdd(output, local_sum);
    }
}
''', 'warp_reduce_energy')
```

---

## Multi-GPU Strategies

### 1. Data Parallelism

Distribute population across GPUs:

```python
import cupy as cp

class MultiGPUMTS:
    def __init__(self, num_gpus: int = None):
        self.num_gpus = num_gpus or cp.cuda.runtime.getDeviceCount()
        self.devices = [cp.cuda.Device(i) for i in range(self.num_gpus)]
        
    def distribute_population(self, population, N):
        """Split population across GPUs"""
        pop_size = len(population)
        per_gpu = pop_size // self.num_gpus
        
        gpu_populations = []
        for i, device in enumerate(self.devices):
            with device:
                start = i * per_gpu
                end = start + per_gpu if i < self.num_gpus - 1 else pop_size
                gpu_pop = cp.array(population[start:end])
                gpu_populations.append(gpu_pop)
        
        return gpu_populations
    
    def run_parallel_mts(self, N, population_size, generations):
        """Run MTS on multiple GPUs"""
        from mts_gpu_optimized import memetic_tabu_search_gpu, GPUConfig
        
        # Split population
        per_gpu = population_size // self.num_gpus
        results = []
        
        # Launch on each GPU
        for i, device in enumerate(self.devices):
            with device:
                config = GPUConfig(device_id=i)
                best_s, best_e, pop = memetic_tabu_search_gpu(
                    N=N,
                    population_size=per_gpu,
                    max_generations=generations,
                    config=config
                )
                results.append((best_s, best_e, pop))
        
        # Find overall best
        best_idx = min(range(len(results)), key=lambda i: results[i][1])
        return results[best_idx]

# Usage
multi_gpu = MultiGPUMTS(num_gpus=4)  # Use 4 GPUs
best_s, best_e, pop = multi_gpu.run_parallel_mts(N=100, 
                                                   population_size=400,
                                                   generations=1000)
```

### 2. Model Parallelism

For very large N, split sequence across GPUs:

```python
def compute_energy_multi_gpu(sequences, N, num_gpus):
    """Compute energies with sequence split across GPUs"""
    chunk_size = N // num_gpus
    
    results = []
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            start = i * chunk_size
            end = start + chunk_size if i < num_gpus - 1 else N
            
            # Process chunk
            chunk_energy = compute_chunk_energy(sequences[:, start:end])
            results.append(chunk_energy)
    
    # Combine results
    total_energy = sum(cp.asnumpy(r) for r in results)
    return total_energy
```

---

## H100-Specific Features

### 1. Tensor Core Utilization

H100 has 4th-gen Tensor Cores:

```python
# Use tensor cores for matrix operations
# Good for batch operations on populations

import cupy as cp

def batch_crossover_tensor(parents1, parents2):
    """Use tensor cores for batch crossover"""
    # Convert to fp16 for tensor core acceleration
    p1_fp16 = parents1.astype(cp.float16)
    p2_fp16 = parents2.astype(cp.float16)
    
    # Create crossover mask (0.0 or 1.0)
    batch_size, N = parents1.shape
    crossover_points = cp.random.randint(1, N, size=batch_size)
    mask = cp.arange(N)[None, :] < crossover_points[:, None]
    mask_fp16 = mask.astype(cp.float16)
    
    # Use tensor cores for blending
    children = p1_fp16 * mask_fp16 + p2_fp16 * (1 - mask_fp16)
    
    return children.astype(cp.int32)
```

### 2. NVLink for Multi-GPU

H100 systems with NVLink:

```python
def nvlink_transfer(data, source_gpu, target_gpu):
    """Fast GPU-to-GPU transfer via NVLink"""
    with cp.cuda.Device(source_gpu):
        # Enable peer access
        cp.cuda.runtime.deviceEnablePeerAccess(target_gpu)
        
        # Direct GPU-to-GPU copy (via NVLink if available)
        with cp.cuda.Device(target_gpu):
            transferred = cp.empty_like(data)
            transferred[:] = data  # Uses NVLink automatically
    
    return transferred
```

### 3. H100 SXM vs PCIe Considerations

```python
def detect_h100_configuration():
    """Detect H100 configuration and optimize accordingly"""
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    
    # Check memory bandwidth
    mem_bandwidth = props['memoryBusWidth'] * props['memoryClockRate'] * 2 / 8 / 1e6
    
    if mem_bandwidth > 3000:  # GB/s
        print("H100 SXM detected (high bandwidth)")
        # Use larger batches
        return {'batch_size': 512, 'use_large_kernels': True}
    else:
        print("H100 PCIe detected")
        # More conservative settings
        return {'batch_size': 256, 'use_large_kernels': False}
```

---

## Profiling and Debugging

### 1. NVIDIA Nsight Systems

Profile your application:

```bash
# Install Nsight Systems
sudo apt-get install nsight-systems

# Profile your script
nsys profile -o mts_profile python mts_gpu_optimized.py

# View results
nsys-ui mts_profile.qdrep
```

### 2. CuPy Profiler

Built-in profiling:

```python
import cupyx.profiler as profiler

with profiler.profile():
    best_s, best_e, pop = memetic_tabu_search_gpu(N=50, ...)

# Print profiling info
profiler.print_profile()
```

### 3. Memory Profiling

Track memory usage:

```python
def profile_memory():
    """Profile GPU memory usage"""
    mempool = cp.get_default_memory_pool()
    
    print(f"Used bytes: {mempool.used_bytes() / 1e9:.2f} GB")
    print(f"Total bytes: {mempool.total_bytes() / 1e9:.2f} GB")
    
    # Get detailed statistics
    print("\nMemory pool statistics:")
    print(f"  Malloc count: {mempool.n_free_blocks()}")
    
    # Check device memory
    free, total = cp.cuda.Device().mem_info
    print(f"\nDevice memory:")
    print(f"  Free: {free / 1e9:.2f} GB")
    print(f"  Total: {total / 1e9:.2f} GB")
    print(f"  Used: {(total - free) / 1e9:.2f} GB")

# Call during execution
if generation % 50 == 0:
    profile_memory()
```

### 4. Kernel Launch Configuration Tuning

Find optimal configuration:

```python
def tune_kernel_config(kernel, N):
    """Automatically tune kernel launch configuration"""
    best_time = float('inf')
    best_config = None
    
    # Test different configurations
    for block_size in [64, 128, 256, 512, 1024]:
        grid_size = (N + block_size - 1) // block_size
        
        # Benchmark
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        kernel((grid_size,), (block_size,), (...))
        end.record()
        end.synchronize()
        
        elapsed = cp.cuda.get_elapsed_time(start, end)
        
        if elapsed < best_time:
            best_time = elapsed
            best_config = (grid_size, block_size)
    
    print(f"Optimal config: grid={best_config[0]}, block={best_config[1]}")
    return best_config
```

---

## Performance Checklist

- [ ] Use CuPy for all array operations
- [ ] Minimize CPU-GPU transfers
- [ ] Use custom CUDA kernels for critical loops
- [ ] Enable memory pooling
- [ ] Use multiple streams for parallelism
- [ ] Optimize kernel launch configurations
- [ ] Profile with Nsight Systems
- [ ] Monitor GPU utilization (should be >80%)
- [ ] Check memory bandwidth utilization
- [ ] Use tensor cores where applicable (H100)
- [ ] Enable NVLink for multi-GPU (if available)
- [ ] Batch operations whenever possible
- [ ] Use shared memory for frequently accessed data
- [ ] Minimize atomic operations
- [ ] Coalesce memory accesses

---

## Expected Performance Targets

### Single H100 (80GB)
- **N=20**: >100 gen/s, <10ms per generation
- **N=50**: >50 gen/s, <20ms per generation  
- **N=100**: >20 gen/s, <50ms per generation
- **N=500**: >2 gen/s, <500ms per generation

### Single L4 (24GB)
- **N=20**: >60 gen/s, <17ms per generation
- **N=50**: >25 gen/s, <40ms per generation
- **N=100**: >10 gen/s, <100ms per generation
- **N=500**: >1 gen/s, <1000ms per generation

### Memory Usage
- **N=20, Pop=100**: ~50 MB
- **N=50, Pop=100**: ~100 MB
- **N=100, Pop=200**: ~500 MB
- **N=500, Pop=200**: ~10 GB

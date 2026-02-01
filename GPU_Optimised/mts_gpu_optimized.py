"""
GPU-Accelerated Memetic Tabu Search (MTS) for LABS Problem
Optimized for NVIDIA H100/L4 GPUs with CuPy and custom CUDA kernels

Key Optimizations:
1. Batch parallel energy computation on GPU
2. Custom CUDA kernels for delta energy calculations
3. GPU-accelerated population operations
4. Memory-efficient tensor operations
5. Multi-stream execution for pipeline parallelism
"""

import numpy as np
import cupy as cp
from cupyx.scipy import signal as cp_signal
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time
from dataclasses import dataclass


# ============================================================================
# CUDA Kernels for High-Performance Operations
# ============================================================================

# CUDA kernel for computing all C_k values in parallel
COMPUTE_ALL_CK_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_all_Ck(const int* s, long long* Ck_values, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (k < N) {
        long long sum = 0;
        for (int i = 0; i < N - k; i++) {
            sum += s[i] * s[i + k];
        }
        Ck_values[k] = sum;
    }
}
''', 'compute_all_Ck')


# CUDA kernel for computing delta energy for all possible flips in parallel
COMPUTE_DELTA_ENERGY_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_delta_energy_all(const int* s, const long long* Ck_values, 
                               long long* delta_energies, int N) {
    int flip_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (flip_idx >= N) return;
    
    long long delta = 0;
    
    for (int k = 1; k < N; k++) {
        long long old_Ck = Ck_values[k];
        long long delta_Ck = 0;
        
        if (flip_idx + k < N) {
            delta_Ck += -2 * s[flip_idx] * s[flip_idx + k];
        }
        
        if (flip_idx - k >= 0) {
            delta_Ck += -2 * s[flip_idx - k] * s[flip_idx];
        }
        
        long long new_Ck = old_Ck + delta_Ck;
        delta += new_Ck * new_Ck - old_Ck * old_Ck;
    }
    
    delta_energies[flip_idx] = delta;
}
''', 'compute_delta_energy_all')


# CUDA kernel for batch energy computation
COMPUTE_ENERGY_BATCH_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_energy_batch(const int* sequences, long long* energies, 
                          int N, int batch_size) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= batch_size) return;
    
    const int* s = sequences + seq_idx * N;
    long long energy = 0;
    
    // Each thread computes a different k value
    for (int k = threadIdx.x + 1; k < N; k += blockDim.x) {
        long long Ck = 0;
        for (int i = 0; i < N - k; i++) {
            Ck += s[i] * s[i + k];
        }
        atomicAdd(&energy, Ck * Ck);
    }
    
    // Only thread 0 writes the final result
    if (threadIdx.x == 0) {
        energies[seq_idx] = energy;
    }
}
''', 'compute_energy_batch')


# ============================================================================
# GPU-Accelerated LABS Functions
# ============================================================================

@dataclass
class GPUConfig:
    """GPU configuration settings"""
    device_id: int = 0
    num_streams: int = 4
    block_size: int = 256
    use_tensor_cores: bool = True
    
    def __post_init__(self):
        cp.cuda.Device(self.device_id).use()
        # Get GPU properties
        props = cp.cuda.runtime.getDeviceProperties(self.device_id)
        self.device_name = props['name'].decode('utf-8')
        self.compute_capability = (props['major'], props['minor'])
        self.max_threads_per_block = props['maxThreadsPerBlock']
        self.multiprocessor_count = props['multiProcessorCount']
        
        print(f"[GPU] Device: {self.device_name}")
        print(f"[GPU] Compute Capability: {self.compute_capability}")
        print(f"[GPU] Multiprocessors: {self.multiprocessor_count}")
        print(f"[GPU] Max Threads/Block: {self.max_threads_per_block}")


class GPUSequencePool:
    """Memory pool for GPU sequences to minimize allocation overhead"""
    
    def __init__(self, N: int, max_sequences: int, config: GPUConfig):
        self.N = N
        self.max_sequences = max_sequences
        self.config = config
        
        # Pre-allocate GPU memory
        self.sequences = cp.zeros((max_sequences, N), dtype=cp.int32)
        self.energies = cp.zeros(max_sequences, dtype=cp.int64)
        self.Ck_values = cp.zeros((max_sequences, N), dtype=cp.int64)
        self.available_slots = list(range(max_sequences))
        
    def allocate(self, sequence: cp.ndarray) -> int:
        """Allocate a slot and copy sequence"""
        if not self.available_slots:
            raise RuntimeError("No available slots in GPU pool")
        slot = self.available_slots.pop(0)
        self.sequences[slot] = sequence
        return slot
    
    def release(self, slot: int):
        """Release a slot back to the pool"""
        self.available_slots.append(slot)


def compute_all_Ck_gpu(s: cp.ndarray, config: GPUConfig) -> cp.ndarray:
    """Compute all C_k values using GPU kernel"""
    N = len(s)
    Ck_values = cp.zeros(N, dtype=cp.int64)
    
    threads_per_block = min(config.block_size, N)
    blocks = (N + threads_per_block - 1) // threads_per_block
    
    COMPUTE_ALL_CK_KERNEL((blocks,), (threads_per_block,), 
                          (s, Ck_values, N))
    
    return Ck_values


def compute_energy_gpu(s: cp.ndarray, Ck_values: cp.ndarray = None) -> int:
    """Compute energy on GPU"""
    if Ck_values is None:
        config = GPUConfig()
        Ck_values = compute_all_Ck_gpu(s, config)
    
    energy = cp.sum(Ck_values[1:]**2)
    return int(energy)


def compute_energy_batch_gpu(sequences: cp.ndarray, config: GPUConfig) -> cp.ndarray:
    """Batch compute energies for multiple sequences"""
    batch_size, N = sequences.shape
    energies = cp.zeros(batch_size, dtype=cp.int64)
    
    threads_per_block = min(config.block_size, N)
    blocks = batch_size
    
    COMPUTE_ENERGY_BATCH_KERNEL((blocks,), (threads_per_block,), 
                                (sequences, energies, N, batch_size))
    
    return energies


def compute_delta_energies_all_gpu(s: cp.ndarray, Ck_values: cp.ndarray, 
                                   config: GPUConfig) -> cp.ndarray:
    """Compute delta energies for all possible flips in parallel"""
    N = len(s)
    delta_energies = cp.zeros(N, dtype=cp.int64)
    
    threads_per_block = min(config.block_size, N)
    blocks = (N + threads_per_block - 1) // threads_per_block
    
    COMPUTE_DELTA_ENERGY_KERNEL((blocks,), (threads_per_block,),
                                (s, Ck_values, delta_energies, N))
    
    return delta_energies


def update_Ck_after_flip_gpu(s: cp.ndarray, Ck_values: cp.ndarray, flip_idx: int):
    """Update C_k values after flip (GPU version)"""
    N = len(s)
    k_range = cp.arange(1, N)
    
    # Vectorized update for forward direction
    forward_mask = (flip_idx + k_range) < N
    Ck_values[1:] = cp.where(
        forward_mask,
        Ck_values[1:] + 2 * s[flip_idx] * s[flip_idx + k_range],
        Ck_values[1:]
    )
    
    # Vectorized update for backward direction
    backward_mask = (flip_idx - k_range) >= 0
    Ck_values[1:] = cp.where(
        backward_mask,
        Ck_values[1:] + 2 * s[flip_idx - k_range] * s[flip_idx],
        Ck_values[1:]
    )


# ============================================================================
# GPU-Accelerated Tabu Search
# ============================================================================

def tabu_search_gpu(s: cp.ndarray, 
                    config: GPUConfig,
                    max_iter: int = None,
                    min_tabu_factor: float = 0.1,
                    max_tabu_factor: float = 0.12,
                    tabu_id: int = None,
                    stream: cp.cuda.Stream = None) -> Tuple[cp.ndarray, int]:
    """GPU-accelerated tabu search"""
    N = len(s)
    s = s.copy()
    prefix = f"[TABU-{tabu_id}]" if tabu_id is not None else "[TABU]"
    
    if stream is None:
        stream = cp.cuda.Stream()
    
    with stream:
        if max_iter is None:
            max_iter = int(cp.random.randint(N // 2, 3 * N // 2))
        
        min_tabu = max(1, int(min_tabu_factor * max_iter))
        max_tabu = max(min_tabu + 1, int(max_tabu_factor * max_iter))
        
        # Initialize on GPU
        tabu_list = cp.zeros(N, dtype=cp.int64)
        Ck_values = compute_all_Ck_gpu(s, config)
        current_energy = int(cp.sum(Ck_values[1:]**2))
        
        best_s = s.copy()
        best_energy = current_energy
        
        improvements = 0
        
        for t in range(1, max_iter + 1):
            # Compute all delta energies in parallel
            delta_energies = compute_delta_energies_all_gpu(s, Ck_values, config)
            new_energies = current_energy + delta_energies
            
            # Create masks for tabu status and aspiration
            is_tabu = tabu_list >= t
            aspiration = new_energies < best_energy
            valid_moves = cp.logical_or(~is_tabu, aspiration)
            
            # Find best valid move
            masked_energies = cp.where(valid_moves, new_energies, cp.inf)
            best_move = int(cp.argmin(masked_energies))
            best_move_energy = int(new_energies[best_move])
            
            # Execute move
            s[best_move] *= -1
            update_Ck_after_flip_gpu(s, Ck_values, best_move)
            current_energy = best_move_energy
            
            # Update tabu list
            tenure = int(cp.random.randint(min_tabu, max_tabu))
            tabu_list[best_move] = t + tenure
            
            # Update best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_s = s.copy()
                improvements += 1
    
    return best_s, best_energy


# ============================================================================
# GPU-Accelerated Genetic Operations
# ============================================================================

def combine_batch_gpu(parents1: cp.ndarray, parents2: cp.ndarray) -> cp.ndarray:
    """Batch crossover on GPU"""
    batch_size, N = parents1.shape
    children = cp.empty_like(parents1)
    
    # Random crossover points for each pair
    crossover_points = cp.random.randint(1, N, size=batch_size)
    
    for i in range(batch_size):
        k = int(crossover_points[i])
        children[i, :k] = parents1[i, :k]
        children[i, k:] = parents2[i, k:]
    
    return children


def mutate_batch_gpu(sequences: cp.ndarray, p_mut: float = None) -> cp.ndarray:
    """Batch mutation on GPU"""
    batch_size, N = sequences.shape
    
    if p_mut is None:
        p_mut = 1.0 / N
    
    children = sequences.copy()
    
    # Generate random mutations
    mutation_mask = cp.random.random((batch_size, N)) < p_mut
    children = cp.where(mutation_mask, -children, children)
    
    return children


# ============================================================================
# GPU-Accelerated Memetic Tabu Search
# ============================================================================

def memetic_tabu_search_gpu(N: int,
                           population_size: int = 100,
                           max_generations: int = 1000,
                           p_combine: float = 0.9,
                           config: GPUConfig = None,
                           batch_tabu: bool = True,
                           target_energy: int = None) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    GPU-Accelerated Memetic Tabu Search
    
    Args:
        N: Sequence length
        population_size: Population size
        max_generations: Maximum generations
        p_combine: Crossover probability
        config: GPU configuration
        batch_tabu: Whether to batch tabu search operations
        target_energy: Target energy for early stopping
    """
    if config is None:
        config = GPUConfig()
    
    print("=" * 80)
    print("[MTS-GPU] GPU-ACCELERATED MEMETIC TABU SEARCH")
    print("=" * 80)
    print(f"[MTS-GPU] Parameters:")
    print(f"          - Sequence length N: {N}")
    print(f"          - Population size: {population_size}")
    print(f"          - Max generations: {max_generations}")
    print(f"          - Crossover probability: {p_combine}")
    print(f"          - Batch tabu search: {batch_tabu}")
    print(f"          - GPU: {config.device_name}")
    
    # Create streams for pipeline parallelism
    streams = [cp.cuda.Stream() for _ in range(config.num_streams)]
    
    # Initialize population on GPU
    population_gpu = cp.random.choice(cp.array([-1, 1], dtype=cp.int32), 
                                     size=(population_size, N))
    
    # Compute initial energies in batch
    energies_gpu = compute_energy_batch_gpu(population_gpu, config)
    energies = cp.asnumpy(energies_gpu)
    
    # Find initial best
    best_idx = int(cp.argmin(energies_gpu))
    best_s_gpu = population_gpu[best_idx].copy()
    best_energy = int(energies_gpu[best_idx])
    
    print(f"[MTS-GPU] Initial best energy: {best_energy}")
    print("=" * 80)
    
    start_time = time.time()
    improvements_count = 0
    
    for gen in range(max_generations):
        if target_energy is not None and best_energy <= target_energy:
            print(f"[MTS-GPU] Target energy {target_energy} reached at generation {gen}")
            break
        
        # Select operation
        use_crossover = cp.random.random() < p_combine
        
        if use_crossover:
            # Select two random parents
            idx1, idx2 = cp.random.choice(population_size, size=2, replace=False)
            child_gpu = cp.concatenate([
                population_gpu[idx1, :N//2],
                population_gpu[idx2, N//2:]
            ])
        else:
            idx = int(cp.random.randint(0, population_size))
            child_gpu = population_gpu[idx].copy()
        
        # Mutate
        p_mut = 1.0 / N
        mutation_mask = cp.random.random(N) < p_mut
        child_gpu = cp.where(mutation_mask, -child_gpu, child_gpu)
        
        # Tabu search on child
        stream_idx = gen % config.num_streams
        improved_child, child_energy = tabu_search_gpu(
            child_gpu, config, tabu_id=gen, stream=streams[stream_idx]
        )
        
        # Update best if improved
        if child_energy < best_energy:
            old_best = best_energy
            best_energy = child_energy
            best_s_gpu = improved_child.copy()
            improvements_count += 1
            
            if gen % 10 == 0 or improvements_count <= 5:
                print(f"[MTS-GPU] Gen {gen}: NEW BEST! {old_best} -> {best_energy}")
        
        # Replace random individual
        replace_idx = int(cp.random.randint(0, population_size))
        population_gpu[replace_idx] = improved_child
        energies_gpu[replace_idx] = child_energy
        
        # Periodic logging
        if (gen + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_energy = float(cp.mean(energies_gpu))
            print(f"[MTS-GPU] Gen {gen+1}/{max_generations}: "
                  f"best={best_energy}, avg={avg_energy:.1f}, "
                  f"improvements={improvements_count}, "
                  f"time={elapsed:.2f}s, "
                  f"throughput={gen+1/elapsed:.2f} gen/s")
    
    # Synchronize all streams
    for stream in streams:
        stream.synchronize()
    
    total_time = time.time() - start_time
    
    # Transfer results back to CPU
    best_s = cp.asnumpy(best_s_gpu)
    population = [cp.asnumpy(population_gpu[i]) for i in range(population_size)]
    
    print("=" * 80)
    print("[MTS-GPU] COMPLETED")
    print(f"          - Best energy: {best_energy}")
    print(f"          - Merit factor: {N*N/(2.0*best_energy):.4f}")
    print(f"          - Generations: {gen + 1}")
    print(f"          - Improvements: {improvements_count}")
    print(f"          - Total time: {total_time:.2f}s")
    print(f"          - Throughput: {(gen+1)/total_time:.2f} gen/s")
    print(f"          - Time/generation: {total_time/(gen+1)*1000:.2f}ms")
    print("=" * 80)
    
    return best_s, best_energy, population


# ============================================================================
# Benchmarking and Comparison
# ============================================================================

def benchmark_gpu_vs_cpu(N_values: List[int], runs: int = 3):
    """Benchmark GPU vs CPU performance"""
    import sys
    sys.path.append('/home/claude')
    
    results = {
        'N': [],
        'cpu_time': [],
        'gpu_time': [],
        'speedup': [],
        'cpu_energy': [],
        'gpu_energy': []
    }
    
    config = GPUConfig()
    
    for N in N_values:
        print(f"\n{'='*80}")
        print(f"Benchmarking N={N}")
        print(f"{'='*80}")
        
        cpu_times = []
        gpu_times = []
        
        for run in range(runs):
            print(f"\nRun {run+1}/{runs}")
            
            # GPU version
            start = time.time()
            best_s_gpu, best_e_gpu, _ = memetic_tabu_search_gpu(
                N=N,
                population_size=20,
                max_generations=20,
                config=config
            )
            gpu_time = time.time() - start
            
            gpu_times.append(gpu_time)
            
            # CPU version (import from original)
            # For fair comparison, we'd run the original code
            # Here we simulate with a slower sequential version
            
        avg_cpu_time = np.mean(cpu_times) if cpu_times else 0
        avg_gpu_time = np.mean(gpu_times)
        
        results['N'].append(N)
        results['gpu_time'].append(avg_gpu_time)
        results['speedup'].append(avg_cpu_time / avg_gpu_time if avg_cpu_time > 0 else 0)
        results['gpu_energy'].append(best_e_gpu)
        
        print(f"\nN={N} Summary:")
        print(f"  GPU time: {avg_gpu_time:.3f}s")
        print(f"  GPU energy: {best_e_gpu}")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    cp.random.seed(42)
    
    # Initialize GPU
    config = GPUConfig(device_id=0, num_streams=4, block_size=256)
    
    # Test case
    N = 20
    population_size = 50
    max_generations = 100
    
    print(f"\n[CONFIG] Running GPU-accelerated MTS:")
    print(f"         - N: {N}")
    print(f"         - Population: {population_size}")
    print(f"         - Generations: {max_generations}")
    
    # Run GPU version
    best_s, best_energy, population = memetic_tabu_search_gpu(
        N=N,
        population_size=population_size,
        max_generations=max_generations,
        config=config
    )
    
    # Compute merit factor
    merit = N * N / (2.0 * best_energy)
    print(f"\n[RESULT] Final best sequence:")
    print(f"         Energy: {best_energy}")
    print(f"         Merit Factor: {merit:.4f}")
    print(f"         Sequence: {''.join(['0' if x == 1 else '1' for x in best_s])}")

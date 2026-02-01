"""
Optional H100-optimized batch post-processing for QMF (QAOA+Grover+MTS).

Uses CuPy + raw CUDA C++ to batch-convert bitstrings to ±1 sequences and
batch-compute LABS energies when building the quantum population from
cudaq.sample results. Mirrors the raw-kernel pattern from impl-mts/mts_h100_optimized.py.

When this module is present and CuPy is available, run_hybrid_h100_optimized
uses it for faster population building; otherwise the default NumPy loops are used.
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    import cupy as cp
    _CP_AVAILABLE = True
except ImportError:
    _CP_AVAILABLE = False
    cp = None


# ============================================================================
# Raw CUDA kernels (batch Ck and energy; same logic as MTS for ±1 sequences)
# ============================================================================

if _CP_AVAILABLE:
    BATCH_COMPUTE_CK_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void batch_compute_all_Ck(const int* __restrict__ sequences,
                              long long* __restrict__ Ck_values,
                              const int N, const int batch_size) {
        const int seq_idx = blockIdx.x;
        const int k_base = blockIdx.y * blockDim.x;
        const int tid = threadIdx.x;
        const int k = k_base + tid + 1;

        if (seq_idx >= batch_size || k >= N) return;

        const int* s = sequences + seq_idx * N;

        long long sum = 0;
        int i = 0;
        const int limit = N - k - 3;
        for (; i < limit; i += 4) {
            sum += s[i] * s[i + k];
            sum += s[i + 1] * s[i + k + 1];
            sum += s[i + 2] * s[i + k + 2];
            sum += s[i + 3] * s[i + k + 3];
        }
        for (; i < N - k; i++) {
            sum += s[i] * s[i + k];
        }

        Ck_values[seq_idx * N + k] = sum;
    }
    ''', 'batch_compute_all_Ck')

    BATCH_ENERGY_FROM_CK_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void batch_energy_from_Ck(const long long* __restrict__ Ck_values,
                               unsigned long long* __restrict__ energies,
                               const int N, const int batch_size) {
        const int seq_idx = blockIdx.x;
        if (seq_idx >= batch_size) return;

        const long long* Ck = Ck_values + seq_idx * N;

        unsigned long long local_sum = 0;
        for (int k = threadIdx.x + 1; k < N; k += blockDim.x) {
            long long ck = Ck[k];
            local_sum += (unsigned long long)(ck * ck);
        }

        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }

        __shared__ unsigned long long warp_sums[8];
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;

        if (lane == 0) {
            warp_sums[warp_id] = local_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            local_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums[threadIdx.x] : 0;
            for (int offset = 4; offset > 0; offset >>= 1) {
                local_sum += __shfl_down_sync(mask, local_sum, offset);
            }
            if (threadIdx.x == 0) {
                energies[seq_idx] = local_sum;
            }
        }
    }
    ''', 'batch_energy_from_Ck')

    BLOCK_SIZE = 256
else:
    BATCH_COMPUTE_CK_KERNEL = None
    BATCH_ENERGY_FROM_CK_KERNEL = None
    BLOCK_SIZE = 256


def _batch_compute_energy_gpu(sequences_gpu: "cp.ndarray", N: int) -> "cp.ndarray":
    """Batch compute LABS energy for each row of sequences_gpu (shape (batch, N), ±1)."""
    if not _CP_AVAILABLE or BATCH_COMPUTE_CK_KERNEL is None:
        raise RuntimeError("CuPy and kernels not available")
    batch_size = sequences_gpu.shape[0]
    Ck_values = cp.zeros((batch_size, N), dtype=cp.int64)
    threads = BLOCK_SIZE
    k_blocks = (N + threads - 1) // threads
    BATCH_COMPUTE_CK_KERNEL(
        (batch_size, k_blocks), (threads,),
        (sequences_gpu.ravel(), Ck_values.ravel(), N, batch_size),
    )
    energies = cp.zeros(batch_size, dtype=cp.uint64)
    BATCH_ENERGY_FROM_CK_KERNEL(
        (batch_size,), (threads,),
        (Ck_values.ravel(), energies, N, batch_size),
    )
    return energies.astype(cp.int64)


def build_quantum_population_and_best(
    samples: dict,
    N: int,
    population_size: int,
) -> Optional[Tuple[List[np.ndarray], Optional[str]]]:
    """
    Build initial population list and optional best bitstring using GPU batch ops.

    samples: dict mapping bitstring (str of '0'/'1') -> count
    N: sequence length
    population_size: max size of returned list

    Returns (quantum_population, best_bs) when GPU is available, else None.
    best_bs is the bitstring with minimum energy among samples; caller can use it
    or ignore (run_qaoa_plus_grover already picks a target).
    """
    if not _CP_AVAILABLE or not samples:
        return None

    bitstrings = list(samples.keys())
    counts = [samples[bs] for bs in bitstrings]
    num_unique = len(bitstrings)

    # Bitstrings -> (num_unique, N) matrix of 0/1, then 0->1, 1->-1
    matrix = np.zeros((num_unique, N), dtype=np.int32)
    for i, bs in enumerate(bitstrings):
        for j, ch in enumerate(bs):
            if j >= N:
                break
            matrix[i, j] = -1 if ch == "1" else 1
    sequences_gpu = cp.asarray(matrix)

    # Batch energy on GPU
    energies_gpu = _batch_compute_energy_gpu(sequences_gpu, N)
    best_idx = int(cp.argmin(energies_gpu))
    best_bs = bitstrings[best_idx]

    # Build population list: repeat each sequence by its count, cap at population_size.
    # Use explicit host copies so no GPU buffer refs leak; sync before return to avoid
    # corrupted size vs. prev_size when mixing cudaq/CuPy.
    population: List[np.ndarray] = []
    for i in range(num_unique):
        seq_np = np.asarray(matrix[i], dtype=np.int32, order="C").copy()
        for _ in range(counts[i]):
            population.append(seq_np.copy())
            if len(population) >= population_size:
                break
        if len(population) >= population_size:
            break

    # Ensure all GPU work and frees complete before returning (avoids heap corruption)
    if _CP_AVAILABLE and cp is not None:
        cp.cuda.Stream.null.synchronize()
    return population, best_bs

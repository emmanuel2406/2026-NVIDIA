"""
Multi-GPU (H100s) Memetic Tabu Search (MTS) for LABS Problem

Drop-in replacement for main.py / mts_h100_optimized.py with the same API,
optimized for multiple NVIDIA H100 (or A100) GPUs using NVLink and peer-to-peer.

Optimizations:
1. Population partitioned across GPUs; each GPU runs batch tabu on its partition.
2. NVLink peer-to-peer (P2P) for fast gather/broadcast of best solutions (no host round-trip).
3. Periodic migration: global best gathered to GPU 0, broadcast to all GPUs and injected.
4. Per-device streams and memory pools; P2P enabled between all pairs where supported.
5. Same H100-oriented kernels as mts_h100_optimized.py (batch Ck, energy, delta, tabu).

Based on: "Scaling advantage with quantum-enhanced memetic tabu search for LABS"
https://arxiv.org/abs/2511.04553

Usage (same as main.py):
    python mts_h100s_mult.py [N] [population_size] [max_generations]
    python mts_h100s_mult.py 30 100 200
"""

import sys
from pathlib import Path

# Ensure impl-mts is on path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cupy as cp
from typing import Tuple, List, Optional
import random
import time
from dataclasses import dataclass, field

# Reuse H100-optimized kernels and helpers from single-GPU module (same directory)
from mts_h100_optimized import (
    compute_merit_factor,
    compute_energy,
    bitstring_to_sequence,
    sequence_to_bitstring,
    _batch_compute_Ck_gpu,
    _batch_compute_energy_gpu,
    _batch_tabu_search_gpu,
)


# ============================================================================
# Multi-GPU configuration: NVLink P2P and per-device configs
# ============================================================================

def _enable_peer_access(dst_device: int, peer_device: int) -> bool:
    """Enable peer access from dst_device to peer_device. Returns True if enabled or already supported."""
    try:
        if dst_device == peer_device:
            return True
        if cp.cuda.runtime.deviceCanAccessPeer(dst_device, peer_device):
            cp.cuda.Device(dst_device).use()
            cp.cuda.runtime.deviceEnablePeerAccess(peer_device)
            return True
    except Exception:
        pass
    return False


@dataclass
class MultiGPUConfig:
    """Multi-GPU configuration with NVLink P2P and per-device settings."""
    device_ids: List[int] = field(default_factory=list)
    configs: List["GPUConfig"] = field(default_factory=list)
    p2p_enabled: bool = False
    migration_interval: int = 1  # migrate (gather/broadcast) every N generations
    _initialized: bool = False

    def __post_init__(self):
        if self._initialized:
            return
        num_gpus = cp.cuda.runtime.getDeviceCount()
        if num_gpus < 1:
            raise RuntimeError("No GPU found")
        self.device_ids = list(range(num_gpus))

        # Per-device config (reuse single-GPU config logic per device)
        self.configs = []
        for i in self.device_ids:
            cp.cuda.Device(i).use()
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode("utf-8")
            total_mem = props["totalGlobalMem"]
            is_h100 = "H100" in name or (props["major"], props["minor"]) >= (9, 0)
            is_a100 = "A100" in name
            batch_tabu = 64 if is_h100 else (48 if is_a100 else 32)
            num_streams = 16 if is_h100 else (12 if is_a100 else 8)
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=int(total_mem * 0.85))
            self.configs.append(
                type("GPUConfig", (), {
                    "device_id": i,
                    "device_name": name,
                    "batch_tabu_size": batch_tabu,
                    "num_streams": num_streams,
                    "block_size": 256,
                    "total_memory": total_mem,
                })()
            )

        # Enable NVLink P2P: device 0 can access 1,2,... for gather; each i can access 0 for broadcast
        cp.cuda.Device(0).use()
        for peer in self.device_ids:
            if peer != 0:
                self.p2p_enabled = _enable_peer_access(0, peer) or self.p2p_enabled
        for i in self.device_ids:
            if i != 0:
                _enable_peer_access(i, 0)
        self._initialized = True


_global_multiconfig: Optional[MultiGPUConfig] = None


def get_multi_config() -> MultiGPUConfig:
    """Get or create global multi-GPU config."""
    global _global_multiconfig
    if _global_multiconfig is None:
        _global_multiconfig = MultiGPUConfig()
        print(f"[MULTI-GPU] Devices: {len(_global_multiconfig.device_ids)}")
        for i, c in enumerate(_global_multiconfig.configs):
            print(f"[MULTI-GPU]   GPU {i}: {c.device_name}, batch_tabu={c.batch_tabu_size}")
        print(f"[MULTI-GPU] NVLink P2P: {_global_multiconfig.p2p_enabled}")
    return _global_multiconfig


# ============================================================================
# P2P gather/broadcast helpers
# ============================================================================

def _gather_bests_to_device_0(
    best_sequences: List[cp.ndarray],
    best_energies: List[cp.ndarray],
    all_best_sequences_d0: cp.ndarray,
    all_best_energies_d0: cp.ndarray,
    num_gpus: int,
    N: int,
) -> None:
    """Gather each GPU's best sequence and energy to GPU 0 (using P2P when available)."""
    with cp.cuda.Device(0):
        all_best_sequences_d0[0] = best_sequences[0]
        all_best_energies_d0[0] = best_energies[0]

    for i in range(1, num_gpus):
        with cp.cuda.Device(i):
            src_s = best_sequences[i]
            src_e = best_energies[i]
            src_ptr_s = src_s.data.ptr
            src_ptr_e = src_e.data.ptr
            nbytes_s = src_s.nbytes
            nbytes_e = src_e.nbytes
        with cp.cuda.Device(0):
            try:
                cp.cuda.runtime.memcpyPeer(
                    all_best_sequences_d0[i].data.ptr, 0, src_ptr_s, i, nbytes_s
                )
                cp.cuda.runtime.memcpyPeer(
                    all_best_energies_d0[i].data.ptr, 0, src_ptr_e, i, nbytes_e
                )
            except Exception:
                all_best_sequences_d0[i].set(cp.asarray(cp.asnumpy(src_s)))
                all_best_energies_d0[i] = cp.asarray(cp.asnumpy(src_e))


def _broadcast_best_from_device_0(
    global_best_sequence: cp.ndarray,
    best_sequences: List[cp.ndarray],
    num_gpus: int,
) -> None:
    """Broadcast global best sequence from GPU 0 to all other GPUs (P2P when available)."""
    with cp.cuda.Device(0):
        src_ptr = global_best_sequence.data.ptr
        nbytes = global_best_sequence.nbytes
    for i in range(1, num_gpus):
        with cp.cuda.Device(i):
            try:
                cp.cuda.runtime.memcpyPeer(
                    best_sequences[i].data.ptr, i, src_ptr, 0, nbytes
                )
            except Exception:
                best_sequences[i].set(cp.array(cp.asnumpy(global_best_sequence)))


# ============================================================================
# LABS API (re-export from single-GPU module for compatibility)
# ============================================================================

def compute_Ck(s: np.ndarray, k: int) -> int:
    N = len(s)
    return int(np.sum(s[: N - k] * s[k:]))


def compute_all_Ck(s: np.ndarray) -> np.ndarray:
    N = len(s)
    Ck_values = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        Ck_values[k] = compute_Ck(s, k)
    return Ck_values


def compute_delta_energy(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int) -> int:
    N = len(s)
    delta = 0
    for k in range(1, N):
        old_Ck = Ck_values[k]
        delta_Ck = 0
        if flip_idx + k < N:
            delta_Ck += -2 * s[flip_idx] * s[flip_idx + k]
        if flip_idx - k >= 0:
            delta_Ck += -2 * s[flip_idx - k] * s[flip_idx]
        new_Ck = old_Ck + delta_Ck
        delta += new_Ck * new_Ck - old_Ck * old_Ck
    return delta


def update_Ck_after_flip(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int) -> None:
    N = len(s)
    for k in range(1, N):
        if flip_idx + k < N:
            Ck_values[k] += 2 * s[flip_idx] * s[flip_idx + k]
        if flip_idx - k >= 0:
            Ck_values[k] += 2 * s[flip_idx - k] * s[flip_idx]


def combine(parent1: np.ndarray, parent2: np.ndarray, verbose: bool = True) -> np.ndarray:
    N = len(parent1)
    k = random.randint(1, N - 1)
    child = np.concatenate([parent1[:k], parent2[k:]])
    if verbose:
        print(f"[COMBINE] Cut point k={k}")
    return child


def mutate(
    s: np.ndarray, p_mut: Optional[float] = None, verbose: bool = True
) -> np.ndarray:
    N = len(s)
    if p_mut is None:
        p_mut = 1.0 / N
    child = s.copy()
    flipped = [i for i in range(N) if random.random() < p_mut]
    for i in flipped:
        child[i] *= -1
    if verbose:
        print(f"[MUTATE] p_mut={p_mut:.4f}, flipped {len(flipped)} bits")
    return child


def random_search(
    N: int, n_samples: int, verbose: bool = True
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    if verbose:
        print(f"[RANDOM] Generating {n_samples} random sequences of length N={N}")
    population = [np.random.choice([-1, 1], size=N) for _ in range(n_samples)]
    energies = [compute_energy(s) for s in population]
    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]
    if verbose:
        print(
            f"[RANDOM] Best energy={best_energy}, merit={compute_merit_factor(best_s, best_energy):.4f}"
        )
    return best_s, best_energy, population


# ============================================================================
# Tabu search (single sequence) – run on device 0 for API compatibility
# ============================================================================

def tabu_search(
    s: np.ndarray,
    max_iter: Optional[int] = None,
    min_tabu_factor: float = 0.1,
    max_tabu_factor: float = 0.12,
    tabu_id: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, int]:
    mconfig = get_multi_config()
    config = mconfig.configs[0]
    cp.cuda.Device(0).use()
    N = len(s)
    if max_iter is None:
        max_iter = random.randint(N // 2, 3 * N // 2)
    s_gpu = cp.array(s.reshape(1, -1), dtype=cp.int32)
    best_s_gpu, best_energy_gpu = _batch_tabu_search_gpu(
        s_gpu, config, max_iter, min_tabu_factor, max_tabu_factor
    )
    best_s = cp.asnumpy(best_s_gpu[0])
    best_energy = int(best_energy_gpu[0])
    if verbose:
        prefix = f"[TABU-{tabu_id}]" if tabu_id is not None else "[TABU]"
        print(
            f"{prefix} Completed: energy={best_energy}, "
            f"merit={compute_merit_factor(best_s, best_energy):.4f}"
        )
    return best_s, best_energy


# ============================================================================
# Multi-GPU Memetic Tabu Search
# ============================================================================

def memetic_tabu_search(
    N: int,
    population_size: int = 100,
    max_generations: int = 1000,
    p_combine: float = 0.9,
    initial_population: Optional[List[np.ndarray]] = None,
    target_energy: Optional[int] = None,
    verbose: bool = True,
    fixed_indices: Optional[List[int]] = None,
    fixed_values: Optional[np.ndarray] = None,
    migration_interval: Optional[int] = None,
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Multi-GPU (H100s) Memetic Tabu Search for LABS.
    Same API as main.py; population is split across GPUs with NVLink P2P gather/broadcast.
    """
    mconfig = get_multi_config()
    num_gpus = len(mconfig.device_ids)
    if migration_interval is not None:
        mconfig.migration_interval = max(1, migration_interval)

    if fixed_indices is not None:
        if fixed_values is None or len(fixed_values) != len(fixed_indices):
            raise ValueError(
                "fixed_values must be provided and match length of fixed_indices"
            )
        fixed_values = np.asarray(fixed_values, dtype=np.int32)
    fixed_indices_np = np.array(fixed_indices, dtype=np.int32) if fixed_indices else None
    fixed_values_np = np.array(fixed_values, dtype=np.int32) if fixed_indices else None

    if verbose:
        print("=" * 70)
        print("[MTS-MULTI] MULTI-GPU (NVLink) MEMETIC TABU SEARCH")
        print("=" * 70)
        print(
            f"[MTS-MULTI] N={N}, population_size={population_size}, "
            f"max_generations={max_generations}, GPUs={num_gpus}"
        )
        if fixed_indices is not None:
            print(f"[MTS-MULTI] Fixed indices: {fixed_indices} -> values {fixed_values.tolist()}")

    # Partition population across GPUs (as equal as possible)
    base_size = population_size // num_gpus
    remainder = population_size % num_gpus
    sizes = [base_size + (1 if i < remainder else 0) for i in range(num_gpus)]

    # Allocate per-GPU population and bests
    population_gpu: List[cp.ndarray] = []
    best_s_gpu: List[cp.ndarray] = []
    best_energy_gpu: List[cp.ndarray] = []
    energies_gpu: List[cp.ndarray] = []

    for i in range(num_gpus):
        with cp.cuda.Device(i):
            pop_i = sizes[i]
            population_gpu.append(
                cp.random.choice(
                    cp.array([-1, 1], dtype=cp.int32), size=(pop_i, N)
                )
            )
            if fixed_indices_np is not None:
                fixed_cp = cp.array(fixed_values_np, dtype=cp.int32)
                fixed_idx_cp = cp.array(fixed_indices_np, dtype=cp.int32)
                population_gpu[i][:, fixed_idx_cp] = fixed_cp
            best_s_gpu.append(cp.zeros(N, dtype=cp.int32))
            best_energy_gpu.append(cp.zeros(1, dtype=cp.int64))
            energies_gpu.append(cp.zeros(pop_i, dtype=cp.int64))

    # Initial population from provided or keep random
    if initial_population is not None:
        idx = 0
        for i in range(num_gpus):
            with cp.cuda.Device(i):
                for j in range(sizes[i]):
                    if idx < len(initial_population):
                        arr = np.asarray(initial_population[idx], dtype=np.int32)
                        arr = arr.reshape(-1)[:N]
                        if arr.size < N:
                            arr = np.pad(arr, (0, N - arr.size), constant_values=1)
                        population_gpu[i][j] = cp.array(arr)
                    idx += 1

    # Initial energies and best per device
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            config = mconfig.configs[i]
            Ck = _batch_compute_Ck_gpu(population_gpu[i], config)
            energies_gpu[i] = _batch_compute_energy_gpu(
                population_gpu[i], Ck, config
            )
            best_idx = int(cp.argmin(energies_gpu[i]))
            best_s_gpu[i] = population_gpu[i][best_idx].copy()
            best_energy_gpu[i][0] = energies_gpu[i][best_idx]

    # On GPU 0: buffers for gather
    with cp.cuda.Device(0):
        all_best_sequences_d0 = cp.zeros((num_gpus, N), dtype=cp.int32)
        all_best_energies_d0 = cp.zeros(num_gpus, dtype=cp.int64)

    # Global best (lived on GPU 0)
    _gather_bests_to_device_0(
        best_s_gpu, best_energy_gpu,
        all_best_sequences_d0, all_best_energies_d0,
        num_gpus, N,
    )
    with cp.cuda.Device(0):
        global_best_idx = int(cp.argmin(all_best_energies_d0))
        global_best_s = all_best_sequences_d0[global_best_idx].copy()
        global_best_energy = int(all_best_energies_d0[global_best_idx])

    # Broadcast so every device has global best for migration
    _broadcast_best_from_device_0(global_best_s, best_s_gpu, num_gpus)

    if verbose:
        init_merit = compute_merit_factor(
            cp.asnumpy(global_best_s), global_best_energy
        )
        print(f"[MTS-MULTI] Initial: energy={global_best_energy}, merit={init_merit:.4f}")
        print("[MTS-MULTI] STARTING EVOLUTION")
        print("=" * 70)

    start_time = time.time()
    improvements_count = 0
    gen = 0

    while gen < max_generations:
        if target_energy is not None and global_best_energy <= target_energy:
            if verbose:
                print(f"[MTS-MULTI] TARGET REACHED at generation {gen}!")
            break

        for i in range(num_gpus):
            with cp.cuda.Device(i):
                config = mconfig.configs[i]
                pop_i, _ = population_gpu[i].shape
                batch_size = min(config.batch_tabu_size, max_generations - gen, pop_i)
                if batch_size < 1:
                    continue
                children_gpu = cp.empty((batch_size, N), dtype=cp.int32)
                use_crossover = cp.random.random(batch_size) < p_combine
                parent_indices = cp.random.randint(0, pop_i, size=(batch_size, 2))
                crossover_points = cp.random.randint(1, N, size=batch_size, dtype=cp.int32)
                fixed_idx_cp = cp.array(fixed_indices_np, dtype=cp.int32) if fixed_indices_np is not None else None
                fixed_cp = cp.array(fixed_values_np, dtype=cp.int32) if fixed_values_np is not None else None

                for b in range(batch_size):
                    if use_crossover[b]:
                        k = int(crossover_points[b])
                        p1, p2 = int(parent_indices[b, 0]), int(parent_indices[b, 1])
                        children_gpu[b, :k] = population_gpu[i][p1, :k]
                        children_gpu[b, k:] = population_gpu[i][p2, k:]
                        if fixed_idx_cp is not None:
                            children_gpu[b, fixed_idx_cp] = population_gpu[i][p1, fixed_idx_cp]
                    else:
                        idx = int(parent_indices[b, 0])
                        children_gpu[b] = population_gpu[i][idx].copy()

                p_mut = 1.0 / N
                mutation_mask = cp.random.random((batch_size, N)) < p_mut
                children_gpu = cp.where(mutation_mask, -children_gpu, children_gpu)
                if fixed_idx_cp is not None:
                    children_gpu[:, fixed_idx_cp] = fixed_cp

                improved_children, child_energies = _batch_tabu_search_gpu(
                    children_gpu, config, max_iter=N
                )
                if fixed_idx_cp is not None:
                    improved_children[:, fixed_idx_cp] = fixed_cp

                batch_best_idx = int(cp.argmin(child_energies))
                batch_best_energy = int(child_energies[batch_best_idx])

                if batch_best_energy < int(best_energy_gpu[i][0]):
                    best_s_gpu[i] = improved_children[batch_best_idx].copy()
                    best_energy_gpu[i][0] = batch_best_energy

                replace_indices = cp.random.randint(0, pop_i, size=batch_size)
                for b in range(batch_size):
                    idx = int(replace_indices[b])
                    population_gpu[i][idx] = improved_children[b]
                    energies_gpu[i][idx] = child_energies[b]

        batch_done = sum(
            min(mconfig.configs[i].batch_tabu_size, max_generations - gen, sizes[i])
            for i in range(num_gpus)
        )
        gen += batch_done

        # Migration: gather to GPU 0, compute global best, broadcast
        if gen % mconfig.migration_interval == 0 or gen >= max_generations:
            _gather_bests_to_device_0(
                best_s_gpu, best_energy_gpu,
                all_best_sequences_d0, all_best_energies_d0,
                num_gpus, N,
            )
            with cp.cuda.Device(0):
                new_global_idx = int(cp.argmin(all_best_energies_d0))
                new_global_energy = int(all_best_energies_d0[new_global_idx])
                if new_global_energy < global_best_energy:
                    old_best = global_best_energy
                    global_best_energy = new_global_energy
                    global_best_s = all_best_sequences_d0[new_global_idx].copy()
                    improvements_count += 1
                    if verbose:
                        print(
                            f"[MTS-MULTI] Gen {gen}: NEW GLOBAL BEST energy "
                            f"{old_best} -> {global_best_energy}, "
                            f"merit={N*N/(2.0*global_best_energy):.4f}"
                        )
            _broadcast_best_from_device_0(global_best_s, best_s_gpu, num_gpus)

            # Inject global best as worst on each GPU (migration)
            for i in range(num_gpus):
                with cp.cuda.Device(i):
                    worst_idx = int(cp.argmax(energies_gpu[i]))
                    population_gpu[i][worst_idx] = best_s_gpu[i]
                    energies_gpu[i][worst_idx] = global_best_energy

    total_time = time.time() - start_time
    best_s = cp.asnumpy(global_best_s)
    population = []
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            population.extend(cp.asnumpy(population_gpu[i]))

    if verbose:
        print("=" * 70)
        print("[MTS-MULTI] EVOLUTION COMPLETE")
        print(
            f"[MTS-MULTI] Best energy={global_best_energy}, "
            f"merit={compute_merit_factor(best_s, global_best_energy):.4f}, "
            f"improvements={improvements_count}, time={total_time:.2f}s"
        )
        print(f"[MTS-MULTI] Throughput: {gen/total_time:.1f} gen/s")
        print("=" * 70)

    return best_s, global_best_energy, population


# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    import sys

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    random.seed(42)
    np.random.seed(42)
    cp.random.seed(42)

    best_s, best_energy, pop = memetic_tabu_search(
        N=N, population_size=pop_size, max_generations=max_gen, p_combine=0.9
    )

    print("Final best:", sequence_to_bitstring(best_s), "E=", best_energy)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA iQuHACK 2026 Challenge: Hybrid quantum-enhanced solver for the Low Autocorrelation Binary Sequences (LABS) problem. The goal is to evolve the classical Memetic Tabu Search (MTS) algorithm by integrating quantum algorithms (QAOA, Grover) and GPU acceleration.

**LABS Problem**: Find a binary sequence minimizing autocorrelation (sidelobe energy):
- Energy: `E(s) = Σ C_k²` where `C_k = Σ s_i × s_{i+k}`
- Merit Factor: `F(s) = N² / (2×E(s))` (higher is better)
- Conversion: Binary {0,1} ↔ Spin {+1,-1}

## Commands

### Running Tests
```bash
# Evaluation utilities tests
python tutorial_notebook/evals/eval_util.py test
cd tutorial_notebook/evals && python -m pytest test_eval_util.py

# Trotter tests
python impl-trotter/test.py
```

### Benchmarking
```bash
# Run benchmark for specific N values
python benchmarks/run_benchmark.py 3 4 5 10 20

# Run benchmark for range
python benchmarks/run_benchmark.py 3-25
```

### Running Implementations
```bash
# Tutorial notebook (Phase 1 - qBraid)
jupyter notebook tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb
# Kernel: Python 3 [cuda q-v0.13.0]

# GPU MTS (Phase 2 - Brev)
python GPU_Optimised/run_mts_gpu.py --N 50 --population_size 100 --generations 1000
```

## Architecture

### Implementation Directories

- **`impl-mts/`** - Memetic Tabu Search (classical baseline)
  - `main.py` - CPU-based MTS with energy utilities
  - `mts_h100_optimized.py` - H100 GPU-optimized version using CuPy

- **`impl-qaoa/`** - Digitized Counterdiabatic QAOA (DC-QAOA)
  - Fixed-parameter QAOA with counterdiabatic Y-rotations
  - Uses CUDA-Q for quantum simulation

- **`impl-qmf/`** - Quantum Minimum Finding
  - Hybrid QAOA + Grover + MTS pipeline
  - Quantum samples seed classical MTS population

- **`impl-trotter/`** - Trotterization approach
  - Image Hamiltonian with 1/2/3/4-body terms
  - `qe_mts_image_hamiltonian.py` - main implementation
  - `generate_trotterization.py` - circuit generation

- **`GPU_Optimised/`** - GPU acceleration toolkit
  - CuPy-based batch parallel energy computation
  - Custom CUDA kernels for autocorrelation
  - H100-specific optimizations (Tensor Cores, NVLink)

### Evaluation & Validation

- **`tutorial_notebook/evals/`** and **`impl-trotter/evals/`**
  - `eval_util.py` - LABS energy computation, merit factor, run-length encoding
  - `answers.csv` - ground truth optimal solutions
  - `physics_tests.py` - physics validation tests

### Key Functions (impl-mts/main.py)
- `compute_energy(s)` - Sidelobe energy E(s)
- `compute_merit_factor(s)` - Merit factor F(s)
- `memetic_tabu_search(N, population_size, max_generations, ...)` - Main MTS algorithm
- `bitstring_to_sequence(bitstring)` - Convert '0'/'1' to +1/-1

## Dependencies

- **CUDA-Q v0.13.0** - Quantum simulation and compilation
- **CuPy** (cupy-cuda12x) - GPU array operations
- **NumPy** - CPU array operations

## Platform Workflow

1. **Phase 1 (qBraid)**: CPU-based prototyping and validation
2. **Phase 2 (Brev)**: GPU acceleration with L4/A100/H100

## Key Files

- `skills.md` - CUDA-Q Python API reference
- `tutorial_notebook/auxiliary_files/labs_utils.py` - LABS utilities
- `team-submissions/README.md` - Deliverables checklist

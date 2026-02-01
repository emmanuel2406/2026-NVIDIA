# CUDA-Q Python API Reference

## Program Construction

### `make_kernel()`
Creates empty kernel functions with optional parameterization via type arguments.

### `PyKernel` (alias: `Kernel`)
Enables programmatic circuit construction using MLIR's Quake dialect.

**Key properties:**
- `name` - kernel name
- `arguments` - kernel arguments
- `argument_count` - number of arguments

### `@kernel()` decorator
Marks Python functions as CUDA-Q kernels with optional verbose logging.

---

## Kernel Execution Primitives

### `sample()`
Executes circuits repeatedly, returning measurement statistics across shots.

### `sample_async()`
Asynchronous variant of `sample()`, returns a future.

### `run()`
Executes circuits and returns custom data type results per execution.

### `run_async()`
Asynchronous variant of `run()`.

### `observe()`
Computes expectation values of spin operators; supports broadcasting.

### `observe_async()`
Asynchronous variant of `observe()`.

### `get_state()`
Retrieves the quantum state vector or density matrix post-execution.

### `get_state_async()`
Asynchronous variant of `get_state()`.

---

## Backend Management

### `set_target()`
Select quantum backend for execution.

### `get_target()`
Query the currently selected backend.

### `set_noise()`
Attach noise model for realistic simulation.

### `unset_noise()`
Remove noise model.

### `num_available_gpus()`
Query GPU availability.

### `set_random_seed()`
Set seed for deterministic simulation.

---

## Advanced Features

### `vqe()`
Variational Quantum Eigensolver with optimizer and gradient strategy support.

### `draw()`
ASCII/LaTeX circuit visualization.

### `translate()`
Convert kernels to QIR, OpenQASM2, and other formats.

### `estimate_resources()`
Gate and qubit counting with measurement-dependent control flow handling.

---

## Dynamics Simulation

### `evolve()`
Time evolution under Hamiltonians with optional collapse operators for open systems.

### `evolve_async()`
Asynchronous variant of `evolve()`.

### `Schedule`
Time-dependent control specifications.

**Supported operator types:**
- Spin operators
- Fermion operators
- Boson operators
- Matrix operators

---

## Operators Module (`cudaq.operators`)

### Spin Operators
Pauli-based operators for spin systems.

### Fermion Operators
Second quantization operators for fermionic systems.

### Boson Operators
Second quantization operators for bosonic systems.

### Matrix Operators
Custom dense matrix operators.

### `SuperOperator`
Lindblad superoperators for open quantum systems.

---

## Data Types

### `SampleResult`
Measurement statistics dictionary containing counts for each bitstring outcome.

### `ObserveResult`
Expectation value with optional shot statistics.

### `State`
Quantum state representation (state vector or density matrix).

### `Resources`
Gate counts and qubit usage metrics.

### `AsyncSampleResult`
Future for non-blocking sample execution.

### `AsyncObserveResult`
Future for non-blocking observe execution.

---

## Noise and Error Modeling

### `NoiseModel`
Composite noise specification for realistic simulation.

### `KrausChannel`
Custom noise via Kraus operators.

**Built-in channels:**
- `Depolarization2`
- Other standard quantum channels

### `apply_noise()`
In-kernel noise injection at specific gates.

---

## MPI and Distributed Computing (`cudaq.mpi`)

### `initialize()`
Initialize MPI environment.

### `finalize()`
Finalize MPI environment.

### `rank()`
Get current MPI rank.

### `num_ranks()`
Get total number of MPI ranks.

### `all_gather()`
Gather data from all ranks.

### `broadcast()`
Broadcast data from one rank to all others.

---

## Quick Reference

| Category | Functions |
|----------|-----------|
| Construction | `make_kernel()`, `Kernel`, `@kernel()` |
| Execution | `sample()`, `run()`, `observe()`, `get_state()` |
| Async | `sample_async()`, `run_async()`, `observe_async()`, `get_state_async()` |
| Backend | `set_target()`, `get_target()`, `set_noise()`, `unset_noise()` |
| VQE | `vqe()` |
| Dynamics | `evolve()`, `evolve_async()`, `Schedule` |
| Utilities | `draw()`, `translate()`, `estimate_resources()` |
| MPI | `cudaq.mpi.initialize()`, `rank()`, `num_ranks()`, `all_gather()`, `broadcast()`, `finalize()` |

---

## Source
[CUDA-Q Python API Documentation](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)

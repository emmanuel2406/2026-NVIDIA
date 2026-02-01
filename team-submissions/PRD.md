# Product Requirements Document (PRD)

**Project Name:** Quantum-go-brr
**Team Name:** Harvard Blocheads
**GitHub Repository:** https://github.com/emmanuel2406/2026-NVIDIA

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | [Emmanuel Rassou] | [@emmanuel2406] | [@Luna_Meme] |
| **GPU Acceleration PIC** (Builder) | [Tarun Sasirekha] | [@tarunsasirekha] | [@tarunsasirekha_50776] |
| **Quality Assurance PIC** (Verifier) | [Anmay Gupta] | [@AnmayG] | [@AnmayG] |
| **Technical Marketing PIC** (Storyteller) | [Hugo Mackay] | [@hugomackay] | [@hugomackay] |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** [Identify the specific algorithm or ansatz]
    * *Example:* "Quantum Approximate Optimization Algorithm (QAOA) with a hardware-efficient ansatz."
    * *Example:* "Variational Quantum Eigensolver (VQE) using a custom warm-start initialization."

* **Motivation:** [Why this algorithm? Connect it to the problem structure or learning goals.]
    * *Example (Metric-driven):* "We chose QAOA because we believe the layer depth corresponds well to the correlation length of the LABS sequences."
    *  Example (Skills-driven):* "We selected VQE to maximize skill transfer. Our senior members want to test a novel 'warm-start' adaptation, while the standard implementation provides an accessible ramp-up for our members new to quantum variational methods."
   
For this hackathon we have chosen to explore two different approaches: one using
the trotterization method and another using QAOA. By exploring both methods, we
aim to gain a deeper understanding of the strengths and limitations of each
approach, as well as improve our probability of success by exploring surer gains
along with more experimental methods.

For the trotterization method, we chose to augment the trotterization method provided with symmetry-based exploitation and a short run of quantum minimum finding to converge even faster. This approach ensures at least a constant factor speedup over the original method (thanks to the symmetry-based exploitation) and can potentially achieve an exponential speedup over the original method by leveraging the power of quantum minimum finding. We have three primary reasons for choosing this approach:

1. **Speedup:** The symmetry-based exploitation and quantum minimum finding can significantly speed up the trotterization method, ensuring at least a constant factor speedup over the original method with relative surety.
2. **Runtime Improvements:** By leveraging Quantum Minimum Finding, we target a
reduction in the scaling base of the problem (theoretically improving from
classical $O(1.34^N)$ to quantum $O(1.21^N)$). This provides a relative
exponential speedup over the original method, as fewer Trotter steps are needed
to generate high-quality "seeds" for the classical solver [4].
3. **Parallelism:** Since QMF (as an adaptation on Grover Search)'s
implementation on GPUs is highly parallelizable, we can reduce our reliance on
the (less parallelizable) Trotterization method. This trading off of speed with
lower accuracy of should give a higher reliance on the classical part consisting
of Memetic Tabu Search.

For the QAOA approach, we chose to use fixed-parameter QAOA with a
contradiabatic ansatz for optimization, along with QMF as seen in prior
literature. We have three primary reasons for choosing this approach:

1. **Circuit Depth Efficiency:** Unlike generic ansatzes, the Counteradiabatic
(CD) terms suppress transitions to excited states during evolution. This allows
us to achieve higher fidelity with significantly shallower circuits (fewer
Trotter steps), which is critical for minimizing noise and simulation overhead [2,3].
2. **Scalability:** By using fixed-parameter schedules (rather than variational
optimization loops), we avoid the high computational cost of training. This
approach has been empirically shown to scale better than classical
branch-and-bound solvers for the LABS problem [1].
3. **Skill-based:** Our team members have extensive experience with QAOA and are
familiar with its implementation details thanks to our experience from past
hackathons.

We also reviewed the following approaches:

### Literature Review
* **Reference:**  [Evidence of scaling advantage for the quantum approximate optimization algorithm on a classically intractable problem, Shaydulin et al., https://www.science.org/doi/10.1126/sciadv.adm6761]
* **Relevance:**
    This paper serves as the primary justification for our "Fixed Parameter QAOA + QMF" approach. It explicitly demonstrates that while standard QAOA is powerful, the combination of QAOA with Quantum Minimum Finding (QMF) yields the best known empirical scaling for the LABS problem.

* **Reference:** [Digitized-counterdiabatic quantum approximate optimization
  algorithm, Chandarana et al.,
  https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013141]
* **Relevance:**
  This paper introduces the digitized counterdiabatic quantum approximate
  optimization algorithm, which is a key component of our approach and provides
  information on its improvements over standard QAOA. It shows that adding CD
  terms suppresses transitions to excited states, allowing us to reach
  high-fidelity solutions with significantly shallower circuits than standard QAOA.

* **Reference:** [Molecular docking via DC-QAOA, NVIDIA Corporation and
  Affiliates,
  https://nvidia.github.io/cuda-quantum/latest/applications/python/digitized_counterdiabatic_qaoa.html]
* **Relevance:**
  This example provides a CUDAQ implementation of the digitized counterdiabatic QAOA algorithm, which is a key component of our approach.

* **Reference:** [A Quantum Algorithm for Finding the Minimum, Durr and Hoyer, https://arxiv.org/abs/quant-ph/9607014] 
* **Relevance:**
  This is the foundational paper for the QMF routine we are using, there's also an implementation of the algorithm in CUDAQ in the NVIDIA examples.

* **Reference:** [Low Autocorrelation Binary Sequences,Packebusch and Mertens, https://arxiv.org/pdf/1512.0247]
* **Relevance:**
  This has a table with precomputed golden answers for sequence lengths up to 66, which saves us time to quickly add an evaluation server, that matches the optimal sequences and energies.

---

## 3. The Acceleration Strategy
*Owner:* GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* *Strategy:* The quantum component is used to generate *high-quality candidate
  LABS bitstrings* that seed the Memetic Tabu Search (MTS). Both of our quantum
  approaches (trotterization-based methods and QAOA) involve repeated circuit
  execution and sampling, making them well-suited for GPU acceleration.

Rather than aiming for deep circuits, we focus on: high-throughput sampling, and
parallel evaluation of circuit instances, allowing the classical MTS to handle
fine-grained optimization.

### Implementation
- Use CUDA-Q with GPU-backed simulation** to accelerate short-depth trotterized
  evolution and fixed-parameter QAOA with a contradiabatic ansatz.
- Execute circuits in *batches* to amortize kernel launch and memory transfer overhead.
- Run multiple parameter sets or shot batches in parallel using GPU streams.

### Classical Component (MTS)
* *Strategy:* The dominant computational bottleneck in classical MTS for LABS is
  neighbor evaluation during local search. Evaluating candidate flips
  sequentially is inefficient due to the global autocorrelation structure of
  LABS.

Our strategy is to batch neighbor evaluations on the GPU, turning the most
expensive loop in MTS into a massively parallel computation.

### Implementation
- Represent LABS sequences as {−1, +1} vectors on the GPU.
- For each MTS iteration:
  - generate a batch of candidate moves (e.g., single-bit flips),
  - compute changes in autocorrelation values ΔC_k and total energy ΔE for all candidates in parallel on the GPU**,
  - return batched scores to the CPU for tabu filtering and move selection.
- Implement the first version using CuPy for rapid development and reliability under time pressure.
- Use incremental energy updates rather than recomputing full LABS energy from scratch.
- Run multiple MTS individuals concurrently by batching populations on the GPU.

Tabu bookkeeping and high-level control logic remain on the CPU to minimize implementation risk while still capturing the majority of the available speedup.

### Hardware Targets
*Context & time budget:* ~12 hours remaining in the hackathon. We are
compute-bound (heavy linear algebra, many batched circuit simulations, and
thousands of neighbor ΔE evaluations) rather than I/O-bound (small transfers,
limited host-device synchronization). That observation drives our hardware
allocation: we use an L4 for rapid iteration and lower-cost batched development,
but shift most heavy compute bursts to an H100 for raw throughput.

## Why Our Workflow Is Compute-Bound
- *Trotterization:* dominated by repeated matrix exponentiation and tensor contractions.
- *QAOA:* dominated by statevector/tensor-network simulation and repeated circuit sampling.
Both approaches involve high arithmetic intensity, repeated linear algebra operations, batched execution with minimal data movement.

As a result, performance scales primarily with *floating-point throughput and memory bandwidth*, not I/O latency.

### Design Implication
Because both components are compute-heavy:
- faster GPUs translate almost directly into faster time-to-solution,
- spending more time on high-throughput hardware (H100) yields meaningful gains,
- cheaper GPUs (L4) remain ideal for *iteration and tuning*, where flexibility and hours-per-dollar matter more than peak throughput.

  ## Raw cost math 

*L4 (4 GPUs × 5 hours):*
- Rate: $0.85 / hr  
- Per-hour for 4 L4s: 4 × $0.85 = $3.40 / hr  


*H100 (block A — 1 GPU × 1 hour):*
- Rate: $2.55 / hr  
- Duration: 1 hour  
- Cost: 1 × $2.55 × 1 = $2.55

*H100 (block B — 2 GPUs × 3 hours):*
- Rate : $4.56/hr
- Duration :3 hrs
- total : 13.68

Therefore total comes to 19.63.

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** : pytest
* **AI Hallucination Guardrails:** We test the solution for multiple values of n and in our case, ensure that it finds the optimal energy solution referenced by our golden answers in `evals/answers.csv`, which has already been peer-reviewed and stress-tested in the paper [2016, Packebush et al.](https://arxiv.org/pdf/1512.0247) with 100 citations.

### Core Correctness Checks
* **Check 1 (Symmetry):** 
    We test both physics symmetry tests that we found in Exercise 1, which tests both complementary and reversal symmetry for some sampled sequences as seen in `evals/physics_tests.py`.
* **Check 2 (Ground Truth):**
    Our core evaluation suite which has been incrementally engineering to integrate well with our milestone 1 is given by `evals/eval_util.py`.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** [How will you orchestrate your tools?]
    We are using a variety of IDEs, with one member using Zed + Claude Code,
    another using Cursor, and the others using VSCode. We have created a
    `skills.md` file containing the CUDA-Q documentation so the agent doesn't
    hallucinate API calls. The QA Lead runs the tests, and if they fail, pastes
    the error log back into the Agent to refactor. We're using git worktrees
    with multiple Claude Code sessions to iterate on both approaches
    simultaneously to maximize efficiency.

### Success Metrics
* **Metric 1 (Approximation):**  Optimal solutions for up to N=25
* **Metric 2 (Speedup):**  4x speedup over the CPU-only Tutorial baseline
* **Metric 3 (Scale):** Successfully run a simulation for N=40

### Visualization Plan
* **Plot 1:**  Time-to-Solution vs Problem Size (N) comparing CPU vs. GPU 
* **Plot 2:**  Number of Gates vs Problem Size (N) comparing QAOA vs. Trotterization
* **Plot 3:**  Convergence Rate (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."

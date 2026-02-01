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
2. **Exponential Speedup:** By leveraging Quantum Minimum Finding, we target a
reduction in the scaling base of the problem (theoretically improving from
classical $O(1.34^N)$ to quantum $O(1.21^N)$). This provides a relative
exponential speedup over the original method, as fewer Trotter steps are needed
to generate high-quality "seeds" for the classical solver.
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
Trotter steps), which is critical for minimizing noise and simulation overhead.
2. **Scalability:** By using fixed-parameter schedules (rather than variational
optimization loops), we avoid the "barren plateau" problem and the high
computational cost of training. This approach has been empirically shown to
scale better than classical branch-and-bound solvers for the LABS problem [2].
3. **Skill-based:** Our team members have extensive experience with QAOA and are
familiar with its implementation details thanks to our experience from past
hackathons.

We also reviewed the following approaches:

### Literature Review
* **Reference:**  [Evidence of scaling advantage for the quantum approximate optimization algorithm on a classically intractable problem, Shaydulin et al., https://www.science.org/doi/10.1126/sciadv.adm6761]
* **Relevance:** [How does this paper support your plan?]
    This paper serves as the primary justification for our "Fixed Parameter QAOA + QMF" approach. It explicitly demonstrates that while standard QAOA is powerful, the combination of QAOA with Quantum Minimum Finding (QMF) yields the best known empirical scaling for the LABS problem.

* **Reference:** [Digitized-counterdiabatic quantum approximate optimization
  algorithm, Chandarana et al.,
  https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013141]
* **Relevance:** [How does this paper support your plan?]
  This paper introduces the digitized counterdiabatic quantum approximate
  optimization algorithm, which is a key component of our approach and provides
  information on its improvements over standard QAOA. It shows that adding CD
  terms suppresses transitions to excited states, allowing us to reach
  high-fidelity solutions with significantly shallower circuits than standard QAOA.

* **Reference:** [Molecular docking via DC-QAOA, NVIDIA Corporation and
  Affiliates,
  https://nvidia.github.io/cuda-quantum/latest/applications/python/digitized_counterdiabatic_qaoa.html]
* **Relevance:** [How does this paper support your plan?]
  This example provides a CUDAQ implementation of the digitized counterdiabatic QAOA algorithm, which is a key component of our approach.

* **Reference:** [A Quantum Algorithm for Finding the Minimum, Durr and Hoyer, https://arxiv.org/abs/quant-ph/9607014] 
* **Relevance:** [How does this paper support your plan?]
  This is the foundational paper for the QMF routine we are using, there's also an implementation of the algorithm in CUDAQ in the NVIDIA examples.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** [How will you use the GPU for the quantum part?]
    * *Example:* "After testing with a single L4, we will target the `nvidia-mgpu` backend to distribute the circuit simulation across multiple L4s for large $N$."
 

### Classical Acceleration (MTS)
* **Strategy:** [The classical search has many opportuntities for GPU acceleration. What will you chose to do?]
    * *Example:* "The standard MTS evaluates neighbors one by one. We will use `cupy` to rewrite the energy function to evaluate a batch of 1,000 neighbor flips simultaneously on the GPU."

### Hardware Targets
* **Dev Environment:** [e.g., Qbraid (CPU) for logic, Brev L4 for initial GPU testing]
* **Production Environment:** [e.g., Brev A100-80GB for final N=50 benchmarks]

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** : pytest
* **AI Hallucination Guardrails:** We test the solution for multiple values of n and in our case, ensure that it finds the optimal energy solution referenced by our golden answers in `evals/answers.csv`.

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

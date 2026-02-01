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
2. **Exponential Speedup:** By leveraging the power of quantum minimum finding, we can potentially achieve an exponential speedup over the original method, as a few iterations of quantum minimum finding allows for a significant reduction in the number of trotter steps required as we require lower accuracy from our Trotterization method.
3. **Parallelism:** Since QMF (as an adaptation on Grover Search)'s implementation on GPUs is highly
parallelizable, we can reduce our reliance on the (less parallelizable)
Trotterization method. This trading off of speed with lower accuracy of should
give a higher reliance on the classical part consisting of Memetic Tabu Search.

For the QAOA approach, we chose to use fixed-parameter QAOA with a
contradiabatic ansatz for optimization, along with QMF as seen in prior
literature. We have three primary reasons for choosing this approach:

1. **Resource Efficiency:** Not only are hardware-efficient ansatzes more resource-efficient, but using an additional contradiabatic term also allows us to reduce the size of the Hamiltonian, which can lead to faster convergence on our ansatz.
2. **Scalability:** Hardware-efficient ansatzes are more likely to scale well with increasing problem size, which is important for tackling larger instances of the LABS problem.
3. **Skill-based:** Our team members have extensive experience with QAOA and are familiar with its implementation details thanks to our experience from past hackathons.

### Literature Review
* **Reference:** [Title, Author, Link]
* **Relevance:** [How does this paper support your plan?]
    * *Example:* "Reference: 'QAOA for MaxCut.' Relevance: Although LABS is different from MaxCut, this paper demonstrates how parameter concentration can speed up optimization, which we hope to replicate."

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
    * *Example:* "We are using Cursor as the IDE. We have created a `skills.md` file containing the CUDA-Q documentation so the agent doesn't hallucinate API calls. The QA Lead runs the tests, and if they fail, pastes the error log back into the Agent to refactor."

### Success Metrics
* **Metric 1 (Approximation):** [e.g., Target Ratio > 0.9 for N=30]
* **Metric 2 (Speedup):** [e.g., 10x speedup over the CPU-only Tutorial baseline]
* **Metric 3 (Scale):** [e.g., Successfully run a simulation for N=40]

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** [e.g., "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed]

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."

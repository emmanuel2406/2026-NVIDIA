> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have employed AI agents in a thoughtful way.

**Required Sections:**
<img width="1381" height="257" alt="Screenshot 2026-02-01 at 4 20 38 AM" src="https://github.com/user-attachments/assets/101d6207-32d9-4054-b74e-83dd342cd69f" />

**The Workflow:**

  We organized our AI agents using a single n8n workflow, which better represents how the system actually runs compared to using separate tools or chat instances. All agents are connected in one executable pipeline rather than working in isolation.

The workflow is managed by an orchestrator agent that handles configuration, run IDs, and execution order. Domain-specific agents (such as QAOA and MTS agents) run their respective code modules. Using this setup, we can directly run all code files across different environments—BERQ, qBraid, or locally—without changing the overall workflow structure.

A verification agent checks the results and branches the workflow into success or failure paths, where summaries, logs, and history are automatically handled.

For development, we primarily used Claude for coding and implementation, while ChatGPT was used for research, learning core concepts, and reasoning through design decisions. This combination allowed us to move quickly while keeping both the code and the underlying understanding aligned.

**Verification Strategy:** 

To validate AI-generated code and reduce logical errors or hallucinations, we relied on symmetry-based checks and simple invariants rather than overly complex testing. These methods are effective because many algorithms are expected to behave consistently under specific transformations, and violations often indicate hidden logic bugs.

- Symmetry checks: Outputs were verified to remain invariant under symmetric transformations such as reordering inputs, permuting indices, or relabeling qubits. Any deviation signaled a logical inconsistency.
- Baseline consistency tests: For small problem sizes, results were compared against simple or known-correct reference implementations to ensure correctness.
- Sanity and range checks: We enforced basic constraints on output shapes, numerical ranges, and the absence of NaN or infinite values to catch silent failures.
- Run-to-run consistency: Identical configurations were re-run to confirm stable outputs, helping detect unintended randomness or incorrect state handling.

These checks were integrated into the verification stage of the workflow, allowing errors to be detected early while keeping the validation process lightweight and reliable.


 **The "Vibe" Log:**

Even though AI handled a significant portion of the coding, we learned that **AI is only effective if the team using it understands what is right and what is wrong**. To benefit from AI acceleration, we still needed to understand the algorithms, the physics, and the system-level constraints well enough to recognize incorrect behavior, subtle bugs, or unrealistic assumptions.

This made our hackathon experience not easier, but **more challenging and more valuable**. The AI accelerated implementation, but correctness, validation, and final decisions always remained human responsibilities.

The examples below show how we used AI effectively — including what worked, what failed, and the concrete steps we took to stay fast, correct, and in control.

---

## Win — AI Saved Us Hours

### What happened
We used an AI coding agent to help implement GPU-accelerated components for both the quantum (CUDA-Q) and classical (Memetic Tabu Search) parts of the system. These components involved nontrivial parallelism, batching, and memory layout decisions, and writing them from scratch would have taken many hours.

### Why it worked
We learned early that AI performs best when it is **guided, not trusted blindly**. Instead of asking the AI to solve the entire problem, we:
- scoped tasks very narrowly (one file or one function at a time),
- provided explicit constraints (GPU batching only, shallow circuits, no unsupported CUDA-Q APIs),
- and gave the AI existing code structure and naming conventions to follow.

This reduced ambiguity and forced the AI to work within the same engineering constraints we were using ourselves.

### Result
The AI helped us:
- scaffold CUDA-Q circuits that matched the official API,
- convert slow CPU loops into batched GPU kernels,
- and refactor evaluation code without changing correctness.

This saved us multiple hours of development time and allowed us to focus on algorithmic choices and system design rather than boilerplate implementation.

---

## Learn — Improving Our Prompting Strategy

### Initial issue
In early iterations, the AI occasionally suggested APIs or helper functions that looked correct but did not actually exist in CUDA-Q or CuPy. These suggestions were syntactically reasonable but failed at runtime.

### What we changed
Rather than repeatedly correcting the AI, we changed our process:
  - verified CUDA-Q API references,
  - examples copied from official NVIDIA documentation,
  - explicit notes on unsupported features,
  - pasted compiler errors and stack traces directly into prompts,
  - required the AI to explain *why* it chose a specific approach before writing code.

This forced the AI to reason within a bounded, verified context.

### Outcome
After this change:
- hallucinated APIs almost completely disappeared,
- suggestions became more conservative and realistic,
- integration time dropped significantly.

The AI transitioned from a trial-and-error assistant into a reliable pair programmer.

---

## Fail — When AI Hallucinated (and How We Fixed It)

### What failed
At one point, the AI proposed a quantum optimization routine that appeared mathematically sound and produced low-energy solutions. However:
- it violated known LABS symmetries,
- and generated solutions that were numerically good but physically invalid.

This was a subtle failure that would have been easy to miss if we only optimized for energy values.

### How we caught it
We deliberately built verification into our workflow:
- ran complement and reversal symmetry tests,
- compared results against known golden answers from prior literature,
- monitored consistency across repeated runs.

These checks revealed inconsistencies that flagged the routine as incorrect.

### How we fixed it
We:
- removed the faulty routine entirely,
- added explicit symmetry unit tests so similar errors could not reappear,


This ensured correctness was enforced before performance.

---

## Additional Example — When We Chose Not to Use AI

### The situation
The AI suggested aggressively parallelizing parts of the Memetic Tabu Search control logic on the GPU.

### Why we rejected it
Although technically feasible, we recognized that:
- tabu bookkeeping involves complex branching and state,
- moving it to the GPU would increase implementation risk,
- and the expected speedup was marginal compared to neighbor evaluation.

### Our decision
We kept high-level control logic on the CPU and focused GPU acceleration only on the true bottleneck: batched energy and autocorrelation evaluation.

This decision reduced complexity, avoided bugs, and still captured most of the performance gains.

---

### Key takeaway
Understanding the system well enough to say “no” was just as important as knowing when to say “yes.”


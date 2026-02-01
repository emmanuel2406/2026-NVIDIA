


**The Workflow:**


<img width="1366" height="259" alt="Screenshot 2026-02-01 at 5 23 43 AM" src="https://github.com/user-attachments/assets/714bcee1-3d54-499d-8f2e-d99639534b07" />

 
This workflow is designed as an end-to-end execution and verification pipeline using n8n. It begins with a manual trigger and a configuration step that defines the execution environment and parameters. The workflow then ensures required directories exist and launches an orchestrator process over SSH, which coordinates the run and outputs a structured JSON description of the task. A unique run directory is created, after which domain-specific agents (such as the quantum seed agent and MTS agent) are executed sequentially via SSH. Once execution completes, a verification agent evaluates the results and writes a verification report, which is parsed to determine whether the run passes or fails.

Based on this verification decision, the workflow branches automatically. If the run passes, key metrics are read, a success summary is generated, and the result is appended to a persistent history log before sending a success notification. If the run fails, logs are collected, a failure summary is built, the failure is recorded in history, and a failure notification is triggered. This design ensures that execution, validation, logging, and reporting are fully automated, reproducible, and environment-agnostic while remaining easy to extend and debug.

For development, we primarily used Claude for coding and implementation, while ChatGPT was used for research, learning core concepts, and reasoning through design decisions. This combination allowed us to move quickly while keeping both the code and the underlying understanding aligned.

**Verification Strategy:** 

To validate AI-generated code and reduce logical errors or hallucinations, we relied on symmetry-based checks and simple invariants rather than overly complex testing. These methods are effective because many algorithms are expected to behave consistently under specific transformations, and violations often indicate hidden logic bugs.

- Symmetry checks: Outputs were verified to remain invariant under symmetric transformations such as reordering inputs, permuting indices, or relabeling qubits. Any deviation signaled a logical inconsistency.
- Baseline consistency tests: For small problem sizes, results were compared against simple or known-correct reference implementations to ensure correctness.
- Sanity and range checks: We enforced basic constraints on output shapes, numerical ranges, and the absence of NaN or infinite values to catch silent failures.
- Run-to-run consistency: Identical configurations were re-run to confirm stable outputs, helping detect unintended randomness or incorrect state handling.

These checks were integrated into the verification stage of the workflow, allowing errors to be detected early while keeping the validation process lightweight and reliable.


 **The "Vibe" Log:**

Even though AI handled a significant portion of the coding, we learned that AI is only effective if the team using it understands what is right and what is wrong. To benefit from AI acceleration, we still needed to understand the algorithms, the physics, and the system-level constraints well enough to recognize incorrect behavior, subtle bugs, or unrealistic assumptions.

This made our hackathon experience not easier, but more challenging and more valuable. The AI accelerated implementation, but correctness, validation, and final decisions always remained human responsibilities.

The examples below show how we used AI effectively — including what worked, what failed, and the concrete steps we took to stay fast, correct, and in control.



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
-created a `skills.md` file containing:
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
 ### When We Chose Not to Use AI

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


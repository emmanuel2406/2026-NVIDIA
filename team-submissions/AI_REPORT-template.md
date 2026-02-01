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

**What happened**  
We used an AI coding agent to help implement GPU-accelerated components for both the quantum (CUDA-Q) and classical (Memetic Tabu Search) parts of the system. Writing and debugging these components manually would have taken many hours.

**Why it worked**  
We did not ask the AI to “solve the problem end-to-end.” Instead, we:
- scoped tasks narrowly (one file or function at a time),
- provided explicit constraints (GPU batching, shallow circuits, no unsupported APIs),
- and gave the AI existing code structure to follow.

**Result**  
The AI helped us:
- scaffold CUDA-Q circuits correctly,
- convert CPU loops into batched GPU operations,
- and refactor evaluation code efficiently.

This saved us multiple hours of development time and allowed us to focus on algorithm design rather than boilerplate.

---

##  Learn — Improving Our Prompting Strategy

**Initial issue**  
Early prompts occasionally caused the AI to suggest APIs or functions that looked plausible but did not exist in CUDA-Q or CuPy.

**What we changed**  
To reduce hallucinations, we:
- created a `skills.md` file containing:
  - CUDA-Q API references,
  - examples from official NVIDIA documentation,
  - explicit constraints on unsupported features,
- pasted error logs and stack traces directly into prompts,
- required the AI to explain its approach before writing code.

**Outcome**  
With a bounded knowledge context, the AI’s suggestions became:
- more conservative,
- more accurate,
- easier to integrate.

This shifted the AI from a guessing assistant to a reliable pair programmer.

---

##  Fail — When AI Hallucinated (and How We Fixed It)

**What failed**  
At one point, the AI proposed a quantum optimization routine that appeared correct but:
- violated known LABS symmetries,
- and produced numerically good but physically invalid solutions.

**How we caught it**  
We did not trust outputs blindly. We:
- ran symmetry checks (complement and reversal),
- compared energies against known golden answers,
- identified inconsistencies early.

**How we fixed it**  
- Removed the faulty routine,
- added explicit symmetry unit tests to prevent regressions,
- updated prompts to require:
  > “Explain how this approach respects LABS symmetries before coding.”





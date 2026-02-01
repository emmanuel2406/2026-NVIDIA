# NVIDIA iQuHACK 2026 Challenge - Team Submission

## 👥 Our Team
- **Team Name:** [Your Team Name]
- **Members:** [Name 1], [Name 2], [Name 3], [Name 4]
- **Discord:** [@username1, @username2, ...]

---

## 🎯 What We Built

We tackled the **Low Autocorrelation of Binary Sequences (LABS)** problem by creating a hybrid quantum-classical solution:

1. **Quantum Part:** Used [QAOA/QMF/Trotter] to generate good starting solutions
2. **Classical Part:** Enhanced Memetic Tabu Search (MTS) with quantum samples
3. **GPU Acceleration:** Sped everything up using NVIDIA GPUs

**Our Approach:** [Briefly explain your strategy in 2-3 sentences]

---

## 📊 Our Results

### Best Solutions Found

| Sequence Length | Energy E(s) | Time (sec) |
|-----------------|-------------|------------|
| 20              | [value]     | [time]     |
| 40              | [value]     | [time]     |
| 60              | [value]     | [time]     |

### GPU Performance

**CPU vs GPU Speedup:** [X]x faster on [GPU type]

**Key Finding:** [One sentence about your main discovery]

---

## 📁 Where to Find Our Work

```
📦 Repository Structure
├── 📄 README.md (this file)
│
├── 📂 team-submissions/[YourTeamName]/
│   ├── final_report.pdf
│   ├── presentation.pdf
│   └── retrospective.md
│
├── 📂 [your-code-folder]/
│   ├── main_implementation.py
│   └── [other code files]
│
├── 📂 results/[YourTeamName]/
│   └── [your experimental results]
│
└── 📂 benchmarks/[YourTeamName]/
    └── [your benchmark data]
```

**Main Implementation:** `[path/to/your/main/code.py]`

---

## 🚀 How to Run Our Code

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run our solution
cd [your-code-folder]
python main_implementation.py --length 40

# Run benchmarks
python benchmarks.py
```

### On qBraid (Phase 1)
1. Open `tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb`
2. Follow the tutorial to understand LABS
3. Run our code with CPU backend

### On Brev (Phase 2)
1. Load GPU environment
2. Run with `--backend nvidia` flag
3. See GPU speedup!

---

## 🤖 Our AI Workflow

**Tools We Used:**
- [Claude/ChatGPT/Copilot] for [specific tasks]
- CODA for [quantum algorithm exploration]

**What Worked:**
- AI helped with: [e.g., code generation, debugging, documentation]
- We verified everything with: [e.g., tests, manual review]

**Key Lesson:** [One sentence about AI-assisted development]

---

## 💡 Key Learnings

**Technical:**
- [Learning 1 about quantum/GPU/hybrid approach]
- [Learning 2]

**Process:**
- [Learning about teamwork/AI tools/hackathon workflow]

**If We Had More Time:**
- [What you'd improve or try next]

---

## 🎉 We Really Enjoyed This Hackathon!

[Write 2-3 sentences about your genuine experience - what was fun, challenging, or memorable]

**Thank you to:**
- NVIDIA for the amazing challenge and GPU credits
- MIT iQuISE for organizing iQuHACK 2026
- qBraid and Brev for the platforms
- All the mentors and fellow participants!

---

## 📖 Quick Reference Guide

### Challenge Overview
The repo contains examples of different approaches:
- `/impl-mts/` - Classical MTS baseline
- `/impl-qaoa/` - QAOA quantum approach
- `/impl-qmf/` - Quantum Mean Field approach
- `/impl-trotter/` - Trotterization approach
- `/GPU_Optimised/` - GPU acceleration examples
- `/tutorial_notebook/` - Getting started tutorial

### Important Files
- `LABS-challenge-Phase1.md` - Phase 1 requirements
- `LABS-challenge-Phase2.md` - Phase 2 requirements
- `team-submissions/README.md` - Deliverables checklist
- `AGENTS.md` - AI workflow guide

### Resources
- [CUDA-Q Docs](https://nvidia.github.io/cuda-quantum/)
- [qBraid Platform](https://account-v2.qbraid.com/)
- [Brev Console](https://brev.nvidia.com/)

---

**Submission Date:** [Date]  
**Repository:** https://github.com/emmanuel2406/2026-NVIDIA

✨ *"Let the agents build the code, you build the architecture."* ✨



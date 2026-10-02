# 🧠📦 LLM-Guided 3D Bin Packing Simulation

**LLM-Robotic-Packer** is an advanced research project that integrates **Large Language Models (LLMs)** with a **custom 3D bin-packing simulation** built using **OpenAI Gymnasium**.  
It demonstrates how language models can operate as intelligent decision-makers in **complex spatial reasoning tasks**, bridging the gap between AI-driven planning and robotic manipulation.

---

## 🚀 Project Overview

This project simulates a **robotic packing environment** where an LLM is responsible for **deciding how to place boxes inside a 3D bin**.  
It is designed to serve as both a **research platform** and a **proof-of-concept** for LLM-powered decision-making in constrained physical environments.

The environment, decision flow, and evaluation metrics are all **fully automated**, allowing reproducible experiments and scalable extensions for future research in:
- AI-guided robotics
- Warehouse automation
- Spatial reasoning for LLMs
- Reinforcement learning with LLM advisors

---

## 🛠 How It Works

1. **Custom Gymnasium Environment**  
   - A 3D bin is initialized as the packing space.  
   - Boxes of varying dimensions are introduced for placement.  
   - Supports both **static** and (soon) **dynamic** box generation.

2. **LLM as the Planner**  
   - The current bin state and **valid anchor positions** are passed to the LLM.  
   - The LLM outputs a **placement path** (and optional rotation) for the next box.  
   - The system logs rotation decisions for further analysis.

3. **Collision & Boundary Validation**  
   - Placement proposals are **validated** against bin boundaries and existing boxes.  
   - Invalid placements trigger retries until a valid one is found or attempts are exhausted.

4. **Visualization & Snapshots**  
   - After each successful placement, a **3D visualization** of the bin state is generated.  
   - Snapshots are saved to:  
     ```
     snapshots/<date_time>/
     ```

5. **Metrics & Logging**  
   - **Space Utilization** – Filled volume / Total bin volume  
   - **Placement Success Rate** – % of boxes placed successfully  
   - **Average Placement Time** – Time per box placement  
   - **Rotation Count** – Number of placements involving rotation  
   - **Box-to-Bin Fit Ratio** – Volume of placed boxes vs. total available box volume  
   - Metrics are stored in JSON format in:  
     ```
     results/<date_time>/metrics.json
     ```

---

## 📊 Current Capabilities

✅ **Pre-generated box sequences** for a complete simulation run  
✅ **Anchor-based placement guidance** for informed LLM decisions  
✅ **Full rotation support** with logging  
✅ **Collision & boundary checks** for safe placements  
✅ **3D visual snapshots** after each placement  
✅ **Detailed performance metrics** stored in structured JSON format

---

## 📊 Results

This is the simulation and evaluation half of **Packi**. Packi fine-tunes a small open LLM with LoRA on placement demonstrations (recorded and trained in the companion repo, [learning-from-demonstration](https://github.com/kasramojallal1/learning-from-demonstration)). Two backbones (Llama 3.2 3B, Qwen3-4B) and two teachers (human demonstrations, privileged-expert demonstrations from an offline beam search) give four policies.

Utilization on the four benchmarks (mean of 5 fixed sequences each; all methods use the same sequences, candidates, retry budget and validator — numbers from `results/`):

| Method | CURRIC.-25 | DATA-1 | DATA-2 | DATA-3 | Mean |
|---|---|---|---|---|---|
| Oracle (sees the whole sequence; upper bound) | 0.846 | 0.848 | 0.898 | 0.793 | 0.846 |
| **Packi-Qwen-E** (Qwen3-4B, expert demos) | 0.827 | 0.757 | 0.795 | 0.734 | **0.778** |
| Packi-Llama-E (Llama 3.2 3B, expert demos) | 0.789 | 0.709 | 0.791 | 0.750 | 0.760 |
| Greedy (top-scoring anchor, no LLM) | 0.707 | 0.727 | 0.828 | 0.689 | 0.738 |
| Best API models (Grok 4.3, Claude Haiku 4.5) | | | | | 0.730 |
| GOPT (dedicated DRL packer, trained by us) | 0.681 | 0.700 | 0.736 | 0.719 | 0.709 |
| Packi-Llama-H (Llama 3.2 3B, human demos) | 0.731 | 0.650 | 0.757 | 0.686 | 0.706 |
| Random anchor | 0.630 | 0.600 | 0.648 | 0.589 | 0.617 |

With expert demonstrations, Packi-Qwen-E is significantly better than greedy, the best of twelve API models and GOPT (paired tests over the 20 sequences, p ≤ 0.033); Packi-Llama-E is on par with them. The fine-tuned policies are valid at the first attempt for every box and run locally (≈2–3 s per box on one GPU). Full tables, controls and statistics are in the paper.

> The submitted version of the paper (and an earlier version of this README) reported 87 % utilization on an online-sampled CURRICULUM-25 and a two-dataset comparison. Those numbers came from an easier sampler, a metrics bug and an older protocol, and are superseded by the fixed-sequence protocol above.

### Trained models

| Model | Hugging Face | Used as |
|---|---|---|
| Packi-Llama-H | `kasramojallal/packi-llama32-3b-lora-v2` | `--method packi` (`models/llama32-3b-v2`) |
| Packi-Llama-E | `kasramojallal/packi-llama32-3b-lora-e` | `--method packi-e` (`models/llama32-3b-e`) |
| Packi-Qwen-H / -E | `kasramojallal/packi-qwen3-4b-lora-h`, `…-lora-e` | `--method qwen3-4b-h` / `qwen3-4b-e` |
| GOPT baseline | `kasramojallal/packi-gopt-baseline` | `--method gopt-sample` (see below) |

The repositories are made public with the paper's release.

### GOPT baseline

[GOPT](https://github.com/Xiong5Heng/GOPT) (Xiong et al., IEEE RA-L 2024) is a dedicated deep-reinforcement-learning packer. Its authors publish code but no weights, so we trained it with their code and unchanged configuration (40 M environment steps, ≈11.6 h on one RTX PRO 4500). Our weights, configuration and logs are on Hugging Face (`kasramojallal/packi-gopt-baseline`).

**Licence: GOPT is released by its authors for academic use only, not for commercial purposes without their authorization.** This repository therefore does not contain any GOPT code; the adapter `harness/gopt_policy.py` loads it from your own clone, and the weights are shared under the same academic-use terms.

```bash
git clone https://github.com/Xiong5Heng/GOPT && git -C GOPT checkout a2e42de1c0ab62c5e0a356e363349c32beb9e05b
export GOPT_DIR=$PWD/GOPT
huggingface-cli download kasramojallal/packi-gopt-baseline --local-dir models/gopt
python evaluate.py --method gopt-sample --all              # main row: actions sampled, as in GOPT's test script
python evaluate.py --method gopt-control-greedy-2rot --all # greedy limited to GOPT's two upright orientations
```

In the harness GOPT chooses among its own candidate placements after they are filtered by our placement rules (containment, non-overlap, full base support, top-down clearance), keeps its published two upright orientations, and its path is the template path (GOPT produces no motion).

### Evaluation harness (paper numbers)

All reported numbers come from `evaluate.py`: headless, one method on one fixed
box sequence, one JSON per run under `results/` (committed). `main.py` and
`run_paper_datasets.py` are the interactive demo (live 3D window) and are not
used for reported results.

```bash
python -m harness.sequences --check                         # the 35 committed sequence files (paper + unseen suites) match their generators
python evaluate.py --method greedy --dataset data1 --seed 0  # no LLM: top-scoring anchor + template path
python evaluate.py --method random --all                     # every dataset x seeds 0-4
python evaluate.py --method packi --dataset curriculum25 --seeds 0 1 2 3 4        # local LoRA model
python evaluate.py --method base-llama --dataset data1 --seed 0                    # same base model, no adapter
python evaluate.py --method api:openai/gpt-4o-mini --dataset data1 --seed 0        # OpenRouter (needs OPENROUTER_API_KEY in .env)
python evaluate.py --method packi --all --shuffle-anchors    # randomized anchor order/ids (shortcut-learning check)
python evaluate.py --method packi --all --no-feedback        # same retry budget, empty feedback history
python evaluate.py --method packi-e --suite unseen          # generalization: unseen bins / item sizes (T10.4)
python evaluate.py --method packi-e --dataset data1-b15 --seeds 0 1 2 3 4 --template-path   # pick-only: template path (T10.5)
python aggregate.py results/ [--latex] [--csv out.csv --per-seed]   # mean +- std over seeds per (method, dataset, flags)
pytest tests/
```

| Piece | Where |
|---|---|
| Fixed sequences: `curriculum25` (25 items), `data1`/`data2`/`data3` (cutting-stock tilings of the bin after Zhao et al. 2021 / PUSNet: sides in [2,5] shuffled; 64-template cut in CUT-1 order; same cut in CUT-2 order — item count set by the cut, 21–44 per file); seeds 0-4 | `data/sequences/`, generators in `harness/sequences.py` |
| Unseen suite (T10.4, not part of `--all`): `data1-b15` (bin 15³), `data1-b12x8x10` (non-cubic bin), `data1-s1to6` (sides in [1,6]) — DATA-1 generator with one factor changed; seeds 0-4 | `data/sequences/data1-*/`, `UNSEEN_SPECS` in `harness/sequences.py` |
| One prompt format for every method (system + compact JSON user message, feedback history list) | `harness/prompts.py` |
| Policies: `greedy`, `random`, `oracle`, `ranker-h`/`ranker-e`, `packi`, `packi-e`, `qwen3-4b-h`/`qwen3-4b-e`, `base-llama`, `base-qwen3-4b`, `gopt-sample`, `gopt-control-greedy-2rot`, `local:<hf-id>[@lora]`, `api:<openrouter-id>` | `harness/policies.py`, `harness/ranker.py`, `harness/gopt_policy.py`, `harness/expert.py` |
| Validator: containment, AABB overlap, full-base support, vertical clearance, path ends at target, swept-AABB collision along every path segment | `harness/validator.py` |
| Loop, retry budget (3 pick x 2 path), reliability counters, run record | `harness/runner.py` |

A run record holds the method and model id, dataset/seed/flags, git commit, host/GPU, timestamps,
every attempt for every box (response, validator code, latency), the placed boxes and paths,
the metrics from `envs/metrics.py`, and reliability counters (first-attempt validity, retries per box,
budget exhaustion, invalid JSON, path collisions, skipped items, latency mean/median/p95).

Paper: *Packi: Robotic 3D Bin Packing with LLMs Fine-tuned by Learning from Demonstration* (under review).

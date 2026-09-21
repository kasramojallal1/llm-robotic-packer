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

This is the simulation half of **Packi**. The planner is a **Llama 3.2 3B model fine-tuned with LoRA on human demonstrations** (recorded and trained in the companion repo, [learning-from-demonstration](https://github.com/kasramojallal1/learning-from-demonstration)); `llm_local.py` runs it, `llm_api.py` runs the hosted baselines listed in `config.py`.

On the paper's three box-sequence datasets the fine-tuned 3B model reached **87% bin utilization** and **outperformed 11 proprietary API models** (including GPT-4o, GPT-5-mini and Claude 3.7 Sonnet) while running on a **single 16 GB consumer GPU**.

### Evaluation harness (paper numbers)

All reported numbers come from `evaluate.py`: headless, one method on one fixed
box sequence, one JSON per run under `results/` (committed). `main.py` and
`run_paper_datasets.py` are the interactive demo (live 3D window) and are not
used for reported results.

```bash
python -m harness.sequences --check                         # the 20 committed sequence files match their generators
python evaluate.py --method greedy --dataset data1 --seed 0  # no LLM: top-scoring anchor + template path
python evaluate.py --method random --all                     # every dataset x seeds 0-4
python evaluate.py --method packi --dataset curriculum25 --seeds 0 1 2 3 4        # local LoRA model
python evaluate.py --method base-llama --dataset data1 --seed 0                    # same base model, no adapter
python evaluate.py --method api:openai/gpt-4o-mini --dataset data1 --seed 0        # OpenRouter (needs OPENROUTER_API_KEY in .env)
python evaluate.py --method packi --all --shuffle-anchors    # randomized anchor order/ids (shortcut-learning check)
python evaluate.py --method packi --all --no-feedback        # same retry budget, empty feedback history
python aggregate.py results/ [--latex] [--csv out.csv --per-seed]   # mean +- std over seeds per (method, dataset, flags)
pytest tests/
```

| Piece | Where |
|---|---|
| Fixed sequences: `curriculum25` (25), `data1` (40, exact tiling), `data2` (60), `data3` (80); seeds 0-4 | `data/sequences/`, generators in `harness/sequences.py` |
| One prompt format for every method (system + compact JSON user message, feedback history list) | `harness/prompts.py` |
| Policies: `greedy`, `random`, `packi`, `base-llama`, `local:<hf-id>[@lora]`, `api:<openrouter-id>` | `harness/policies.py` |
| Validator: containment, AABB overlap, full-base support, vertical clearance, path ends at target, swept-AABB collision along every path segment | `harness/validator.py` |
| Loop, retry budget (3 pick x 2 path), reliability counters, run record | `harness/runner.py` |

A run record holds the method and model id, dataset/seed/flags, git commit, host/GPU, timestamps,
every attempt for every box (response, validator code, latency), the placed boxes and paths,
the metrics from `envs/metrics.py`, and reliability counters (first-attempt validity, retries per box,
budget exhaustion, invalid JSON, path collisions, skipped items, latency mean/median/p95).

Paper: *Packi: Robotic 3D Bin Packing with LLMs Fine-tuned by Learning from Demonstration* (under review).

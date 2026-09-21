#!/usr/bin/env python
"""
Headless evaluation of one method on one fixed sequence (T0.7; R1.3/R1.5/R1.6).

    python evaluate.py --method greedy --dataset data1 --seed 0
    python evaluate.py --method api:openai/gpt-4o-mini --dataset curriculum25 --seed 0 --shuffle-anchors
    python evaluate.py --method packi --dataset data1 --seeds 0 1 2 3 4
    python evaluate.py --method greedy --all            # every dataset x seeds 0-4
    python evaluate.py --method api:openai/gpt-5-mini --reasoning low --all

Writes results/<method>/<dataset>/seed<k>[.shuffle][.nofb].json (one file per run).
No matplotlib window, no sleeps.  --render saves a PNG of the final bin next to the JSON
(under results/**/renders/, which is git-ignored).

Methods: see harness/policies.py.  Paper numbers come only from this script; main.py
is the interactive demo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from harness.policies import REASONING_SETTINGS, make_policy   # noqa: E402
from harness.runner import N_PATH, N_PICK, build_run_record, run_episode, run_file_name  # noqa: E402
from harness.sequences import DATASETS, SEEDS, load_sequence  # noqa: E402


def render_final(placed, bin_dims, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, bin_dims[0]); ax.set_ylim(0, bin_dims[1]); ax.set_zlim(0, bin_dims[2])
    ax.bar3d(0, 0, 0, *bin_dims, alpha=0.05, color="gray", edgecolor="black")
    for b in placed:
        ax.bar3d(*b["position"], *b["size"], color="green", alpha=0.6, edgecolor="k", linewidth=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_one(args, dataset: str, seed: int, policy=None) -> str:
    policy = policy or make_policy(args.method, seed=seed, reasoning=args.reasoning)
    sequence = load_sequence(dataset, seed)
    flags = {"shuffle_anchors": args.shuffle_anchors, "feedback": not args.no_feedback,
             "n_pick": args.n_pick, "n_path": args.n_path, "top_k": 8, "reasoning": args.reasoning}
    started = datetime.now(timezone.utc).isoformat()
    print(f"== {args.method} | {dataset} | seed {seed} | shuffle={args.shuffle_anchors} feedback={not args.no_feedback}")
    log = (lambda *a, **k: None) if args.quiet else print
    episode = run_episode(policy, sequence, shuffle_anchors=args.shuffle_anchors,
                          feedback=not args.no_feedback, n_pick=args.n_pick, n_path=args.n_path, log=log)
    record = build_run_record(policy, sequence, episode, flags, REPO_ROOT, started)

    rel = os.path.join(args.out, run_file_name(args.method, dataset, seed, args.shuffle_anchors, not args.no_feedback))
    out_path = os.path.join(REPO_ROOT, rel) if not os.path.isabs(rel) else rel
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1)
    m, r = record["metrics"], record["reliability"]
    print(f"   utilization={m['utilization_final']:.3f} LEC={m['largest_empty_cavity_ratio']:.3f} "
          f"placed={r['items_placed']}/{r['items_total']} skipped={r['items_skipped_no_anchor']} "
          f"exhausted={r['items_budget_exhausted']} first_try={r['first_attempt_validity']:.2f} "
          f"path_collisions={r['path_collisions']} api_errors={r['api_errors']} -> {rel}")
    if record["api_usage"]:
        u = record["api_usage"]
        cost = f"${u['cost_usd']:.4f}" if u["cost_usd"] is not None else "n/a"
        print(f"   api: calls={u['calls']} retried={u['retried_calls']} failed={u['failed_calls']} "
              f"tokens in/out/reasoning={u['prompt_tokens']}/{u['completion_tokens']}/{u['reasoning_tokens']} cost={cost}")
    if args.render:
        png = os.path.join(os.path.dirname(out_path), "renders", os.path.basename(out_path)[:-5] + ".png")
        render_final(record["placed_boxes"], record["bin_dims"], png)
    return out_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="greedy | random | packi | base-llama | local:<hf>[@lora] | api:<openrouter-id>")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", help="several seeds for --dataset")
    ap.add_argument("--all", action="store_true", help="every dataset x seeds 0-4")
    ap.add_argument("--shuffle-anchors", action="store_true", help="randomize anchor order and ids (T3.6, R1.3)")
    ap.add_argument("--no-feedback", action="store_true", help="same retry budget, empty feedback history (R1.6 control)")
    ap.add_argument("--reasoning", choices=sorted(REASONING_SETTINGS),
                    help="api:* only - reasoning effort sent to OpenRouter (D46: 'low' for reasoning models, 'off' for Gemini thinking)")
    ap.add_argument("--n-pick", type=int, default=N_PICK)
    ap.add_argument("--n-path", type=int, default=N_PATH)
    ap.add_argument("--out", default="results")
    ap.add_argument("--render", action="store_true", help="save a PNG of the final bin")
    ap.add_argument("--quiet", action="store_true", help="no per-box lines")
    args = ap.parse_args(argv)
    if args.all:
        args.jobs = [(d, s) for d in DATASETS for s in SEEDS]
    elif args.dataset and (args.seeds or args.seed is not None):
        args.jobs = [(args.dataset, s) for s in (args.seeds or [args.seed])]
    else:
        ap.error("give --dataset with --seed/--seeds, or --all")
    return args


def main(argv=None):
    args = parse_args(argv)
    policy = None
    for dataset, seed in args.jobs:
        # local/API models are loaded once and reused across runs; random re-seeds per run
        if args.method != "random":
            policy = policy or make_policy(args.method, seed=seed, reasoning=args.reasoning)
        run_one(args, dataset, seed, policy)


if __name__ == "__main__":
    main()

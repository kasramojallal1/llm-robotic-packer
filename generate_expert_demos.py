#!/usr/bin/env python
"""
Expert demonstrations for Packi-E (T3.7, D49 B, D52; R1.3, R1.9).

    python generate_expert_demos.py --out ../learning-from-demonstration/data/demos/expert_beam1000.jsonl.gz \
        --episodes 250 --seed-start 1000 --width 1000 --workers 8

For every dataset in harness.sequences.DATASETS and every generator seed in
[seed_start, seed_start + episodes) a fresh sequence is built with the same
generator as the evaluation files (seeds 0-4 are refused), the privileged
beam-search expert (harness/expert.py) packs it, and each placed box becomes one
record in the LfD demo format:

    {"ts", "task": "pick_and_path", "episode": "expert/<dataset>/seed<k>",
     "state": harness top-8 view (unshuffled; sft_prepare.py shuffles),
     "label": {"rotation_index", "anchor_id", "path": overhead -> pre-descend -> target},
     "meta": {"source": "expert", "dataset", "seed", "box_index", "n_items", "beam_width",
              "top_k", "expert_utilization", "util_gain", ...}}

Skipped boxes (no feasible anchor) produce no record.  A manifest JSON next to
the output records the command, the packer commit, per-episode utilization of
the expert and of greedy on the same sequences, record counts and the output's
sha256.  Output is gzip-compressed when the path ends in .gz.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from harness.expert import BEAM_WIDTH, plan_sequence, replay_plan       # noqa: E402
from harness.policies import template_path                              # noqa: E402
from harness.sequences import DATASETS, SEEDS, load_sequence, make_sequence  # noqa: E402
from harness.state import TOP_K, build_state                            # noqa: E402


def _vol(s) -> int:
    return int(s[0] * s[1] * s[2])



def expert_episode(job: Tuple[str, int, int, str]) -> Dict:
    """One (dataset, seed): plan, verify, serialize.  Runs in a worker process."""
    dataset, seed, width, ts = job
    seq = make_sequence(dataset, seed)
    bin_dims = seq["bin_dims"]
    t0 = time.perf_counter()
    plan = plan_sequence(seq["boxes"], bin_dims, width=width)
    if not replay_plan(plan, bin_dims):
        raise RuntimeError(f"{dataset}/seed{seed}: plan not reproducible through build_state")
    episode = f"expert/{dataset}/seed{seed}"
    records: List[Dict] = []
    placed: List[Dict] = []
    for st in plan.steps:
        if st.pos is None:
            continue
        state = build_state(placed, st.size, bin_dims)              # unshuffled top-8 view
        anchor = next(a for a in state["anchors_indexed"]
                      if a["rotation_index"] == st.rotation_index and list(a["pos"]) == list(st.pos))
        records.append({
            "ts": ts,
            "task": "pick_and_path",
            "episode": episode,
            "state": state,
            "label": {"rotation_index": st.rotation_index, "anchor_id": anchor["id"],
                      "path": template_path(state, st.pos)},
            "meta": {"source": "expert", "dataset": dataset, "seed": seed, "box_index": st.index,
                     "n_items": seq["n_items"], "beam_width": width, "top_k": TOP_K,
                     "expert_utilization": plan.utilization, "util_gain": _vol(st.chosen_size),
                     "n_anchors_offered": st.n_anchors_offered},
        })
        placed.append({"position": list(st.pos), "size": list(st.chosen_size)})
    return {
        "dataset": dataset, "seed": seed, "episode": episode, "n_items": seq["n_items"],
        "n_placed": len(records), "n_skipped": seq["n_items"] - len(records),
        "expert_utilization": plan.utilization, "greedy_utilization": plan.greedy_volume / plan.bin_volume,
        "used_greedy": plan.used_greedy,
        "boxes": seq["boxes"], "seconds": time.perf_counter() - t0, "records": records,
    }


def eval_box_lists() -> Dict[Tuple[str, Tuple], int]:
    """Box lists of the 20 evaluation files, to refuse/flag any accidental overlap."""
    out = {}
    for ds in DATASETS:
        for s in SEEDS:
            out[(ds, tuple(map(tuple, load_sequence(ds, s)["boxes"])))] = s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .jsonl or .jsonl.gz (manifest written next to it)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--episodes", type=int, default=250, help="episodes per dataset (D52: 250)")
    ap.add_argument("--seed-start", type=int, default=1000, help="first generator seed (D52: 1000)")
    ap.add_argument("--width", type=int, default=BEAM_WIDTH)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    clash = sorted(set(seeds) & set(SEEDS))
    if clash:
        ap.error(f"training seeds must not include the evaluation seeds {SEEDS}: {clash}")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    jobs = [(ds, s, args.width, ts) for ds in args.datasets for s in seeds]
    print(f"{len(jobs)} episodes ({len(args.datasets)} datasets x {args.episodes}), beam width {args.width}, "
          f"{args.workers} workers -> {args.out}", flush=True)

    t0 = time.perf_counter()
    results: List[Dict] = []
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(expert_episode, jobs), 1):
            results.append(res)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s elapsed", flush=True)
    results.sort(key=lambda r: (args.datasets.index(r["dataset"]), r["seed"]))

    eval_lists = eval_box_lists()
    overlaps = [r["episode"] for r in results if (r["dataset"], tuple(map(tuple, r["boxes"]))) in eval_lists]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    opener = gzip.open if args.out.endswith(".gz") else open
    n_records = 0
    with opener(args.out, "wt") as f:
        for r in results:
            for rec in r["records"]:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n_records += 1

    def commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    per_dataset = {}
    for ds in args.datasets:
        rs = [r for r in results if r["dataset"] == ds]
        n = len(rs)
        per_dataset[ds] = {
            "episodes": n,
            "records": sum(r["n_placed"] for r in rs),
            "skipped_boxes": sum(r["n_skipped"] for r in rs),
            "expert_utilization_mean": sum(r["expert_utilization"] for r in rs) / n,
            "greedy_utilization_mean": sum(r["greedy_utilization"] for r in rs) / n,
            "expert_beats_greedy": sum(r["expert_utilization"] > r["greedy_utilization"] for r in rs),
            "expert_ties_greedy": sum(r["expert_utilization"] == r["greedy_utilization"] for r in rs),
            "episodes_using_greedy_plan": sum(r["used_greedy"] for r in rs),
            "seconds_per_episode_mean": sum(r["seconds"] for r in rs) / n,
        }
    manifest = {
        "output": os.path.abspath(args.out),
        "output_sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
        "records": n_records,
        "episodes": len(results),
        "command": " ".join(["python", os.path.basename(__file__), *(argv if argv is not None else sys.argv[1:])]),
        "packer_commit": commit(),
        "generated_at": ts,
        "beam_width": args.width, "top_k": TOP_K,
        "datasets": args.datasets, "seed_start": args.seed_start, "episodes_per_dataset": args.episodes,
        "evaluation_seeds_excluded": SEEDS,
        "episodes_identical_to_an_evaluation_file": overlaps,
        "path_template": "overhead (z = bin depth + 2) -> pre-descend (z + 1) -> target; the shape of all human demos",
        "per_dataset": per_dataset,
        "wall_time_s": time.perf_counter() - t0,
        "episodes_detail": [{k: r[k] for k in ("dataset", "seed", "episode", "n_items", "n_placed", "n_skipped",
                                                "expert_utilization", "greedy_utilization", "used_greedy", "seconds")} for r in results],
    }
    mpath = args.out[:-len(".jsonl.gz")] if args.out.endswith(".jsonl.gz") else os.path.splitext(args.out)[0]
    mpath += ".manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes_detail"}, indent=1))
    if overlaps:
        print(f"WARNING: {len(overlaps)} training episode(s) have the same box list as an evaluation file", file=sys.stderr)
    return manifest


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Launch one API model's sweep as parallel `evaluate.py` processes (T5.2/T6.2; D46/D47).

    python run_api_sweep.py --model openai/gpt-4o-mini --modes plain shuffle nofb --parallel 6
    python run_api_sweep.py --model openai/gpt-5-mini --reasoning low --parallel 20
    python run_api_sweep.py --model openai/gpt-4o-mini --datasets curriculum25 --seeds 0   # probe
    python run_api_sweep.py --report                     # $ and tokens per model x mode from results/

One process per (dataset, seed, mode); a run whose JSON already exists is skipped, so a
killed sweep is resumed by re-running the same command.  Each process logs to
results/<slug>/logs/<dataset>-seed<k>[.shuffle][.nofb].log (git-ignored).  Runs never
share output files, so any parallelism is safe; the API rate limit is the only coupling
(handled by the policy's backoff).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from harness.runner import run_file_name  # noqa: E402
from harness.sequences import DATASETS, SEEDS  # noqa: E402

MODES = {"plain": [], "shuffle": ["--shuffle-anchors"], "nofb": ["--no-feedback"]}


def job_cmd(model, dataset, seed, mode, reasoning, out_root, no_json_mode=False):
    cmd = [sys.executable, os.path.join(REPO_ROOT, "evaluate.py"), "--method", f"api:{model}",
           "--dataset", dataset, "--seed", str(seed), "--quiet", "--out", out_root, *MODES[mode]]
    if reasoning:
        cmd += ["--reasoning", reasoning]
    if no_json_mode:
        cmd += ["--no-json-mode"]
    return cmd


def run_job(model, dataset, seed, mode, reasoning, out_root, no_json_mode=False):
    rel = run_file_name(f"api:{model}", dataset, seed, mode == "shuffle", mode != "nofb")
    out_json = os.path.join(out_root, rel)
    if os.path.exists(out_json):
        return rel, "exists", 0.0
    log_dir = os.path.join(os.path.dirname(os.path.dirname(out_json)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset}-{os.path.basename(rel)[:-5]}.log")
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(job_cmd(model, dataset, seed, mode, reasoning, out_root, no_json_mode), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if rc == 0 and os.path.exists(out_json) else f"FAILED rc={rc} (see {log_path})"
    return rel, status, time.time() - t0


def report(out_root):
    rows = {}
    for path in sorted(glob.glob(os.path.join(out_root, "api-*", "*", "seed*.json"))):
        rec = json.load(open(path))
        u = rec.get("api_usage") or {}
        f = rec["flags"]
        mode = "shuffle" if f["shuffle_anchors"] else ("nofb" if not f["feedback"] else "plain")
        key = (rec["method"] + ("" if f.get("json_mode", True) else " [no-json-mode]"), mode, f.get("reasoning") or "-")
        r = rows.setdefault(key, {"runs": 0, "calls": 0, "retried": 0, "failed": 0, "in": 0, "out": 0, "reason": 0, "cost": 0.0, "wall_s": 0.0})
        r["runs"] += 1
        r["calls"] += u.get("calls") or 0
        r["retried"] += u.get("retried_calls") or 0
        r["failed"] += u.get("failed_calls") or 0
        r["in"] += u.get("prompt_tokens") or 0
        r["out"] += u.get("completion_tokens") or 0
        r["reason"] += u.get("reasoning_tokens") or 0
        r["cost"] += u.get("cost_usd") or 0.0
        r["wall_s"] += rec["total_wall_time_s"]
    print(f"{'method':38s} {'mode':7s} {'reas':5s} {'runs':>4s} {'calls':>6s} {'retried':>7s} {'failed':>6s} "
          f"{'tok_in':>9s} {'tok_out':>8s} {'tok_reas':>8s} {'cost_$':>8s} {'wall_min':>8s}")
    total = 0.0
    for (m, mode, reas), r in sorted(rows.items()):
        total += r["cost"]
        print(f"{m:38s} {mode:7s} {reas:5s} {r['runs']:4d} {r['calls']:6d} {r['retried']:7d} {r['failed']:6d} "
              f"{r['in']:9d} {r['out']:8d} {r['reason']:8d} {r['cost']:8.3f} {r['wall_s'] / 60:8.1f}")
    print(f"{'TOTAL':38s} {'':7s} {'':5s} {'':4s} {'':6s} {'':7s} {'':6s} {'':9s} {'':8s} {'':8s} {total:8.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="OpenRouter id, e.g. openai/gpt-4o-mini")
    ap.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["plain"])
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--reasoning", help="passed to evaluate.py --reasoning")
    ap.add_argument("--no-json-mode", action="store_true", help="passed to evaluate.py")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="results")
    ap.add_argument("--report", action="store_true", help="only print $ / tokens per model x mode from --out")
    args = ap.parse_args(argv)
    out_root = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    if args.report:
        report(out_root)
        return
    if not args.model:
        ap.error("--model is required (or --report)")

    jobs = [(d, s, m) for m in args.modes for d in args.datasets for s in args.seeds]
    print(f"{args.model}: {len(jobs)} runs, {args.parallel} in parallel, reasoning={args.reasoning}", flush=True)
    t0 = time.time()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_job, args.model, d, s, m, args.reasoning, out_root, args.no_json_mode) for d, s, m in jobs]
        for fut in as_completed(futs):
            rel, status, secs = fut.result()
            failed += status.startswith("FAILED")
            print(f"  {rel:60s} {status:12s} {secs / 60:5.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {failed} failed", flush=True)
    report(out_root)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

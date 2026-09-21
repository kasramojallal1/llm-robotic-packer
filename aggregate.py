#!/usr/bin/env python
"""
Aggregate harness runs: mean +- std over seeds per (method, dataset, flags).

    python aggregate.py                       # results/ -> markdown table on stdout
    python aggregate.py results/ --latex      # LaTeX tabular rows
    python aggregate.py results/ --csv out.csv --per-seed

Columns (all means over seeds unless noted):
  util   final utilization            LEC    largest empty cavity ratio (lower = one big cavity is NOT preserved; see T2.2)
  N      packed items                 skip   items with no feasible anchor
  exh    items that exhausted the budget     first  first-attempt validity rate
  retry  extra calls per attempted box       ijson  invalid-JSON responses
  pcol   path collisions (rejected paths)    lat    mean per-decision latency (pick+path calls), seconds
  p95    p95 per-call latency
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

COLUMNS = [
    ("util", lambda r: r["metrics"]["utilization_final"], "{:.3f}"),
    ("LEC", lambda r: r["metrics"]["largest_empty_cavity_ratio"], "{:.3f}"),
    ("N", lambda r: r["reliability"]["items_placed"], "{:.1f}"),
    ("skip", lambda r: r["reliability"]["items_skipped_no_anchor"], "{:.1f}"),
    ("exh", lambda r: r["reliability"]["items_budget_exhausted"], "{:.1f}"),
    ("first", lambda r: r["reliability"]["first_attempt_validity"], "{:.2f}"),
    ("retry", lambda r: r["reliability"]["retries_per_box"], "{:.2f}"),
    ("ijson", lambda r: r["reliability"]["invalid_json"], "{:.1f}"),
    ("pcol", lambda r: r["reliability"]["path_collisions"], "{:.1f}"),
    ("lat", lambda r: _call_latency(r), "{:.2f}"),
    ("p95", lambda r: max(r["reliability"]["latency_pick_s"]["p95"], r["reliability"]["latency_path_s"]["p95"]), "{:.2f}"),
]


def _call_latency(r):
    a, b = r["reliability"]["latency_pick_s"], r["reliability"]["latency_path_s"]
    n = a["n"] + b["n"]
    return (a["mean"] * a["n"] + b["mean"] * b["n"]) / n if n else 0.0


def load_runs(root: str) -> List[Dict]:
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        if r.get("schema", "").startswith("packi-eval-run/"):
            r["_path"] = p
            runs.append(r)
    return runs


def group_key(r: Dict) -> tuple:
    f = r["flags"]
    variant = "".join(["+shuffle" if f.get("shuffle_anchors") else "", "+nofb" if not f.get("feedback", True) else ""])
    return (r["method"], r["dataset"], variant)


def aggregate(runs: List[Dict]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    rows = []
    for key in sorted(groups):
        rs = groups[key]
        row = {"method": key[0], "dataset": key[1], "variant": key[2], "n_seeds": len(rs),
               "seeds": sorted(r["seed"] for r in rs), "commits": sorted({(r.get("git") or {}).get("commit") or "?" for r in rs})}
        for name, fn, _ in COLUMNS:
            vals = [float(fn(r)) for r in rs]
            row[name] = (float(np.mean(vals)), float(np.std(vals, ddof=0)))
        rows.append(row)
    return rows


def fmt(row: Dict, name: str, f: str, pm: str = "±") -> str:
    m, s = row[name]
    return f"{f.format(m)} {pm} {f.format(s)}"


def to_markdown(rows: List[Dict]) -> str:
    head = ["method", "dataset", "variant", "seeds"] + [c[0] for c in COLUMNS]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["method"], r["dataset"], r["variant"] or "-", str(r["n_seeds"])] + [fmt(r, n, f) for n, _, f in COLUMNS]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def to_latex(rows: List[Dict]) -> str:
    out = ["% method & dataset & variant & " + " & ".join(c[0] for c in COLUMNS) + r" \\"]
    for r in rows:
        cells = [r["method"].replace("_", r"\_"), r["dataset"], r["variant"] or "--"] + \
                [fmt(r, n, f, pm=r"$\pm$") for n, _, f in COLUMNS]
        out.append(" & ".join(cells) + r" \\")
    return "\n".join(out)


def per_seed_rows(runs: List[Dict]) -> List[Dict]:
    rows = []
    for r in sorted(runs, key=lambda r: (group_key(r), r["seed"])):
        row = {"method": r["method"], "dataset": r["dataset"], "variant": group_key(r)[2], "seed": r["seed"],
               "commit": (r.get("git") or {}).get("commit"), "file": r["_path"]}
        for name, fn, _ in COLUMNS:
            row[name] = float(fn(r))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--csv", help="write aggregated rows (or per-seed rows with --per-seed) to this CSV")
    ap.add_argument("--per-seed", action="store_true", help="one row per run instead of mean +- std")
    args = ap.parse_args(argv)

    runs = load_runs(args.root)
    if not runs:
        print(f"no run files under {args.root}")
        return
    rows = aggregate(runs)
    print(to_latex(rows) if args.latex else to_markdown(rows))
    mixed = [r for r in rows if len(r["commits"]) > 1]
    if mixed:
        print("\nWARNING: these groups mix runs from different commits:")
        for r in mixed:
            print(f"  {r['method']} {r['dataset']} {r['variant'] or '-'}: {r['commits']}")

    if args.csv:
        data = per_seed_rows(runs) if args.per_seed else [
            {**{k: v for k, v in r.items() if k not in ("seeds", "commits")},
             **{n: r[n][0] for n, _, _ in COLUMNS}, **{n + "_std": r[n][1] for n, _, _ in COLUMNS}} for r in rows]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

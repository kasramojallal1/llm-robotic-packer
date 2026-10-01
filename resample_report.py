#!/usr/bin/env python
"""
Equal-budget resampling control report (T6.4, R1.6, D100-D104).

    python resample_report.py                    # the three D90 models, results/
    python resample_report.py --models openai/gpt-4o-mini --out results

Per model x mode (plain = feedback, nofb = no feedback at temperature 0, resample = no
feedback, retries sampled at T): utilization mean +- std over seeds per dataset and the
mean over datasets, first-attempt validity, retries per box, share of boxes with a retry,
budget-exhausted boxes per run, how often a retry repeated the previous attempt's decision
(parsed, so whitespace does not count), agreement of each run's first decision with the
plain run on the same sequence, cost.  Then paired per-sequence differences over the 20
shared sequences (Wilcoxon signed-rank, paired t as a check).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASETS = ("curriculum25", "data1", "data2", "data3")
SEEDS = range(5)
MODELS = ("openai/gpt-4o-mini", "meta-llama/llama-4-maverick", "deepseek/deepseek-v3.1-terminus")
MODES = {"plain": "", "nofb": ".nofb", "resample": ".resample", "resample-t1": ".resample-t1"}


def load(out, model, dataset, seed, mode):
    slug = "api-" + model.replace("/", "-")
    p = os.path.join(out, slug, dataset, f"seed{seed}{MODES[mode]}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def decision(att):
    d = att.get("response")
    if not isinstance(d, dict):
        return ("invalid", att.get("raw"))
    if att["stage"] == "pick":
        return (d.get("rotation_index"), d.get("anchor_id"))
    return json.dumps(d.get("path"))


def retry_repeats(run):
    """(pick retries, of which repeat the previous pick, path retries, of which repeat the previous
    path for the same pick) -- parsed decisions."""
    n = rep = pn = prep = 0
    for b in run["boxes"]:
        picks = [a for a in b["attempts"] if a["stage"] == "pick"]
        for p, q in zip(picks, picks[1:]):
            n += 1
            rep += decision(p) == decision(q)
        paths = [a for a in b["attempts"] if a["stage"] == "path"]
        for p, q in zip(paths, paths[1:]):
            if q["pick_attempt"] == p["pick_attempt"]:
                pn += 1
                prep += decision(p) == decision(q)
    return n, rep, pn, prep


def first_decision(run):
    for b in run["boxes"]:
        if b["attempts"]:
            return decision(b["attempts"][0])
    return None


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def paired(a, b):
    from scipy import stats
    d = [x - y for x, y in zip(a, b)]
    w = stats.wilcoxon(a, b).pvalue if any(d) else 1.0
    t = stats.ttest_rel(a, b).pvalue if any(d) else 1.0
    return st.mean(d), w, t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args(argv)

    print("| Model | mode | runs | C-25 | DATA-1 | DATA-2 | DATA-3 | mean | first-try | retries/box | boxes w/ retry | exhausted/run | retry = previous (pick · path) | 1st decision = plain | cost $ |")
    print("|" + "---|" * 15)
    seq_util, seq_exh = {}, {}
    for model in args.models:
        for mode in MODES:
            runs = {(d, s): load(args.out, model, d, s, mode) for d in DATASETS for s in SEEDS}
            runs = {k: v for k, v in runs.items() if v}
            if not runs:
                continue
            per_ds = []
            cells = []
            for d in DATASETS:
                u = [runs[(d, s)]["metrics"]["utilization_final"] for s in SEEDS if (d, s) in runs]
                m, sd = ms(u)
                per_ds.append(m)
                cells.append(f"{m:.3f} ± {sd:.3f}")
            seq_util[(model, mode)] = {k: v["metrics"]["utilization_final"] for k, v in runs.items()}
            seq_exh[(model, mode)] = {k: v["reliability"]["items_budget_exhausted"] for k, v in runs.items()}
            rel = [r["reliability"] for r in runs.values()]
            boxes = [b for r in runs.values() for b in r["boxes"] if b["outcome"] != "skipped_no_anchor"]
            with_retry = sum(1 for b in boxes if len(b["attempts"]) > 2) / len(boxes)
            n, rep, pn, prep = map(sum, zip(*(retry_repeats(r) for r in runs.values())))
            agree = [first_decision(r) == first_decision(load(args.out, model, d, s, "plain"))
                     for (d, s), r in runs.items()]
            cost = sum((r.get("api_usage") or {}).get("cost_usd") or 0 for r in runs.values())
            print(f"| {model} | {mode} | {len(runs)} | " + " | ".join(cells) +
                  f" | **{st.mean(per_ds):.3f}** | {st.mean(x['first_attempt_validity'] for x in rel):.2f}"
                  f" | {st.mean(x['retries_per_box'] for x in rel):.2f} | {with_retry:.1%}"
                  f" | {st.mean(x['items_budget_exhausted'] for x in rel):.2f}"
                  f" | {f'{rep}/{n}' if n else '–'} · {f'{prep}/{pn}' if pn else '–'} | {sum(agree)}/{len(agree)} | {cost:.3f} |")

    for title, table, fmt in (("utilization", seq_util, "+.3f"), ("budget-exhausted boxes per run", seq_exh, "+.2f")):
        print(f"\nPaired {title} differences over shared sequences (mean diff, Wilcoxon p, paired-t p):")
        for model in args.models:
            for a, b in (("plain", "resample"), ("plain", "nofb"), ("resample", "nofb"),
                         ("plain", "resample-t1"), ("resample-t1", "nofb"), ("resample-t1", "resample")):
                ua, ub = table.get((model, a)), table.get((model, b))
                if not ua or not ub:
                    continue
                keys = sorted(set(ua) & set(ub))
                d, w, t = paired([ua[k] for k in keys], [ub[k] for k in keys])
                print(f"  {model:34s} {a:8s} - {b:8s} n={len(keys):2d}  {d:{fmt}}  p_W={w:.3f}  p_t={t:.3f}")


if __name__ == "__main__":
    main()

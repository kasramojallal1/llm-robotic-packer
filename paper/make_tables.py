#!/usr/bin/env python3
"""Generate every results table of the revised Packi paper from harness run records.

Numbers are never typed by hand: rerun this script after new results are merged.

    python revision/make_tables.py --roots llm-robotic-packer/results [extra roots...] \
        --out Wiley_Journal_of_Robotics/tables

Each root is a directory laid out as <method>/<dataset>/seed<k>[.shuffle][.nofb][.tpath][.resample].json
(the packer harness layout). Later roots override earlier ones for the same run file.
Missing cells are emitted as \\TBD{...} so they stay visible in the PDF.

Conventions (decision D95): utilization reported as mean and *sample* standard deviation (ddof=1)
over the five sequences of a benchmark; "Mean" column = mean of the four benchmark means;
paired tests use the 20 per-sequence utilizations (Wilcoxon signed-rank, paired t as a check).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

DATASETS = ["curriculum25", "data1", "data2", "data3"]
DS_LABEL = {"curriculum25": "CURRIC.-25", "data1": "DATA-1", "data2": "DATA-2", "data3": "DATA-3"}
UNSEEN = ["data1", "data1-b15", "data1-b12x8x10", "data1-s1to6"]
UNSEEN_LABEL = {"data1": "DATA-1 (seen)", "data1-b15": "Bin $15^3$", "data1-b12x8x10": "Bin $12{\\times}8{\\times}10$",
                "data1-s1to6": "Sides 1--6"}
SEEDS = [0, 1, 2, 3, 4]

API = [  # folder, display, footnote mark
    ("api-anthropic-claude-haiku-4.5", "Claude Haiku 4.5", "$^{c}$"),
    ("api-x-ai-grok-4.3", "Grok 4.3", "$^{a}$"),
    ("api-openai-gpt-5-mini", "GPT-5-mini", "$^{a}$"),
    ("api-google-gemini-2.5-flash", "Gemini 2.5 Flash", "$^{b}$"),
    ("api-qwen-qwen3-max", "Qwen3-Max", ""),
    ("api-openai-gpt-4o", "GPT-4o", ""),
    ("api-openai-gpt-4o-mini", "GPT-4o-mini", ""),
    ("api-qwen-qwen3-vl-32b-instruct", "Qwen3-VL-32B", ""),
    ("api-deepseek-deepseek-v3.1-terminus", "DeepSeek V3.1", ""),
    ("api-meta-llama-llama-4-maverick", "Llama 4 Maverick", ""),
    ("api-meta-llama-llama-4-scout", "Llama 4 Scout", "$^{d}$"),
    ("api-google-gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", ""),
]
GROUPS = [
    ("Reference (not online)", [("oracle", "Oracle (privileged, sees full sequence)")]),
    ("Non-LLM baselines", [("random", "Random"), ("greedy", "Greedy (Eq.~\\ref{eq:score})"),
                           ("ranker-h", "Ranker-H"), ("ranker-e", "Ranker-E")]),
    ("Dedicated DRL packer", [("gopt-sample", "GOPT (2 orientations)"),
                              ("gopt-control-greedy-2rot", "Greedy, 2 orientations (control)")]),
    ("Unadapted open LLMs", [("base-llama", "Llama 3.2 3B"), ("base-qwen3-4b", "Qwen3-4B")]),
    ("Packi (ours): fine-tuned open LLMs", [("packi", "Packi-Llama-H (human demos)"),
                                     ("packi-e", "Packi-Llama-E (expert demos)"),
                                     ("qwen3-4b-h", "Packi-Qwen-H (human demos)"),
                                     ("qwen3-4b-e", "Packi-Qwen-E (expert demos)")]),
]
NAME = {m: d for _, rows in GROUPS for m, d in rows}
NAME.update({f: d for f, d, _ in API})
NAME["human"] = "Human (first author)"
SHORT = {"packi": "P-Llama-H", "packi-e": "P-Llama-E", "qwen3-4b-h": "P-Qwen-H", "qwen3-4b-e": "P-Qwen-E",
         "greedy": "Greedy", "ranker-h": "Ranker-H", "ranker-e": "Ranker-E", "gopt-sample": "GOPT",
         "api-anthropic-claude-haiku-4.5": "Claude Haiku 4.5", "api-x-ai-grok-4.3": "Grok 4.3",
         "api-openai-gpt-4o-mini": "GPT-4o-mini", "api-meta-llama-llama-4-maverick": "Llama 4 Maverick",
         "api-deepseek-deepseek-v3.1-terminus": "DeepSeek V3.1"}
SHORT.update({"oracle": "Oracle (privileged)", "random": "Random", "gopt-control-greedy-2rot": "Greedy, 2 orient.",
              "base-llama": "Llama 3.2 3B (unadapted)", "base-qwen3-4b": "Qwen3-4B (unadapted)", "human": "Human"})
SHORT.update({f: d for f, d, _ in API if f not in SHORT})

VAR_RE = re.compile(r"seed(\d+)((?:\.[a-z]+)*)\.json$")


def load(roots):
    runs = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "*", "*", "seed*.json")):
            m = VAR_RE.search(os.path.basename(f))
            if not m:
                continue
            method = os.path.basename(os.path.dirname(os.path.dirname(f)))
            dataset = os.path.basename(os.path.dirname(f))
            variant = m.group(2).lstrip(".") or "plain"
            with open(f) as fh:
                runs[(method, dataset, variant, int(m.group(1)))] = json.load(fh)
    return runs


def series(runs, method, dataset, variant="plain", key=lambda r: r["metrics"]["utilization_final"]):
    vals = [key(runs[(method, dataset, variant, s)]) for s in SEEDS if (method, dataset, variant, s) in runs]
    return vals if len(vals) == len(SEEDS) else None


def ms(vals, nd=3):
    if vals is None:
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def fmt(x, tbd, nd=3, bold=False):
    if x is None:
        return f"\\TBD{{{tbd}}}"
    m, s = x
    core = f"{m:.{nd}f}" if s is None else f"{m:.{nd}f}\\,$\\pm$\\,{s:.{nd}f}"
    return f"\\textbf{{{core}}}" if bold else core


def method_means(runs, method, datasets=DATASETS, variant="plain"):
    cells = [ms(series(runs, method, d, variant)) for d in datasets]
    mean = float(np.mean([c[0] for c in cells])) if all(cells) else None
    return cells, mean


# ------------------------------------------------------------------ main table
def table_main(runs):
    online = [m for _, rows in GROUPS[1:] for m, _ in rows] + [f for f, _, _ in API]
    best = {}
    for d in DATASETS:
        vals = [(ms(series(runs, m, d)) or (-1, 0))[0] for m in online]
        best[d] = max(vals)
    means = {m: method_means(runs, m)[1] for m in online}
    best_mean = max(v for v in means.values() if v is not None)

    lines = []
    def row(method, label):
        cells, mean = method_means(runs, method)
        out = [label]
        for d, c in zip(DATASETS, cells):
            out.append(fmt(c, "?", bold=(c is not None and method != "oracle" and abs(c[0] - best[d]) < 5e-4)))
        out.append("\\TBD{?}" if mean is None else
                   (f"\\textbf{{{mean:.3f}}}" if method != "oracle" and abs(mean - best_mean) < 5e-4 else f"{mean:.3f}"))
        lines.append(" & ".join(out) + " \\\\")

    for title, rows in GROUPS:
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for m, label in rows:
            row(m, label)
        lines.append("\\addlinespace[2pt]")
    lines.append("\\multicolumn{6}{@{}l}{\\textit{API LLMs (OpenRouter)}} \\\\")
    api_sorted = sorted(API, key=lambda a: -(method_means(runs, a[0])[1] or 0))
    for f, d, mark in api_sorted:
        row(f, d + mark)
    lines.append("\\addlinespace[2pt]")
    h = ms(series(runs, "human", "curriculum25"))
    lines.append("\\multicolumn{6}{@{}l}{\\textit{Human reference}} \\\\")
    lines.append(f"Human (first author) & {fmt(h, 'pending')} & -- & -- & -- & -- \\\\")
    return "\n".join(lines)


# ------------------------------------------------------------ reliability table
REL_METHODS = (["packi", "packi-e", "qwen3-4b-h", "qwen3-4b-e", "base-llama", "base-qwen3-4b"]
               + [f for f, _, _ in API] + ["gopt-sample"])


def pooled(runs, method, fn):
    vals = [fn(runs[k]) for k in runs if k[0] == method and k[1] in DATASETS and k[2] == "plain"]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def table_reliability(runs):
    hw_short = {"NVIDIA GeForce RTX 5090": "RTX 5090", "NVIDIA RTX PRO 4500 Blackwell": "RTX PRO 4500"}
    lines = []
    for m in REL_METHODS:
        fa, n = pooled(runs, m, lambda r: r["reliability"]["first_attempt_validity"])
        if fa is None:
            lines.append(f"{NAME.get(m, m)} & \\multicolumn{{6}}{{l}}{{\\TBD{{pending}}}} \\\\")
            continue
        extra = pooled(runs, m, lambda r: r["reliability"]["retries_per_box"])[0]
        drop = pooled(runs, m, lambda r: r["reliability"]["items_budget_exhausted"])[0]
        ijs = pooled(runs, m, lambda r: r["reliability"]["invalid_json"])[0]
        col = pooled(runs, m, lambda r: r["reliability"]["path_collisions"])[0]
        lat = pooled(runs, m, lambda r: (r["reliability"]["latency_box_end_to_end_s"] or {}).get("mean") or 0.0)[0]
        gpus = {runs[k]["hardware"].get("gpu") for k in runs if k[0] == m and k[1] in DATASETS and k[2] == "plain"}
        hw = "API" if m.startswith("api-") else "/".join(sorted(hw_short.get(g, g or "CPU") for g in gpus))
        lines.append(f"{NAME.get(m, m)} & {100*fa:.0f}\\% & {extra:.2f} & {drop:.1f} & {ijs:.1f} & {col:.1f} & "
                     f"{lat:.2f} & {hw} \\\\")
        if n != 20:
            lines[-1] = lines[-1].replace(" \\\\", f" \\TBD{{{n}/20 runs}} \\\\")
    return "\n".join(lines)


# --------------------------------------------------------------- ablation table
def table_ablation(runs):
    lines = []
    for m in ["packi", "packi-e", "qwen3-4b-h", "qwen3-4b-e", "api-openai-gpt-4o-mini",
              "api-meta-llama-llama-4-maverick", "api-deepseek-deepseek-v3.1-terminus"]:
        p = method_means(runs, m)[1]
        s = method_means(runs, m, variant="shuffle")[1]
        nf = method_means(runs, m, variant="nofb")[1]
        rs = method_means(runs, m, variant="resample")[1]
        def c(x, want=True):
            return "--" if not want else ("\\TBD{?}" if x is None else f"{x:.3f}")
        api = m.startswith("api-")
        d = "\\TBD{?}" if (p is None or s is None) else f"{s - p:+.3f}"
        lines.append(f"{SHORT.get(m, NAME.get(m, m))} & {c(p)} & {c(s)} & {d} & {c(nf, api or m == 'packi')} & {c(rs, api)} \\\\")
    return "\n".join(lines)


# ------------------------------------------------------------ paired statistics
def paired(runs, a, b):
    xa, xb = [], []
    for d in DATASETS:
        sa, sb = series(runs, a, d), series(runs, b, d)
        if sa is None or sb is None:
            return None
        xa += sa
        xb += sb
    xa, xb = np.array(xa), np.array(xb)
    diff = xa - xb
    w = stats.wilcoxon(xa, xb, zero_method="wilcox") if np.any(diff != 0) else None
    t = stats.ttest_rel(xa, xb)
    return dict(mean=float(diff.mean()), win=int((diff > 1e-9).sum()), tie=int((abs(diff) <= 1e-9).sum()),
                loss=int((diff < -1e-9).sum()), p_w=(w.pvalue if w else 1.0), p_t=float(t.pvalue))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run, m = [0.0] * len(ps), 0.0, len(ps)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


STATS_PAIRS = [("packi-e", "greedy"), ("packi-e", "packi"), ("packi-e", "ranker-e"),
               ("packi-e", "api-anthropic-claude-haiku-4.5"), ("packi-e", "api-x-ai-grok-4.3"),
               ("packi-e", "gopt-sample"), ("packi-e", "qwen3-4b-e"),
               ("qwen3-4b-e", "greedy"), ("qwen3-4b-e", "api-x-ai-grok-4.3"), ("qwen3-4b-e", "api-anthropic-claude-haiku-4.5"),
               ("qwen3-4b-e", "gopt-sample"), ("qwen3-4b-e", "qwen3-4b-h"),
               ("packi", "greedy"), ("packi", "ranker-h"), ("qwen3-4b-h", "packi"), ("gopt-sample", "greedy")]


def table_stats(runs):
    res = [paired(runs, a, b) for a, b in STATS_PAIRS]
    ok = [r for r in res if r is not None]
    adj = iter(holm([r["p_w"] for r in ok]))
    lines = []
    for (a, b), r in zip(STATS_PAIRS, res):
        lab = f"{SHORT.get(a, a)} vs {SHORT.get(b, b)}"
        if r is None:
            lines.append(f"{lab} & \\multicolumn{{5}}{{l}}{{\\TBD{{?}}}} \\\\")
        else:
            h = next(adj)
            hs = f"\\textbf{{{h:.3f}}}" if h < 0.05 else f"{h:.3f}"
            lines.append(f"{lab} & {r['mean']:+.3f} & {r['win']}/{r['tie']}/{r['loss']} & {r['p_w']:.3f} & {hs} & {r['p_t']:.3f} \\\\")
    return "\n".join(lines)


# ------------------------------------------------------------ appendix tables
PERSEED_ORDER = (["oracle", "random", "greedy", "ranker-h", "ranker-e", "gopt-sample", "gopt-control-greedy-2rot",
                  "base-llama", "base-qwen3-4b", "packi", "packi-e", "qwen3-4b-h", "qwen3-4b-e"]
                 + [f for f, _, _ in API] + ["human"])


def table_perseed(runs, dataset):
    lines = []
    for m in PERSEED_ORDER:
        cells = []
        for s in SEEDS:
            k = (m, dataset, "plain", s)
            if k not in runs:
                cells.append("--")
                continue
            r = runs[k]
            cells.append(f"{r['metrics']['utilization_final']:.3f} ({r['reliability']['items_placed']})")
        if all(c == "--" for c in cells):
            continue
        n = runs.get((m, dataset, "plain", 0), {}).get("n_items", "")
        lines.append(f"{SHORT.get(m, NAME.get(m, m))} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_allpairs(runs):
    lines = []
    for m in [x for x in PERSEED_ORDER if x not in ("human", "oracle")] + ["oracle"]:
        cells = []
        for ref in ["qwen3-4b-e", "packi-e"]:
            if m == ref:
                cells.append("\\multicolumn{2}{c}{--}")
                continue
            r = paired(runs, ref, m)
            cells.append("\\multicolumn{2}{c}{\\TBD{?}}" if r is None else f"{r['mean']:+.3f} & {r['p_w']:.3f}")
        lines.append(f"{SHORT.get(m, NAME.get(m, m))} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


# ----------------------------------------------------------- PUSNet comparison
PUSNET = [  # copied verbatim from PUSNet Table 1 (Yang et al. 2025): (Uti, Num, Sta) x DATA-1/2/3
    ("Random", [(0.363, 15.66, 0.131), (0.348, 9.61, 0.134), (0.366, 10.16, 0.127)]),
    ("Column Building~\\cite{mahvash2018column}", [(0.629, 25.75, 0.125), (0.566, 14.76, 0.124), (0.571, 15.26, 0.127)]),
    ("Floor Building~\\cite{sweep2003three}", [(0.634, 25.91, 0.127), (0.557, 14.52, 0.139), (0.568, 15.17, 0.136)]),
    ("First Fit~\\cite{falkenauer1996hybrid}", [(0.611, 24.92, 0.129), (0.571, 14.94, 0.132), (0.572, 15.23, 0.136)]),
    ("Corner Point~\\cite{martello2000three}", [(0.662, 26.76, 0.121), (0.654, 17.05, 0.139), (0.641, 17.10, 0.124)]),
    ("Extreme Point~\\cite{crainic2008extreme}", [(0.667, 27.73, 0.126), (0.584, 15.20, 0.115), (0.586, 15.62, 0.129)]),
    ("Empty Max Space~\\cite{ha2017online}", [(0.669, 27.79, 0.114), (0.652, 16.99, 0.118), (0.649, 17.31, 0.124)]),
    ("Zhao et al.~\\cite{zhao2021online}", [(0.687, 28.61, 0.103), (0.632, 16.39, 0.114), (0.642, 16.97, 0.111)]),
    ("Yang et al.~\\cite{yang2021packerbot}", [(0.704, 29.13, 0.097), (0.667, 17.31, 0.104), (0.675, 18.05, 0.105)]),
    ("Zhao et al.~\\cite{zhao2022learning}", [(0.834, 32.91, 0.084), (0.819, 20.93, 0.087), (0.813, 21.37, 0.092)]),
    ("PUSNet~\\cite{yang2025learning}", [(0.861, 35.42, 0.063), (0.835, 22.57, 0.068), (0.836, 22.15, 0.070)]),
]


def table_pusnet(runs):
    lines = ["\\multicolumn{10}{@{}l}{\\textit{Published results, copied from PUSNet Table 1~\\cite{yang2025learning} "
             "(their sequences and rules)}} \\\\"]
    for name, vals in PUSNET:
        lines.append(name + " & " + " & ".join(f"{u:.3f} & {n:.2f} & {s:.3f}" for u, n, s in vals) + " \\\\")
    lines.append("\\addlinespace[2pt]")
    lines.append("\\multicolumn{10}{@{}l}{\\textit{Run by us on our sequences built with the same recipe "
                 "(5 sequences each)}} \\\\")
    for m in ["greedy", "gopt-sample", "packi", "packi-e", "qwen3-4b-e", "oracle"]:
        cells = []
        for d in ["data1", "data2", "data3"]:
            u = series(runs, m, d)
            n = series(runs, m, d, key=lambda r: r["reliability"]["items_placed"])
            if u is None:
                cells.append("\\multicolumn{3}{c}{\\TBD{?}}")
            else:
                cells.append(f"{np.mean(u):.3f} & {np.mean(n):.2f} & {np.std(u, ddof=1):.3f}")
        lines.append(NAME.get(m, m) + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


# ------------------------------------------------------------- unseen settings
def table_unseen(runs):
    lines = []
    for m in ["greedy", "random", "packi", "packi-e", "api-google-gemini-2.5-flash", "api-openai-gpt-4o-mini", "oracle"]:
        cells = [fmt(ms(series(runs, m, d)), "?") for d in UNSEEN]
        lines.append(NAME.get(m, m) + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\addlinespace[2pt]")
    lines.append("\\multicolumn{5}{@{}l}{\\textit{Pick by the LLM, path by the template}} \\\\")
    for m in ["packi", "packi-e"]:
        cells = ["--"] + [fmt(ms(series(runs, m, d, "tpath")), "n/a") if d == "data1-b15" else "--" for d in UNSEEN[1:]]
        lines.append(NAME.get(m, m) + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = load(a.roots)
    os.makedirs(a.out, exist_ok=True)
    for name, fn in [("main", table_main), ("reliability", table_reliability), ("ablation", table_ablation),
                     ("stats", table_stats), ("pusnet", table_pusnet), ("unseen", table_unseen),
                     ("allpairs", table_allpairs)] + [(f"perseed_{d}", (lambda d: lambda r: table_perseed(r, d))(d)) for d in DATASETS]:
        with open(os.path.join(a.out, f"rows_{name}.tex"), "w") as f:
            f.write(f"% generated by revision/make_tables.py -- do not edit by hand\n{fn(runs)}\n")
    print(f"{len(runs)} run records from {len(a.roots)} roots -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Qualitative figure for the Packi paper: final packings of four methods on two shared sequences,
drawn directly from the harness run records (placed_boxes). Rerun to regenerate.

    python revision/make_qual_figure.py --results llm-robotic-packer/results \
        --out Wiley_Journal_of_Robotics/images/qualitative_packings.pdf
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

ROWS = [("curriculum25", 1, "CURRICULUM-25, sequence 1"), ("data2", 1, "DATA-2, sequence 1")]
COLS = [("greedy", "Greedy"), ("packi", "Packi-Llama-H"), ("qwen3-4b-e", "Packi-Qwen-E"), ("oracle", "Oracle (privileged)")]


def draw(ax, run):
    L, W, H = run["bin_dims"]
    boxes = run["placed_boxes"]
    n = max(1, len(boxes) - 1)
    for k, b in enumerate(boxes):
        (x, y, z), (sx, sy, sz) = b["position"], b["size"]
        ax.bar3d(x, y, z, sx, sy, sz, color=cm.viridis(k / n), alpha=0.85, edgecolor="k", linewidth=0.25, shade=True)
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))
    ax.view_init(elev=28, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.08)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig = plt.figure(figsize=(7.2, 3.9))
    for r, (ds, seed, rlabel) in enumerate(ROWS):
        for c, (m, clabel) in enumerate(COLS):
            run = json.load(open(os.path.join(a.results, m, ds, f"seed{seed}.json")))
            ax = fig.add_subplot(len(ROWS), len(COLS), r * len(COLS) + c + 1, projection="3d")
            draw(ax, run)
            u = run["metrics"]["utilization_final"]
            n = run["reliability"]["items_placed"]
            ax.set_title(f"{clabel}\n$U$ = {u:.3f}, {n}/{run['n_items']} boxes", fontsize=7, pad=-2)
            if c == 0:
                ax.text2D(-0.12, 0.5, rlabel, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=7)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.01, wspace=0.0, hspace=0.12)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

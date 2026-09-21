"""
Fixed box sequences (D21, D22, D23, D28).

One JSON file per (dataset, seed) under data/sequences/<dataset>/seed<k>.json.
Every method reads the same file; a box that fits nowhere at evaluation time is
skipped and counted.  Regenerate (and verify nothing drifted) with

    python -m harness.sequences --write        # writes all datasets x seeds 0-4
    python -m harness.sequences --check        # fails if files differ from generator

Generators:
  curriculum25  port of main._sample_size_for_phase with bin fill replaced by
                cumulative sampled volume / bin volume (D23)
  data1         exact tiling of the bin, dims in [2, dim/2] (Yang et al.)
  data2         64-template catalog (sides 1..5, at most one side == 1), random order
  data3         same catalog, sorted by decreasing volume
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from typing import Dict, List, Tuple

BIN_DIMS = [10, 10, 10]
SEEDS = [0, 1, 2, 3, 4]
N_ITEMS = {"curriculum25": 25, "data1": 40, "data2": 60, "data3": 80}
DATASETS = list(N_ITEMS)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEQ_DIR = os.path.join(REPO_ROOT, "data", "sequences")


# ------------------------ CURRICULUM-25 (offline port of main.py) ------------------------

def _sample_size_for_phase(rng: random.Random, fill: float, big_cubes_used: int) -> List[int]:
    """Verbatim logic of main._sample_size_for_phase, using an explicit rng."""
    if fill < 0.35:
        weights = {2: 1, 3: 3, 4: 4, 5: 3}
        allow_555 = big_cubes_used < 2 and rng.random() < 0.05
    elif fill < 0.7:
        weights = {2: 2, 3: 4, 4: 4, 5: 2}
        allow_555 = False
    else:
        weights = {2: 5, 3: 4, 4: 3, 5: 1}
        allow_555 = False

    def draw_dim() -> int:
        bag = []
        for v, w in weights.items():
            bag.extend([v] * w)
        return rng.choice(bag)

    if allow_555 and rng.random() < 0.5:
        return [5, 5, 5]

    dims = []
    count5 = 0
    for _ in range(3):
        v = draw_dim()
        if v == 5:
            if count5 >= 1:
                v = rng.choice([2, 3, 4])
            else:
                count5 += 1
        dims.append(v)
    return dims


def gen_curriculum25(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    """
    main.py's curriculum sampler without the bin: the phase is chosen from the
    cumulative volume of the boxes sampled so far (D23). The state-dependent
    shrink loop is dropped; infeasible boxes are skipped at evaluation time.
    """
    rng = random.Random(seed)
    V = bin_dims[0] * bin_dims[1] * bin_dims[2]
    boxes: List[List[int]] = []
    sampled_volume = 0
    big_cubes = 0
    for _ in range(n_items):
        fill = sampled_volume / V
        size = _sample_size_for_phase(rng, fill, big_cubes)
        # main._generate_feasible_box: avoid degenerate small volume early
        if fill < 0.2 and size[0] * size[1] * size[2] < 24:
            i = rng.randrange(3)
            size[i] = min(5, max(size[i], 4))
        boxes.append(size)
        sampled_volume += size[0] * size[1] * size[2]
        if sorted(size) == [5, 5, 5]:
            big_cubes += 1
    return boxes


# ------------------------ DATA-1 / DATA-2 / DATA-3 (from run_paper_datasets.py) ------------------------

def _vol(s) -> int:
    return s[0] * s[1] * s[2]


def _template_sizes_1to5() -> List[Tuple[int, int, int]]:
    sizes = []
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                if [a, b, c].count(1) <= 1:
                    sizes.append((a, b, c))
    return sizes


def _balanced_sample_templates(k: int, seed: int) -> List[Tuple[int, int, int]]:
    rng = random.Random(seed)
    all_sizes = sorted(_template_sizes_1to5(), key=_vol)
    n = len(all_sizes)
    t1, t2, t3 = all_sizes[: n // 3], all_sizes[n // 3 : 2 * n // 3], all_sizes[2 * n // 3 :]
    k1, k2 = int(round(0.30 * k)), int(round(0.40 * k))
    out: List[Tuple[int, int, int]] = []
    out += rng.sample(t1, min(k1, len(t1)))
    out += rng.sample(t2, min(k2, len(t2)))
    remain3 = [s for s in t3 if s not in out]
    need = k - len(out)
    if need > 0:
        out += rng.sample(remain3, min(need, len(remain3)))
    remain = [s for s in all_sizes if s not in out]
    while len(out) < k and remain:
        out.append(remain.pop(0))
    return out[:k]


def gen_data1_exact(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    """
    DATA-1: exactly n_items boxes with each dim in [2, floor(dim/2)] that
    exactly tile the bin (sum of volumes == L*W*H), then shuffled.
    """
    rng = random.Random(seed)
    L, W, H = bin_dims
    if min(L, W, H) < 4:
        raise ValueError("Each bin dimension must be >= 4 to allow parts in [2, floor(dim/2)].")

    def factor_triples(n: int):
        out = []
        for a in range(2, n + 1):
            if n % a:
                continue
            n2 = n // a
            for b in range(2, n2 + 1):
                if n2 % b:
                    continue
                c = n2 // b
                if c >= 2:
                    out.append((a, b, c))
        return out

    def feasible_k(dim: int, k: int) -> bool:
        return (2 * k <= dim) and (k * (dim // 2) >= dim)

    def partition_dim(dim: int, k: int) -> List[int]:
        if not feasible_k(dim, k):
            raise ValueError("No per-axis partition fits the bounds.")
        max_part = dim // 2
        parts = [2] * k
        remaining = dim - 2 * k
        i = 0
        while remaining > 0:
            give = min(remaining, max_part - parts[i])
            parts[i] += give
            remaining -= give
            i = (i + 1) % k
        return parts

    triples = factor_triples(n_items)
    if not triples:
        raise ValueError("n_items must factor as a*b*c with a,b,c >= 2.")
    triples.sort(key=lambda t: max(t) - min(t))
    choice = None
    for a, b, c in triples:
        for kL, kW, kH in sorted(set(itertools.permutations((a, b, c)))):
            if feasible_k(L, kL) and feasible_k(W, kW) and feasible_k(H, kH):
                choice = (kL, kW, kH)
                break
        if choice:
            break
    if not choice:
        raise ValueError("n_items cannot fit given bin and DATA-1 bounds (try a different n_items).")

    kL, kW, kH = choice
    segL, segW, segH = partition_dim(L, kL), partition_dim(W, kW), partition_dim(H, kH)
    boxes = [[l, w, h] for l in segL for w in segW for h in segH]
    rng.shuffle(boxes)
    return boxes


def gen_data2(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    templates = _balanced_sample_templates(64, seed)
    boxes = [list(rng.choice(templates)) for _ in range(n_items)]
    rng.shuffle(boxes)
    return boxes


def gen_data3(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    templates = _balanced_sample_templates(64, seed ^ 1337)
    boxes = [list(rng.choice(templates)) for _ in range(n_items)]
    boxes.sort(key=lambda s: _vol(s), reverse=True)
    return boxes


GENERATORS = {
    "curriculum25": gen_curriculum25,
    "data1": gen_data1_exact,
    "data2": gen_data2,
    "data3": gen_data3,
}


# ------------------------ files ------------------------

def sequence_path(dataset: str, seed: int) -> str:
    return os.path.join(SEQ_DIR, dataset, f"seed{seed}.json")


def make_sequence(dataset: str, seed: int, bin_dims: List[int] = BIN_DIMS) -> Dict:
    if dataset not in GENERATORS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    n = N_ITEMS[dataset]
    boxes = GENERATORS[dataset](list(bin_dims), n, seed)
    return {
        "dataset": dataset,
        "seed": seed,
        "bin_dims": list(bin_dims),
        "n_items": n,
        "generator": GENERATORS[dataset].__name__,
        "boxes": [list(map(int, b)) for b in boxes],
    }


def dumps_sequence(seq: Dict) -> str:
    return json.dumps(seq, indent=1) + "\n"


def write_sequence(dataset: str, seed: int) -> str:
    path = sequence_path(dataset, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(dumps_sequence(make_sequence(dataset, seed)))
    return path


def load_sequence(dataset: str, seed: int) -> Dict:
    path = sequence_path(dataset, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run `python -m harness.sequences --write`"
        )
    with open(path) as f:
        return json.load(f)


def check_all() -> List[str]:
    """Return the list of committed files that differ from the generator output."""
    bad = []
    for ds in DATASETS:
        for s in SEEDS:
            path = sequence_path(ds, s)
            expected = dumps_sequence(make_sequence(ds, s))
            if not os.path.exists(path) or open(path).read() != expected:
                bad.append(path)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write all datasets x seeds")
    ap.add_argument("--check", action="store_true", help="verify committed files match the generators")
    args = ap.parse_args()
    if args.write:
        for ds in DATASETS:
            for s in SEEDS:
                print("wrote", os.path.relpath(write_sequence(ds, s), REPO_ROOT))
    if args.check:
        bad = check_all()
        if bad:
            print("MISMATCH:", *bad, sep="\n  ")
            raise SystemExit(1)
        print(f"all {len(DATASETS) * len(SEEDS)} sequence files match the generators")
    if not (args.write or args.check):
        ap.print_help()


if __name__ == "__main__":
    main()

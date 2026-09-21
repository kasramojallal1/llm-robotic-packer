"""
Fixed box sequences (D21, D22, D23, D28).

One JSON file per (dataset, seed) under data/sequences/<dataset>/seed<k>.json.
Every method reads the same file; a box that fits nowhere at evaluation time is
skipped and counted.  Regenerate (and verify nothing drifted) with

    python -m harness.sequences --write        # writes all datasets x seeds 0-4
    python -m harness.sequences --check        # fails if files differ from generator

Generators:
  curriculum25  port of main._sample_size_for_phase with bin fill replaced by
                cumulative sampled volume / bin volume (D23); 25 items
  data1         cutting stock (Zhao et al. 2021, Alg. 3): the bin is cut into
                pieces with every side in [2, 5]; random shuffle (PUSNet DATA-1)
  data2         cutting stock constrained to the 64 PUSNet templates (sides 1..5,
                at most one side == 1, 30/40/30 by volume class); CUT-1 order
  data3         the same cut as data2 for the same seed; CUT-2 order
  (D31, T0.12, T0.13.)  data1/2/3 tile the bin exactly, so a 100 % packing exists
  and the item count is whatever the cut yields (`n_items` = actual count).
  Each file stores the cut position of every box (`positions`) and the cut
  parameters (`cut`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

BIN_DIMS = [10, 10, 10]
SEEDS = [0, 1, 2, 3, 4]
N_ITEMS = {"curriculum25": 25}          # data1/2/3: the count emerges from the cut
DATASETS = ["curriculum25", "data1", "data2", "data3"]

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


# ------------------------ DATA-1 / DATA-2 / DATA-3: cutting stock (D31) ------------------------
#
# Zhao et al. 2021 (AAAI, supplement Algorithm 3 "Benchmark Construction"):
#   L_invalid = {bin}, L_valid = {}
#   while L_invalid: pop a random piece; pick a random axis whose extent > a_max;
#     split the piece at a random point along it; a sub-piece with a_min <= every
#     side <= a_max goes to L_valid, otherwise back to L_invalid.
#   CUT-1: sort L_valid by the z of each piece's front-left-bottom corner, ties random.
#   CUT-2: height map H = 0; repeatedly pop a random piece whose corner z equals H
#          under its footprint; raise H there by its height.
# PUSNet (Sec. 4.1) uses this for DATA-1 (sides in [2, L/2], shuffled) and, with a
# 64-template validity rule, for DATA-2 / DATA-3 ("sorted according to different
# rules described in [10]" -- read as CUT-1 / CUT-2).
#
# Two points the papers leave open, decided 2026-09-21 (D32):
#   * "randomly split": we split only at points that leave both halves >= a_min
#     along the cut axis. A literal uniform split can create a piece with a side
#     < a_min that can never become valid (Algorithm 3 does not terminate).
#   * "according to the templates": a piece is valid iff its sorted sides equal a
#     template's sorted sides (rotation-insensitive). A template piece can still
#     need splitting (all sides <= a_max but not a template); then an axis with
#     extent >= 2*a_min is cut. A piece with no legal cut is dead and the whole
#     cut restarts with the next RNG draw; restarts are counted in the file.

Piece = Tuple[Tuple[int, int, int], Tuple[int, int, int]]   # (FLB position, size)

TEMPLATE_SEED = 0            # one fixed template set for every seed (D32)
N_TEMPLATES = 64
CUT_MAX_RESTARTS = 10_000


def _vol(s) -> int:
    return s[0] * s[1] * s[2]


def _template_sizes_1to5() -> List[Tuple[int, int, int]]:
    """PUSNet's candidate pool: sides 1..5, at most one side equal to 1 (100 sizes)."""
    sizes = []
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                if [a, b, c].count(1) <= 1:
                    sizes.append((a, b, c))
    return sizes


def _balanced_sample_templates(k: int, seed: int) -> List[Tuple[int, int, int]]:
    """k templates, ~30 % / 40 % / 30 % from the small / medium / large volume terciles
    of the candidate pool (PUSNet Sec. 4.1). Ported unchanged from run_paper_datasets.py."""
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


def pusnet_templates() -> List[Tuple[int, int, int]]:
    return _balanced_sample_templates(N_TEMPLATES, TEMPLATE_SEED)


def cut_bin(rng: random.Random, bin_dims: List[int], is_valid, a_min: int, a_max: int):
    """
    One attempt at Algorithm 3. Returns the list of valid pieces (position, size),
    which tile the bin, or None if a piece with no legal cut appeared (dead piece).
    """
    invalid: List[Piece] = [((0, 0, 0), tuple(bin_dims))]
    valid: List[Piece] = []
    while invalid:
        pos, size = invalid.pop(rng.randrange(len(invalid)))
        if is_valid(size):
            valid.append((pos, size))
            continue
        axes = [a for a in range(3) if size[a] > a_max]
        if not axes:
            axes = [a for a in range(3) if size[a] >= 2 * a_min]
        if not axes:
            return None
        a = rng.choice(axes)
        k = rng.randint(a_min, size[a] - a_min)
        s1, s2 = list(size), list(size)
        s1[a], s2[a] = k, size[a] - k
        p2 = list(pos)
        p2[a] += k
        invalid.append((pos, tuple(s1)))
        invalid.append((tuple(p2), tuple(s2)))
    return valid


def cut_bin_with_restarts(rng: random.Random, bin_dims: List[int], is_valid, a_min: int, a_max: int):
    """Algorithm 3 with restart on a dead piece; returns (pieces, n_restarts)."""
    for restarts in range(CUT_MAX_RESTARTS):
        pieces = cut_bin(rng, bin_dims, is_valid, a_min, a_max)
        if pieces is not None:
            return pieces, restarts
    raise RuntimeError(f"cut did not succeed within {CUT_MAX_RESTARTS} restarts")


def order_shuffle(rng: random.Random, pieces: List[Piece]) -> List[Piece]:
    out = list(pieces)
    rng.shuffle(out)
    return out


def order_cut1(rng: random.Random, pieces: List[Piece]) -> List[Piece]:
    """Ascending FLB z; ties in random order (shuffle, then stable sort)."""
    out = list(pieces)
    rng.shuffle(out)
    out.sort(key=lambda pc: pc[0][2])
    return out


def order_cut2(rng: random.Random, pieces: List[Piece], bin_dims: List[int]) -> List[Piece]:
    """Random among the pieces whose FLB z equals the height map under their footprint."""
    L, W, _ = bin_dims
    hmap = [[0] * W for _ in range(L)]
    remaining = list(pieces)
    out: List[Piece] = []

    def resting(pc: Piece) -> bool:
        (x, y, z), (l, w, _h) = pc
        return all(hmap[i][j] == z for i in range(x, x + l) for j in range(y, y + w))

    while remaining:
        ready = [i for i, pc in enumerate(remaining) if resting(pc)]
        if not ready:                      # cannot happen for a tiling; guard anyway
            raise RuntimeError("CUT-2: no piece is supported by the height map")
        pc = remaining.pop(rng.choice(ready))
        (x, y, z), (l, w, h) = pc
        for i in range(x, x + l):
            for j in range(y, y + w):
                hmap[i][j] = z + h
        out.append(pc)
    return out


def _pack_pieces(pieces: List[Piece], extra: Dict) -> Tuple[List[List[int]], Dict]:
    boxes = [list(map(int, size)) for _, size in pieces]
    positions = [list(map(int, pos)) for pos, _ in pieces]
    return boxes, {"positions": positions, **extra}


DATA1_A_MIN, DATA1_A_MAX = 2, 5
DATA23_A_MIN, DATA23_A_MAX = 1, 5

GENERATOR_NOTES = {
    "data1": "Zhao et al. 2021 Alg. 3: cut the bin into pieces with every side in [2,5] "
             "(split points keep both halves >= 2); random shuffle (PUSNet DATA-1).",
    "data2": "Zhao et al. 2021 Alg. 3 with PUSNet templates: a piece is valid iff its sorted sides "
             "match one of the 64 templates (split points keep both halves >= 1; a piece with no legal "
             "cut restarts the cut); CUT-1 order = ascending FLB z, ties random.",
    "data3": "Same cut as data2 for the same seed; CUT-2 order = random among pieces whose FLB z "
             "equals the height map under their footprint (all supporting pieces come first).",
}


def gen_data1(bin_dims: List[int], seed: int) -> Tuple[List[List[int]], Dict]:
    rng = random.Random(seed)
    a_min, a_max = DATA1_A_MIN, DATA1_A_MAX
    pieces, restarts = cut_bin_with_restarts(
        rng, bin_dims, lambda s: all(a_min <= v <= a_max for v in s), a_min, a_max)
    return _pack_pieces(order_shuffle(rng, pieces),
                        {"cut": {"a_min": a_min, "a_max": a_max, "restarts": restarts}})


def _cut_to_templates(rng: random.Random, bin_dims: List[int]) -> Tuple[List[Piece], Dict]:
    templates = pusnet_templates()
    shapes = {tuple(sorted(t)) for t in templates}
    a_min, a_max = DATA23_A_MIN, DATA23_A_MAX
    pieces, restarts = cut_bin_with_restarts(
        rng, bin_dims, lambda s: tuple(sorted(s)) in shapes, a_min, a_max)
    cut = {"a_min": a_min, "a_max": a_max, "restarts": restarts,
           "template_seed": TEMPLATE_SEED, "templates": [list(t) for t in templates]}
    return pieces, cut


def gen_data2(bin_dims: List[int], seed: int) -> Tuple[List[List[int]], Dict]:
    rng = random.Random(seed)
    pieces, cut = _cut_to_templates(rng, bin_dims)
    return _pack_pieces(order_cut1(rng, pieces), {"cut": cut})


def gen_data3(bin_dims: List[int], seed: int) -> Tuple[List[List[int]], Dict]:
    rng = random.Random(seed)
    pieces, cut = _cut_to_templates(rng, bin_dims)     # same RNG stream -> same pieces as data2
    return _pack_pieces(order_cut2(rng, pieces, bin_dims), {"cut": cut})


GENERATORS = {
    "data1": gen_data1,
    "data2": gen_data2,
    "data3": gen_data3,
}


# ------------------------ files ------------------------

def sequence_path(dataset: str, seed: int) -> str:
    return os.path.join(SEQ_DIR, dataset, f"seed{seed}.json")


def make_sequence(dataset: str, seed: int, bin_dims: List[int] = BIN_DIMS) -> Dict:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    if dataset == "curriculum25":
        boxes = gen_curriculum25(list(bin_dims), N_ITEMS[dataset], seed)
        generator, extra = gen_curriculum25.__name__, {}
    else:
        boxes, extra = GENERATORS[dataset](list(bin_dims), seed)
        generator = GENERATOR_NOTES[dataset]
    return {
        "dataset": dataset,
        "seed": seed,
        "bin_dims": list(bin_dims),
        "n_items": len(boxes),
        "generator": generator,
        "boxes": [list(map(int, b)) for b in boxes],
        **extra,
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

"""
Run main.py on the three box-sequence datasets used in the paper comparison.

Only the box-sampling step is swapped: main.py is imported as a module and its
box generator is patched at runtime, so rendering, collision and support checks,
retries, metrics and the LLM backends all run exactly as in main.py.

Usage (from the repo root):
  python run_paper_datasets.py --dataset data1 --n_items 40 --seed 123
  python run_paper_datasets.py --dataset data2 --n_items 60
  python run_paper_datasets.py --dataset data3 --n_items 80

The bin size is read from main.py; dataset items are generated to match it.
"""

import argparse
import importlib
import random
from typing import List, Tuple


from typing import List, Optional
import itertools, random, math


# ------------------------ Dataset generators (paper-aligned) ------------------------
# DATA-1: li ∈ [2, L/2], wi ∈ [2, W/2], hi ∈ [2, H/2], shuffled.
# DATA-2: draw from a catalog: sides in {1..5}, with at most one side == 1 (balanced volume mix), random order.
# DATA-3: same as DATA-2 but sort the stream by decreasing volume.


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _vol(s: Tuple[int, int, int]) -> int:
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
    rng = _rng(seed)
    all_sizes = sorted(_template_sizes_1to5(), key=_vol)
    n = len(all_sizes)
    t1, t2, t3 = all_sizes[: n // 3], all_sizes[n // 3 : 2 * n // 3], all_sizes[2 * n // 3 :]
    k1, k2, k3 = int(round(0.30 * k)), int(round(0.40 * k)), k - int(round(0.30 * k)) - int(round(0.40 * k))
    out: List[Tuple[int, int, int]] = []
    out += rng.sample(t1, min(k1, len(t1)))
    out += rng.sample(t2, min(k2, len(t2)))
    remain3 = [s for s in t3 if s not in out]
    need = k - len(out)
    if need > 0:
        out += rng.sample(remain3, min(need, len(remain3)))
    # top up if still short
    remain = [s for s in all_sizes if s not in out]
    while len(out) < k and remain:
        out.append(remain.pop(0))
    return out[:k]


def gen_data1(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    rng = _rng(seed)
    bin_length, bin_width, bin_height = bin_dims
    max_length, max_width, max_height = max(2, bin_length // 2), max(2, bin_width // 2), max(2, bin_height // 2)
    boxes = [[rng.randint(2, max_length), rng.randint(2, max_width), rng.randint(2, max_height)] for _ in range(n_items)]
    rng.shuffle(boxes)
    return boxes



def gen_data1_exact(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    """
    DATA-1 generator:
      - bin_dims = [L, W, H] (ints)
      - exactly n_items boxes
      - each box dim in [2, floor(dim/2)]
      - boxes exactly tile the bin (sum volume == L*W*H)
    """
    rng = random.Random(seed)
    L, W, H = bin_dims
    if min(L, W, H) < 4:
        raise ValueError("Each bin dimension must be >= 4 to allow parts in [2, floor(dim/2)].")

    # --- helpers ---
    def factor_triples(n: int):
        out = []
        for a in range(2, n+1):
            if n % a: continue
            n2 = n // a
            for b in range(2, n2+1):
                if n2 % b: continue
                c = n2 // b
                if c >= 2:
                    out.append((a, b, c))
        return out

    def feasible_k(dim: int, k: int) -> bool:
        # Need: 2*k <= dim  and  k*floor(dim/2) >= dim
        return (2*k <= dim) and (k*(dim//2) >= dim)

    def partition_dim(dim: int, k: int) -> List[int]:
        # Create k integers summing to dim with each in [2, floor(dim/2)]
        if not feasible_k(dim, k):
            raise ValueError("No per-axis partition fits the bounds.")
        max_part = dim // 2
        parts = [2]*k
        remaining = dim - 2*k
        i = 0
        while remaining > 0:
            give = min(remaining, max_part - parts[i])
            parts[i] += give
            remaining -= give
            i = (i + 1) % k
        return parts

    # 1) find (a,b,c) with a*b*c == n_items and per-axis feasibility
    triples = factor_triples(n_items)
    if not triples:
        raise ValueError("n_items must factor as a*b*c with a,b,c >= 2.")
    # Prefer balanced counts
    triples.sort(key=lambda t: max(t)-min(t))
    choice = None
    for a,b,c in triples:
        for kL,kW,kH in set(itertools.permutations((a,b,c))):
            if feasible_k(L,kL) and feasible_k(W,kW) and feasible_k(H,kH):
                choice = (kL,kW,kH)
                break
        if choice: break
    if not choice:
        raise ValueError("n_items cannot fit given bin and DATA-1 bounds (try a different n_items).")

    kL,kW,kH = choice
    segL = partition_dim(L, kL)
    segW = partition_dim(W, kW)
    segH = partition_dim(H, kH)

    boxes = [[l,w,h] for l in segL for w in segW for h in segH]
    print(boxes)
    print(len(boxes))
    rng.shuffle(boxes)
    return boxes



def gen_data2(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    rng = _rng(seed)
    templates = _balanced_sample_templates(64, seed)
    boxes = [list(rng.choice(templates)) for _ in range(n_items)]
    rng.shuffle(boxes)
    return boxes


def gen_data3(bin_dims: List[int], n_items: int, seed: int) -> List[List[int]]:
    rng = _rng(seed)
    templates = _balanced_sample_templates(64, seed ^ 1337)
    boxes = [list(rng.choice(templates)) for _ in range(n_items)]
    boxes.sort(key=lambda s: _vol(tuple(s)), reverse=True)
    return boxes

# DATASET_MAP = {"data1": gen_data1, "data2": gen_data2, "data3": gen_data3}
DATASET_MAP = {"data1": gen_data1_exact, "data2": gen_data2, "data3": gen_data3}


# ------------------------ Patch main.py's box generator ONLY ------------------------

def _discover_bin_dims(pm) -> List[int]:
    # Try common names used in your code; fall back to 10^3 if not found.
    for name in ("BIN_DIMS", "BIN", "BIN_SIZE", "BIN_DIMENSIONS", "BIN_SHAPE"):
        if hasattr(pm, name):
            val = getattr(pm, name)
            if isinstance(val, (list, tuple)) and len(val) == 3:
                return [int(val[0]), int(val[1]), int(val[2])]
            if isinstance(val, int):
                return [val, val, val]
    return [10, 10, 10]


def _patch_box_sampler(pm, sequence: List[List[int]]):
    """Replace pm.generate_smart_box (and a couple of common aliases) so that
    main.py draws sizes from our prebuilt `sequence`. No other behavior changes.
    """
    pm._DATASET_SEQUENCE = [list(map(int, s)) for s in sequence]
    pm._DATASET_INDEX = 0

    def _next_size():
        i = getattr(pm, "_DATASET_INDEX", 0)
        seq = getattr(pm, "_DATASET_SEQUENCE")
        if i >= len(seq):
            return None
        s = seq[i]
        setattr(pm, "_DATASET_INDEX", i + 1)
        return s

    def _make_patched(fn_name: str):
        orig = getattr(pm, fn_name, None)
        if orig is None:
            return None

        def patched(*args, **kwargs):
            s = _next_size()
            if s is None:
                # Fall back to original generator when sequence is exhausted
                return orig(*args, **kwargs)
            return {"size": s}

        setattr(pm, f"_ORIG_{fn_name}", orig)
        setattr(pm, fn_name, patched)
        return patched

    # Patch the most likely entry points used in your main.py
    for name in (
        "generate_smart_box",    # our summary indicated this is used
        "generate_random_box",   # common alias
        "sample_next_box",       # sometimes used
        "generate_box",          # catch‑all
    ):
        if hasattr(pm, name):
            _make_patched(name)

    # Ensure loop bounds match our sequence length if a MAX_BOXES (or similar) exists
    for name in ("MAX_BOXES", "N_BOXES", "NUM_BOXES"):
        if hasattr(pm, name):
            setattr(pm, name, len(sequence))


# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run main.py with the paper's three box-sequence datasets.")
    p.add_argument("--dataset", choices=["data1", "data2", "data3"], default="data1")
    p.add_argument("--n_items", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 10_000_000)

    # Import your project main (does not execute its __main__ block)
    pm = importlib.import_module("main")

    bin_dims = _discover_bin_dims(pm)
    sequence = DATASET_MAP[args.dataset](bin_dims, n_items=args.n_items, seed=seed)

    # Patch ONLY the box sampling step; keep all other behavior identical.
    _patch_box_sampler(pm, sequence)

    # Hand control to your real main. All prints, rendering, metrics, checks are unchanged.
    if hasattr(pm, "main") and callable(pm.main):
        pm.main()
    else:
        raise RuntimeError("Your project main.py does not expose a main() function to call.")


if __name__ == "__main__":
    main()

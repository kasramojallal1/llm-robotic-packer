"""
Run main.py (interactive demo, live window) on the three box-sequence datasets.
Paper numbers come from evaluate.py, not from this script.

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


# ------------------------ Dataset generators ------------------------
# The generators live in harness/sequences.py (single source of truth for the
# committed sequence files under data/sequences/). This runner is the
# interactive/demo path; paper numbers come from evaluate.py.
from harness.sequences import gen_data1_exact, gen_data2, gen_data3  # noqa: E402

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

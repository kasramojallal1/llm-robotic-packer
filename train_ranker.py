#!/usr/bin/env python
"""
Train the lightweight anchor ranker (T3.4; R1.3; D60-D62) on the same demonstrations as Packi.

    python train_ranker.py --teacher human    # -> rankers/ranker-h/   (Packi-H's 646 demos)
    python train_ranker.py --teacher expert   # -> rankers/ranker-e/   (Packi-E's 24,753 demos)

Data = exactly Packi's SFT data: the records kept by the LfD repo's
training/sft_prepare.py (same replay, same drops, same D33 force-include), the
packer's top-8 view of each record, and the same 90/10 split by episode
(random.Random(13)).  The split is checked against data/processed_{v2,e}/manifest.json
when that file is on disk.  The evaluation sequences are never touched.

Model selection: a small grid over HistGradientBoostingClassifier settings, scored by
top-1 agreement with the teacher on the held-out episodes (ties -> smaller model).
The chosen model is the one fitted on the training episodes only (as Packi).
CPU, deterministic (random_state fixed, no early stopping).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from harness.ranker import FEATURE_NAMES, FEATURE_VERSION, RANKER_DIR, anchor_features, choose  # noqa: E402

DEFAULT_LFD = os.path.abspath(os.path.join(REPO_ROOT, "..", "learning-from-demonstration"))
TEACHERS = {
    "human": ("ranker-h", "data/demos/binpack_lfd.jsonl", "data/processed_v2"),
    "expert": ("ranker-e", "data/demos/expert_beam1000.jsonl.gz", "data/processed_e"),
}
GRID = {
    "max_iter": [100, 300],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [7, 15, 31],
    "l2_regularization": [0.0, 1.0],
}
RANDOM_STATE = 0


def _find_lfd(path: str) -> str:
    """The LfD repo: --lfd-repo, else a sibling of this repo (walking up out of a worktree)."""
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    d = REPO_ROOT
    while d != os.path.dirname(d):
        cand = os.path.join(os.path.dirname(d), "learning-from-demonstration")
        if os.path.isdir(os.path.join(cand, "training")):
            return cand
        d = os.path.dirname(d)
    raise FileNotFoundError("learning-from-demonstration repo not found; pass --lfd-repo")


def load_demos(lfd: str, demo_rel: str):
    """
    Records kept by sft_prepare.prepare(), in file order, as (idx, episode, state, (ri, pos)).
    Mirrors prepare()'s filter loop exactly, without the per-record id shuffle (the
    ranker does not read ids).
    """
    sys.path.append(lfd)
    from training import sft_prepare as sp

    path = os.path.join(lfd, demo_rel)
    raw = open(path, "rb").read()
    text = gzip.decompress(raw).decode() if path.endswith(".gz") else raw.decode()
    records = [json.loads(l) for l in text.splitlines() if l.strip()]
    kept, dropped = [], Counter()
    for idx, rec, episode, placed_before, status in sp.replay(records):
        if rec.get("task") != "pick_and_path":
            dropped["not pick_and_path"] += 1; continue
        if rec["ts"] < sp.PRE_FIX_CUTOFF_TS:
            dropped["pre-fix (D32)"] += 1; continue
        if status != "ok":
            dropped["unreproduced (D32)"] += 1; continue
        state, info = sp.packer_view(placed_before, rec, None)
        if state is None:
            dropped[info] += 1; continue
        ri, _, pos = sp.chosen_anchor(rec)
        kept.append((idx, episode, state, (ri, list(pos))))
    split = dict(seed=sp.SPLIT_SEED, frac=sp.TEST_FRACTION)
    return kept, dict(dropped), hashlib.sha256(raw).hexdigest(), len(records), split


def split_episodes(kept, seed: int, frac: float):
    """sft_prepare.prepare()'s split by episode, verbatim."""
    episodes = sorted({ep for _, ep, _, _ in kept}, key=lambda e: (isinstance(e, str), e))
    per_ep = Counter(ep for _, ep, _, _ in kept)
    order = list(episodes)
    random.Random(seed).shuffle(order)
    target = frac * len(kept)
    test_eps, n_test = set(), 0
    for ep in order:
        if n_test >= target:
            break
        test_eps.add(ep); n_test += per_ep[ep]
    return test_eps


def featurize(kept):
    """Stack per-anchor rows; group g = record; y = 1 for the demonstrated anchor (exactly one per record)."""
    Xs, ys, groups, states = [], [], [], []
    for g, (_, _, state, (ri, pos)) in enumerate(kept):
        X, anchors = anchor_features(state)
        y = np.array([a["rotation_index"] == ri and list(a["pos"]) == pos for a in anchors], dtype=np.int8)
        assert y.sum() == 1, f"record {kept[g][0]}: label must be exactly one offered anchor"
        Xs.append(X); ys.append(y); groups.append(np.full(len(y), g)); states.append(state)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups), states


def top1(model, X, y, groups, states, idx):
    """Fraction of records in `idx` where the model's pick (with the policy's tie-break) is the teacher's."""
    proba = model.predict_proba(X)[:, 1] if model is not None else None
    hits = 0
    for g in idx:
        m = groups == g
        if proba is None:     # greedy reference: the is_greedy feature
            i = int(np.argmax(X[m, FEATURE_NAMES.index("is_greedy")]))
        else:
            i = choose(proba[m], states[g])
        hits += int(y[m][i] == 1)
    return hits / len(idx)


def git_commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":(exclude)results",
                                         ":(exclude)rankers"], cwd=REPO_ROOT).decode().strip()
        return {"commit": c, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", choices=sorted(TEACHERS), required=True)
    ap.add_argument("--lfd-repo", default=None, help=f"default: sibling repo ({DEFAULT_LFD})")
    ap.add_argument("--out", default=None, help="default: rankers/<ranker-h|ranker-e>/")
    args = ap.parse_args(argv)

    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    name, demo_rel, processed_rel = TEACHERS[args.teacher]
    lfd = _find_lfd(args.lfd_repo)
    t0 = time.time()
    kept, dropped, sha, n_in, split = load_demos(lfd, demo_rel)
    test_eps = split_episodes(kept, split["seed"], split["frac"])
    print(f"[data] {demo_rel}: {n_in} records, kept {len(kept)}, dropped {dropped}, "
          f"episodes {len({e for _, e, _, _ in kept})}, held-out {len(test_eps)} ({time.time() - t0:.1f}s)")

    mpath = os.path.join(lfd, processed_rel, "manifest.json")
    split_check = None
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        same = (man["input_sha256"] == sha and man["kept_records"] == len(kept)
                and sorted(map(str, man["test_episodes"])) == sorted(map(str, test_eps)))
        if not same:
            raise SystemExit(f"split/data differ from Packi's {mpath}")
        split_check = f"identical to {processed_rel}/manifest.json (Packi's SFT split)"
        print(f"[data] {split_check}")

    X, y, groups, states = featurize(kept)
    is_val = np.array([kept[g][1] in test_eps for g in range(len(kept))])
    tr_g, va_g = np.where(~is_val)[0], np.where(is_val)[0]
    tr = np.isin(groups, tr_g)
    print(f"[data] rows {len(y)} ({X.shape[1]} features); train records {len(tr_g)}, val records {len(va_g)}")

    greedy_val = top1(None, X, y, groups, states, va_g)
    greedy_tr = top1(None, X, y, groups, states, tr_g)
    print(f"[ref ] greedy's anchor = teacher's: train {greedy_tr:.3f}, val {greedy_val:.3f}")

    results, best = [], None
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, vals))
        t1 = time.time()
        model = HistGradientBoostingClassifier(early_stopping=False, random_state=RANDOM_STATE,
                                               min_samples_leaf=20, **params)
        model.fit(X[tr], y[tr])
        va = top1(model, X, y, groups, states, va_g)
        trn = top1(model, X, y, groups, states, tr_g)
        size = params["max_iter"] * params["max_leaf_nodes"]
        results.append({"params": params, "val_top1": va, "train_top1": trn, "fit_s": round(time.time() - t1, 2)})
        print(f"[grid] {params} train {trn:.3f} val {va:.3f} ({time.time() - t1:.1f}s)")
        rank = (-va, size, params["learning_rate"], params["l2_regularization"])
        if best is None or rank < best[0]:
            best = (rank, params, model, va, trn)

    _, params, model, va, trn = best
    out = args.out or os.path.join(RANKER_DIR, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "model.pkl"), "wb") as f:
        pickle.dump(model, f, protocol=4)
    model_sha = hashlib.sha256(open(os.path.join(out, "model.pkl"), "rb").read()).hexdigest()
    lfd_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=lfd).decode().strip()
    manifest = {
        "name": name, "teacher": args.teacher, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "python train_ranker.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "packer_git": git_commit(), "lfd_commit": lfd_commit,
        "demos": demo_rel, "demos_sha256": sha, "input_records": n_in, "kept_records": len(kept),
        "dropped": dropped, "split": dict(split, by="episode", held_out_episodes=len(test_eps), check=split_check),
        "train_records": int(len(tr_g)), "val_records": int(len(va_g)), "rows": int(len(y)),
        "model": "sklearn.ensemble.HistGradientBoostingClassifier", "objective": "pointwise: demonstrated anchor vs the rest",
        "fixed_params": {"early_stopping": False, "random_state": RANDOM_STATE, "min_samples_leaf": 20},
        "params": params, "selection": "max val top-1, ties -> fewer iterations x leaves",
        "val_top1": va, "train_top1": trn, "greedy_val_top1": greedy_val, "greedy_train_top1": greedy_tr,
        "grid": results, "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
        "model_sha256": model_sha, "sklearn": sklearn.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[done] {name}: {params} val top-1 {va:.3f} (greedy {greedy_val:.3f}) -> {os.path.relpath(out, REPO_ROOT)} "
          f"sha256 {model_sha[:12]} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""
Lightweight learned anchor ranker (T3.4; R1.3 "a lightweight classifier or ranking model").

A non-LLM policy trained on the same demonstrations as Packi (D60-D62):

    ranker-h   trained on the human demos Packi-H saw (646 records, same episode split)
    ranker-e   trained on the expert demos Packi-E saw (24,753 records, same episode split)

Input = exactly what the LLM sees (D60): the compact state of harness/state.py
(bin size, incoming box with its rotations, the offered anchor shortlist).  No
placed-box geometry, no fill level, no sequence look-ahead.  Every feature is a
function of an anchor's position and rotated size plus summary statistics of the
whole offered list, so the ranker reads neither anchor ids nor list order and is
invariant to --shuffle-anchors by construction (tested).

Model (D61): scikit-learn HistGradientBoostingClassifier, pointwise "was this the
demonstrated anchor" over every offered anchor; at test time the anchor with the
highest predicted probability is picked; ties -> greedy's key (Eq. 5 score, then
lowest z, y, x, rotation_index).  The path stage is the deterministic template
(T3.3), as for greedy and random.  Deterministic, CPU only.

Trained by train_ranker.py; the fitted model + manifest live in rankers/<name>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from envs.state_manager import score_anchor
from harness.policies import Policy, PolicyOutput, _TemplatePathMixin, _repo_relative

FEATURE_VERSION = 1
RANKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rankers")

FEATURE_NAMES: List[str] = [
    # the anchor itself
    "x", "y", "z", "w", "h", "d",
    "top_z",                    # z + d: height the box reaches
    "footprint",                # w * h
    "d_is_min_dim", "d_is_max_dim",
    "gap_x_far", "gap_y_far",   # free distance to the far walls
    "wall_x0", "wall_y0", "floor", "wall_x1", "wall_y1", "ceiling",
    "walls_touched",
    "score",                    # Eq. 5
    # the anchor relative to the whole offered list
    "score_minus_max", "score_rank",        # dense rank, 0 = best score
    "z_minus_min", "top_minus_min_top",
    "is_greedy",
    "n_same_rotation",          # anchors offered for this rotation (<= 8)
    # the offered list as a whole (a coarse view of how full the bin is)
    "n_offered", "n_rotations", "frac_offered_above_floor", "max_offered_z",
    "box_volume", "box_min_dim", "box_max_dim",
]


def _greedy_key(score: float, pos: Sequence[int], rotation_index: int) -> Tuple:
    x, y, z = pos
    return (-score, z, y, x, rotation_index)


def anchor_features(state: Dict) -> Tuple[np.ndarray, List[Dict]]:
    """
    One feature row per offered anchor, rows in the state's own anchor order.
    Returns (X [n_anchors, n_features] float64, anchors).  Uses only positions,
    rotated sizes and bin size -- never ids or list positions.
    """
    anchors = state["anchors_indexed"]
    W, H, D = state["bin"]["w"], state["bin"]["h"], state["bin"]["d"]
    rots = state["incoming_box"]["rotations"]
    orig = state["incoming_box"]["original_size"]
    if not anchors:
        return np.zeros((0, len(FEATURE_NAMES))), anchors

    sizes = [rots[a["rotation_index"]] for a in anchors]
    scores = [score_anchor(a["pos"], s, [W, H, D]) for a, s in zip(anchors, sizes)]
    keys = [_greedy_key(sc, a["pos"], a["rotation_index"]) for sc, a in zip(scores, anchors)]
    greedy_i = min(range(len(anchors)), key=lambda i: keys[i])
    distinct = sorted(set(scores), reverse=True)
    rank_of = {s: r for r, s in enumerate(distinct)}
    max_score = distinct[0]
    min_z = min(a["pos"][2] for a in anchors)
    min_top = min(a["pos"][2] + s[2] for a, s in zip(anchors, sizes))
    per_rot: Dict[int, int] = {}
    for a in anchors:
        per_rot[a["rotation_index"]] = per_rot.get(a["rotation_index"], 0) + 1
    n = len(anchors)
    frac_above = sum(1 for a in anchors if a["pos"][2] > 0) / n
    max_z = max(a["pos"][2] for a in anchors)
    vol = orig[0] * orig[1] * orig[2]

    rows = []
    for i, (a, (w, h, d), sc) in enumerate(zip(anchors, sizes, scores)):
        x, y, z = a["pos"]
        walls = [x == 0, y == 0, z == 0, x + w == W, y + h == H, z + d == D]
        rows.append([
            x, y, z, w, h, d,
            z + d, w * h,
            d == min(w, h, d), d == max(w, h, d),
            W - (x + w), H - (y + h),
            *walls, sum(walls),
            sc,
            sc - max_score, rank_of[sc],
            z - min_z, (z + d) - min_top,
            i == greedy_i,
            per_rot[a["rotation_index"]],
            n, len(rots), frac_above, max_z,
            vol, min(orig), max(orig),
        ])
    return np.asarray(rows, dtype=np.float64), anchors


def choose(proba: np.ndarray, state: Dict) -> int:
    """Index of the highest-probability anchor; exact ties -> greedy's key."""
    W, H, D = state["bin"]["w"], state["bin"]["h"], state["bin"]["d"]
    rots = state["incoming_box"]["rotations"]
    anchors = state["anchors_indexed"]

    def key(i):
        a = anchors[i]
        return (-float(proba[i]), *_greedy_key(score_anchor(a["pos"], rots[a["rotation_index"]], [W, H, D]),
                                               a["pos"], a["rotation_index"]))
    return min(range(len(anchors)), key=key)


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class RankerPolicy(_TemplatePathMixin, Policy):
    """`ranker-h` / `ranker-e` (T3.4): learned anchor scorer + template path."""

    def __init__(self, name: str, model_dir: Optional[str] = None, model=None):
        self.name = name
        self.model_dir = model_dir or os.path.join(RANKER_DIR, name)
        self._model = model
        self._manifest: Dict = {}
        if model is None:
            path = os.path.join(self.model_dir, "model.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found - train it with train_ranker.py")
            # pickle is safe here: the file is our own, written by train_ranker.py and committed
            # with its sha256 in manifest.json (recorded again in every run file)
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            self._model_sha256 = file_sha256(path)
            mpath = os.path.join(self.model_dir, "manifest.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    self._manifest = json.load(f)
            if self._manifest.get("feature_names", FEATURE_NAMES) != FEATURE_NAMES:
                raise ValueError(f"{self.model_dir}: model was trained on a different feature set")
        else:
            self._model_sha256 = None
        self.model_id = _repo_relative(self.model_dir)

    def describe(self):
        d = super().describe()
        m = self._manifest
        d.update({"model": "sklearn.HistGradientBoostingClassifier", "model_sha256": self._model_sha256,
                  "feature_version": FEATURE_VERSION, "n_features": len(FEATURE_NAMES),
                  "teacher": m.get("teacher"), "demos_sha256": m.get("demos_sha256"),
                  "train_records": m.get("train_records"), "params": m.get("params"),
                  "val_top1": m.get("val_top1"), "path": "template (T3.3)"})
        return d

    def pick(self, state, feedback):
        X, anchors = anchor_features(state)
        if not anchors:
            return PolicyOutput(None)
        proba = self._model.predict_proba(X)[:, 1]
        a = anchors[choose(proba, state)]
        return PolicyOutput({"rotation_index": a["rotation_index"], "anchor_id": a["id"]})

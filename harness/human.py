"""
Human-expert row (T3.8, D49 C, D80): replay a human's recorded choices through
the normal harness, so the row is scored exactly like every other method.

The choices file (schema packi-human-choices/1) is written by the
learning-from-demonstration repo's `human_eval.py`, where the operator packs the
evaluation sequence choosing only among this harness's top-8-per-rotation
shortlist (no lookahead, no voluntary skip, no undo).  Here every recorded
choice is fed back as the pick for its box and validated by the same validator;
the path is the template path, like greedy/random.

    evaluate.py --method human --dataset curriculum25 --seeds 0 1 2 3 4
        reads results/human/choices/<dataset>/seed<k>.json
    evaluate.py --method human:<dir> ...   reads <dir>/<dataset>/seed<k>.json

The replay is strict: if the harness offers a different shortlist than the one
the operator saw (box size, anchor id or position differ), the run stops with an
error instead of producing a number.  Harness latency here is replay time only;
the operator's own decision times are in `policy.human_decision_time_s`.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

from harness.policies import Policy, PolicyOutput, _repo_relative, template_path
from harness.state import lookup_anchor

SCHEMA = "packi-human-choices/1"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DIR = os.path.join(_REPO_ROOT, "results", "human", "choices")


class HumanReplayPolicy(Policy):
    name = "human"
    deterministic = True

    def __init__(self, choices_dir: str = DEFAULT_DIR):
        self.choices_dir = os.path.abspath(choices_dir)
        self.model_id = None
        self._file: Optional[str] = None
        self._record: Optional[Dict] = None
        self._queue: List[Dict] = []
        self._state = None
        self._cur: Optional[Dict] = None

    @classmethod
    def from_spec(cls, spec: str) -> "HumanReplayPolicy":
        _, _, d = spec.partition(":")
        return cls(d or DEFAULT_DIR)

    def begin_episode(self, sequence: Dict):
        path = os.path.join(self.choices_dir, sequence["dataset"], f"seed{sequence['seed']}.json")
        with open(path) as f:
            rec = json.load(f)
        seq_file = os.path.join(_REPO_ROOT, "data", "sequences", sequence["dataset"], f"seed{sequence['seed']}.json")
        with open(seq_file, "rb") as f:
            seq_sha = hashlib.sha256(f.read()).hexdigest()
        if rec.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        if not rec.get("complete"):
            raise ValueError(f"{path}: the episode is not finished ({len(rec['decisions'])}/{rec['n_items']} boxes)")
        if rec["sequence_sha256"] != seq_sha or [d["size"] for d in rec["decisions"]] != [list(map(int, b)) for b in sequence["boxes"]]:
            raise ValueError(f"{path}: recorded on a different sequence than {seq_file}")
        self._file, self._record = path, rec
        self._queue = [d for d in rec["decisions"] if d["outcome"] == "placed"]
        self._state, self._cur = None, None
        self.model_id = f"human:{rec['operator']}"

    def pick(self, state, feedback):
        if state is not self._state:              # a new box (the runner builds one state per box)
            if not self._queue:
                raise RuntimeError(f"{self._file}: harness asks for more placements than were recorded")
            self._state, self._cur = state, self._queue.pop(0)
            d = self._cur
            if list(state["incoming_box"]["original_size"]) != d["size"]:
                raise RuntimeError(f"{self._file}: box {d['index']} size differs from the harness's box")
            size, pos = lookup_anchor(state, d["rotation_index"], d["anchor_id"])
            if pos != d["pos"] or size != d["chosen_size"] or len(state["anchors_indexed"]) != d["n_anchors_offered"]:
                raise RuntimeError(f"{self._file}: box {d['index']}: the harness's shortlist differs from the one recorded")
        else:
            # a recorded choice is always an offered, validated anchor; a second call means it was rejected
            why = feedback[-1] if feedback else "(no feedback)"
            raise RuntimeError(f"{self._file}: box {self._cur['index']} rejected by the harness: {why}")
        return PolicyOutput({"rotation_index": self._cur["rotation_index"], "anchor_id": self._cur["anchor_id"]})

    def path(self, state, target, feedback):
        return PolicyOutput({"path": template_path(state, target)})

    def describe(self) -> Dict:
        d = super().describe()
        if self._record is None:
            return d
        rec = self._record
        with open(self._file, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        times = [x["decision_time_s"] for x in rec["decisions"]
                 if x["outcome"] == "placed" and x.get("decision_time_s") is not None]
        d.update({
            "choices_file": _repo_relative(self._file), "choices_sha256": sha,
            "operator": rec["operator"], "rules": rec["rules"], "recorder_git": rec["recorder_git"],
            "sessions": rec["sessions"],
            "human_decision_time_s": {
                "n": len(times),
                "mean": float(np.mean(times)) if times else None,
                "median": float(np.median(times)) if times else None,
                "p95": float(np.percentile(times, 95)) if times else None,
                "max": float(max(times)) if times else None,
                "total": float(sum(times)) if times else None,
            },
        })
        return d

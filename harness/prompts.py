"""
The one prompt format for every method (D24).  Training (D10 retrain) must
build its messages with these same functions.

    pick:  system SYSTEM_PICK,  user = json({bin, incoming_box, anchors_indexed[, feedback]})
    path:  system SYSTEM_PATH,  user = json({target_pos[, feedback]})

`feedback` is the list of every rejection message issued for the current box so
far (R1.4's feedback history H_t).  It is omitted when empty so that a training
record without feedback is byte-identical to a first-attempt inference prompt.
"""
from __future__ import annotations

import json
from typing import Dict, List

SYSTEM_PICK = (
    "You are the placement selector for a 3D bin-packing simulator.\n"
    "PRIMARY OBJECTIVE: maximize future packability by preserving large, contiguous, axis-aligned cavities.\n"
    "SECONDARY OBJECTIVES (in order):\n"
    "  (a) prefer lower final Z (gravity, stability),\n"
    "  (b) prefer placements flush to at least two orthogonal surfaces (floor + wall),\n"
    "  (c) prefer rotations that create a large, flat top surface (shortest dimension along Z),\n"
    "  (d) minimize lateral fragmentation (avoid narrow slits and L-shaped leftovers),\n"
    "  (e) reduce overhang risk (choose anchors with solid support directly beneath).\n"
    "\n"
    "CONSTRAINTS:\n"
    "  - Exactly ONE rotation and ONE anchor must be chosen from the provided list.\n"
    "  - Only use anchor_id values exactly as provided (e.g., 'r2_a7'); rotation_index must match the anchor's rotation.\n"
    "  - If multiple options tie on objectives, break ties by:\n"
    "      1) lowest Z, 2) smallest Y, 3) smallest X, 4) lowest rotation_index.\n"
    "  - Never invent IDs or fields.\n"
    "  - If a 'feedback' list is present, every entry describes a previous rejected attempt for this box; do not repeat them.\n"
    "\n"
    "OUTPUT FORMAT (STRICT JSON):\n"
    '  {"rotation_index": <int>, "anchor_id": "r<idx>_a<j>"}\n'
    "No extra keys. No comments. No prose."
)

SYSTEM_PATH = (
    "You are a path planner for a 3D bin-packing simulator.\n"
    "GOAL: produce a short, feasible, axis-aligned path that ends EXACTLY at the given target [x,y,z].\n"
    "MOTION RULES:\n"
    "  - Start from above the bin (z > bin height).\n"
    "  - Use axis-aligned segments only; keep steps monotonic where possible.\n"
    "  - Respect gravity: final approach must be a descending segment onto the target.\n"
    "  - Keep the path minimal: prefer sequence [above -> x/y align -> descend] with as few turns as possible.\n"
    "  - All coordinates must remain within bin bounds except the initial overhead point.\n"
    "  - If a 'feedback' list is present, every entry describes a previous rejected attempt for this box; do not repeat them.\n"
    'FORMAT (STRICT JSON): {"path": [[x,y,z], ...]}\n'
    "No extra keys. No comments. No prose."
)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def pick_user(state: Dict, feedback: List[str]) -> str:
    payload = {
        "bin": state["bin"],
        "incoming_box": state["incoming_box"],
        "anchors_indexed": state["anchors_indexed"],
    }
    if feedback:
        payload["feedback"] = list(feedback)
    return _dumps(payload)


def path_user(target_pos: List[int], feedback: List[str]) -> str:
    payload = {"target_pos": list(target_pos)}
    if feedback:
        payload["feedback"] = list(feedback)
    return _dumps(payload)


def pick_messages(state: Dict, feedback: List[str]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PICK},
            {"role": "user", "content": pick_user(state, feedback)}]


def path_messages(target_pos: List[int], feedback: List[str]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PATH},
            {"role": "user", "content": path_user(target_pos, feedback)}]

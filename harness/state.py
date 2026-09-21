"""
Compact state shown to every policy (D24 layout = the demos' layout).

    {"bin": {"w": 10, "h": 10, "d": 10},
     "incoming_box": {"original_size": [..], "rotations": [[..], ..]},
     "anchors_indexed": [{"id": "r0_a0", "rotation_index": 0, "pos": [x, y, z]}, ...]}

Anchors come from envs/state_manager.py exactly as main.py builds them
(all feasible -> vertical clearance filter -> score -> top-K per rotation).
`bin.w/h/d` are bin_dims[0..2]; the third axis (d) is vertical, like everywhere
else in the simulator.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from envs.state_manager import (
    filter_anchors_with_clearance,
    generate_anchor_positions,
    generate_orientations,
    topk_anchors,
)

TOP_K = 8


def build_state(
    placed_boxes: List[Dict],
    box_size: List[int],
    bin_dims: List[int],
    top_k: int = TOP_K,
    shuffle_rng: Optional[random.Random] = None,
) -> Dict:
    """
    Enumerate anchors for every rotation and serialize them.  With `shuffle_rng`
    (T3.6 / --shuffle-anchors) the anchor order is shuffled and the "a<j>" part
    of each id is a random permutation within its rotation, so neither position
    in the list nor the id index carries the score rank.
    """
    rotations = generate_orientations(box_size)
    anchors: List[Dict] = []
    for r_idx, rot_size in enumerate(rotations):
        cand = generate_anchor_positions(placed_boxes, rot_size, bin_dims)
        cand = filter_anchors_with_clearance(cand, rot_size, placed_boxes, bin_dims)
        cand = topk_anchors(cand, rot_size, bin_dims, k=top_k)
        ids = list(range(len(cand)))
        if shuffle_rng is not None:
            shuffle_rng.shuffle(ids)
        for j, pos in zip(ids, cand):
            anchors.append({"id": f"r{r_idx}_a{j}", "rotation_index": r_idx, "pos": list(map(int, pos))})
    if shuffle_rng is not None:
        shuffle_rng.shuffle(anchors)
    return {
        "bin": {"w": int(bin_dims[0]), "h": int(bin_dims[1]), "d": int(bin_dims[2])},
        "incoming_box": {"original_size": list(map(int, box_size)), "rotations": rotations},
        "anchors_indexed": anchors,
    }


def lookup_anchor(state: Dict, rotation_index, anchor_id) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Return (size, pos) for a pick, or (None, None) if the id/rotation pair is not offered."""
    try:
        r = int(rotation_index)
    except (TypeError, ValueError):
        return None, None
    for a in state["anchors_indexed"]:
        if a["id"] == anchor_id and a["rotation_index"] == r:
            return list(state["incoming_box"]["rotations"][r]), list(a["pos"])
    return None, None


def has_anchors(state: Dict) -> bool:
    return len(state["anchors_indexed"]) > 0

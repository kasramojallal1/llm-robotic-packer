"""
Privileged offline expert (T3.7, D49 B, D52): beam search over the whole box
sequence, choosing only among the harness's own top-8-per-rotation shortlist.

The expert sees the entire sequence in advance -- information no online policy
(greedy, Packi, API models) has -- and maximises the final placed volume.  It is
used in two ways:

    oracle    `evaluate.py --method oracle`: the expert packs the 20 evaluation
              sequences itself.  An upper-bound row that uses future information.
    teacher   `generate_expert_demos.py`: the expert packs fresh training
              sequences (seeds disjoint from the evaluation files) and every
              placement becomes an LfD demonstration for Packi-E.

Search (D52 point 1): at every box, every kept partial packing is expanded by
every offered (rotation, anchor) of `harness.state.build_state` (all feasible ->
vertical clearance -> Eq. 5 score -> top-8 per rotation); identical packings are
merged; the `width` packings with the most placed volume survive (ties: larger
sum of Eq. 5 scores, then a fixed lexicographic order).  A packing whose next box
has no anchor carries forward with the box skipped, exactly as the harness skips
it.  Deterministic; no randomness anywhere.

Every planned move is by construction a shortlist entry validated by the same
generator the harness uses, so it is expressible by the policy and never needs
D33's force-include.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from envs.state_manager import score_anchor
from harness.state import TOP_K, build_state

BEAM_WIDTH = 1000

Placement = Tuple[Tuple[int, int, int], Tuple[int, int, int]]   # ((x, y, z), (w, h, d))


@dataclass
class Step:
    """The expert's move for one box of the sequence (None fields = skipped, no anchor)."""
    index: int
    size: List[int]
    rotation_index: Optional[int]
    pos: Optional[List[int]]
    chosen_size: Optional[List[int]]
    n_anchors_offered: int


@dataclass
class Plan:
    steps: List[Step]
    final_volume: int
    bin_volume: int
    width: int
    top_k: int
    n_expanded: int = 0                    # child packings generated over the whole search
    placed_boxes: List[Dict] = field(default_factory=list)
    greedy_volume: int = 0                 # what the corner heuristic reaches on the same sequence
    used_greedy: bool = False              # True when greedy's packing beat the beam and was taken instead

    @property
    def utilization(self) -> float:
        return self.final_volume / self.bin_volume


class _Node:
    __slots__ = ("volume", "heur", "key", "placed", "parent", "move")

    def __init__(self, volume, heur, key, placed, parent, move):
        self.volume = volume          # placed volume
        self.heur = heur              # sum of Eq. 5 scores of the placed boxes (tie-break only)
        self.key = key                # sorted tuple of placements: identity of the packing
        self.placed = placed          # list of {"position", "size"} for build_state
        self.parent = parent
        self.move = move              # (rotation_index, pos, chosen_size) or None for a skip


def _vol(size) -> int:
    return size[0] * size[1] * size[2]


def greedy_plan(boxes: List[List[int]], bin_dims: List[int], top_k: int = TOP_K) -> Plan:
    """The harness's `greedy` policy (top Eq. 5 anchor) as a Plan, for the never-worse-than-greedy guard."""
    from harness.policies import GreedyPolicy
    pol = GreedyPolicy()
    placed: List[Dict] = []
    steps: List[Step] = []
    for i, size in enumerate(boxes):
        size = list(map(int, size))
        state = build_state(placed, size, bin_dims, top_k=top_k)
        anchors = state["anchors_indexed"]
        if not anchors:
            steps.append(Step(i, size, None, None, None, 0))
            continue
        o = pol.pick(state, []).data
        a = next(x for x in anchors if x["id"] == o["anchor_id"] and x["rotation_index"] == o["rotation_index"])
        rs = list(state["incoming_box"]["rotations"][a["rotation_index"]])
        steps.append(Step(i, size, int(a["rotation_index"]), list(a["pos"]), rs, len(anchors)))
        placed.append({"position": list(a["pos"]), "size": rs})
    vol = sum(_vol(b["size"]) for b in placed)
    return Plan(steps=steps, final_volume=vol, bin_volume=_vol(bin_dims), width=1, top_k=top_k,
                placed_boxes=placed, greedy_volume=vol, used_greedy=True)


def plan_sequence(boxes: List[List[int]], bin_dims: List[int], width: int = BEAM_WIDTH,
                  top_k: int = TOP_K) -> Plan:
    """
    Beam search over the whole sequence; returns the best plan found.  If the
    corner heuristic packs more volume on this sequence than the beam (possible at
    small widths), its plan is returned instead, so the expert is never worse than
    greedy (`Plan.used_greedy`).
    """
    bin_dims = list(map(int, bin_dims))
    g = greedy_plan(boxes, bin_dims, top_k=top_k)
    root = _Node(0, 0.0, (), [], None, None)
    beam: List[_Node] = [root]
    offered: List[int] = []
    n_expanded = 0

    for i, size in enumerate(boxes):
        size = list(map(int, size))
        children: Dict[Tuple, _Node] = {}
        n_offered_root = None
        for node in beam:
            state = build_state(node.placed, size, bin_dims, top_k=top_k)
            anchors = state["anchors_indexed"]
            if n_offered_root is None:
                n_offered_root = len(anchors)
            if not anchors:                                   # skipped, as the harness would
                children.setdefault(node.key, _Node(node.volume, node.heur, node.key, node.placed, node, None))
                continue
            rotations = state["incoming_box"]["rotations"]
            for a in anchors:
                rs = rotations[a["rotation_index"]]
                pos = a["pos"]
                pl: Placement = (tuple(pos), tuple(rs))
                key = tuple(sorted(node.key + (pl,)))
                n_expanded += 1
                if key in children:
                    continue
                children[key] = _Node(
                    node.volume + _vol(rs),
                    node.heur + score_anchor(pos, rs, bin_dims),
                    key,
                    node.placed + [{"position": list(pos), "size": list(rs)}],
                    node,
                    (a["rotation_index"], list(pos), list(rs)),
                )
        offered.append(n_offered_root or 0)
        beam = sorted(children.values(), key=lambda n: (-n.volume, -n.heur, n.key))[:width]

    best = beam[0]
    # walk back to the root to recover the moves
    moves: List[Optional[Tuple]] = []
    node = best
    while node.parent is not None:
        moves.append(node.move)
        node = node.parent
    moves.reverse()
    assert len(moves) == len(boxes)

    steps = []
    for i, (size, mv) in enumerate(zip(boxes, moves)):
        if mv is None:
            steps.append(Step(i, list(map(int, size)), None, None, None, offered[i]))
        else:
            r, pos, rs = mv
            steps.append(Step(i, list(map(int, size)), int(r), list(pos), list(rs), offered[i]))
    if g.final_volume > best.volume:
        g.width = width
        g.n_expanded = n_expanded
        return g
    return Plan(steps=steps, final_volume=best.volume, bin_volume=_vol(bin_dims), width=width,
                top_k=top_k, n_expanded=n_expanded, placed_boxes=list(best.placed),
                greedy_volume=g.final_volume, used_greedy=False)


def replay_plan(plan: Plan, bin_dims: List[int], top_k: int = TOP_K) -> bool:
    """
    Safety net: re-run the plan through build_state step by step and confirm every
    move is an offered anchor of the shortlist (and every skip has no anchor).
    """
    placed: List[Dict] = []
    for st in plan.steps:
        state = build_state(placed, st.size, bin_dims, top_k=top_k)
        anchors = state["anchors_indexed"]
        if st.pos is None:
            if anchors:
                return False
            continue
        rs = state["incoming_box"]["rotations"][st.rotation_index]
        if list(rs) != list(st.chosen_size):
            return False
        if not any(a["rotation_index"] == st.rotation_index and list(a["pos"]) == list(st.pos) for a in anchors):
            return False
        placed.append({"position": list(st.pos), "size": list(rs)})
    return True


# ------------------------ oracle policy (evaluate.py --method oracle) ------------------------

from harness.policies import Policy, PolicyOutput, template_path   # noqa: E402


class OraclePolicy(Policy):
    """
    The expert as a harness method.  `begin_episode(sequence)` (called by the runner)
    plans the whole sequence; `pick` then returns the planned move for the current
    box, looked up by (rotation, position) so it is unaffected by --shuffle-anchors.
    Uses future information: an upper bound, not a competitor.
    """
    name = "oracle"

    def __init__(self, width: Optional[int] = None):
        self.width = int(width or BEAM_WIDTH)
        self.model_id = f"oracle-beam{self.width}"
        self.plan: Optional[Plan] = None
        self._i = 0

    def describe(self):
        d = super().describe()
        d.update({"privileged": "sees the whole box sequence", "search": "beam", "beam_width": self.width,
                  "top_k": TOP_K, "plan_utilization": self.plan.utilization if self.plan else None,
                  "greedy_utilization_same_sequence": self.plan.greedy_volume / self.plan.bin_volume if self.plan else None,
                  "used_greedy_plan": self.plan.used_greedy if self.plan else None})
        return d

    def begin_episode(self, sequence: Dict):
        self.plan = plan_sequence(sequence["boxes"], sequence["bin_dims"], width=self.width)
        self._i = 0

    def pick(self, state, feedback):
        if self.plan is None:
            raise RuntimeError("OraclePolicy.begin_episode(sequence) must be called first")
        # the runner calls pick only for boxes with anchors; the plan's box index is
        # recovered from the incoming box order (skips have no pick call)
        size = list(state["incoming_box"]["original_size"])
        while self._i < len(self.plan.steps) and self.plan.steps[self._i].pos is None:
            self._i += 1
        st = self.plan.steps[self._i]
        self._i += 1
        assert st.size == size, f"oracle plan out of sync: expected box {st.size}, got {size}"
        for a in state["anchors_indexed"]:
            if a["rotation_index"] == st.rotation_index and list(a["pos"]) == list(st.pos):
                return PolicyOutput({"rotation_index": st.rotation_index, "anchor_id": a["id"]})
        return PolicyOutput(None)   # cannot happen if the harness state matches the plan's state

    def path(self, state, target, feedback):
        return PolicyOutput({"path": template_path(state, target)})

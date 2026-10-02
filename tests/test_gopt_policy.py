"""GOPT baseline plumbing (T10.x; R1.10; D70-D73).

The runner hooks (observe / has_placement / free placement) and the D71 rotation control
are tested without GOPT.  The GOPT policy itself is tested only when GOPT_DIR and
GOPT_CKPT point at a checkout and an exported actor (skipped otherwise).
"""
import json
import os

import pytest

import evaluate
from harness.gopt_policy import Greedy2RotPolicy, upright_rotation_indices
from harness.policies import GreedyPolicy, Policy, PolicyOutput, _TemplatePathMixin
from harness.runner import run_episode
from harness.sequences import load_sequence
from harness.state import build_state, lookup_position


def _run(tmp_path, method):
    evaluate.main(["--method", method, "--dataset", "data1", "--seed", "0", "--quiet", "--out", str(tmp_path)])
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp_path) for f in fs if f.endswith(".json")]
    assert len(files) == 1
    return json.load(open(files[0])), files[0]


def test_upright_rotation_indices():
    st = build_state([], [2, 3, 4], [10, 10, 10])
    m = upright_rotation_indices(st)
    rots = st["incoming_box"]["rotations"]
    assert rots[m[0]] == [2, 3, 4] and rots[m[1]] == [3, 2, 4]
    st = build_state([], [3, 3, 2], [10, 10, 10])          # square footprint: both yaws are one rotation
    m = upright_rotation_indices(st)
    assert m[0] == m[1] and st["incoming_box"]["rotations"][m[0]] == [3, 3, 2]


def test_lookup_position():
    st = build_state([], [2, 3, 4], [10, 10, 10])
    assert lookup_position(st, 0, [1, 2, 0]) == (st["incoming_box"]["rotations"][0], [1, 2, 0])
    assert lookup_position(st, 99, [0, 0, 0]) == (None, None)
    assert lookup_position(st, 0, [0, 0]) == (None, None)
    assert lookup_position(st, "x", [0, 0, 0]) == (None, None)


def test_greedy_2rot_only_uses_upright_yaws(tmp_path):
    rec, path = _run(tmp_path, "gopt-control-greedy-2rot")
    assert path.endswith(os.path.join("gopt-control-greedy-2rot", "data1", "seed0.json"))
    seq = load_sequence("data1", 0)
    for b in rec["boxes"]:
        if b["outcome"] == "placed":
            l, w, h = seq["boxes"][b["index"]]
            assert b["placement"]["size"] in ([l, w, h], [w, l, h])
    r = rec["reliability"]
    assert r["items_budget_exhausted"] == 0 and r["first_attempt_validity"] == 1.0


def test_greedy_2rot_equals_greedy_when_all_rotations_upright():
    # a cube has one rotation, which is upright: identical choice to greedy
    st = build_state([], [2, 2, 2], [10, 10, 10])
    assert Greedy2RotPolicy("x").pick(st, []).data == GreedyPolicy().pick(st, []).data


class _FreeCorner(_TemplatePathMixin, Policy):
    """Free-placement stub: proposes an illegal floating spot first, then the floor corner; skips when told."""
    name = "free-stub"
    free_placement = True

    def __init__(self):
        self.observed = 0
        self.calls = 0

    def observe(self, placed, size, bin_dims):
        self.observed += 1
        self.placed = placed

    def has_placement(self, state):
        return len(self.placed) < 2

    def pick(self, state, feedback):
        self.calls += 1
        if not feedback:
            return PolicyOutput({"rotation_index": 0, "position": [0, 0, 5]})       # floating -> unsupported
        x = 5 * len(self.placed)
        return PolicyOutput({"rotation_index": 0, "position": [x, 0, 0]}, info={"note": "retry"})


def test_runner_free_placement_hooks():
    seq = {"dataset": "data1", "seed": 0, "bin_dims": [10, 10, 10], "boxes": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]}
    pol = _FreeCorner()
    ep = run_episode(pol, seq, log=lambda *a, **k: None)
    outs = [b["outcome"] for b in ep["boxes"]]
    assert outs == ["placed", "placed", "skipped_no_anchor"]        # has_placement decides the skip
    assert pol.observed == 3
    b0 = ep["boxes"][0]
    assert [a["code"] for a in b0["attempts"]] == ["unsupported", "ok", "ok"]   # validator still judges
    assert b0["placement"]["position"] == [0, 0, 0] and b0["placement"]["anchor_id"] is None
    assert b0["attempts"][1]["info"] == {"note": "retry"} and "observe_s" in b0


def test_anchor_policies_unaffected_by_free_placement_path():
    # an anchor policy returning "position" is still treated as malformed (no free_placement flag)
    class P(_TemplatePathMixin, Policy):
        name = "p"
        def pick(self, state, feedback):
            return PolicyOutput({"rotation_index": 0, "position": [0, 0, 0]})
    seq = {"dataset": "data1", "seed": 0, "bin_dims": [10, 10, 10], "boxes": [[2, 2, 2]]}
    ep = run_episode(P(), seq, log=lambda *a, **k: None)
    assert ep["boxes"][0]["outcome"] == "budget_exhausted"
    assert {a["code"] for a in ep["boxes"][0]["attempts"]} == {"invalid_json"}


@pytest.mark.skipif(not (os.environ.get("GOPT_DIR") and os.environ.get("GOPT_CKPT")),
                    reason="needs a GOPT checkout (GOPT_DIR) and an exported actor (GOPT_CKPT)")
def test_gopt_end_to_end(tmp_path):
    rec, _ = _run(tmp_path / "a", "gopt")
    rec2, _ = _run(tmp_path / "b", "gopt")
    assert rec["placed_boxes"] == rec2["placed_boxes"]                  # argmax is deterministic
    seq = load_sequence("data1", 0)
    for b in rec["boxes"]:
        if b["outcome"] == "placed":
            l, w, h = seq["boxes"][b["index"]]
            assert b["placement"]["size"] in ([l, w, h], [w, l, h])     # GOPT's two upright yaws only
    assert rec["reliability"]["first_attempt_validity"] == 1.0         # D72: choices pre-filtered by our rules
    assert rec["policy"]["gopt_commit"].startswith("a2e42de")

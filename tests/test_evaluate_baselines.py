"""End-to-end: greedy and random on data1/seed0 with no LLM; run JSON has the documented contents."""
import json
import os

import pytest

import evaluate
from harness.state import build_state
import random


def _run(tmp_path, method, extra=()):
    evaluate.main(["--method", method, "--dataset", "data1", "--seed", "0", "--quiet",
                   "--out", str(tmp_path), *extra])
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp_path) for f in fs if f.endswith(".json")]
    assert len(files) == 1
    return json.load(open(files[0])), files[0]


def test_greedy_end_to_end(tmp_path):
    rec, path = _run(tmp_path, "greedy")
    assert path.endswith(os.path.join("greedy", "data1", "seed0.json"))
    assert rec["schema"] == "packi-eval-run/1"
    assert rec["method"] == "greedy" and rec["dataset"] == "data1" and rec["seed"] == 0
    assert rec["git"]["commit"] and rec["hardware"]["hostname"]
    assert rec["budget"] == {"n_pick": 3, "n_path": 2}
    r = rec["reliability"]
    assert r["items_total"] == 40
    assert r["items_placed"] + r["items_skipped_no_anchor"] + r["items_budget_exhausted"] == 40
    assert r["items_placed"] > 20
    assert r["first_attempt_validity"] == 1.0 and r["invalid_json"] == 0 and r["path_collisions"] == 0
    assert 0.5 < rec["metrics"]["utilization_final"] <= 1.0
    assert rec["metrics"]["support_coverage_full_rate"] == 1.0
    assert len(rec["boxes"]) == 40 and len(rec["placed_boxes"]) == r["items_placed"]
    placed = [b for b in rec["boxes"] if b["outcome"] == "placed"]
    assert all(b["placement"]["path"][-1] == [float(v) for v in b["placement"]["position"]] for b in placed)


def test_greedy_is_deterministic(tmp_path):
    a, _ = _run(tmp_path / "a", "greedy")
    b, _ = _run(tmp_path / "b", "greedy")
    assert a["placed_boxes"] == b["placed_boxes"]


def test_random_end_to_end_and_seeded(tmp_path):
    a, _ = _run(tmp_path / "a", "random")
    b, _ = _run(tmp_path / "b", "random")
    assert a["reliability"]["items_placed"] > 10
    assert a["placed_boxes"] == b["placed_boxes"]      # same seed -> same draws


def test_flags_in_file_name_and_record(tmp_path):
    rec, path = _run(tmp_path, "greedy", ["--shuffle-anchors", "--no-feedback"])
    assert path.endswith("seed0.shuffle.nofb.json")
    assert rec["flags"]["shuffle_anchors"] is True and rec["flags"]["feedback"] is False


def test_shuffle_changes_ids_and_order_but_not_geometry():
    plain = build_state([], [4, 2, 3], [10, 10, 10])
    shuf = build_state([], [4, 2, 3], [10, 10, 10], shuffle_rng=random.Random(1))
    assert len(plain["anchors_indexed"]) == len(shuf["anchors_indexed"])
    key = lambda s: sorted((a["rotation_index"], tuple(a["pos"])) for a in s["anchors_indexed"])
    assert key(plain) == key(shuf)
    assert [a["id"] for a in plain["anchors_indexed"]] != [a["id"] for a in shuf["anchors_indexed"]]
    # plain ids are rank-ordered per rotation
    assert plain["anchors_indexed"][0]["id"] == "r0_a0"


def test_unknown_method():
    with pytest.raises(KeyError):
        evaluate.make_policy("nope")

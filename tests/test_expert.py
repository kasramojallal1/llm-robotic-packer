"""Privileged beam-search expert and the `oracle` method (T3.7, D52)."""
import json
import os

import evaluate
from harness.expert import OraclePolicy, plan_sequence, replay_plan
from harness.policies import GreedyPolicy, make_policy
from harness.runner import run_episode
from harness.sequences import load_sequence
from harness.state import build_state

BIN = [10, 10, 10]


def _greedy_utilization(seq):
    placed, pol = [], GreedyPolicy()
    for size in seq["boxes"]:
        st = build_state(placed, size, seq["bin_dims"])
        if not st["anchors_indexed"]:
            continue
        o = pol.pick(st, []).data
        a = next(x for x in st["anchors_indexed"] if x["id"] == o["anchor_id"])
        placed.append({"position": a["pos"], "size": st["incoming_box"]["rotations"][a["rotation_index"]]})
    return sum(b["size"][0] * b["size"][1] * b["size"][2] for b in placed) / 1000


def test_plan_is_valid_and_deterministic():
    seq = load_sequence("data1", 0)
    p1 = plan_sequence(seq["boxes"], seq["bin_dims"], width=50)
    p2 = plan_sequence(seq["boxes"], seq["bin_dims"], width=50)
    assert len(p1.steps) == len(seq["boxes"])
    assert replay_plan(p1, seq["bin_dims"])                       # every move is an offered shortlist anchor
    assert [(s.rotation_index, s.pos) for s in p1.steps] == [(s.rotation_index, s.pos) for s in p2.steps]
    assert p1.final_volume == sum(s.chosen_size[0] * s.chosen_size[1] * s.chosen_size[2]
                                  for s in p1.steps if s.pos is not None)
    assert 0 < p1.utilization <= 1.0


def test_width_one_equals_a_volume_greedy_and_wider_is_not_worse():
    seq = load_sequence("curriculum25", 0)
    u1 = plan_sequence(seq["boxes"], seq["bin_dims"], width=1).utilization
    u50 = plan_sequence(seq["boxes"], seq["bin_dims"], width=50).utilization
    assert u50 >= u1
    assert u50 >= _greedy_utilization(seq)          # the eval-set headroom the experiment relies on


def test_exact_tiling_is_found_when_reachable():
    # two 10x10x5 slabs: the shortlist offers the floor corner and then the top; beam must reach 100 %
    p = plan_sequence([[10, 10, 5], [5, 10, 10]], BIN, width=10)
    assert p.utilization == 1.0 and replay_plan(p, BIN)


def test_skip_when_no_anchor_matches_harness():
    p = plan_sequence([[10, 10, 10], [2, 2, 2]], BIN, width=10)
    assert p.steps[0].pos == [0, 0, 0] and p.steps[1].pos is None and p.steps[1].n_anchors_offered == 0


def test_oracle_policy_matches_plan_with_and_without_shuffle():
    seq = load_sequence("data3", 1)
    pol = OraclePolicy(width=50)
    ep_plain = run_episode(pol, seq, log=lambda *a, **k: None)
    plan_placed = [{"position": s.pos, "size": s.chosen_size} for s in pol.plan.steps if s.pos is not None]
    assert ep_plain["placed_boxes"] == plan_placed
    ep_shuf = run_episode(pol, seq, shuffle_anchors=True, log=lambda *a, **k: None)
    assert ep_shuf["placed_boxes"] == plan_placed              # looked up by position, ids irrelevant
    codes = [a["code"] for b in ep_plain["boxes"] for a in b["attempts"]]
    assert set(codes) == {"ok"}


def test_oracle_end_to_end(tmp_path, monkeypatch):
    import harness.expert as ex
    monkeypatch.setattr(ex, "BEAM_WIDTH", 30)
    evaluate.main(["--method", "oracle", "--dataset", "data2", "--seed", "0", "--quiet", "--out", str(tmp_path)])
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp_path) for f in fs if f.endswith(".json")]
    assert len(files) == 1 and files[0].endswith(os.path.join("oracle", "data2", "seed0.json"))
    rec = json.load(open(files[0]))
    assert rec["method"] == "oracle" and rec["policy"]["beam_width"] == 30
    assert rec["policy"]["privileged"] and rec["policy"]["model_id"] == "oracle-beam30"
    r = rec["reliability"]
    assert r["first_attempt_validity"] == 1.0 and r["items_budget_exhausted"] == 0
    assert abs(rec["metrics"]["utilization_final"] - rec["policy"]["plan_utilization"]) < 1e-9


def test_registry():
    assert isinstance(make_policy("oracle"), OraclePolicy)
    p = make_policy("packi-e")
    assert p.name == "packi-e" and p.lora_dir.endswith(os.path.join("models", "llama32-3b-e"))


def test_expert_is_never_worse_than_greedy_on_training_seeds():
    from harness.expert import greedy_plan
    from harness.sequences import make_sequence
    for ds in ("curriculum25", "data1", "data2", "data3"):
        seq = make_sequence(ds, 1000)
        p = plan_sequence(seq["boxes"], seq["bin_dims"], width=5)
        g = greedy_plan(seq["boxes"], seq["bin_dims"])
        assert p.final_volume >= g.final_volume and p.greedy_volume == g.final_volume
        assert replay_plan(p, seq["bin_dims"])
        # greedy_plan reproduces the harness's greedy policy exactly
        ep = run_episode(GreedyPolicy(), seq, log=lambda *a, **k: None)
        assert ep["placed_boxes"] == g.placed_boxes

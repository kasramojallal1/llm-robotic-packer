"""Human-expert row (T3.8, D80): recorded choices replay through the harness exactly and strictly."""
import hashlib
import json
import os

import pytest

from harness.human import HumanReplayPolicy
from harness.policies import GreedyPolicy, make_policy
from harness.runner import run_episode, run_file_name
from harness.sequences import load_sequence
from harness.state import build_state

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_choices(tmp_path, dataset, seed, chooser, complete=True):
    """A choices file as human_eval.py writes it, played by `chooser(state) -> anchor`."""
    seq = load_sequence(dataset, seed)
    seq_file = os.path.join(REPO, "data", "sequences", dataset, f"seed{seed}.json")
    placed, decisions = [], []
    for i, box in enumerate(seq["boxes"]):
        st = build_state(placed, list(box), seq["bin_dims"])
        if not st["anchors_indexed"]:
            decisions.append({"index": i, "size": list(box), "outcome": "skipped_no_anchor", "n_anchors_offered": 0})
            continue
        a = chooser(st)
        size = st["incoming_box"]["rotations"][a["rotation_index"]]
        placed.append({"position": a["pos"], "size": size})
        decisions.append({"index": i, "size": list(box), "outcome": "placed", "rotation_index": a["rotation_index"],
                          "anchor_id": a["id"], "pos": a["pos"], "chosen_size": size,
                          "n_anchors_offered": len(st["anchors_indexed"]), "decision_time_s": 1.0 + i})
    rec = {"schema": "packi-human-choices/1", "dataset": dataset, "seed": seed,
           "sequence_sha256": hashlib.sha256(open(seq_file, "rb").read()).hexdigest(),
           "n_items": len(seq["boxes"]), "operator": "test", "rules": {}, "recorder_git": {}, "sessions": [],
           "complete": complete, "decisions": decisions}
    path = tmp_path / dataset / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return path


def _greedy_anchor(st):
    out = GreedyPolicy().pick(st, []).data
    return next(a for a in st["anchors_indexed"] if a["id"] == out["anchor_id"] and a["rotation_index"] == out["rotation_index"])


@pytest.mark.parametrize("seed", [0, 3])
def test_greedy_choices_replay_to_greedy_run(tmp_path, seed):
    _write_choices(tmp_path, "curriculum25", seed, _greedy_anchor)
    seq = load_sequence("curriculum25", seed)
    human = run_episode(HumanReplayPolicy(str(tmp_path)), seq, log=lambda *a: None)
    greedy = run_episode(GreedyPolicy(), seq, log=lambda *a: None)
    assert human["placed_boxes"] == greedy["placed_boxes"]
    assert [b["outcome"] for b in human["boxes"]] == [b["outcome"] for b in greedy["boxes"]]


def test_non_greedy_choices_replay_as_recorded(tmp_path):
    path = _write_choices(tmp_path, "curriculum25", 2, lambda st: st["anchors_indexed"][-1])
    rec = json.loads(path.read_text())
    pol = HumanReplayPolicy(str(tmp_path))
    ep = run_episode(pol, load_sequence("curriculum25", 2), log=lambda *a: None)
    want = [{"position": d["pos"], "size": d["chosen_size"]} for d in rec["decisions"] if d["outcome"] == "placed"]
    assert ep["placed_boxes"] == want
    assert all(b["placement"]["pick_attempt"] == 1 for b in ep["boxes"] if b["outcome"] == "placed")
    t = pol.describe()["human_decision_time_s"]
    assert t["n"] == len(want) and t["mean"] > 1.0


def test_divergent_or_incomplete_files_are_refused(tmp_path):
    seq = load_sequence("curriculum25", 0)
    path = _write_choices(tmp_path, "curriculum25", 0, lambda st: st["anchors_indexed"][0])
    rec = json.loads(path.read_text())
    rec["decisions"][3]["pos"] = [0, 0, 9]
    path.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError):
        run_episode(HumanReplayPolicy(str(tmp_path)), seq, log=lambda *a: None)

    _write_choices(tmp_path, "curriculum25", 0, lambda st: st["anchors_indexed"][0], complete=False)
    with pytest.raises(ValueError):
        run_episode(HumanReplayPolicy(str(tmp_path)), seq, log=lambda *a: None)


def test_registration_and_output_path():
    pol = make_policy("human")
    assert isinstance(pol, HumanReplayPolicy)
    assert pol.choices_dir == os.path.join(REPO, "results", "human", "choices")
    assert isinstance(make_policy("human:/some/dir"), HumanReplayPolicy)
    assert run_file_name("human", "curriculum25", 0, False, True) == os.path.join("human", "curriculum25", "seed0.json")

"""Learned anchor ranker (T3.4, D60-D62): deterministic, picks only offered anchors, blind to ids/order."""
import json
import os
import pickle
import random

import numpy as np
import pytest

import evaluate
from harness.policies import GreedyPolicy, make_policy
from harness.ranker import FEATURE_NAMES, RANKER_DIR, RankerPolicy, anchor_features, file_sha256
from harness.runner import run_episode
from harness.sequences import load_sequence
from harness.state import build_state, lookup_anchor

BIN = [10, 10, 10]


def _synthetic_states(n_episodes=6, n_boxes=14, seed=123):
    """States from greedy-packed random boxes (not the evaluation files), with random labels."""
    rng = random.Random(seed)
    greedy = GreedyPolicy()
    states, labels = [], []
    for _ in range(n_episodes):
        placed = []
        for _ in range(n_boxes):
            size = [rng.randint(1, 5) for _ in range(3)]
            st = build_state(placed, size, BIN)
            if not st["anchors_indexed"]:
                continue
            states.append(st)
            labels.append(rng.randrange(len(st["anchors_indexed"])))
            g = greedy.pick(st, []).data
            s, p = lookup_anchor(st, g["rotation_index"], g["anchor_id"])
            placed.append({"position": p, "size": s})
    return states, labels


@pytest.fixture(scope="module")
def fitted():
    from sklearn.ensemble import HistGradientBoostingClassifier
    states, labels = _synthetic_states()
    X = np.vstack([anchor_features(s)[0] for s in states])
    y = np.concatenate([np.eye(len(s["anchors_indexed"]), dtype=int)[l] for s, l in zip(states, labels)])
    model = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=7, early_stopping=False, random_state=0)
    model.fit(X, y)
    return model, states


def _picks(policy, states):
    return [tuple(policy.pick(s, []).data.items()) for s in states]


def test_features_shape_and_no_id_dependence(fitted):
    _, states = fitted
    st = states[5]
    X, anchors = anchor_features(st)
    assert X.shape == (len(anchors), len(FEATURE_NAMES))
    renamed = json.loads(json.dumps(st))
    for k, a in enumerate(renamed["anchors_indexed"]):
        a["id"] = f"zz{k * 7}"
    assert np.array_equal(anchor_features(renamed)[0], X)


def test_picks_only_from_shortlist(fitted):
    model, states = fitted
    pol = RankerPolicy("ranker-test", model_dir="unused", model=model)
    for st in states:
        d = pol.pick(st, []).data
        size, pos = lookup_anchor(st, d["rotation_index"], d["anchor_id"])
        assert size is not None and pos is not None
    assert pol.pick({**states[0], "anchors_indexed": []}, []).data is None


def test_deterministic_across_instances_and_reload(fitted):
    model, states = fitted
    a = RankerPolicy("ranker-test", model_dir="unused", model=model)
    b = RankerPolicy("ranker-test", model_dir="unused", model=pickle.loads(pickle.dumps(model)))
    assert _picks(a, states) == _picks(a, states) == _picks(b, states)


def test_shuffle_anchors_does_not_change_the_packing(fitted):
    model, _ = fitted
    pol = RankerPolicy("ranker-test", model_dir="unused", model=model)
    seq = load_sequence("data1", 0)
    quiet = lambda *a, **k: None
    plain = run_episode(pol, seq, shuffle_anchors=False, log=quiet)
    shuf = run_episode(pol, seq, shuffle_anchors=True, log=quiet)
    assert plain["placed_boxes"] == shuf["placed_boxes"]
    assert run_episode(pol, seq, log=quiet)["placed_boxes"] == plain["placed_boxes"]


@pytest.mark.parametrize("name", ["ranker-h", "ranker-e"])
def test_committed_ranker_end_to_end(tmp_path, name):
    model_dir = os.path.join(RANKER_DIR, name)
    if not os.path.exists(os.path.join(model_dir, "model.pkl")):
        pytest.skip(f"{name} not trained")
    man = json.load(open(os.path.join(model_dir, "manifest.json")))
    assert file_sha256(os.path.join(model_dir, "model.pkl")) == man["model_sha256"]
    assert man["feature_names"] == FEATURE_NAMES
    pol = make_policy(name)
    assert pol.describe()["model_sha256"] == man["model_sha256"]
    evaluate.main(["--method", name, "--dataset", "data1", "--seed", "0", "--quiet", "--out", str(tmp_path)])
    rec = json.load(open(os.path.join(tmp_path, name, "data1", "seed0.json")))
    r = rec["reliability"]
    assert r["first_attempt_validity"] == 1.0 and r["unknown_anchor"] == 0 and r["path_collisions"] == 0
    assert r["items_placed"] + r["items_skipped_no_anchor"] == r["items_total"]
    assert rec["policy"]["teacher"] == man["teacher"]

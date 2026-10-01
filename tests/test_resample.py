"""Equal-budget resampling control (T6.4, R1.6, D100-D103) with a stubbed OpenRouter client:
the first pick of a box and the first path for it go out exactly as the plain request
(temperature 0, no seed); every retry is sampled at T with a fixed per-call seed; no
feedback is sent.  No network, no key."""
import json
import os
from types import SimpleNamespace

import pytest

import evaluate
from harness.policies import OpenRouterPolicy, template_path
from harness.runner import run_episode, run_file_name, sampling_for
from harness.sequences import load_sequence

BAD_PICK = '{"rotation_index": 0, "anchor_id": "nope"}'
BAD_PATH = '{"route": []}'                     # no "path" key -> invalid_json


class EchoClient:
    """Answers like a sensible model (first offered anchor, template path) unless the call index
    is in `bad`, which maps call index -> a rejected answer.  Records every request."""

    def __init__(self, bad=None):
        self.bad = dict(bad or {})
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        n = len(self.calls)
        self.calls.append(kw)
        user = json.loads(kw["messages"][-1]["content"])
        if n in self.bad:
            content = self.bad[n]
        elif "anchors_indexed" in user:
            a = user["anchors_indexed"][0]
            content = json.dumps({"rotation_index": a["rotation_index"], "anchor_id": a["id"]})
        else:
            content = json.dumps({"path": template_path({"bin": user.get("bin", {"d": 10})}, user["target_pos"])})
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0001})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                               usage=usage, provider="StubProvider")


def _policy(bad=None):
    p = OpenRouterPolicy("vendor/model", sleep=lambda s: None)
    p._client = EchoClient(bad)
    return p, p._client


def _run(p, **kw):
    return run_episode(p, load_sequence("curriculum25", 0), log=lambda *a, **k: None, **kw)


def test_schedule_rule():
    assert sampling_for(None, "d", 0, 3, "pick", 2) == (0.0, None)            # control off: always plain
    assert sampling_for(0.7, "d", 0, 3, "pick", 1) == (0.0, None)             # first pick
    assert sampling_for(0.7, "d", 0, 3, "path", 1, 1) == (0.0, None)          # first path of the first pick
    t, s = sampling_for(0.7, "d", 0, 3, "path", 1, 2)
    assert t == 0.7 and isinstance(s, int) and 0 <= s < 2 ** 31
    assert sampling_for(0.7, "d", 0, 3, "pick", 2)[0] == 0.7
    assert sampling_for(0.7, "d", 0, 3, "path", 2, 1)[0] == 0.7               # path after a resampled pick
    # seeds are fixed (reproducible) and differ between calls, boxes and sequences
    assert sampling_for(0.7, "d", 0, 3, "pick", 2) == sampling_for(0.7, "d", 0, 3, "pick", 2)
    seeds = {sampling_for(0.7, d, sd, b, "pick", a)[1] for d in ("x", "y") for sd in (0, 1) for b in (0, 1) for a in (2, 3)}
    assert len(seeds) == 16


def test_first_attempt_request_is_byte_identical_to_plain():
    p_plain, c_plain = _policy()
    p_res, c_res = _policy()
    _run(p_plain, feedback=True)
    _run(p_res, feedback=False, resample_temperature=0.7)
    # no retries happen with a sensible model, so every request of the episode must match the plain one
    assert len(c_plain.calls) == len(c_res.calls)
    for a, b in zip(c_plain.calls, c_res.calls):
        assert a == b and a["temperature"] == 0.0 and "seed" not in a


def test_retries_are_sampled_without_feedback():
    # box 0: pick 1 rejected (unknown anchor) -> pick 2 valid -> path 1 rejected -> path 2 valid
    p, c = _policy({0: BAD_PICK, 2: BAD_PATH})
    ep = _run(p, feedback=False, resample_temperature=0.7)
    b0 = ep["boxes"][0]
    assert [(a["stage"], a["attempt"], a["temperature"]) for a in b0["attempts"]] == \
        [("pick", 1, 0.0), ("pick", 2, 0.7), ("path", 1, 0.7), ("path", 2, 0.7)]
    assert b0["attempts"][0]["sampling_seed"] is None and all(isinstance(a["sampling_seed"], int) for a in b0["attempts"][1:])
    assert b0["outcome"] == "placed"
    assert [k["temperature"] for k in c.calls[:4]] == [0.0, 0.7, 0.7, 0.7]
    assert "seed" not in c.calls[0] and [k["seed"] for k in c.calls[1:4]] == [a["sampling_seed"] for a in b0["attempts"][1:]]
    for k in c.calls:                                     # history kept for the record, never sent
        assert '"feedback"' not in k["messages"][-1]["content"]
    assert len(b0["feedback_history"]) == 2
    # box 1 starts again at temperature 0 with no seed; provider is recorded
    b1 = ep["boxes"][1]
    assert b1["attempts"][0]["temperature"] == 0.0 and b1["attempts"][0]["api"]["provider"] == "StubProvider"
    assert c.calls[4]["temperature"] == 0.0 and "seed" not in c.calls[4]


def test_path_retry_after_first_pick_is_sampled():
    p, c = _policy({1: BAD_PATH})
    ep = _run(p, feedback=False, resample_temperature=0.7)
    assert [(a["stage"], a["temperature"]) for a in ep["boxes"][0]["attempts"]] == \
        [("pick", 0.0), ("path", 0.0), ("path", 0.7)]


def test_feedback_prompt_identical_to_nofb_prompt():
    """The resample run's prompts equal the no-feedback run's, temperature/seed aside."""
    p1, c1 = _policy({0: BAD_PICK})
    p2, c2 = _policy({0: BAD_PICK})
    _run(p1, feedback=False)
    _run(p2, feedback=False, resample_temperature=0.7)
    assert [k["messages"] for k in c1.calls] == [k["messages"] for k in c2.calls]
    assert c1.calls[1]["temperature"] == 0.0 and c2.calls[1]["temperature"] == 0.7


def test_resample_requires_no_feedback_and_api_policy():
    p, _ = _policy()
    with pytest.raises(ValueError):
        _run(p, feedback=True, resample_temperature=0.7)
    from harness.policies import GreedyPolicy
    with pytest.raises(ValueError):
        _run(GreedyPolicy(), feedback=False, resample_temperature=0.7)
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--method", "greedy", "--resample", "0.7", "--all"])


def test_file_name_and_record(tmp_path, monkeypatch):
    assert run_file_name("api:a/b", "data1", 2, False, False, resample=True).endswith("data1/seed2.resample.json")
    assert run_file_name("api:a/b", "data1", 2, False, False).endswith("seed2.nofb.json")
    monkeypatch.setattr(evaluate, "make_policy", lambda spec, seed=0, reasoning=None, json_mode=True: _policy({0: BAD_PICK})[0])
    evaluate.main(["--method", "api:stub/model", "--dataset", "curriculum25", "--seed", "0", "--quiet",
                   "--resample", "0.7", "--out", str(tmp_path)])
    rec = json.load(open(os.path.join(tmp_path, "api-stub-model", "curriculum25", "seed0.resample.json")))
    assert rec["flags"]["feedback"] is False and rec["flags"]["resample_temperature"] == 0.7
    import aggregate
    assert aggregate.group_key(rec)[2] == "+resample"

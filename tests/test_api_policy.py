"""OpenRouterPolicy with a stubbed client: backoff, usage/cost bookkeeping, reasoning setting,
api_error inside the budget (T5.2; D46/D47).  No network, no key."""
import json
import os
from types import SimpleNamespace

import httpx
import openai
import pytest

import evaluate
from harness import policies
from harness.policies import OpenRouterPolicy, PolicyOutput, template_path
from harness.runner import run_episode
from harness.sequences import load_sequence
from harness.state import build_state

STATE = build_state([], [2, 3, 4], [10, 10, 10])   # a real pick prompt input


def _resp(content, prompt=700, completion=20, reasoning=None, cost=0.0003, finish="stop"):
    details = {"reasoning_tokens": reasoning} if reasoning is not None else None
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": prompt, "completion_tokens": completion,
                                                "completion_tokens_details": details, "cost": cost})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                           usage=usage)


def _http_error(status, cls=None):
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    cls = cls or openai.APIStatusError
    return cls(f"http {status}", response=resp, body=None)


class StubClient:
    """Scripted responses: each item is a response object or an exception to raise."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else _resp('{"rotation_index": 0, "anchor_id": "r0_a0"}')
        if isinstance(item, Exception):
            raise item
        return item


def _policy(script, reasoning=None):
    sleeps = []
    p = OpenRouterPolicy("vendor/model", reasoning=reasoning, sleep=sleeps.append)
    p._client = StubClient(script)
    return p, p._client, sleeps


def test_usage_and_cost_are_recorded():
    p, client, _ = _policy([_resp('{"rotation_index": 1, "anchor_id": "r1_a0"}', prompt=800, completion=25, reasoning=300, cost=0.0012)])
    out = p.pick(STATE, [])
    assert out.data == {"rotation_index": 1, "anchor_id": "r1_a0"}
    assert out.meta == {"api_retries": 0, "prompt_tokens": 800, "completion_tokens": 25,
                        "reasoning_tokens": 300, "cost_usd": 0.0012, "finish_reason": "stop"}
    kw = client.calls[0]
    assert kw["model"] == "vendor/model" and kw["temperature"] == 0.0
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["extra_body"] == {"usage": {"include": True}}       # no reasoning field unless asked


def test_backoff_on_429_then_success():
    p, client, sleeps = _policy([_http_error(429, openai.RateLimitError), _http_error(503),
                                 _resp('{"path": [[0, 0, 12], [0, 0, 1], [0, 0, 0]]}')])
    out = p.path({"bin": {"w": 10, "h": 10, "d": 10}}, [0, 0, 0], [])
    assert out.data == {"path": [[0, 0, 12], [0, 0, 1], [0, 0, 0]]}
    assert out.meta["api_retries"] == 2 and "api_error" not in out.meta
    assert len(client.calls) == 3
    assert len(sleeps) == 2 and 2.0 <= sleeps[0] < 3.0 and 4.0 <= sleeps[1] < 5.0   # 2, 4 (+ jitter)


def test_gives_up_after_bounded_retries():
    p, client, sleeps = _policy([_http_error(502)] * 10)
    out = p.pick(STATE, [])
    assert out.data is None and out.raw == ""
    assert out.meta["api_retries"] == policies.API_MAX_RETRIES
    assert out.meta["api_error"].startswith("APIStatusError")
    assert len(client.calls) == policies.API_MAX_RETRIES + 1
    assert len(sleeps) == policies.API_MAX_RETRIES


def test_402_inflight_reservation_is_retried():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(402, request=req)
    err = openai.APIStatusError("This request would exceed your available credits given your current in-flight requests.",
                                response=resp, body=None)
    p, client, sleeps = _policy([err, _resp('{"rotation_index": 0, "anchor_id": "r0_a0"}')])
    out = p.pick(STATE, [])
    assert out.data is not None and out.meta["api_retries"] == 1 and len(sleeps) == 1


def test_max_tokens_is_sent():
    p, client, _ = _policy([])
    p.pick(STATE, [])
    assert client.calls[0]["max_tokens"] == policies.API_MAX_TOKENS
    p2, client2, _ = _policy([], reasoning="low")
    p2.pick(STATE, [])
    assert client2.calls[0]["max_tokens"] == policies.API_MAX_TOKENS_REASONING


def test_non_retryable_error_fails_immediately():
    p, client, sleeps = _policy([_http_error(400, openai.BadRequestError)])
    out = p.pick(STATE, [])
    assert out.data is None and out.meta["api_retries"] == 0 and "BadRequestError" in out.meta["api_error"]
    assert len(client.calls) == 1 and sleeps == []


def test_connection_error_is_retried():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    p, client, sleeps = _policy([openai.APIConnectionError(request=req), _resp('{"rotation_index": 0, "anchor_id": "r0_a0"}')])
    out = p.pick(STATE, [])
    assert out.data is not None and out.meta["api_retries"] == 1


@pytest.mark.parametrize("setting, expected", [
    ("low", {"effort": "low", "exclude": True}),
    ("off", {"enabled": False, "exclude": True}),
])
def test_reasoning_setting_goes_into_the_request(setting, expected):
    p, client, _ = _policy([], reasoning=setting)
    p.pick(STATE, [])
    assert client.calls[0]["extra_body"]["reasoning"] == expected
    assert p.describe()["reasoning"] == policies.REASONING_SETTINGS[setting]


def test_no_json_mode_omits_response_format():
    p, client, _ = _policy([], reasoning=None)
    p.json_mode = False
    p.pick(STATE, [])
    assert "response_format" not in client.calls[0]
    assert p.describe()["response_format"] is None


def test_unknown_reasoning_setting_rejected():
    with pytest.raises(ValueError):
        OpenRouterPolicy("vendor/model", reasoning="max")


class ScriptedPolicy(OpenRouterPolicy):
    """A stubbed API model that answers like greedy-ish: first offered anchor, template path."""

    def __init__(self, fail_first_pick=False):
        super().__init__("stub/model", reasoning="low", sleep=lambda s: None)
        self.fail_first_pick = fail_first_pick
        self._n = 0

    def _complete(self, messages, max_new_tokens):
        self._n += 1
        if self.fail_first_pick and self._n == 1:
            return "", {"api_retries": policies.API_MAX_RETRIES, "api_error": "APIStatusError: http 503"}
        user = json.loads(messages[-1]["content"])
        meta = {"api_retries": 0, "prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 5, "cost_usd": 0.0001}
        if "anchors_indexed" in user:
            a = user["anchors_indexed"][0]
            return json.dumps({"rotation_index": a["rotation_index"], "anchor_id": a["id"]}), meta
        return json.dumps({"path": template_path({"bin": user.get("bin", {"d": 10})}, user["target_pos"])}), meta


def test_api_error_consumes_one_attempt_and_run_continues():
    seq = load_sequence("data1", 0)
    ep = run_episode(ScriptedPolicy(fail_first_pick=True), seq, log=lambda *a, **k: None)
    b0 = ep["boxes"][0]
    assert b0["attempts"][0]["code"] == "api_error" and b0["attempts"][0]["api"]["api_error"]
    assert b0["feedback_history"] == []                       # the model never answered: no feedback message
    assert b0["outcome"] == "placed" and b0["placement"]["pick_attempt"] == 2
    assert all(b["outcome"] in ("placed", "skipped_no_anchor") for b in ep["boxes"][1:])


def test_run_record_has_api_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "make_policy", lambda spec, seed=0, reasoning=None, json_mode=True: ScriptedPolicy(fail_first_pick=True))
    evaluate.main(["--method", "api:stub/model", "--dataset", "data1", "--seed", "0", "--quiet",
                   "--reasoning", "low", "--out", str(tmp_path)])
    path = os.path.join(tmp_path, "api-stub-model", "data1", "seed0.json")
    rec = json.load(open(path))
    assert rec["flags"]["reasoning"] == "low"
    assert rec["policy"]["reasoning"] == {"effort": "low"} and rec["policy"]["provider"] == "openrouter"
    u = rec["api_usage"]
    assert u["calls"] == rec["reliability"]["pick_calls"] + rec["reliability"]["path_calls"]
    assert u["failed_calls"] == 1 and u["retried_calls"] == 1 and u["retries_total"] == policies.API_MAX_RETRIES
    assert u["prompt_tokens"] == 100 * (u["calls"] - 1) and u["reasoning_tokens"] == 5 * (u["calls"] - 1)
    assert abs(u["cost_usd"] - 0.0001 * (u["calls"] - 1)) < 1e-9
    assert rec["reliability"]["api_errors"] == 1 and rec["reliability"]["attempt_codes"]["pick:api_error"] == 1
    assert rec["metrics"]["utilization_final"] > 0.5


def test_greedy_record_has_no_api_usage(tmp_path):
    evaluate.main(["--method", "greedy", "--dataset", "data1", "--seed", "0", "--quiet", "--out", str(tmp_path)])
    rec = json.load(open(os.path.join(tmp_path, "greedy", "data1", "seed0.json")))
    assert rec["api_usage"] is None and rec["reliability"]["api_errors"] == 0

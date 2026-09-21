"""
Policies: everything the harness can evaluate, behind one interface.

    pick(state, feedback) -> PolicyOutput(data={"rotation_index", "anchor_id"} | None, raw=str | None)
    path(state, target, feedback) -> PolicyOutput(data={"path": [...]} | None, raw=str | None)

`data` is the parsed JSON exactly as the policy produced it (or None when it
could not be parsed); the runner validates it.  Policies never see the placed
boxes directly, only the compact state.

Registered names (evaluate.py --method):
    greedy            top-scoring anchor of Eq. 5 + template path      (T3.1 + T3.3)
    random            uniform anchor + template path
    packi             local base model + LoRA adapter (config.BASE_MODEL, config.LORA_DIR)
    packi-e           same base model + the expert-demo adapter (config.LORA_DIR_E)   (T3.7)
    oracle            privileged beam-search expert over the whole sequence (T3.7, upper bound)
    base-llama        the same base model, no adapter                   (T3.2)
    local:<hf-id>[@<lora-dir>]   any local HF model, optional adapter    (backbone study, R1.9)
    api:<openrouter-id>          OpenRouter model via the openai client
    ranker            reserved (T3.4)
    gopt              reserved (T10.x)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from envs.state_manager import score_anchor
from harness import prompts
from harness.jsonutil import parse_json_lenient

OVERHEAD_Z_OFFSET = 2   # demos start at z = bin height + 2
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_relative(path: str) -> str:
    """Stable model ids across machines: paths inside the repo are written relative to it."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, _REPO_ROOT) if ap.startswith(_REPO_ROOT + os.sep) else path


@dataclass
class PolicyOutput:
    data: Optional[Dict]
    raw: Optional[str] = None


class Policy:
    name: str = "policy"
    model_id: Optional[str] = None
    deterministic: bool = True

    def pick(self, state: Dict, feedback: List[str]) -> PolicyOutput:
        raise NotImplementedError

    def path(self, state: Dict, target: List[int], feedback: List[str]) -> PolicyOutput:
        raise NotImplementedError

    def describe(self) -> Dict:
        return {"name": self.name, "model_id": self.model_id}


# ------------------------ template path (T3.3) ------------------------

def template_path(state: Dict, target: List[int]) -> List[List[int]]:
    """overhead -> pre-descend -> target, exactly the shape of all 667 demos."""
    x, y, z = map(int, target)
    top = int(state["bin"]["d"]) + OVERHEAD_Z_OFFSET
    return [[x, y, top], [x, y, z + 1], [x, y, z]]


class _TemplatePathMixin:
    def path(self, state, target, feedback):
        return PolicyOutput({"path": template_path(state, target)})


# ------------------------ greedy (T3.1) ------------------------

class GreedyPolicy(_TemplatePathMixin, Policy):
    """Highest score_anchor (Eq. 5); ties -> lowest z, y, x, rotation_index (the prompt's tie-break)."""
    name = "greedy"

    def pick(self, state, feedback):
        bin_dims = [state["bin"]["w"], state["bin"]["h"], state["bin"]["d"]]
        best, best_key = None, None
        for a in state["anchors_indexed"]:
            size = state["incoming_box"]["rotations"][a["rotation_index"]]
            x, y, z = a["pos"]
            key = (-score_anchor(a["pos"], size, bin_dims), z, y, x, a["rotation_index"])
            if best_key is None or key < best_key:
                best, best_key = a, key
        if best is None:
            return PolicyOutput(None)
        return PolicyOutput({"rotation_index": best["rotation_index"], "anchor_id": best["id"]})


# ------------------------ random ------------------------

class RandomPolicy(_TemplatePathMixin, Policy):
    """Uniform over the offered anchors.  Seeded from the run seed by the runner."""
    name = "random"
    deterministic = False

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def pick(self, state, feedback):
        if not state["anchors_indexed"]:
            return PolicyOutput(None)
        a = self.rng.choice(state["anchors_indexed"])
        return PolicyOutput({"rotation_index": a["rotation_index"], "anchor_id": a["id"]})


# ------------------------ LLM policies ------------------------

class _LLMPolicy(Policy):
    """Shared: build the unified messages, call `_complete`, parse leniently."""

    def _complete(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        raise NotImplementedError

    def pick(self, state, feedback):
        raw = self._complete(prompts.pick_messages(state, feedback), max_new_tokens=64)
        data = parse_json_lenient(raw)
        return PolicyOutput(data if isinstance(data, dict) else None, raw)

    def path(self, state, target, feedback):
        raw = self._complete(prompts.path_messages(target, feedback), max_new_tokens=192)
        data = parse_json_lenient(raw)
        return PolicyOutput(data if isinstance(data, dict) else None, raw)


class OpenRouterPolicy(_LLMPolicy):
    """`api:<model>`: OpenRouter via the openai client, temperature 0, JSON mode (as llm_api.py)."""

    def __init__(self, model: str):
        self.name = f"api:{model}"
        self.model_id = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv()
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        return self._client

    def _complete(self, messages, max_new_tokens):
        resp = self._get_client().chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class LocalHFPolicy(_LLMPolicy):
    """
    `packi`, `base-llama`, `local:<hf-id>[@<lora-dir>]`: a local HF causal LM,
    greedy decoding, optional LoRA adapter.  Loads on first use.
    No hidden self-repair call and no silent path fixing (D24).
    """

    def __init__(self, name: str, base_model: str, lora_dir: Optional[str]):
        self.name = name
        self.base_model = base_model
        self.lora_dir = lora_dir
        self.model_id = base_model if not lora_dir else f"{base_model}+{_repo_relative(lora_dir)}"
        self._model = self._tok = self._device = None

    def describe(self):
        d = super().describe()
        d.update({"base_model": self.base_model, "lora_dir": self.lora_dir})
        return d

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

        tok = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        model.to(device)
        if self.lora_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.lora_dir)
        model.eval()
        self._model, self._tok, self._device = model, tok, device

    def _complete(self, messages, max_new_tokens):
        import torch
        self._load()
        tok, model = self._tok, self._model
        import config
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False,
                                      date_string=config.CHAT_TEMPLATE_DATE)   # D42: pinned date
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids             # transformers 4.x / 5.x
        inputs = torch.tensor([ids], device=self._device)
        eos = [tok.eos_token_id]
        try:
            eot = tok.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id:
                eos.append(eot)
        except Exception:
            pass
        with torch.no_grad():
            out = model.generate(
                inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id, eos_token_id=eos,
            )
        return tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)


# ------------------------ registry ------------------------

def make_policy(spec: str, seed: int = 0) -> Policy:
    import config

    if spec == "greedy":
        return GreedyPolicy()
    if spec == "random":
        return RandomPolicy(seed=seed)
    if spec == "packi":
        return LocalHFPolicy("packi", config.BASE_MODEL, config.LORA_DIR)
    if spec == "packi-e":
        return LocalHFPolicy("packi-e", config.BASE_MODEL, config.LORA_DIR_E)
    if spec == "base-llama":
        return LocalHFPolicy("base-llama", config.BASE_MODEL, None)
    if spec == "oracle":
        from harness.expert import OraclePolicy
        return OraclePolicy()
    if spec.startswith("local:"):
        rest = spec[len("local:"):]
        base, _, lora = rest.partition("@")
        return LocalHFPolicy(spec, base, lora or None)
    if spec.startswith("api:"):
        return OpenRouterPolicy(spec[len("api:"):])
    if spec in ("ranker", "gopt"):
        raise NotImplementedError(f"{spec!r} is reserved (T3.4 / T10.x) and not implemented yet")
    raise KeyError(f"unknown method {spec!r}")

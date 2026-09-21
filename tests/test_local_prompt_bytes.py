"""The local policy's prompt tokens must not depend on the day the run happens (D42)."""
import os

import pytest

import config
from harness import prompts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIRS = [os.path.join(REPO, "models", d) for d in ("llama32-3b-v2", "llama32-3b")]


def _tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception:
        pytest.skip("transformers not installed")
    for d in TOK_DIRS:
        if os.path.exists(os.path.join(d, "tokenizer.json")):
            return AutoTokenizer.from_pretrained(d)
    pytest.skip("no local Llama tokenizer under models/")


def test_pinned_date_in_system_header():
    tok = _tokenizer()
    state = {"bin": {"w": 10, "h": 10, "d": 10},
             "incoming_box": {"original_size": [2, 2, 2], "rotations": [[2, 2, 2]]},
             "anchors_indexed": [{"id": "r0_a0", "rotation_index": 0, "pos": [0, 0, 0]}]}
    text = tok.apply_chat_template(prompts.pick_messages(state, []), add_generation_prompt=True,
                                   tokenize=False, date_string=config.CHAT_TEMPLATE_DATE)
    assert f"Today Date: {config.CHAT_TEMPLATE_DATE}" in text
    assert text.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")


def test_lora_dir_is_the_revision_adapter():
    assert config.LORA_DIR.endswith(os.path.join("models", "llama32-3b-v2"))

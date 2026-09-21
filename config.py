import os
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


API_GPT_5mini = 'openai/gpt-5-mini'
API_GPT_5nano = 'openai/gpt-5-nano'
API_GPT_oss120 = 'openai/gpt-oss-120b'
API_GPT_oss20 = 'openai/gpt-oss-20b'
API_GPT_41 = 'openai/gpt-4.1'
API_GPT_41_mini = 'openai/gpt-4.1-mini'
API_GPT_4o_mini = 'openai/gpt-4o-mini'
API_GPT_4o = 'openai/gpt-4o'

API_CLAUDE_4 = 'anthropic/claude-sonnet-4'
API_CLAUDE_37 = 'anthropic/claude-3.7-sonnet'
API_CLAUDE_35 = 'anthropic/claude-3.5-sonnet'

API_GEMINI_FLASH_25 = 'google/gemini-2.5-flash'
API_GEMINI_FLASH_25_LITE = 'google/gemini-2.5-flash-lite'

API_QWEN_25_32 = 'qwen/qwen2.5-vl-32b-instruct'
API_QWEN_vl_max = 'qwen/qwen-vl-max'

API_DEEPSEEK_v31 = 'deepseek/deepseek-v3.1-terminus'
API_DEEPSEEK_r1 = 'deepseek/deepseek-r1-0528'
API_DEEPSEEK_v3 = 'deepseek/deepseek-chat-v3-0324'

API_LLAMA_4_MAVERICK = 'meta-llama/llama-4-maverick'
API_LLAMA_4_SCOUT = 'meta-llama/llama-4-scout'

API_GROK_3 = 'x-ai/grok-3-mini'
API_GROK_4_FAST = 'x-ai/grok-4-fast'


API_MODEL = API_GROK_3


LOCAL_MODEL = "llama32-3b"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# D41: `packi` = the adapter retrained for the revision (T0.10).  The adapter of the
# original submission stays at models/llama32-3b and runs as
#   --method local:meta-llama/Llama-3.2-3B-Instruct@models/llama32-3b
LORA_DIR   = os.path.join(REPO_ROOT, "models", "llama32-3b-v2")
# T3.7 / D52: `packi-e` = the adapter trained on privileged-expert demonstrations.
LORA_DIR_E = os.path.join(REPO_ROOT, "models", "llama32-3b-e")
# D42: chat-template date pinned so prompts are byte-identical at training and at
# every evaluation run (the same constant is used by the LfD repo's trainer).
CHAT_TEMPLATE_DATE = "26 Jul 2024"
ATTN_IMPL = "sdpa"  # eager or sdpa


USE_LOCAL_LLM: bool = False

if USE_LOCAL_LLM:
    from llm_local import (
        choose_rotation_and_anchor,
        generate_path,
    )
else:
    from llm_api import (
        choose_rotation_and_anchor,
        generate_path,
    )

__all__ = ["choose_rotation_and_anchor", "generate_path"]

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import config

BIN_STATE_PATH = "instructions/bin_state.json"

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
open_router_key = os.getenv("OPENROUTER_API_KEY")
# Without a key (e.g. a GPU box that only runs local policies) the client is still
# constructed; any API call then fails with 401 instead of breaking `import config`.
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=open_router_key or "missing-OPENROUTER_API_KEY")


SYSTEM_PICK = (
    "You are the placement selector for a 3D bin‑packing simulator.\n"
    "PRIMARY OBJECTIVE: maximize future packability by preserving large, contiguous, axis‑aligned cavities.\n"
    "SECONDARY OBJECTIVES (in order):\n"
    "  (a) prefer lower final Z (gravity, stability),\n"
    "  (b) prefer placements flush to at least two orthogonal surfaces (floor + wall),\n"
    "  (c) prefer rotations that create a large, flat top surface (shortest dimension along Z),\n"
    "  (d) minimize lateral fragmentation (avoid narrow slits and L‑shaped leftovers),\n"
    "  (e) reduce overhang risk (choose anchors with solid support directly beneath).\n"
    "\n"
    "CONSTRAINTS:\n"
    "  • Exactly ONE rotation and ONE anchor must be chosen from the provided lists.\n"
    "  • Only use anchor_id values exactly as provided (e.g., 'r2_a7').\n"
    "  • If multiple options tie on objectives, break ties by:\n"
    "      1) lowest Z, 2) smallest Y, 3) smallest X, 4) lowest rotation_index.\n"
    "  • Never invent IDs or fields.\n"
    "\n"
    "OUTPUT FORMAT (STRICT JSON):\n"
    '  {\"rotation_index\": <int>, \"anchor_id\": \"r<idx>_a<j>\"}\n'
    "No extra keys. No comments. No prose."
)

SYSTEM_PATH = (
    "You are a path planner for a 3D bin‑packing simulator.\n"
    "GOAL: produce a short, feasible, axis‑aligned path that ends EXACTLY at the given target [x,y,z].\n"
    "MOTION RULES:\n"
    "  • Start from above the bin (z > bin_height) or current lift height if provided.\n"
    "  • Use axis‑aligned segments only; keep steps monotonic where possible.\n"
    "  • Respect gravity: final approach must be a descending segment onto the target.\n"
    "  • Keep the path minimal: prefer sequence [above->x/y align->descend] with as few turns as possible.\n"
    "  • All coordinates must remain within bin bounds except the initial overhead point.\n"
    "FORMAT (STRICT JSON): {\"path\": [[x,y,z], ...]}\n"
    "No extra keys. No comments. No prose."
)


def choose_rotation_and_anchor(feedback: str = ""):
    """
    Ask the model to pick a rotation and an anchor by ID.
    Uses JSON response mode to ensure strict JSON output.
    """
    with open(BIN_STATE_PATH, "r") as f:
        state = json.load(f)

    # Keep only essentials in the prompt (compact JSON)
    compact = json.dumps(
        {
            "bin": state.get("bin", {}),
            "incoming_box": {
                "original_size": state.get("incoming_box", {}).get("original_size", []),
                "rotations": state.get("incoming_box", {}).get("rotations", []),
            },
            "anchors_indexed": state.get("anchors_indexed", []),
        },
        separators=(",", ":"),
    )

    user = (
        "Select exactly one rotation_index and one anchor_id from anchors_indexed.\n"
        f"{('Feedback: ' + feedback) if feedback else ''}\n"
        f"{compact}"
    )

    resp = client.chat.completions.create(
        model=config.API_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PICK},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# noinspection PyTypeChecker
def generate_path(target_pos, feedback: str = ""):
    """
    Ask the model for a gravity-like path that ends exactly at target_pos.
    Uses JSON response mode to ensure strict JSON output.
    """
    user = (
        f"Final target position (must end here): {target_pos}\n"
        f"{('Feedback: ' + feedback) if feedback else ''}"
    )

    resp = client.chat.completions.create(
        model=config.API_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PATH},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

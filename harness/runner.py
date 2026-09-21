"""
The evaluation loop: one policy, one fixed sequence, one run record.

Per box (same budget for every method, D28):
    up to N_PICK pick attempts; for each valid pick up to N_PATH path attempts.
    Every rejection appends a message to the box's feedback history H_t, and
    every later call for that box receives the whole history (R1.4).
    --no-feedback keeps the budget but always sends an empty history.

Box outcomes:  placed | skipped_no_anchor | budget_exhausted
Attempt codes: pick  -> invalid_json, unknown_anchor, out_of_bounds, collision,
                        unsupported, no_clearance, ok
               path  -> invalid_json, malformed, not_at_target,
                        path_out_of_bounds, path_collision, ok
               both  -> api_error: the API call failed after the bounded backoff
                        (T5.2); consumes the attempt, no feedback message (the
                        model never answered).  API attempts also carry `api`:
                        {api_retries, prompt_tokens, completion_tokens,
                        reasoning_tokens, cost_usd, finish_reason}.
"""
from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from envs.metrics import compute_and_package_metrics
from harness.policies import Policy
from harness.state import build_state, has_anchors, lookup_anchor
from harness.validator import validate_path, validate_pick

N_PICK = 3
N_PATH = 2

PICK_FEEDBACK_INVALID_JSON = ("Return JSON: {'rotation_index': <int>, 'anchor_id': 'rX_aY'}. Do not invent IDs.")
PICK_FEEDBACK_UNKNOWN = "Choose an anchor_id from anchors_indexed with its matching rotation_index. Do not invent IDs."
PICK_FEEDBACK_PATH_FAILED = ("Failed to produce a valid path to your selected anchor. "
                             "Pick a different anchor that is fully supported and has clear vertical access.")
PATH_FEEDBACK_INVALID_JSON = "Return JSON with 'path': [[x,y,z], ...] ending exactly at the target."


# ------------------------ provenance ------------------------

def git_info(repo_root: str) -> Dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    # run outputs under results/ do not make the code dirty
    dirty = run("status", "--porcelain", "--", ".", ":(exclude)results")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty) if dirty is not None else None}


def hardware_info() -> Dict:
    info = {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "gpu": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu"] = "apple-mps"
    except Exception:
        pass
    return info


# ------------------------ episode ------------------------

def _timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def run_episode(
    policy: Policy,
    sequence: Dict,
    *,
    shuffle_anchors: bool = False,
    feedback: bool = True,
    n_pick: int = N_PICK,
    n_path: int = N_PATH,
    log=print,
) -> Dict:
    bin_dims = list(sequence["bin_dims"])
    seed = int(sequence["seed"])
    shuffle_rng = random.Random(10_000 + seed) if shuffle_anchors else None

    placed: List[Dict] = []
    boxes_out: List[Dict] = []
    t_run0 = time.perf_counter()

    for i, size in enumerate(sequence["boxes"]):
        size = list(map(int, size))
        t_box0 = time.perf_counter()
        state = build_state(placed, size, bin_dims, shuffle_rng=shuffle_rng)
        rec: Dict = {
            "index": i, "size": size, "n_anchors_offered": len(state["anchors_indexed"]),
            "attempts": [], "outcome": None, "placement": None, "feedback_history": [],
        }
        boxes_out.append(rec)

        if not has_anchors(state):
            rec["outcome"] = "skipped_no_anchor"
            rec["wall_time_s"] = time.perf_counter() - t_box0
            log(f"[{i + 1:>3}/{len(sequence['boxes'])}] {size} skipped: no feasible anchor")
            continue

        history: List[str] = rec["feedback_history"]
        hist = (lambda: list(history)) if feedback else (lambda: [])

        for pick_attempt in range(1, n_pick + 1):
            out, lat = _timed(policy.pick, state, hist())
            att = {"stage": "pick", "attempt": pick_attempt, "latency_s": lat,
                   "response": out.data, "raw": out.raw, "code": None}
            if out.meta:
                att["api"] = out.meta
            rec["attempts"].append(att)

            data = out.data
            if out.meta and out.meta.get("api_error"):
                att["code"] = "api_error"
                continue
            if not isinstance(data, dict) or "rotation_index" not in data or "anchor_id" not in data:
                att["code"] = "invalid_json"
                history.append(f"[pick attempt {pick_attempt}] {PICK_FEEDBACK_INVALID_JSON}")
                continue
            chosen_size, pos = lookup_anchor(state, data["rotation_index"], data["anchor_id"])
            if chosen_size is None:
                att["code"] = "unknown_anchor"
                history.append(f"[pick attempt {pick_attempt}] {PICK_FEEDBACK_UNKNOWN}")
                continue
            v = validate_pick(pos, chosen_size, placed, bin_dims)
            att["code"] = v.code
            att["rotation_index"] = int(data["rotation_index"])
            att["anchor_id"] = data["anchor_id"]
            att["target"] = pos
            att["chosen_size"] = chosen_size
            if not v.ok:
                history.append(f"[pick attempt {pick_attempt}] {v.message}")
                continue

            path_ok = False
            for path_attempt in range(1, n_path + 1):
                pout, plat = _timed(policy.path, state, pos, hist())
                patt = {"stage": "path", "attempt": path_attempt, "pick_attempt": pick_attempt,
                        "latency_s": plat, "response": pout.data, "raw": pout.raw, "code": None}
                if pout.meta:
                    patt["api"] = pout.meta
                rec["attempts"].append(patt)
                pdata = pout.data
                if pout.meta and pout.meta.get("api_error"):
                    patt["code"] = "api_error"
                    continue
                if not isinstance(pdata, dict) or "path" not in pdata:
                    patt["code"] = "invalid_json"
                    history.append(f"[path attempt {path_attempt}] {PATH_FEEDBACK_INVALID_JSON}")
                    continue
                pv = validate_path(pdata["path"], chosen_size, pos, placed, bin_dims)
                patt["code"] = pv.code
                patt["info"] = pv.info
                if not pv.ok:
                    history.append(f"[path attempt {path_attempt}] {pv.message}")
                    continue
                path = [[float(c) for c in p] for p in pdata["path"]]
                placed.append({"position": list(pos), "size": list(chosen_size)})
                rec["outcome"] = "placed"
                rec["placement"] = {"position": list(pos), "size": list(chosen_size),
                                    "rotation_index": int(data["rotation_index"]),
                                    "anchor_id": data["anchor_id"], "path": path,
                                    "pick_attempt": pick_attempt, "path_attempt": path_attempt,
                                    "diagonal_segments": pv.info.get("diagonal_segments", 0),
                                    "starts_above_bin": pv.info.get("starts_above_bin")}
                path_ok = True
                break

            if path_ok:
                break
            history.append(f"[pick attempt {pick_attempt}] {PICK_FEEDBACK_PATH_FAILED}")

        if rec["outcome"] is None:
            rec["outcome"] = "budget_exhausted"
        rec["wall_time_s"] = time.perf_counter() - t_box0
        fill = sum(b["size"][0] * b["size"][1] * b["size"][2] for b in placed) / (bin_dims[0] * bin_dims[1] * bin_dims[2])
        log(f"[{i + 1:>3}/{len(sequence['boxes'])}] {size} {rec['outcome']:<17} "
            f"calls={len(rec['attempts'])} fill={fill:.3f}")

    total_s = time.perf_counter() - t_run0
    return {"placed_boxes": placed, "boxes": boxes_out, "total_wall_time_s": total_s}


# ------------------------ reliability counters (T6.3) ------------------------

def _pct(x: List[float], q: float) -> float:
    return float(np.percentile(x, q)) if x else 0.0


def _lat_stats(x: List[float]) -> Dict:
    return {"n": len(x), "mean": float(np.mean(x)) if x else 0.0, "median": _pct(x, 50),
            "p95": _pct(x, 95), "max": float(max(x)) if x else 0.0}


def reliability(boxes: List[Dict]) -> Dict:
    attempted = [b for b in boxes if b["outcome"] != "skipped_no_anchor"]
    placed = [b for b in boxes if b["outcome"] == "placed"]
    first_try = [b for b in placed if b["placement"]["pick_attempt"] == 1 and b["placement"]["path_attempt"] == 1]
    picks = [a for b in boxes for a in b["attempts"] if a["stage"] == "pick"]
    paths = [a for b in boxes for a in b["attempts"] if a["stage"] == "path"]
    codes: Dict[str, int] = {}
    for a in picks + paths:
        key = f"{a['stage']}:{a['code']}"
        codes[key] = codes.get(key, 0) + 1
    calls_per_box = [len(b["attempts"]) for b in attempted]
    return {
        "items_total": len(boxes),
        "items_attempted": len(attempted),
        "items_placed": len(placed),
        "items_skipped_no_anchor": len(boxes) - len(attempted),
        "items_budget_exhausted": sum(1 for b in boxes if b["outcome"] == "budget_exhausted"),
        "first_attempt_validity": (len(first_try) / len(attempted)) if attempted else 0.0,
        "retries_per_box": (float(np.mean([c - 2 for c in calls_per_box])) if calls_per_box else 0.0),
        "calls_per_box": float(np.mean(calls_per_box)) if calls_per_box else 0.0,
        "pick_calls": len(picks),
        "path_calls": len(paths),
        "invalid_json": sum(1 for a in picks + paths if a["code"] == "invalid_json"),
        "unknown_anchor": sum(1 for a in picks if a["code"] == "unknown_anchor"),
        "pick_rejected_by_validator": sum(1 for a in picks if a["code"] in ("out_of_bounds", "collision", "unsupported", "no_clearance")),
        "path_collisions": sum(1 for a in paths if a["code"] == "path_collision"),
        "path_not_at_target": sum(1 for a in paths if a["code"] == "not_at_target"),
        "path_out_of_bounds": sum(1 for a in paths if a["code"] == "path_out_of_bounds"),
        "path_malformed": sum(1 for a in paths if a["code"] == "malformed"),
        "diagonal_segments_in_placed_paths": sum(b["placement"]["diagonal_segments"] for b in placed),
        "placed_paths_starting_above_bin": sum(1 for b in placed if b["placement"]["starts_above_bin"]),
        "api_errors": sum(1 for a in picks + paths if a["code"] == "api_error"),
        "attempt_codes": codes,
        "latency_pick_s": _lat_stats([a["latency_s"] for a in picks]),
        "latency_path_s": _lat_stats([a["latency_s"] for a in paths]),
        "latency_box_end_to_end_s": _lat_stats([b["wall_time_s"] for b in attempted]),
    }


def api_usage(boxes: List[Dict]) -> Optional[Dict]:
    """Totals over every API call of the run (T5.2 budget tracking); None for non-API policies."""
    calls = [a["api"] for b in boxes for a in b["attempts"] if a.get("api")]
    if not calls:
        return None

    def total(key):
        vals = [c.get(key) for c in calls if c.get(key) is not None]
        return (sum(vals), len(vals)) if vals else (None, 0)

    out = {"calls": len(calls),
           "retried_calls": sum(1 for c in calls if c.get("api_retries")),
           "retries_total": sum(c.get("api_retries", 0) for c in calls),
           "failed_calls": sum(1 for c in calls if c.get("api_error"))}
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"):
        s, n = total(key)
        out[key] = s
        out[key + "_reported_calls"] = n
    return out


# ------------------------ run record ------------------------

def build_run_record(policy: Policy, sequence: Dict, episode: Dict, flags: Dict, repo_root: str,
                     started_at: str) -> Dict:
    boxes = episode["boxes"]
    placed = episode["placed_boxes"]
    rel = reliability(boxes)
    run_stats = {
        "paths": [b["placement"]["path"] for b in boxes if b["outcome"] == "placed"],
        "pick_calls": rel["pick_calls"], "path_calls": rel["path_calls"],
        "pick_invalid_json": sum(1 for b in boxes for a in b["attempts"] if a["stage"] == "pick" and a["code"] == "invalid_json"),
        "pick_unknown_anchor": rel["unknown_anchor"],
        "path_invalid_json": sum(1 for b in boxes for a in b["attempts"] if a["stage"] == "path" and a["code"] == "invalid_json"),
        "path_not_at_target": rel["path_not_at_target"],
        "pick_latency": [a["latency_s"] for b in boxes for a in b["attempts"] if a["stage"] == "pick"],
        "path_latency": [a["latency_s"] for b in boxes for a in b["attempts"] if a["stage"] == "path"],
        "rotation_hist": {},
    }
    for b in boxes:
        if b["outcome"] == "placed":
            k = str(b["placement"]["rotation_index"])
            run_stats["rotation_hist"][k] = run_stats["rotation_hist"].get(k, 0) + 1
    metrics = compute_and_package_metrics(sequence["bin_dims"], placed, run_stats, model=policy.model_id or policy.name)

    return {
        "schema": "packi-eval-run/1",
        "method": policy.name,
        "policy": policy.describe(),
        "dataset": sequence["dataset"],
        "seed": sequence["seed"],
        "sequence_file": os.path.relpath(os.path.join(repo_root, "data", "sequences", sequence["dataset"], f"seed{sequence['seed']}.json"), repo_root),
        "n_items": sequence["n_items"],
        "bin_dims": sequence["bin_dims"],
        "flags": flags,
        "budget": {"n_pick": flags.get("n_pick", N_PICK), "n_path": flags.get("n_path", N_PATH)},
        "git": git_info(repo_root),
        "hardware": hardware_info(),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_wall_time_s": episode["total_wall_time_s"],
        "metrics": metrics,
        "reliability": rel,
        "api_usage": api_usage(boxes),
        "placed_boxes": placed,
        "boxes": boxes,
    }


def run_file_name(method: str, dataset: str, seed: int, shuffle_anchors: bool, feedback: bool) -> str:
    slug = method.replace("/", "-").replace(":", "-").replace("@", "-")
    name = f"seed{seed}"
    if shuffle_anchors:
        name += ".shuffle"
    if not feedback:
        name += ".nofb"
    return os.path.join(slug, dataset, name + ".json")

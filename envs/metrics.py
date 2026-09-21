# envs/metrics.py

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------- helpers -----------------------------
#
# Axis convention (must match envs/state_manager.py and envs/bin_packing_env.py):
#   position = [x, y, z]      size = [sx, sy, sz]      bin_dims = [X, Y, Z]
# The THIRD component is vertical (z). A box occupies
#   [x, x+sx) x [y, y+sy) x [z, z+sz)   and its top surface is at z + sz.

def _bin_volume(bin_dims: List[int]) -> int:
    return int(bin_dims[0] * bin_dims[1] * bin_dims[2])


def _volume(size: List[float]) -> float:
    return float(size[0] * size[1] * size[2])


def _round_int(x: float) -> int:
    # Boxes/anchors are integral in this sim; be tolerant to tiny FP.
    return int(round(x + 1e-9))


def _as_int_triplet(v: List[float]) -> Tuple[int, int, int]:
    return (_round_int(v[0]), _round_int(v[1]), _round_int(v[2]))


# ----------------------------- voxelizer -----------------------------

def voxelize(bin_dims: List[int], boxes: List[Dict]) -> np.ndarray:
    """
    Return a boolean occupancy grid occ[x, y, z] with 6-neighbor connectivity.
    Resolution = 1. Assumes integer sizes/positions.
    """
    W, D, H = map(int, bin_dims)  # extents along x, y, z
    occ = np.zeros((W, D, H), dtype=bool)
    for b in boxes:
        x0, y0, z0 = map(_round_int, b["position"])
        sx, sy, sz = map(_round_int, b["size"])  # size = [sx, sy, sz], sz is vertical (see convention above)
        x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz
        # clamp defensively
        x0c, y0c, z0c = max(0, x0), max(0, y0), max(0, z0)
        x1c, y1c, z1c = min(W, x1), min(D, y1), min(H, z1)
        if x1c > x0c and y1c > y0c and z1c > z0c:
            occ[x0c:x1c, y0c:y1c, z0c:z1c] = True
    return occ


# ----------------------------- empty-space metrics -----------------------------

def _neighbors_6(x: int, y: int, z: int, W: int, D: int, H: int):
    if x > 0: yield (x-1, y, z)
    if x+1 < W: yield (x+1, y, z)
    if y > 0: yield (x, y-1, z)
    if y+1 < D: yield (x, y+1, z)
    if z > 0: yield (x, y, z-1)
    if z+1 < H: yield (x, y, z+1)


def largest_empty_cavity_and_frag(occ: np.ndarray) -> Tuple[int, int]:
    """
    Return (largest_component_voxels, num_components) for the empty space.
    """
    inv = ~occ
    W, D, H = occ.shape
    seen = np.zeros_like(inv, dtype=bool)
    largest = 0
    comps = 0
    for x in range(W):
        for y in range(D):
            for z in range(H):
                if not inv[x, y, z] or seen[x, y, z]:
                    continue
                # BFS
                comps += 1
                q = [(x, y, z)]
                seen[x, y, z] = True
                size = 0
                while q:
                    cx, cy, cz = q.pop()
                    size += 1
                    for nx, ny, nz in _neighbors_6(cx, cy, cz, W, D, H):
                        if inv[nx, ny, nz] and not seen[nx, ny, nz]:
                            seen[nx, ny, nz] = True
                            q.append((nx, ny, nz))
                largest = max(largest, size)
    return largest, comps


# ----------------------------- top planarity -----------------------------

def _largest_connected_area_2d(mask2d: np.ndarray) -> int:
    """Area (cells) of the largest 4-neighbor component in a 2D boolean mask."""
    H, W = mask2d.shape  # (rows, cols)
    seen = np.zeros_like(mask2d, dtype=bool)
    best = 0
    for r in range(H):
        for c in range(W):
            if not mask2d[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            area = 0
            while q:
                rr, cc = q.pop()
                area += 1
                if rr > 0 and mask2d[rr-1, cc] and not seen[rr-1, cc]:
                    seen[rr-1, cc] = True; q.append((rr-1, cc))
                if rr+1 < H and mask2d[rr+1, cc] and not seen[rr+1, cc]:
                    seen[rr+1, cc] = True; q.append((rr+1, cc))
                if cc > 0 and mask2d[rr, cc-1] and not seen[rr, cc-1]:
                    seen[rr, cc-1] = True; q.append((rr, cc-1))
                if cc+1 < W and mask2d[rr, cc+1] and not seen[rr, cc+1]:
                    seen[rr, cc+1] = True; q.append((rr, cc+1))
            best = max(best, area)
    return best


def top_surface_planarity(occ: np.ndarray) -> int:
    """
    Largest connected top-surface area (in cells).
    A top cell is occupied at z and empty at z+1 (or z is the topmost layer).
    """
    W, D, H = occ.shape
    best = 0
    for z in range(H):
        top_mask = occ[:, :, z] & (z == H-1 or ~occ[:, :, z+1])
        if not np.any(top_mask):
            continue
        # convert to (rows, cols) = (D, W) for the 2D helper
        area = _largest_connected_area_2d(top_mask.T)
        best = max(best, area)
    return best


# ----------------------------- support coverage -----------------------------

def support_coverage_for_box(idx: int, placed_boxes: List[Dict]) -> float:
    """
    Fraction of base area supported by floor or tops of *previous* boxes
    (i.e., no support from boxes placed after index idx).
    """
    EPS = 1e-6
    box = placed_boxes[idx]
    x0, y0, z0 = map(float, box["position"])
    sx, sy, sz = map(float, box["size"])  # sz vertical
    base_area = sx * sy
    if base_area <= 0:
        return 0.0
    if z0 <= EPS:
        return 1.0

    prev = placed_boxes[:idx]
    covered = 0.0
    x1, y1 = x0 + sx, y0 + sy

    for b in prev:
        bx, by, bz = map(float, b["position"])
        bsx, bsy, bsz = map(float, b["size"])
        top_z = bz + bsz
        if abs(top_z - z0) > EPS:
            continue
        ix = max(0.0, min(x1, bx + bsx) - max(x0, bx))
        iy = max(0.0, min(y1, by + bsy) - max(y0, by))
        if ix > 0.0 and iy > 0.0:
            covered += ix * iy

    return max(0.0, min(1.0, covered / base_area))


def support_coverage_series(placed_boxes: List[Dict]) -> List[float]:
    return [support_coverage_for_box(i, placed_boxes) for i in range(len(placed_boxes))]


# ----------------------------- corner / layer metrics -----------------------------

def corner_flush_rate(placed_boxes: List[Dict], bin_dims: List[int]) -> float:
    """
    % placements that are flush to at least two orthogonal bin faces
    (e.g., floor + one wall, or two walls).
    """
    EPS = 1e-6
    cnt = 0
    for b in placed_boxes:
        x, y, z = map(float, b["position"])
        flush_walls = int(abs(x - 0.0) <= EPS) + int(abs(y - 0.0) <= EPS)
        flush_floor = int(abs(z - 0.0) <= EPS)
        if flush_floor + flush_walls >= 2:
            cnt += 1
    return cnt / max(1, len(placed_boxes))


def layer_completion_events(placed_boxes: List[Dict], bin_dims: List[int]) -> Tuple[int, float]:
    """
    Count events where the XY coverage of bases at a given base-z plane reaches full W*D.
    Return (num_events, ratio_over_placements).
    """
    W, D, _ = map(int, bin_dims)  # x, y extents
    events = 0
    coverage: Dict[int, np.ndarray] = {}

    for b in placed_boxes:
        x, y, z = map(_round_int, b["position"])
        sx, sy, _sz = map(_round_int, b["size"])  # sz vertical
        if z not in coverage:
            coverage[z] = np.zeros((W, D), dtype=bool)
        cov = coverage[z]
        cov[x:x+sx, y:y+sy] = True
        if cov.all():
            events += 1
            # reset to avoid double counting on the same plane
            coverage[z] = np.zeros((W, D), dtype=bool)

    ratio = events / max(1, len(placed_boxes))
    return events, ratio


# ----------------------------- fill curve -----------------------------

def fill_curve(placed_boxes: List[Dict], bin_dims: List[int]) -> List[float]:
    Vbin = _bin_volume(bin_dims)
    acc = 0.0
    curve = []
    for b in placed_boxes:
        acc += _volume(b["size"])
        curve.append(acc / max(1, Vbin))
    return curve


# ----------------------------- path metrics -----------------------------

def path_length_and_turns(path: List[List[float]]) -> Tuple[float, int]:
    """
    Manhattan length & number of turns (axis changes).
    """
    if not path or len(path) < 2:
        return 0.0, 0
    length = 0.0
    turns = 0
    prev_dir = None
    for i in range(1, len(path)):
        x0, y0, z0 = path[i-1]
        x1, y1, z1 = path[i]
        dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
        length += dx + dy + dz
        cur_dir = (dx > 0, dy > 0, dz > 0)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return float(length), int(turns)


# ----------------------------- aggregate + save -----------------------------

def compute_and_package_metrics(
    bin_dims: List[int],
    placed_boxes: List[Dict],
    run_stats: Dict,
    model: str | None = None,
) -> Dict:
    """
    Produce a serializable dict with all metrics.
    `run_stats` may contain:
      - 'paths': List[List[List[float]]]
      - counters: 'pick_calls','path_calls','pick_invalid_json','pick_unknown_anchor',
                  'path_invalid_json','path_not_at_target'
      - 'pick_latency', 'path_latency' (lists of floats)
      - 'rotation_hist' (dict)
    """
    Vbin = _bin_volume(bin_dims)
    Vused = sum(_volume(b["size"]) for b in placed_boxes)
    U = Vused / max(1, Vbin)

    curve = fill_curve(placed_boxes, bin_dims)

    # voxel metrics
    occ = voxelize(bin_dims, placed_boxes) if Vbin > 0 else None
    if occ is not None:
        empty_total = int(np.prod(occ.shape) - occ.sum())
        largest, comps = largest_empty_cavity_and_frag(occ)
        LEC = (largest / empty_total) if empty_total > 0 else 0.0
        EFI = int(comps)
        TPS = int(top_surface_planarity(occ))
    else:
        LEC, EFI, TPS = 1.0, 1, 0

    # stability / support
    SC_series = support_coverage_series(placed_boxes)
    SC_mean = float(np.mean(SC_series)) if SC_series else 0.0
    SC_full_rate = float(np.mean([c >= 0.999 for c in SC_series])) if SC_series else 0.0

    # corner & layer structure
    CFR = corner_flush_rate(placed_boxes, bin_dims)
    layer_events, layer_ratio = layer_completion_events(placed_boxes, bin_dims)

    # paths
    paths = run_stats.get("paths", [])
    path_lengths, path_turns = [], []
    for p in paths:
        L, T = path_length_and_turns(p)
        path_lengths.append(L)
        path_turns.append(T)
    mean_path_len = float(np.mean(path_lengths)) if path_lengths else 0.0
    mean_path_turns = float(np.mean(path_turns)) if path_turns else 0.0

    # LLM reliability & efficiency
    pick_calls = int(run_stats.get("pick_calls", 0))
    path_calls = int(run_stats.get("path_calls", 0))
    a_total = pick_calls + path_calls
    AER = (100.0 * U / a_total) if a_total > 0 else 0.0

    final_model = model
    if final_model is None:
        # main.py demo: name the backend config.py selected (harness passes `model` explicitly)
        import config
        final_model = config.LOCAL_MODEL if config.USE_LOCAL_LLM else config.API_MODEL

    metrics = {
        "model": final_model,
        "bin_dims": bin_dims,
        "placements": len(placed_boxes),
        "utilization_final": U,
        "utilization_curve": curve,
        "packed_volume": Vused,
        "empty_volume": Vbin - Vused,
        "largest_empty_cavity_ratio": LEC,
        "empty_fragmentation_count": EFI,
        "top_planarity_area": TPS,
        "support_coverage_mean": SC_mean,
        "support_coverage_full_rate": SC_full_rate,
        "corner_flush_rate": CFR,
        "layer_completion_events": layer_events,
        "layer_completion_ratio": layer_ratio,
        "actions": {
            "pick_calls": pick_calls,
            "path_calls": path_calls,
            "AER": AER,
            "pick_invalid_json": int(run_stats.get("pick_invalid_json", 0)),
            "pick_unknown_anchor": int(run_stats.get("pick_unknown_anchor", 0)),
            "path_invalid_json": int(run_stats.get("path_invalid_json", 0)),
            "path_not_at_target": int(run_stats.get("path_not_at_target", 0)),
            "rotation_hist": run_stats.get("rotation_hist", {}),
            "mean_pick_latency_s": float(np.mean(run_stats.get("pick_latency", []) or [0.0])),
            "mean_path_latency_s": float(np.mean(run_stats.get("path_latency", []) or [0.0])),
        },
        "paths": {
            "count": len(paths),
            "mean_length_L1": mean_path_len,
            "mean_turns": mean_path_turns,
        },
    }
    return metrics


def save_run_metrics(
    bin_dims: List[int],
    placed_boxes: List[Dict],
    run_stats: Dict,
    snapshot_dir: str | None = None,
    out_dir: str = "runs"
):
    """
    Compute metrics and persist them as a JSON file, next to snapshots by default.
    """
    metrics = compute_and_package_metrics(bin_dims, placed_boxes, run_stats)

    # output dir
    if snapshot_dir:
        out_dir = os.path.abspath(os.path.join(snapshot_dir, "..", "metrics"))
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"metrics_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


    print(f"[metrics] Saved metrics to {out_path}")
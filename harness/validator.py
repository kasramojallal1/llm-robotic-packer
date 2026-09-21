"""
The placement contract (paper Sec. 4.5 (i)-(v) + continuous AABB checks, D25).

validate_pick  : anchor exists, containment, no overlap, full-base support,
                 vertical clearance
validate_path  : well-formed, ends exactly at the target, stays inside the bin
                 in x/y and above the floor (z above the bin is allowed for the
                 overhead entry), and no collision along any segment (exact
                 swept-AABB interval test; touching is not a collision).
                 Diagonal segments and a start below the bin top are counted
                 in `info`, not rejected.

Every rejection carries a short machine code and the human-readable feedback
message that goes into the policy's feedback history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from envs.state_manager import check_collision, has_vertical_clearance, is_supported, is_within_bounds

EPS = 1e-9


@dataclass
class Verdict:
    ok: bool
    code: str
    message: str = ""
    info: Dict = field(default_factory=dict)


# ------------------------ pick ------------------------

def validate_pick(pos: List[int], size: List[int], placed: List[Dict], bin_dims: List[int]) -> Verdict:
    if not is_within_bounds(pos, size, bin_dims):
        return Verdict(False, "out_of_bounds",
                       "Selected anchor is out of bounds. Pick another anchor from the list.")
    if check_collision(pos, size, placed):
        return Verdict(False, "collision",
                       "Selected anchor collides with existing boxes. Pick another.")
    if not is_supported(pos, size, placed):
        return Verdict(False, "unsupported",
                       "NO OVERHANGS: the entire base must be supported by floor or box tops at the same Z. "
                       "Pick a different anchor with full support.")
    if not has_vertical_clearance(pos, size, placed, bin_dims):
        return Verdict(False, "no_clearance",
                       "Top-down access blocked: there is geometry above this anchor within the box's XY footprint. "
                       "Choose a different anchor with clear vertical access.")
    return Verdict(True, "ok")


# ------------------------ path ------------------------

def coerce_path(raw) -> Optional[List[List[float]]]:
    """Return a list of numeric [x,y,z] points, or None if malformed."""
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            return None
        try:
            q = [float(v) for v in p]
        except (TypeError, ValueError):
            return None
        if isinstance(p[0], bool) or isinstance(p[1], bool) or isinstance(p[2], bool):
            return None
        out.append(q)
    return out


def _sweep_interval(p0: float, p1: float, s: float, q: float, t: float) -> Tuple[float, float]:
    """
    Parameter interval tau in [0,1] during which [p(tau), p(tau)+s) and [q, q+t)
    overlap, for p(tau) = p0 + (p1-p0)*tau.  Empty -> (1, 0).
    Overlap iff  p(tau) < q + t  and  p(tau) + s > q.
    """
    v = p1 - p0
    lo, hi = 0.0, 1.0
    # p0 + v*tau < q + t
    if abs(v) < EPS:
        if not (p0 < q + t - EPS):
            return 1.0, 0.0
    elif v > 0:
        hi = min(hi, (q + t - p0) / v)
    else:
        lo = max(lo, (q + t - p0) / v)
    # p0 + v*tau > q - s
    if abs(v) < EPS:
        if not (p0 + s > q + EPS):
            return 1.0, 0.0
    elif v > 0:
        lo = max(lo, (q - s - p0) / v)
    else:
        hi = min(hi, (q - s - p0) / v)
    return lo, hi


def segment_collides(p0: List[float], p1: List[float], size: List[float], placed: List[Dict]) -> Optional[Dict]:
    """First placed box the moving AABB overlaps (with positive duration) on p0->p1, else None."""
    for b in placed:
        lo, hi = 0.0, 1.0
        for a in range(3):
            l, h = _sweep_interval(p0[a], p1[a], size[a], float(b["position"][a]), float(b["size"][a]))
            lo, hi = max(lo, l), min(hi, h)
            if hi - lo <= EPS:
                break
        else:
            return b
    return None


def _point_in_bin_xy(p: List[float], size: List[float], bin_dims: List[int]) -> bool:
    return (p[0] >= -EPS and p[0] + size[0] <= bin_dims[0] + EPS
            and p[1] >= -EPS and p[1] + size[1] <= bin_dims[1] + EPS
            and p[2] >= -EPS)


def validate_path(raw_path, size: List[int], target: List[int], placed: List[Dict], bin_dims: List[int]) -> Verdict:
    path = coerce_path(raw_path)
    if path is None:
        return Verdict(False, "malformed",
                       "Return JSON with 'path': [[x,y,z], ...] of numeric triplets ending exactly at the target.")

    if any(abs(a - b) > EPS for a, b in zip(path[-1], target)):
        return Verdict(False, "not_at_target",
                       f"Your path must end exactly at the target coordinates {list(target)}.")

    info = {
        "n_waypoints": len(path),
        "diagonal_segments": 0,
        "starts_above_bin": path[0][2] >= bin_dims[2] - EPS,
    }
    for i in range(1, len(path)):
        moved = sum(1 for a in range(3) if abs(path[i][a] - path[i - 1][a]) > EPS)
        if moved > 1:
            info["diagonal_segments"] += 1

    # containment along the path (linear motion -> endpoints suffice; z may exceed the bin top)
    for p in path:
        if not _point_in_bin_xy(p, size, bin_dims):
            return Verdict(False, "path_out_of_bounds",
                           f"Waypoint {[round(v, 3) for v in p]} leaves the bin in x/y or goes below the floor. "
                           "Keep every waypoint inside the bin (only z above the bin is allowed).", info)

    # continuous AABB collision along every segment
    fsize = [float(v) for v in size]
    for i in range(1, len(path)):
        hit = segment_collides(path[i - 1], path[i], fsize, placed)
        if hit is not None:
            return Verdict(False, "path_collision",
                           f"Path collides with an existing box at position {hit['position']} (size {hit['size']}) "
                           f"between waypoints {[round(v, 3) for v in path[i - 1]]} and {[round(v, 3) for v in path[i]]}. "
                           "Approach from above and descend straight down onto the target.", info)

    return Verdict(True, "ok", "", info)

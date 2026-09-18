# envs/state_manager.py

import json
from itertools import permutations

# ---------- Small helpers ----------

def generate_orientations(size):
    """All unique 90° axis-aligned orientations of [w,h,d]."""
    if not size or len(size) != 3:
        return []
    return [list(p) for p in sorted(set(permutations(size, 3)))]

def is_within_bounds(pos, size, bin_dims):
    """Check if box at pos with size fits entirely in bin."""
    for i in range(3):
        if pos[i] < 0:
            return False
        if pos[i] + size[i] > bin_dims[i]:
            return False
    return True

# ---------- Core placement utilities ----------

def check_collision(new_pos, new_size, placed_boxes):
    """AABB overlap test against placed boxes."""
    nx, ny, nz = new_pos
    nsx, nsy, nsz = new_size

    for box in placed_boxes:
        px, py, pz = box["position"]
        psx, psy, psz = box["size"]

        overlap_x = not (nx + nsx <= px or nx >= px + psx)
        overlap_y = not (ny + nsy <= py or ny >= py + psy)
        overlap_z = not (nz + nsz <= pz or nz >= pz + psz)

        if overlap_x and overlap_y and overlap_z:
            return True
    return False

def is_supported(new_pos, new_size, placed_boxes, eps=1e-6):
    """
    Full-base support: on the floor (z=0), or the ENTIRE base [nx,nx+nsx)x[ny,ny+nsy)
    is covered by the top surfaces of placed boxes whose top is exactly at z=nz.

    Placed boxes never overlap (check_collision), so the covered area is the sum
    of the per-box intersection areas. Anchors from generate_anchor_positions lie
    inside a single supporting box's footprint and therefore always pass.
    """
    nx, ny, nz = new_pos
    nsx, nsy, nsz = new_size

    if nz == 0:
        return True

    base_area = nsx * nsy
    if base_area <= 0:
        return False

    covered = 0.0
    for box in placed_boxes:
        px, py, pz = box["position"]
        psx, psy, psz = box["size"]
        top_z = pz + psz
        if abs(nz - top_z) > 0.1:
            continue
        ix = max(0, min(nx + nsx, px + psx) - max(nx, px))
        iy = max(0, min(ny + nsy, py + psy) - max(ny, py))
        covered += ix * iy

    return covered >= base_area - eps

def generate_anchor_positions(placed_boxes, new_box_size, bin_dims):
    """
    Candidate anchors for a given oriented box size:
    - Floor (z=0)
    - Top of boxes, within bounds, non-colliding
    """
    anchors = []
    bin_w, bin_h, bin_d = bin_dims
    bw, bh, bd = new_box_size

    # Floor anchors
    for x in range(0, max(0, bin_w - bw) + 1):
        for y in range(0, max(0, bin_h - bh) + 1):
            z = 0
            pos = [x, y, z]
            if is_within_bounds(pos, new_box_size, bin_dims) and not check_collision(pos, new_box_size, placed_boxes):
                anchors.append(pos)

    # Top anchors
    for box in placed_boxes:
        px, py, pz = box["position"]
        psx, psy, psz = box["size"]
        top_z = pz + psz

        # Only slide the footprint INSIDE the supporting box's top face so that
        # every top anchor has full-base support. If the new box is wider/deeper
        # than the support, the ranges below are empty and no anchor is offered.
        # (The previous max(0, ...) offered one overhanging corner anchor here.)
        for dx in range(0, psx - bw + 1):
            for dy in range(0, psy - bh + 1):
                x = px + dx
                y = py + dy
                z = top_z
                pos = [x, y, z]
                if (is_within_bounds(pos, new_box_size, bin_dims) and
                    not check_collision(pos, new_box_size, placed_boxes)):
                    anchors.append(pos)

    return anchors

# ---------- Small scoring (corners/edges first) + top-K ----------

def score_anchor(pos, size, bin_dims):
    """Higher score = more preferred (corners/edges)."""
    x, y, z = pos
    w, h, d = size
    bw, bh, bd = bin_dims
    score = 0
    # Touching walls (prefer corners/edges)
    score += (x == 0) + (y == 0) + (z == 0)
    score += (x + w == bw) + (y + h == bh) + (z + d == bd)
    # Light bias to lower coords to keep center open
    score += max(0, (bw - (x + w))) * 0.01
    score += max(0, (bh - (y + h))) * 0.01
    return score

def topk_anchors(anchors, size, bin_dims, k=8):
    ranked = sorted(anchors, key=lambda p: score_anchor(p, size, bin_dims), reverse=True)
    return ranked[:k]

# ---------- State writer with rotations + indexed anchors ----------

def save_bin_state(placed_boxes, new_box, bin_dims, path="instructions/bin_state.json", top_k=8, require_vertical_clearance=True):
    rotations = generate_orientations(new_box["size"])

    anchors_indexed = []
    for r_idx, rot_size in enumerate(rotations):
        anchors_all = generate_anchor_positions(placed_boxes, rot_size, bin_dims)

        if require_vertical_clearance:
            anchors_all = filter_anchors_with_clearance(anchors_all, rot_size, placed_boxes, bin_dims)

        anchors_k = topk_anchors(anchors_all, rot_size, bin_dims, k=top_k)
        anchors_with_ids = [{"id": f"r{r_idx}_a{j}", "pos": pos} for j, pos in enumerate(anchors_k)]
        anchors_indexed.append({
            "rotation_index": r_idx,
            "size": rot_size,
            "anchors": anchors_with_ids
        })

    state = {
        "bin": {"width": bin_dims[0], "height": bin_dims[1], "depth": bin_dims[2]},
        "placed_boxes": placed_boxes,
        "incoming_box": {"original_size": new_box["size"], "rotations": rotations},
        "anchors_indexed": anchors_indexed
    }

    with open(path, "w") as f:
        json.dump(state, f, indent=None, separators=(",", ": "))





# ---------- Vertical top-down clearance ----------

def has_vertical_clearance(anchor_pos, box_size, placed_boxes, bin_dims, eps=1e-6):
    """
    Return True iff a box placed at anchor_pos with box_size can be inserted
    straight down from the bin top (z = bin_dims[2]) without hitting anything.

    Logic: for any existing box that overlaps the new box's XY footprint,
    its top surface (pz + psz) must be <= target z. If any overlap has top_z > z,
    the descent path is obstructed.
    """
    x, y, z = anchor_pos
    w, h, d = box_size
    x_max, y_max = x + w, y + h

    for b in placed_boxes:
        px, py, pz = b["position"]
        psx, psy, psz = b["size"]

        # XY overlap with the new box footprint?
        overlap_x = (x < px + psx) and (x_max > px)
        overlap_y = (y < py + psy) and (y_max > py)
        if not (overlap_x and overlap_y):
            continue

        top_z = pz + psz
        # If any overlapping box protrudes above target z, insertion path is blocked
        if top_z > z + eps:
            return False

    # Also implicitly relies on is_within_bounds() you already run earlier,
    # which guarantees z + d <= bin_dims[2] (ceiling) so there is room to start from the top.
    return True



def filter_anchors_with_clearance(anchors, box_size, placed_boxes, bin_dims):
    """Return only anchors that have straight-down vertical clearance."""
    return [pos for pos in anchors
            if has_vertical_clearance(pos, box_size, placed_boxes, bin_dims)]


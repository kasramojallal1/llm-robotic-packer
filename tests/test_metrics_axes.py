"""
Regression tests for the axis convention shared by the simulator and the metrics.

Convention (envs/state_manager.py, envs/bin_packing_env.py, envs/metrics.py):
    position = [x, y, z], size = [sx, sy, sz], the THIRD component is vertical.

History: envs/metrics.py used to treat size[1] as vertical, so every voxel-based
metric (LEC, support coverage, layer completion, top planarity) was computed on a
grid with y and z swapped (revision task T0.8, reviewer comment R1.2).

Run from the repo root:  pytest tests/
"""
import numpy as np
import pytest

from envs import metrics
from envs import state_manager as sm

BIN = [10, 10, 10]

# A: 4 along x, 2 along y, 3 tall -> top surface at z=3.  B: 2x2x2 cube on A's top.
A = {"position": [0, 0, 0], "size": [4, 2, 3]}
B = {"position": [0, 0, 3], "size": [2, 2, 2]}


def _extents(occ):
    xs, ys, zs = occ.nonzero()
    return (int(xs.max()) + 1, int(ys.max()) + 1, int(zs.max()) + 1)


# ----------------------------- voxelizer -----------------------------

def test_voxelize_uses_third_component_as_height():
    occ = metrics.voxelize(BIN, [A])
    assert occ.shape == tuple(BIN)
    assert _extents(occ) == (4, 2, 3)
    assert int(occ.sum()) == 4 * 2 * 3
    assert occ[3, 1, 2] and not occ[0, 2, 0] and not occ[0, 0, 3]


def test_voxelize_two_boxes_stack_without_gap_or_overlap():
    occ = metrics.voxelize(BIN, [A, B])
    assert int(occ.sum()) == 24 + 8
    # B sits directly on A: column (0,0) is filled from z=0 to z=4 with nothing above.
    assert occ[0, 0, :5].all() and not occ[0, 0, 5:].any()


# ----------------------------- support coverage -----------------------------

def test_support_coverage_full_when_resting_on_top():
    assert metrics.support_coverage_for_box(1, [A, B]) == pytest.approx(1.0)
    assert metrics.support_coverage_series([A, B]) == pytest.approx([1.0, 1.0])


def test_support_coverage_partial_overhang():
    # C hangs 1 unit over A's +x edge: 3 wide, only 3-1=2 columns supported -> 2/3
    C = {"position": [2, 0, 3], "size": [3, 2, 1]}
    assert metrics.support_coverage_for_box(1, [A, C]) == pytest.approx(2 / 3)


def test_support_coverage_zero_when_floating():
    F = {"position": [0, 0, 4], "size": [2, 2, 2]}  # 1 unit above A's top
    assert metrics.support_coverage_for_box(1, [A, F]) == 0.0


# ----------------------------- cavity / fragmentation / planarity -----------------------------

def test_lec_and_fragmentation_sane_for_two_box_case():
    occ = metrics.voxelize(BIN, [A, B])
    largest, comps = metrics.largest_empty_cavity_and_frag(occ)
    empty = int(np.prod(occ.shape) - occ.sum())
    assert empty == 1000 - 32
    # Two boxes in a corner leave one connected empty region.
    assert comps == 1
    assert largest == empty
    assert largest / empty == pytest.approx(1.0)


def test_lec_detects_sealed_cavity():
    # A 2x2x1 slab sealing a 1x1x1 hole: floor ring of four 1x1x1 blocks, lid on top.
    ring = [
        {"position": [0, 0, 0], "size": [1, 1, 1]},
        {"position": [1, 0, 0], "size": [1, 1, 1]},
        {"position": [0, 1, 0], "size": [1, 1, 1]},
    ]
    # leave (1,1,0) empty and seal it with a lid at z=1
    lid = {"position": [0, 0, 1], "size": [2, 2, 1]}
    # close the open sides of the hole at (1,1,0) with two wall blocks
    walls = [
        {"position": [2, 1, 0], "size": [1, 1, 1]},
        {"position": [1, 2, 0], "size": [1, 1, 1]},
    ]
    occ = metrics.voxelize([4, 4, 3], ring + walls + [lid])
    assert not occ[1, 1, 0]  # the hole
    largest, comps = metrics.largest_empty_cavity_and_frag(occ)
    assert comps == 2  # the sealed hole + the rest of the bin
    empty = int(np.prod(occ.shape) - occ.sum())
    assert largest == empty - 1


def test_top_planarity_reads_true_top_surface():
    occ = metrics.voxelize(BIN, [A])
    # A's top at z=3 is a 4x2 = 8-cell plateau (and it is the largest one).
    assert metrics.top_surface_planarity(occ) == 8


# ----------------------------- layer completion -----------------------------

def test_layer_completion_uses_xy_footprint():
    bin_dims = [4, 2, 5]
    # Two boxes of size [2, 2, 3] tile the 4x2 floor exactly -> one layer event.
    boxes = [
        {"position": [0, 0, 0], "size": [2, 2, 3]},
        {"position": [2, 0, 0], "size": [2, 2, 3]},
    ]
    events, ratio = metrics.layer_completion_events(boxes, bin_dims)
    assert events == 1 and ratio == pytest.approx(0.5)
    # With the old (size[1]=vertical) reading the footprint would be 2x3 and never tile a 4x2 floor.


# ----------------------------- simulator <-> metrics agreement -----------------------------

def test_simulator_and_metrics_agree_on_two_box_case():
    assert sm.is_supported(B["position"], B["size"], [A])
    anchors = sm.generate_anchor_positions([A], B["size"], BIN)
    assert [0, 0, 3] in anchors
    assert metrics.support_coverage_for_box(1, [A, B]) == pytest.approx(1.0)


def test_all_generated_top_anchors_have_full_support():
    """Every top anchor the simulator offers must score coverage 1.0 in the metrics
    and pass the tightened is_supported (T0.9)."""
    rng = np.random.default_rng(0)
    placed = [A, B, {"position": [5, 5, 0], "size": [3, 4, 2]}]
    for _ in range(50):
        size = [int(v) for v in rng.integers(1, 4, size=3)]
        for pos in sm.generate_anchor_positions(placed, size, BIN):
            if pos[2] == 0:
                continue
            new = {"position": list(pos), "size": size}
            assert sm.is_supported(pos, size, placed), (pos, size)
            assert metrics.support_coverage_for_box(len(placed), placed + [new]) == pytest.approx(1.0), (pos, size)


# ----------------------------- T0.9: full-base support rule -----------------------------

def test_is_supported_floor_and_full_top():
    assert sm.is_supported([7, 7, 0], [2, 2, 2], [A])
    assert sm.is_supported([0, 0, 3], [4, 2, 1], [A])  # exactly A's footprint


def test_is_supported_rejects_partial_overhang():
    # Previously accepted (any xy overlap counted); now rejected.
    assert not sm.is_supported([2, 0, 3], [3, 2, 1], [A])
    assert not sm.is_supported([0, 1, 3], [2, 2, 1], [A])


def test_is_supported_rejects_wrong_height():
    assert not sm.is_supported([0, 0, 4], [2, 2, 2], [A])
    assert not sm.is_supported([0, 0, 2], [2, 2, 2], [A])


def test_is_supported_accepts_bridge_over_two_equal_height_boxes():
    left = {"position": [0, 0, 0], "size": [2, 2, 3]}
    right = {"position": [2, 0, 0], "size": [2, 2, 3]}
    assert sm.is_supported([1, 0, 3], [2, 2, 1], [left, right])
    gap_right = {"position": [3, 0, 0], "size": [2, 2, 3]}
    assert not sm.is_supported([1, 0, 3], [3, 2, 1], [left, gap_right])  # 1-unit gap under it


# ----------------------------- anchor generator: no overhanging anchors -----------------------------

def test_generator_offers_no_anchor_when_box_is_wider_than_support():
    # Regression: with the old max(0, psx - bw) + 1 the 3x2x2 box got [0,0,5] on top of the 2x2 cube B.
    tops = [a for a in sm.generate_anchor_positions([A, B], [3, 2, 2], BIN) if a[2] > 0]
    # Nothing fits on B (too narrow) and every spot on A's top collides with B.
    assert tops == []


def test_generator_still_offers_exact_fit_on_top():
    tops = [a for a in sm.generate_anchor_positions([A, B], [2, 2, 1], BIN) if a[2] == 5]
    assert tops == [[0, 0, 5]]

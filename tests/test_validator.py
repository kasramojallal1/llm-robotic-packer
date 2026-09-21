"""The placement contract (D25): pose checks and continuous (swept-AABB) path checks."""
import pytest

from harness.validator import segment_collides, validate_path, validate_pick

BIN = [10, 10, 10]
# A: 4 along x, 2 along y, 3 tall -> top at z=3
A = {"position": [0, 0, 0], "size": [4, 2, 3]}
PLACED = [A]


# ------------------------ pick ------------------------

def test_pick_ok_on_floor_and_on_top():
    assert validate_pick([4, 0, 0], [2, 2, 2], PLACED, BIN).ok
    assert validate_pick([0, 0, 3], [2, 2, 2], PLACED, BIN).ok


def test_pick_rejects_overhang():
    v = validate_pick([2, 0, 3], [3, 2, 2], PLACED, BIN)   # hangs 1 unit past A's edge
    assert not v.ok and v.code == "unsupported"


def test_pick_rejects_collision():
    v = validate_pick([1, 0, 1], [2, 2, 2], PLACED, BIN)
    assert not v.ok and v.code == "collision"


def test_pick_rejects_out_of_bounds():
    v = validate_pick([9, 0, 0], [2, 2, 2], PLACED, BIN)
    assert not v.ok and v.code == "out_of_bounds"


def test_pick_rejects_floating():
    v = validate_pick([6, 6, 2], [2, 2, 2], PLACED, BIN)
    assert not v.ok and v.code == "unsupported"


def test_pick_rejects_blocked_descent():
    tall = [{"position": [0, 0, 0], "size": [2, 2, 2]}, {"position": [0, 0, 4], "size": [2, 2, 2]}]
    # the gap at z=2..4 under the second box is unreachable from above (and unsupported? no: top of first is at 2)
    v = validate_pick([0, 0, 2], [2, 2, 2], tall, BIN)
    assert not v.ok and v.code in ("collision", "no_clearance")


# ------------------------ path ------------------------

def test_path_straight_descent_ok():
    v = validate_path([[0, 0, 12], [0, 0, 4], [0, 0, 3]], [2, 2, 2], [0, 0, 3], PLACED, BIN)
    assert v.ok and v.info["diagonal_segments"] == 0 and v.info["starts_above_bin"]


def test_path_through_a_box_rejected():
    # descend into A at (0,0,1), then slide sideways to the (valid) target: the descent crosses A
    v = validate_path([[0, 0, 12], [0, 0, 1], [4, 0, 1], [4, 0, 0]], [2, 2, 2], [4, 0, 0], PLACED, BIN)
    assert not v.ok and v.code == "path_collision"


def test_path_sliding_along_a_face_is_not_a_collision():
    # descend touching A's x=4 face the whole way
    v = validate_path([[4, 0, 12], [4, 0, 0]], [2, 2, 2], [4, 0, 0], PLACED, BIN)
    assert v.ok


def test_diagonal_clip_between_waypoints_is_caught():
    # endpoints are collision-free, the straight line between them cuts A's top corner
    assert segment_collides([5, 0, 4], [3, 0, 2], [2, 2, 2], PLACED) is A
    v = validate_path([[5, 0, 4], [3, 0, 2], [3, 0, 0]], [2, 2, 2], [3, 0, 0], [], BIN)
    assert v.ok and v.info["diagonal_segments"] == 1   # counted, not rejected (D25)


def test_diagonal_endpoints_free_but_sweep_collides():
    v = validate_path([[5, 0, 4], [3, 0, 2], [3, 0, 0]], [2, 2, 2], [3, 0, 0], PLACED, BIN)
    assert not v.ok and v.code == "path_collision"


def test_path_must_end_at_target():
    v = validate_path([[0, 0, 12], [0, 0, 4]], [2, 2, 2], [0, 0, 3], PLACED, BIN)
    assert not v.ok and v.code == "not_at_target"


def test_path_out_of_bin_in_xy_rejected():
    v = validate_path([[-1, 0, 12], [0, 0, 12], [0, 0, 3]], [2, 2, 2], [0, 0, 3], PLACED, BIN)
    assert not v.ok and v.code == "path_out_of_bounds"


def test_path_above_bin_is_allowed_below_floor_is_not():
    assert validate_path([[0, 0, 20], [0, 0, 3]], [2, 2, 2], [0, 0, 3], PLACED, BIN).ok
    v = validate_path([[0, 0, -1], [0, 0, 3]], [2, 2, 2], [0, 0, 3], PLACED, BIN)
    assert not v.ok and v.code == "path_out_of_bounds"


@pytest.mark.parametrize("bad", [None, [], "x", [[0, 0]], [[0, "a", 0]], [[0, 0, 0, 0]], [[0, 0, True]]])
def test_malformed_paths(bad):
    v = validate_path(bad, [2, 2, 2], [0, 0, 3], PLACED, BIN)
    assert not v.ok and v.code == "malformed"

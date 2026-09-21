"""Fixed sequence files (D21-D23, D28, D31): deterministic, committed, and shaped as documented."""
import hashlib
import json
import os
import random

import pytest

from harness import sequences as sq

CUT_DATASETS = ["data1", "data2", "data3"]
BIN_VOLUME = 1000

# md5 of the CURRICULUM-25 files committed at ccd8fdc; D31 leaves them byte-identical.
CURRICULUM25_MD5 = {
    0: "9a2959311adc1731322355c564d615eb",
    1: "46888f83db23035e43bb0b4b2a32e682",
    2: "5f091f5c549a9a9e42013b4eb66f5166",
    3: "f3e304c729036ad78176b908fe6e090e",
    4: "f34011402d74e15e7e95558252d248d7",
}


@pytest.mark.parametrize("dataset", sq.DATASETS)
def test_same_seed_same_sequence(dataset):
    a = sq.make_sequence(dataset, 0)
    b = sq.make_sequence(dataset, 0)
    assert a == b


@pytest.mark.parametrize("dataset", sq.DATASETS)
def test_different_seeds_differ(dataset):
    assert sq.make_sequence(dataset, 0)["boxes"] != sq.make_sequence(dataset, 1)["boxes"]


@pytest.mark.parametrize("dataset", sq.DATASETS)
def test_n_items_is_the_actual_count(dataset):
    seq = sq.make_sequence(dataset, 0)
    assert seq["n_items"] == len(seq["boxes"])
    if dataset == "curriculum25":
        assert seq["n_items"] == 25


def test_curriculum25_files_unchanged_by_d31():
    for s in sq.SEEDS:
        digest = hashlib.md5(open(sq.sequence_path("curriculum25", s), "rb").read()).hexdigest()
        assert digest == CURRICULUM25_MD5[s]


def test_curriculum_sizes_follow_main_rules():
    for s in sq.SEEDS:
        for b in sq.make_sequence("curriculum25", s)["boxes"]:
            assert all(v in (2, 3, 4, 5) for v in b)
            assert b.count(5) <= 1 or b == [5, 5, 5]


# ---------------- cutting-stock datasets (D31) ----------------

def _cells(pos, size):
    return {(x, y, z)
            for x in range(pos[0], pos[0] + size[0])
            for y in range(pos[1], pos[1] + size[1])
            for z in range(pos[2], pos[2] + size[2])}


@pytest.mark.parametrize("dataset", CUT_DATASETS)
@pytest.mark.parametrize("seed", sq.SEEDS)
def test_cut_tiles_the_bin(dataset, seed):
    seq = sq.make_sequence(dataset, seed)
    boxes, positions = seq["boxes"], seq["positions"]
    assert len(boxes) == len(positions) == seq["n_items"]
    assert sum(a * b * c for a, b, c in boxes) == BIN_VOLUME
    occupied = set()
    for pos, size in zip(positions, boxes):
        assert all(0 <= p and p + s <= 10 for p, s in zip(pos, size)), "outside the bin"
        cells = _cells(pos, size)
        assert not (occupied & cells), "overlap"
        occupied |= cells
    assert len(occupied) == BIN_VOLUME


def test_data1_sides_in_2_to_5():
    for s in sq.SEEDS:
        seq = sq.make_sequence("data1", s)
        assert all(2 <= v <= 5 for b in seq["boxes"] for v in b)
        assert seq["cut"] == {"a_min": 2, "a_max": 5, "restarts": 0}


def test_templates_are_pusnet_shaped():
    t = sq.pusnet_templates()
    assert len(t) == 64 and len(set(t)) == 64
    assert all(1 <= v <= 5 for s in t for v in s)
    assert all(list(s).count(1) <= 1 for s in t)
    vols = sorted(sq._vol(s) for s in sq._template_sizes_1to5())
    lo, hi = vols[len(vols) // 3], vols[2 * len(vols) // 3]
    small = sum(sq._vol(s) < lo for s in t)
    large = sum(sq._vol(s) >= hi for s in t)
    assert 15 <= small <= 23 and 15 <= large <= 23      # ~30 % / 40 % / 30 %


@pytest.mark.parametrize("dataset", ["data2", "data3"])
def test_data23_items_are_templates(dataset):
    shapes = {tuple(sorted(t)) for t in sq.pusnet_templates()}
    for s in sq.SEEDS:
        seq = sq.make_sequence(dataset, s)
        assert all(tuple(sorted(b)) in shapes for b in seq["boxes"])
        assert seq["cut"]["templates"] == [list(t) for t in sq.pusnet_templates()]
        assert seq["cut"]["a_min"] == 1 and seq["cut"]["a_max"] == 5


def test_data2_and_data3_share_the_cut():
    for s in sq.SEEDS:
        a, b = sq.make_sequence("data2", s), sq.make_sequence("data3", s)
        assert sorted(zip(map(tuple, a["positions"]), map(tuple, a["boxes"]))) == \
               sorted(zip(map(tuple, b["positions"]), map(tuple, b["boxes"])))
        assert a["cut"] == b["cut"]
        assert a["boxes"] != b["boxes"]        # different order


def test_data2_is_cut1_order():
    for s in sq.SEEDS:
        z = [p[2] for p in sq.make_sequence("data2", s)["positions"]]
        assert z == sorted(z)


def test_data3_is_cut2_order():
    """Every item rests on the bin floor or on items that came earlier in the sequence."""
    for s in sq.SEEDS:
        seq = sq.make_sequence("data3", s)
        hmap = [[0] * 10 for _ in range(10)]
        for pos, size in zip(seq["positions"], seq["boxes"]):
            x, y, z = pos
            l, w, h = size
            assert all(hmap[i][j] == z for i in range(x, x + l) for j in range(y, y + w))
            for i in range(x, x + l):
                for j in range(y, y + w):
                    hmap[i][j] = z + h


def test_cut_restarts_on_dead_piece():
    """A validity rule nothing can satisfy makes cut_bin return None; the wrapper counts restarts."""
    assert sq.cut_bin(random.Random(0), [10, 10, 10], lambda s: False, 1, 5) is None
    pieces, restarts = sq.cut_bin_with_restarts(
        random.Random(0), [10, 10, 10], lambda s: all(2 <= v <= 5 for v in s), 2, 5)
    assert restarts == 0 and sum(sq._vol(s) for _, s in pieces) == BIN_VOLUME


def test_committed_files_match_generators():
    """The files under data/sequences/ are what the generators produce at this commit."""
    assert sq.check_all() == []


def test_committed_file_header():
    seq = json.load(open(sq.sequence_path("data1", 0)))
    assert seq["dataset"] == "data1" and seq["seed"] == 0 and seq["bin_dims"] == [10, 10, 10]
    assert seq["generator"] == sq.GENERATOR_NOTES["data1"]
    assert seq["n_items"] == len(seq["boxes"]) == len(seq["positions"])

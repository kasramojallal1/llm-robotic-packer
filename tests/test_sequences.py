"""Fixed sequence files (D21-D23, D28): deterministic, committed, and shaped as documented."""
import json
import os

import pytest

from harness import sequences as sq


@pytest.mark.parametrize("dataset", sq.DATASETS)
def test_same_seed_same_sequence(dataset):
    a = sq.make_sequence(dataset, 0)
    b = sq.make_sequence(dataset, 0)
    assert a == b


@pytest.mark.parametrize("dataset", sq.DATASETS)
def test_different_seeds_differ(dataset):
    assert sq.make_sequence(dataset, 0)["boxes"] != sq.make_sequence(dataset, 1)["boxes"]


@pytest.mark.parametrize("dataset", sq.DATASETS)
def test_item_counts(dataset):
    assert len(sq.make_sequence(dataset, 0)["boxes"]) == sq.N_ITEMS[dataset]


def test_data1_tiles_the_bin_exactly():
    for s in sq.SEEDS:
        boxes = sq.make_sequence("data1", s)["boxes"]
        assert sum(a * b * c for a, b, c in boxes) == 1000
        assert all(2 <= v <= 5 for b in boxes for v in b)


def test_curriculum_sizes_follow_main_rules():
    for s in sq.SEEDS:
        for b in sq.make_sequence("curriculum25", s)["boxes"]:
            assert all(v in (2, 3, 4, 5) for v in b)
            assert b.count(5) <= 1 or b == [5, 5, 5]


def test_data3_sorted_by_volume_desc():
    boxes = sq.make_sequence("data3", 0)["boxes"]
    vols = [a * b * c for a, b, c in boxes]
    assert vols == sorted(vols, reverse=True)


def test_committed_files_match_generators():
    """The files under data/sequences/ are what the generators produce at this commit."""
    assert sq.check_all() == []


def test_committed_file_header():
    seq = json.load(open(sq.sequence_path("data1", 0)))
    assert seq["dataset"] == "data1" and seq["seed"] == 0 and seq["bin_dims"] == [10, 10, 10]
    assert seq["generator"] == "gen_data1_exact"

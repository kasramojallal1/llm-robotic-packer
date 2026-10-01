"""Unseen-suite sequences (T10.4, R1.10, D75): new names only, the 20 paper files untouched."""
import hashlib
import os

import pytest

import evaluate
from harness import sequences as sq

# md5 of every paper sequence file at packer main 114dc5c; adding the unseen suite must not change any byte.
PAPER_MD5 = {
    ("curriculum25", 0): "9a2959311adc1731322355c564d615eb",
    ("curriculum25", 1): "46888f83db23035e43bb0b4b2a32e682",
    ("curriculum25", 2): "5f091f5c549a9a9e42013b4eb66f5166",
    ("curriculum25", 3): "f3e304c729036ad78176b908fe6e090e",
    ("curriculum25", 4): "f34011402d74e15e7e95558252d248d7",
    ("data1", 0): "8541055df11df79e0c522a4e8a47e877",
    ("data1", 1): "e9db546d8f27e959febccb834a95b5fc",
    ("data1", 2): "d978d1ef0ee73515f166ecdbd888542f",
    ("data1", 3): "ec36131ea561a819ccdc97a1901b99fb",
    ("data1", 4): "947ed225190326ab0c73d1817cc4ab04",
    ("data2", 0): "d0533a090a4779dfacbfe20f57def7b4",
    ("data2", 1): "00a12a2d717b7738bc1657ed03abe73b",
    ("data2", 2): "cdf651d2f72cfad1807da9a5625c975b",
    ("data2", 3): "c5eafe4ec94d4a17ae1ab66799b8aff7",
    ("data2", 4): "277513f60b500837b45ac08afad3f469",
    ("data3", 0): "118920452ac568e49cba909adc78c59d",
    ("data3", 1): "17a6e3b7c350f94b8012bbbcb43967df",
    ("data3", 2): "1681637bd30d83d453edd355ebfcbbf3",
    ("data3", 3): "bf4619998730e8b651ae9ee6f4f3e165",
    ("data3", 4): "9b924d4bc90f03360069764b533ae6d5",
}

EXPECTED_UNSEEN = {
    "data1-b15": ([15, 15, 15], 2, 5),
    "data1-b12x8x10": ([12, 8, 10], 2, 5),
    "data1-s1to6": ([10, 10, 10], 1, 6),
}


def test_paper_datasets_unchanged():
    assert sq.DATASETS == ["curriculum25", "data1", "data2", "data3"]
    assert not set(sq.UNSEEN_DATASETS) & set(sq.DATASETS)


@pytest.mark.parametrize("dataset,seed", sorted(PAPER_MD5))
def test_paper_files_byte_identical(dataset, seed):
    path = sq.sequence_path(dataset, seed)
    assert hashlib.md5(open(path, "rb").read()).hexdigest() == PAPER_MD5[(dataset, seed)]


def test_unseen_specs():
    assert sq.UNSEEN_SPECS == EXPECTED_UNSEEN


def test_variant_reproduces_data1():
    """The variant generator is data1's own: same bin and range -> identical boxes and cut."""
    for s in sq.SEEDS:
        a = sq.gen_data1_variant([10, 10, 10], s, 2, 5)
        b = sq.gen_data1([10, 10, 10], s)
        assert a == b


@pytest.mark.parametrize("dataset", sq.UNSEEN_DATASETS)
@pytest.mark.parametrize("seed", sq.SEEDS)
def test_unseen_file_matches_generator_and_tiles(dataset, seed):
    seq = sq.load_sequence(dataset, seed)
    assert sq.dumps_sequence(sq.make_sequence(dataset, seed)) == open(sq.sequence_path(dataset, seed)).read()
    bd, a_min, a_max = EXPECTED_UNSEEN[dataset]
    assert seq["bin_dims"] == bd and seq["dataset"] == dataset and seq["seed"] == seed
    assert seq["n_items"] == len(seq["boxes"])
    assert all(a_min <= v <= a_max for b in seq["boxes"] for v in b)
    assert sum(b[0] * b[1] * b[2] for b in seq["boxes"]) == bd[0] * bd[1] * bd[2]
    occ = set()
    for (x, y, z), (l, w, h) in zip(seq["positions"], seq["boxes"]):
        cells = {(i, j, k) for i in range(x, x + l) for j in range(y, y + w) for k in range(z, z + h)}
        assert not occ & cells
        assert all(0 <= c[a] < bd[a] for c in cells for a in range(3))
        occ |= cells


def test_evaluate_all_still_means_the_paper_datasets():
    args = evaluate.parse_args(["--method", "greedy", "--all"])
    assert args.jobs == [(d, s) for d in sq.DATASETS for s in sq.SEEDS]


def test_evaluate_suite_unseen():
    args = evaluate.parse_args(["--method", "greedy", "--suite", "unseen"])
    assert args.jobs == [(d, s) for d in sq.UNSEEN_DATASETS for s in sq.SEEDS]


def test_evaluate_single_unseen_dataset():
    args = evaluate.parse_args(["--method", "greedy", "--dataset", "data1-b15", "--seed", "3"])
    assert args.jobs == [("data1-b15", 3)]


# ------------------------ --template-path pick-only diagnostic (T10.5, D78) ------------------------

def test_template_path_wrapper_keeps_pick_replaces_path():
    from harness.policies import RandomPolicy, TemplatePathPolicy, template_path
    from harness.state import build_state

    class BadPath(RandomPolicy):
        def path(self, state, target, feedback):
            raise AssertionError("inner path must not be called")

    inner, ref = BadPath(seed=3), RandomPolicy(seed=3)
    wrapped = TemplatePathPolicy(inner)
    state = build_state([], [2, 3, 4], [15, 15, 15])
    assert wrapped.pick(state, []).data == ref.pick(state, []).data
    assert wrapped.path(state, [1, 2, 13], []).data == {"path": template_path(state, [1, 2, 13])}
    assert template_path(state, [1, 2, 13])[0] == [1, 2, 17]       # starts above the 15-high bin
    assert wrapped.name == "random" and wrapped.describe()["path_source"] == "template"


def test_template_path_file_name_and_variant(tmp_path):
    import aggregate
    from harness.runner import run_file_name
    assert run_file_name("packi-e", "data1-b15", 2, False, True, template_path=True) == "packi-e/data1-b15/seed2.tpath.json"
    assert run_file_name("packi-e", "data1-b15", 2, False, True) == "packi-e/data1-b15/seed2.json"
    assert aggregate.group_key({"method": "m", "dataset": "d", "flags": {"feedback": True, "template_path": True}})[2] == "+tpath"
    assert aggregate.group_key({"method": "m", "dataset": "d", "flags": {"feedback": True}})[2] == ""


def test_template_path_end_to_end(tmp_path):
    out = evaluate.run_one(evaluate.parse_args(["--method", "greedy", "--dataset", "data1-b12x8x10", "--seed", "0",
                                                "--template-path", "--quiet", "--out", str(tmp_path)]),
                           "data1-b12x8x10", 0)
    import json
    r = json.load(open(out))
    assert out.endswith("seed0.tpath.json") and r["flags"]["template_path"] is True
    assert r["policy"]["path_source"] == "template"

"""Per-example migration diff: fixes, negative flips and copied mistakes."""
from basediff.data_gen import FIELDS
from basediff.eval.diff import compare


def row(i, correct, pred=None, slices=None):
    r = {f: correct for f in FIELDS}
    r.update(id=i, all_correct=correct, valid_json=True, pred=pred or {}, slices=slices or {"hard": False})
    return r


def test_outcome_counts():
    wrong_a = {"amount": "1.00"}
    wrong_b = {"amount": "2.00"}
    old = [row(0, True), row(1, True), row(2, False, wrong_a), row(3, False, wrong_a), row(4, False, wrong_a)]
    new = [row(0, True), row(1, False, wrong_b), row(2, True), row(3, False, wrong_a), row(4, False, wrong_b)]
    d = compare(old, new)
    assert d["counts"] == {"both_right": 1, "flip": 1, "fix": 1, "both_wrong_same": 1, "both_wrong_diff": 1}
    assert d["flip_rate"] == 0.2 and d["fix_rate"] == 0.2
    assert d["inherited"] == 1 / 3  # one of the old model's three mistakes is repeated exactly


def test_slices_count_flips_per_slice():
    old = [row(0, True, slices={"hard": True}), row(1, True, slices={"hard": False})]
    new = [row(0, False, slices={"hard": True}), row(1, True, slices={"hard": False})]
    d = compare(old, new)
    assert d["slices"]["hard"]["flips"] == 1 and d["slices"]["hard"]["n"] == 1

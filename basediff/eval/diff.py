"""Per-example migration diff between the old specialist and a new model (the core output).

For each test item: fix (old wrong, new right), negative flip (old right, new wrong),
both right, both wrong with the same answer, both wrong with different answers.
Flip rates come with Wilson CIs, overall and per slice, next to the seed-noise floor
(old model vs the same recipe retrained with another seed), plus inherited errors and
an exact McNemar test of fixes vs flips.
"""
import json
from collections import Counter

from basediff.common import _norm
from basediff.data_gen import FIELDS
from basediff.eval.stats import mcnemar_exact, paired_bootstrap, wilson
from basediff.eval.task import graded_rows


def _same_answer(p, q) -> bool:
    if p is None or q is None:
        return p is None and q is None
    return all(_norm(f, p.get(f)) == _norm(f, q.get(f)) for f in FIELDS)


def compare(old: list[dict], new: list[dict]) -> dict:
    """old/new: graded rows (basediff.eval.task.graded_rows) over the same items."""
    assert [r["id"] for r in old] == [r["id"] for r in new]
    n = len(old)
    kinds = []
    for a, b in zip(old, new):
        if a["all_correct"] and b["all_correct"]:
            kinds.append("both_right")
        elif a["all_correct"]:
            kinds.append("flip")
        elif b["all_correct"]:
            kinds.append("fix")
        else:
            kinds.append("both_wrong_same" if _same_answer(a["pred"], b["pred"]) else "both_wrong_diff")
    c = Counter(kinds)
    old_right = sum(r["all_correct"] for r in old)
    old_wrong = n - old_right
    res = {
        "n": n, "counts": dict(c),
        "old_acc": old_right / n, "new_acc": sum(r["all_correct"] for r in new) / n,
        "acc_diff": paired_bootstrap([r["all_correct"] for r in old], [r["all_correct"] for r in new]),
        # flip rate per item (share of all items that flipped), the plan's headline rate
        "flip_rate": c["flip"] / n, "flip_ci": wilson(c["flip"], n),
        "fix_rate": c["fix"] / n,
        "mcnemar_p": mcnemar_exact(c["flip"], c["fix"]),
        # of the old model's errors, how many the new model reproduces exactly
        "inherited": c["both_wrong_same"] / old_wrong if old_wrong else 0.0,
        "inherited_ci": wilson(c["both_wrong_same"], old_wrong) if old_wrong else (0.0, 0.0),
    }
    slices = {}
    for name in old[0]["slices"]:
        idx = [i for i, r in enumerate(old) if r["slices"][name]]
        if idx:
            k = sum(kinds[i] == "flip" for i in idx)
            slices[name] = {"n": len(idx), "flips": k, "flip_rate": k / len(idx), "flip_ci": wilson(k, len(idx)),
                            "fixes": sum(kinds[i] == "fix" for i in idx),
                            "old_acc": sum(old[i]["all_correct"] for i in idx) / len(idx),
                            "new_acc": sum(new[i]["all_correct"] for i in idx) / len(idx)}
    res["slices"] = slices
    # which fields break in negative flips
    res["flip_fields"] = dict(Counter(f for a, b, k in zip(old, new, kinds) if k == "flip"
                                      for f in FIELDS if not b[f]))
    res["flip_ids"] = [a["id"] for a, k in zip(old, kinds) if k == "flip"][:30]
    return res


def diff_files(old_preds, new_preds, gold) -> dict:
    return compare(graded_rows(old_preds, gold), graded_rows(new_preds, gold))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = diff_files(a.old, a.new, a.gold)
    if a.out:
        open(a.out, "w").write(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k not in ("slices", "flip_ids")}, indent=1))

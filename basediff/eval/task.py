"""Task accuracy for one predictions file against its gold split, overall and per slice.

Usage: python -m portcheck.eval.task --preds runs/A/test_id.preds.jsonl --gold data/test_id.jsonl
"""
import argparse
import json
from pathlib import Path

from portcheck.common import grade
from portcheck.data_gen import FIELDS, load


def graded_rows(preds_path, gold_path) -> list[dict]:
    """One dict per example: id, slices and per-field correctness."""
    gold = {r["id"]: r for r in load(gold_path)}
    out = []
    for p in load(preds_path):
        g = gold[p["id"]]
        out.append({"id": p["id"], "slices": g["slices"], "pred": p["pred"], **grade(p["pred"], g["output"])})
    assert len(out) == len(gold), f"{preds_path}: {len(out)} preds for {len(gold)} gold rows"
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    s = {"n": n, "all_correct": sum(r["all_correct"] for r in rows) / n,
         "valid_json": sum(r["valid_json"] for r in rows) / n}
    s.update({f: sum(r[f] for r in rows) / n for f in FIELDS})
    slices = {}
    for name in rows[0]["slices"]:
        for val in (True, False):
            sub = [r for r in rows if r["slices"][name] == val]
            if sub:
                slices[f"{name}={val}"] = {"n": len(sub), "all_correct": sum(r["all_correct"] for r in sub) / len(sub)}
    s["slices"] = slices
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = summarize(graded_rows(a.preds, a.gold))
    out = Path(a.out) if a.out else Path(a.preds).with_suffix(".score.json")
    out.write_text(json.dumps(s, indent=2))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "slices"}))


if __name__ == "__main__":
    main()

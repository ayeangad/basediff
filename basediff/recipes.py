"""Build the training file for each rebuild recipe from the old specialist's logs.

Inputs: data/log_inputs.jsonl (gold kept hidden except where a recipe "buys" labels) and
runs/A/logs.preds.jsonl + runs/A/logs.preds.topk.pt (A's greedy outputs and token log-probs).
Each output row is {"id", "input", "output" (gold), "target": "log" | "gold"}; train.py
trains "log" rows on A's exact tokens and "gold" rows on the gold JSON.

Recipes (PORT_PLAN v2 §5.3):
  R1-log  gold labels on all log inputs (clean reference, 1,500 labels)
  R2      A's outputs as-is (also used by R2-KD with kd_lambda > 0)
  R3      R2 minus rows that fail the validators
  R4      R3 + gold labels on the LABEL_BUDGET log rows where A was least confident
  R4-rand R3 + gold labels on LABEL_BUDGET random log rows
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch

from portcheck.common import grade, load_tokenizer
from portcheck.data_gen import load
from portcheck.validators import check

LABEL_BUDGET = 200
SELECT_FIELDS = ("purpose_code", "risk_flag")


def field_confidence(tok, d: dict) -> float:
    """Mean log-prob A gave to the tokens of the purpose_code and risk_flag values.
    Unparseable / missing fields count as least confident (-inf)."""
    ids, lp = d["ids"].tolist(), d["chosen_lp"].float().tolist()
    pieces = [tok.decode([i]) for i in ids]
    text = "".join(pieces)
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    vals = []
    for f in SELECT_FIELDS:
        m = re.search(rf'"{f}"\s*:\s*("?[^",}}]+"?)', text)
        if not m:
            return float("-inf")
        a, b = m.span(1)
        vals += [lp[k] for k, s in enumerate(starts) if s < b and s + len(pieces[k]) > a]
    return sum(vals) / len(vals) if vals else float("-inf")


def build(log_inputs: str, preds: str, topk: str, base: str, out_dir: str, seed: int = 0) -> dict:
    rows = {r["id"]: r for r in load(log_inputs)}
    preds = {p["id"]: p for p in load(preds)}
    dists = torch.load(topk)
    tok = load_tokenizer(base)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def row(i, target):
        r = rows[i]
        return {"id": i, "input": r["input"], "output": r["output"], "target": target}

    ids = sorted(rows)
    fails = {i: check(rows[i]["input"], preds[i]["pred"]) for i in ids}
    kept = [i for i in ids if not fails[i]]
    conf = {i: field_confidence(tok, dists[i]) for i in ids}
    by_conf = sorted(ids, key=lambda i: (conf[i], i))[:LABEL_BUDGET]
    rand = random.Random(seed).sample(ids, LABEL_BUDGET)

    def with_labels(chosen):
        chosen = set(chosen)
        return [row(i, "gold") for i in ids if i in chosen] + [row(i, "log") for i in kept if i not in chosen]

    files = {
        "R1-log": [row(i, "gold") for i in ids],
        "R2": [row(i, "log") for i in ids],
        "R3": [row(i, "log") for i in kept],
        "R4": with_labels(by_conf),
        "R4-rand": with_labels(rand),
    }
    for name, rs in files.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # how good are the validators and the selection rule at finding A's real errors?
    wrong = {i for i in ids if not grade(preds[i]["pred"], rows[i]["output"])["all_correct"]}
    rejected = {i for i in ids if fails[i]}
    stats = {
        "log_rows": len(ids), "A_wrong": len(wrong), "validator_rejected": len(rejected),
        "validator_precision": len(rejected & wrong) / max(1, len(rejected)),
        "validator_recall": len(rejected & wrong) / max(1, len(wrong)),
        "R4_selected_wrong": len(set(by_conf) & wrong), "R4rand_selected_wrong": len(set(rand) & wrong),
        "sizes": {k: len(v) for k, v in files.items()},
        "gold_labels": {k: sum(r["target"] == "gold" for r in v) for k, v in files.items()},
    }
    (out / "recipes_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-inputs", default="data/log_inputs.jsonl")
    ap.add_argument("--preds", default="runs/A/logs.preds.jsonl")
    ap.add_argument("--topk", default="runs/A/logs.preds.topk.pt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="data/recipes")
    a = ap.parse_args()
    print(json.dumps(build(a.log_inputs, a.preds, a.topk, a.base, a.out), indent=2))

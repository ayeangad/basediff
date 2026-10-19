"""Off-task retention: ARC-Easy (first 500 items, log-likelihood only) via lm-evaluation-harness.

Saves per-item correctness so models can be compared PAIRED on the same items
(PORT_PLAN v2 §5.4: descriptive, not a pass/fail hypothesis).
"""
import argparse
import json
from pathlib import Path


def run(base: str, adapter: str | None, out: str, limit: int = 500, batch_size: int = 4) -> dict:
    import lm_eval
    args = f"pretrained={base},dtype=float32" + (f",peft={adapter}" if adapter else "")
    res = lm_eval.simple_evaluate(model="hf", model_args=args, tasks=["arc_easy"], limit=limit,
                                  batch_size=batch_size, device="cuda", log_samples=True)
    items = {int(s["doc_id"]): bool(s["acc"]) for s in res["samples"]["arc_easy"]}
    summary = {"base": base, "adapter": adapter, "n": len(items), "acc": sum(items.values()) / len(items),
               "items": items}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    s = run(a.base, a.adapter, a.out, a.limit, a.batch_size)
    print(json.dumps({k: v for k, v in s.items() if k != "items"}))

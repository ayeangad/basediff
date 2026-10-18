"""Batched greedy generation for a base model (+ optional adapter) over a jsonl split.

Used both to create the old specialist's production logs and to evaluate every
model. Writes one row per example: {"id", "raw", "pred"} where pred is the parsed
JSON (or null), plus timing in a sidecar json.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from portcheck.common import load_model, load_tokenizer, parse_output, render_prompt
from portcheck.data_gen import load


def few_shot_examples(path: str, k: int) -> list:
    rows = load(path)[:k]
    return [(r["input"], json.dumps(r["output"], ensure_ascii=False)) for r in rows]


@torch.inference_mode()
def run(base, adapter, rows, batch_size=16, max_new=96, system=None, shots=None, device="cuda"):
    tok = load_tokenizer(base)
    tok.padding_side = "left"
    model = load_model(base, adapter, device=device)
    kw = {"system": system} if system else {}
    prompts = [render_prompt(tok, r["input"], shots=shots, **kw) for r in rows]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(rows)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--system", help="override system prompt (prompt-only baselines)")
    ap.add_argument("--shots-from", help="jsonl to draw few-shot examples from")
    ap.add_argument("--k-shots", type=int, default=0)
    a = ap.parse_args()

    rows = load(a.data)[:a.limit] if a.limit else load(a.data)
    shots = few_shot_examples(a.shots_from, a.k_shots) if a.shots_from and a.k_shots else None
    t0 = time.time()
    outs = run(a.base, a.adapter, rows, a.batch_size, system=a.system, shots=shots)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "raw": o, "pred": parse_output(o)}, ensure_ascii=False) + "\n")
    meta = {"base": a.base, "adapter": a.adapter, "data": a.data, "n": len(rows),
            "minutes": round((time.time() - t0) / 60, 2), "k_shots": a.k_shots,
            "system_override": bool(a.system)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

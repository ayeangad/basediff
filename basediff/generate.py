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
def run(base, adapter, rows, batch_size=16, max_new=96, system=None, shots=None, device="cuda",
        topk=0):
    """Greedy outputs; with topk > 0 also returns, per row, the generated token ids and the
    top-k log-probs (raw logits, before any processor) at each generated position."""
    tok = load_tokenizer(base)
    tok.padding_side = "left"
    model = load_model(base, adapter, device=device)
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    kw = {"system": system} if system else {}
    prompts = [render_prompt(tok, r["input"], shots=shots, **kw) for r in rows]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs, dists = [None] * len(rows), [None] * len(rows)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        # explicit greedy settings: the shipped generation configs sample, and Qwen2.5's
        # adds repetition_penalty=1.1, which would distort repeated digits
        res = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                             top_p=None, top_k=None, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos,
                             return_dict_in_generate=True, output_logits=topk > 0)
        new = res.sequences[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        if topk:  # per step, to avoid a batch x steps x 152k tensor
            tl, ti, cl = [], [], []
            for step, lg in enumerate(res.logits):
                lp = lg.float().log_softmax(-1)
                v, k = lp.topk(topk, -1)
                tl.append(v), ti.append(k)
                cl.append(lp.gather(1, new[:, step:step + 1]).squeeze(1))
            top_lp, top_id, chosen = torch.stack(tl, 1), torch.stack(ti, 1), torch.stack(cl, 1)
        for j, (i, t) in enumerate(zip(idx, texts)):
            outs[i] = t
            if topk:
                ids = new[j].tolist()
                n = next((k + 1 for k, x in enumerate(ids) if x in eos), len(ids))  # keep the eos step
                dists[i] = {"ids": torch.tensor(ids[:n]), "top_id": top_id[j, :n].cpu(),
                            "top_lp": top_lp[j, :n].cpu().half(),
                            "chosen_lp": chosen[j, :n].cpu()}
    return (outs, dists) if topk else outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--system", help="override system prompt (prompt-only baselines)")
    ap.add_argument("--r0", help="named R0 system prompt from portcheck.prompts.R0_VARIANTS")
    ap.add_argument("--shots-from", help="jsonl to draw few-shot examples from")
    ap.add_argument("--k-shots", type=int, default=0)
    ap.add_argument("--save-topk", type=int, default=0, help="also save top-k log-probs per step")
    ap.add_argument("--max-new", type=int, default=96, help="untuned models write long, pretty-printed JSON")
    a = ap.parse_args()

    if a.r0:
        from portcheck.prompts import R0_VARIANTS
        a.system = R0_VARIANTS[a.r0]
    rows = load(a.data)[:a.limit] if a.limit else load(a.data)
    shots = few_shot_examples(a.shots_from, a.k_shots) if a.shots_from and a.k_shots else None
    t0 = time.time()
    outs = run(a.base, a.adapter, rows, a.batch_size, max_new=a.max_new, system=a.system, shots=shots,
               topk=a.save_topk)
    if a.save_topk:  # old-model logs for KD (R2-KD) and confidence-based selection (R4)
        outs, dists = outs
        torch.save({r["id"]: d for r, d in zip(rows, dists)}, Path(a.out).with_suffix(".topk.pt"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "raw": o, "pred": parse_output(o)}, ensure_ascii=False) + "\n")
    meta = {"base": a.base, "adapter": a.adapter, "data": a.data, "n": len(rows),
            "minutes": round((time.time() - t0) / 60, 2), "k_shots": a.k_shots,
            "system_override": bool(a.system), "r0": a.r0, "max_new": a.max_new}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

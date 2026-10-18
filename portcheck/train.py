"""LoRA SFT with loss on the assistant answer only.

A small hand-written loop instead of TRL so precision, masking and memory are
explicit: fp32 by default (fp16 autocast + GradScaler optional), fp32 LoRA weights, and
the LM head is applied only at answer positions (the 150k-vocab logits for the
prompt are never materialised).
"""
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model

from portcheck.common import load_tokenizer, render_prompt, render_target
from portcheck.data_gen import load

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, rows, max_len):
    out = []
    for r in rows:
        p = tok(render_prompt(tok, r["input"]), add_special_tokens=False)["input_ids"]
        t = tok(render_target(tok, r["output"]), add_special_tokens=False)["input_ids"]
        ids = (p + t)[:max_len]
        labels = ([-100] * len(p) + t)[:max_len]
        out.append((ids, labels))
    return out


def batches(examples, mb, rng):
    """Length-bucketed micro-batches in random order (less padding)."""
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    chunks = [idx[i:i + mb] for i in range(0, len(idx), mb)]
    rng.shuffle(chunks)
    return chunks


def collate(examples, chunk, pad_id, device):
    L = max(len(examples[i][0]) for i in chunk)
    ids = torch.full((len(chunk), L), pad_id)
    lab = torch.full((len(chunk), L), -100)
    att = torch.zeros((len(chunk), L), dtype=torch.long)
    for j, i in enumerate(chunk):
        a, b = examples[i]
        ids[j, :len(a)] = torch.tensor(a)
        lab[j, :len(b)] = torch.tensor(b)
        att[j, :len(a)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def answer_loss(model, ids, labels, att):
    """Cross-entropy over answer tokens only, LM head applied only where needed."""
    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    logits = inner.lm_head(h).float()
    return torch.nn.functional.cross_entropy(logits, tgt[mask])


def main(cfg: dict):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    dev = "cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu"
    # fp32 by default: on the GTX 1650 Ti (no tensor cores) fp16 autocast measured ~4x slower
    use_fp16 = dev == "cuda" and cfg.get("precision", "fp32") == "fp16"
    dtype = torch.float16 if use_fp16 else torch.float32

    tok = load_tokenizer(cfg["base"])
    rows = load(cfg["train_file"])
    if cfg.get("limit"):
        rows = rows[:cfg["limit"]]
    if cfg.get("output_field"):  # e.g. train on the old model's logged outputs
        rows = [{**r, "output": r[cfg["output_field"]]} for r in rows]
    ex = encode(tok, rows, cfg.get("max_len", 384))

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=dtype).to(dev)
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(r=cfg.get("r", 16), lora_alpha=cfg.get("alpha", 32),
                      lora_dropout=cfg.get("dropout", 0.05), target_modules=TARGETS,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4), weight_decay=0.0)

    mb, eff = cfg.get("micro_batch", 2), cfg.get("effective_batch", 16)
    accum = max(1, eff // mb)
    epochs = cfg.get("epochs", 2)
    steps_per_epoch = math.ceil(len(ex) / mb / accum)
    total = cfg.get("max_steps") or steps_per_epoch * epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    log = open(out / "train_log.jsonl", "w")
    step, t0, ntok, done = 0, time.time(), 0, False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for ep in range(epochs if not cfg.get("max_steps") else 10 ** 6):
        chunks = batches(ex, mb, rng)
        for k in range(0, len(chunks), accum):
            group = chunks[k:k + accum]
            opt.zero_grad(set_to_none=True)
            tot_loss = 0.0
            for ch in group:
                ids, lab, att = collate(ex, ch, tok.pad_token_id, dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=use_fp16):
                    loss = answer_loss(model, ids, lab, att) / len(group)
                scaler.scale(loss).backward()
                tot_loss += loss.item()
                ntok += int(att.sum())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 10 == 0 or step == total:
                rec = {"step": step, "epoch": ep, "loss": round(tot_loss, 4),
                       "lr": sched.get_last_lr()[0], "tok_s": round(ntok / (time.time() - t0), 1),
                       "elapsed_min": round((time.time() - t0) / 60, 2)}
                if dev == "cuda":
                    rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step >= total:
                done = True
                break
        if done:
            break

    model.save_pretrained(out / "adapter")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"config": cfg, "steps": step, "train_rows": len(ex), "train_tokens_seen": ntok,
               "minutes": round((time.time() - t0) / 60, 2), "tok_s": round(ntok / (time.time() - t0), 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None,
               "git": commit or None}
    (out / "cost.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    main(cfg)

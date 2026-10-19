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

from basediff.common import load_tokenizer, render_prompt, render_target
from basediff.data_gen import load

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, rows, max_len, logs=None):
    """(ids, labels, kd) per row. Rows with target == "log" train on the old model's exact
    generated token ids (shared tokenizer); kd = (top_id, top_lp) for those rows if `logs`
    carries distributions, else None. Other rows train on the gold JSON."""
    out = []
    for r in rows:
        p = tok(render_prompt(tok, r["input"]), add_special_tokens=False)["input_ids"]
        kd = None
        if r.get("target") == "log":
            d = logs[r["id"]]
            t = d["ids"].tolist()
            kd = (d["top_id"], d["top_lp"].float())
        else:
            t = tok(render_target(tok, r["output"]), add_special_tokens=False)["input_ids"]
        assert len(p) + len(t) <= max_len, f"row {r['id']} would be truncated"
        out.append((p + t, [-100] * len(p) + t, kd))
    return out


def batches(examples, mb, rng):
    """Length-bucketed micro-batches in random order (less padding)."""
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    chunks = [idx[i:i + mb] for i in range(0, len(idx), mb)]
    rng.shuffle(chunks)
    return chunks


def collate(examples, chunk, pad_id, device):
    """Padded batch plus KD targets aligned with the answer tokens in row-major order."""
    L = max(len(examples[i][0]) for i in chunk)
    ids = torch.full((len(chunk), L), pad_id)
    lab = torch.full((len(chunk), L), -100)
    att = torch.zeros((len(chunk), L), dtype=torch.long)
    kd_id, kd_lp, kd_on = [], [], []
    for j, i in enumerate(chunk):
        a, b, kd = examples[i]
        ids[j, :len(a)] = torch.tensor(a)
        lab[j, :len(b)] = torch.tensor(b)
        att[j, :len(a)] = 1
        n_ans = sum(x != -100 for x in b)
        if kd is not None:
            assert kd[0].shape[0] == n_ans, "KD positions must match answer tokens"
            kd_id.append(kd[0]), kd_lp.append(kd[1]), kd_on.append(torch.ones(n_ans, dtype=torch.bool))
        else:
            kd_id.append(torch.zeros(n_ans, KD_K, dtype=torch.long))
            kd_lp.append(torch.zeros(n_ans, KD_K))
            kd_on.append(torch.zeros(n_ans, dtype=torch.bool))
    kd = (torch.cat(kd_id).to(device), torch.cat(kd_lp).to(device), torch.cat(kd_on).to(device))
    return ids.to(device), lab.to(device), att.to(device), kd


KD_K = 20


def answer_loss(model, ids, labels, att, kd=None, kd_lambda=0.0):
    """Cross-entropy over answer tokens only, LM head applied only where needed.

    With kd_lambda > 0, adds KL(p_old || p_new) over the old model's top-k token ids at each
    answer position, both distributions renormalised over those k ids (rows without KD
    targets are masked out)."""
    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    logits = inner.lm_head(h).float()
    loss = torch.nn.functional.cross_entropy(logits, tgt[mask])
    if kd_lambda and kd is not None and kd[2].any():
        top_id, top_lp, on = kd
        p_old = top_lp[on].log_softmax(-1)
        q_new = logits[on].gather(1, top_id[on]).log_softmax(-1)
        kl = (p_old.exp() * (p_old - q_new)).sum(-1).mean()
        loss = loss + kd_lambda * kl
    return loss


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
    logs = torch.load(cfg["logs_file"]) if cfg.get("logs_file") else None
    ex = encode(tok, rows, cfg.get("max_len", 384), logs)
    if not cfg.get("kd_lambda"):
        ex = [(a, b, None) for a, b, _ in ex]

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
                ids, lab, att, kd = collate(ex, ch, tok.pad_token_id, dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=use_fp16):
                    loss = answer_loss(model, ids, lab, att, kd, cfg.get("kd_lambda", 0.0)) / len(group)
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

"""Prompt rendering, model loading and output grading shared by every script."""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from portcheck.data_gen import FIELDS, SYSTEM_PROMPT


def load_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def end_of_turn(tok) -> str:
    # Qwen and SmolLM chat templates both close a turn with <|im_end|>
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


def render_prompt(tok, user_text: str, system: str = SYSTEM_PROMPT, shots: list | None = None) -> str:
    """Chat prompt up to the start of the assistant answer (thinking disabled for Qwen3)."""
    msgs = [{"role": "system", "content": system}]
    for ex_in, ex_out in shots or []:
        msgs += [{"role": "user", "content": ex_in}, {"role": "assistant", "content": ex_out}]
    msgs.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def render_target(tok, output: dict) -> str:
    return json.dumps(output, ensure_ascii=False) + end_of_turn(tok)


def load_model(base: str, adapter: str | None = None, dtype=torch.float32, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


# --- grading ---------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_output(text: str) -> dict | None:
    text = _THINK.sub("", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _norm(field: str, v):
    if v is None:
        return None
    if field == "beneficiary_name":
        return " ".join(str(v).casefold().split())
    if field == "amount":
        try:
            return f"{float(str(v).replace(',', '')):.2f}"
        except ValueError:
            return str(v)
    if field == "risk_flag":
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes", "1"}
        return bool(v)
    return str(v).strip().upper() if field in {"currency", "purpose_code"} else str(v).strip()


def grade(pred: dict | None, gold: dict) -> dict:
    """Per-field correctness; a missing/unparseable prediction is wrong on every field."""
    res = {f: (pred is not None and _norm(f, pred.get(f)) == _norm(f, gold[f])) for f in FIELDS}
    res["valid_json"] = pred is not None and all(f in pred for f in FIELDS)
    res["all_correct"] = all(res[f] for f in FIELDS)
    return res

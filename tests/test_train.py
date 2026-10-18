"""Loss masking and the shared-tokenizer assumption behind R2-KD (PORT_PLAN v2 §5.2)."""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from portcheck.common import render_target
from portcheck.train import collate, encode

DATA = Path(__file__).resolve().parents[1] / "data"
OLD, NEW = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def rows():
    with open(DATA / "gold_train.jsonl") as f:
        return [json.loads(next(f)) for _ in range(3)]


@pytest.mark.parametrize("base", [OLD, NEW])
def test_loss_only_on_answer(base, rows):
    tok = AutoTokenizer.from_pretrained(base)
    for (ids, labels, _), r in zip(encode(tok, rows, 384), rows):
        active = [t for t in labels if t != -100]
        assert tok.decode(active) == render_target(tok, r["output"])
        assert ids[-len(active):] == active


def test_shared_tokenizer_for_answers(rows):
    """A's generated ids are reused as Qwen3 targets, so both tokenizers must agree."""
    a, b = AutoTokenizer.from_pretrained(OLD), AutoTokenizer.from_pretrained(NEW)
    for r in rows:
        text = render_target(a, r["output"])
        assert a(text, add_special_tokens=False)["input_ids"] == b(text, add_special_tokens=False)["input_ids"]
    assert a.convert_tokens_to_ids("<|im_end|>") == b.convert_tokens_to_ids("<|im_end|>")


def test_log_rows_use_old_model_ids_and_align_kd(rows):
    tok = AutoTokenizer.from_pretrained(NEW)
    ans = tok(render_target(tok, rows[0]["output"]), add_special_tokens=False)["input_ids"]
    n = len(ans)
    logs = {rows[0]["id"]: {"ids": torch.tensor(ans), "top_id": torch.zeros(n, 20, dtype=torch.long),
                            "top_lp": torch.zeros(n, 20)}}
    ex = encode(tok, [{**rows[0], "target": "log"}, rows[1]], 384, logs)
    assert [t for t in ex[0][1] if t != -100] == ans and ex[0][2] is not None and ex[1][2] is None
    _, lab, _, kd = collate(ex, [0, 1], tok.pad_token_id, "cpu")
    n_ans = int((lab[:, 1:] != -100).sum())
    assert kd[0].shape[0] == n_ans and int(kd[2].sum()) == n

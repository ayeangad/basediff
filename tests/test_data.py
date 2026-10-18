"""Data checks from PORT_PLAN v2 §5.1. Run after `python -m portcheck.data_gen --out data`."""
import json
from pathlib import Path

import pytest

from portcheck.data_gen import HELDOUT, NEW_TEMPLATES, OUTPUT_KEYS, POLICY

DATA = Path(__file__).resolve().parents[1] / "data"
SPLITS = ["gold_train", "log_inputs", "test_id", "test_drift", "val"]
MAX_LEN = 384


@pytest.fixture(scope="module")
def splits():
    return {s: [json.loads(l) for l in open(DATA / f"{s}.jsonl")] for s in SPLITS}


def test_no_input_overlap(splits):
    seen = {}
    for s, rows in splits.items():
        for r in rows:
            assert r["input"] not in seen, f"{s} row {r['id']} duplicates {seen.get(r['input'])}"
            seen[r["input"]] = s


def test_new_templates_and_heldout_wordings_never_in_training(splits):
    new = {f"{st.__name__}:{v}" for st, v in NEW_TEMPLATES}
    for s in ["gold_train", "val"]:
        for r in splits[s]:
            assert r["template"] not in new
            assert not r["slices"]["heldout_wording"]
            assert not any(h in r["input"].lower() for h in HELDOUT.values()), r["id"]


def test_policies(splits):
    assert {r["policy"] for r in splits["gold_train"] + splits["val"]} == {"old"}
    assert {r["policy"] for s in ["log_inputs", "test_id", "test_drift"] for r in splits[s]} == {"new"}
    assert POLICY["new"] < POLICY["old"]
    # the drift band must actually be present where it matters
    assert sum(r["slices"]["policy_affected"] for r in splits["test_id"]) >= 20


def test_gold_schema_and_last4(splits):
    for rows in splits.values():
        for r in rows:
            o = r["output"]
            assert list(o) == OUTPUT_KEYS
            assert isinstance(o["risk_flag"], bool)
            assert o["risk_flag"] == bool(o["risk_reasons"])
            assert len(o["account_last4"]) == 4 and o["account_last4"].isdigit()
            compact = r["input"].replace(" ", "")
            assert o["account_last4"] in compact, r["id"]


@pytest.mark.slow  # ~12 min: tokenises every row with both chat templates
def test_no_row_truncated():
    """TRL-style truncation cuts the end of the sequence, i.e. the JSON answer."""
    from transformers import AutoTokenizer
    from portcheck.common import render_prompt, render_target
    for base in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"]:
        tok = AutoTokenizer.from_pretrained(base)
        for s in SPLITS:
            for r in (json.loads(l) for l in open(DATA / f"{s}.jsonl")):
                n = len(tok(render_prompt(tok, r["input"]) + render_target(tok, r["output"]),
                            add_special_tokens=False)["input_ids"])
                assert n <= MAX_LEN, f"{base} {s} row {r['id']} has {n} tokens"

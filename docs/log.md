# portcheck log

Plan: `~/Downloads/projects/companies/hyde2/PORT_PLAN.md` (v2, LEAN mode).

## DECISIONS

**D0 (2026-10-19): continue from the earlier session's scaffold.** The repo already held a data generator, a hand-written LoRA trainer (loss on answer tokens only, LM head applied only at answer positions) and batched generation from an earlier session that followed plan v1. Nothing was running. I committed it as-is (ab8936a) and upgraded it to v2. The trainer stays hand-written instead of TRL, which keeps precision and masking explicit.

**D1 (G0): train and evaluate in fp32, not fp16.** On this GTX 1650 Ti, fp16 autocast is about 4× slower than fp32. Measured, Qwen2.5-0.5B, 64 rows:

| precision | grad ckpt | micro-batch | tok/s | peak GB |
|---|---|---|---|---|
| fp32 | on | 2 | 486 | 2.45 |
| fp32 | on | 8 | 510 | 3.36 |
| fp16 | on | 8 | 116 | 2.30 |
| fp32 / fp16 | off | 4 | OOM | — |

Qwen3-0.6B, fp32, grad ckpt on: micro-batch 2 → 302 tok/s, 2.90 GB; micro-batch 4 → 345 tok/s, 3.21 GB. Plan §7 says "fp16 + GradScaler"; this measurement overrides it. Since everything runs in fp32, the plan's fp16-vs-fp32 numeric check is unnecessary. Mean row length is ~232 tokens (Qwen3), so one 2-epoch run on 1,500 rows is ~0.7M tokens: about 23 min for Qwen2.5 and 34 min for Qwen3.

**D2 (G0): the GPU is not throttled.** It is on AC power with no throttle reasons. Low clocks during training come from small kernels (CPU-bound), not power limits. The earlier session's 102 tok/s smoke run was fp16.

**D3 (G0): Qwen3.x ≤ 1B check.** `Qwen/Qwen3.5-0.8B` exists (2026-02-28, `qwen3_5_text`, hidden 1024, 24 layers, vocab 248,320, multimodal `Qwen3_5ForConditionalGeneration`). This is the exact next step for Hyde's fine-tuned Qwen3-0.6B, with a tokenizer break. It stays STRETCH: fp32 training of 0.8B params plus 248k-wide logits will not fit in 4 GB, and fp16 is 4× slower here. The main pair stays Qwen2.5-0.5B → Qwen3-0.6B, as the plan says.

**D4 (data v2): policy drift applies to all production traffic.** The plan said "half of log_inputs dated after the change". But tickets carry no policy date the model could use, so a half/half mix would make the gold labels unlearnable noise. Instead, the change happens at launch: `gold_train` and `val` use the old policy (new-beneficiary threshold 50,000); `log_inputs`, `test_id` and `test_drift` use the new one (25,000). The old specialist's logs are therefore stale on exactly the policy-affected rows.

**D5 (data v2): tuned the difficulty mix so the policy slice is large enough.** The first draft had only 1% policy-affected rows, because random currencies made the cross-currency rule flag almost everything. Now: currency = local currency 75% of the time, 30% of amounts in the 25k–50k band, 35% new beneficiaries. Result: 56/600 policy-affected rows in `test_id` (≥ 50 for the slice threshold), risk_flag positive rate 30–40%. Data hash: see `data/MANIFEST.json` (commit 9773eb4).

**D6 (validators): written before seeing any old-specialist errors.** They use only the ticket text, the extracted fields and the written policy, never the generator's phrase lists. They reject 12 of 3,600 gold answers (0.3%): misspelled currency words such as "poudns sterling". I left that in rather than tune it away. By design they cannot catch wrong-but-valid purpose codes or missed new-beneficiary flags, including the policy-drift band.

## PREREG

(to be written after the difficulty probe, before the overnight run)

## FINDINGS

(none yet)

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

**D7 (G1, first attempt): A never learned risk_flag. Added a `risk_reasons` field.** The difficulty probe (1 epoch, 500 rows) scored 22.5% all-fields-correct on val, mostly from `account_last4` (35%), which was fixed by full training. The full A (1,500 rows, 2 epochs, 36 min, 322 tok/s) scored 69.0% on val: every field 97–100% except `risk_flag` at 71.5%.
- A predicted `false` for 56 of 57 risky val tickets, including plain watch-list countries (Panama).
- On 200 of its own training rows it predicted `true` 0 times for 66 risky rows. So this is underfitting, not a hard test: the flag is one token driven by four rules, too little signal for a 0.5B LoRA in 2 epochs.
- Plan §10 G1 allows one adjustment ("raise epochs to 3 or simplify the risk rules"). Three epochs would add ~50% to every overnight run.
- **Fix instead:** the answer now includes `risk_reasons` (the list of rules that fired, in a fixed order) just before `risk_flag`, so each rule gets its own supervised tokens. Real ops systems record reasons as well.
- **Grading is unchanged:** all-fields-correct still uses the original 7 fields. `risk_reasons` is not graded.
- Data regenerated: new hashes in `data/MANIFEST.json`, max sequence 327 tokens.

**D8: generation decodes greedily with an explicit `repetition_penalty=1.0`.** Qwen2.5's shipped generation config sets `repetition_penalty=1.1` and sampling. On the probe it made no difference to the scores, but the setting is now explicit for every model.

**D9: added one more validator check before seeing the new A's errors.** `risk_flag` must equal `bool(risk_reasons)`. Gold answers still fail only on the 12 misspelled-currency rows (0.3%).

**D10 (G1): A after the fix scores 79.5% on val** (n = 200; risk_flag 83.5%, purpose_code 94.5%, hard slice 59%, ambiguous-purpose slice 68%). That is at the lower edge of the 80–90% target, and inside its sampling error (Wilson 95% CI about ±6 points). The formal G1 check is on test_id (n = 600) in the overnight run. No further tuning, so the test set stays untouched.

## PREREG

Written before the overnight run. Committed and tagged `prereg` together with the frozen code: data generator, validators (`portcheck/validators.py`), R4 selection rule (`portcheck/recipes.py`), configs and statistics.

- **Data:** `data/MANIFEST.json` hashes at the `prereg` tag; seed 0.
- **Models:** old A = Qwen2.5-0.5B-Instruct + LoRA on `gold_train` (seed 0); A′ = same with seed 1 (the noise floor); new base B = Qwen3-0.6B. All runs: LoRA r16/α32, lr 2e-4, 2 epochs, effective batch 16, fp32, loss on answer tokens only.
- **Recipes on B:**
  - R0a: zero-shot, prompt variant chosen on val excluding policy-affected rows;
  - R0b: the same prompt + 3 shots;
  - R1-log: gold labels on all 1,500 log inputs;
  - R2: A's log outputs (A's exact tokens);
  - R2-KD: R2 + KL to A's top-20, λ = 0.5;
  - R3: R2 minus validator-rejected rows;
  - R4: R3 + gold on the 200 log rows with the lowest A mean log-prob on the `purpose_code` and `risk_flag` value tokens;
  - R4-rand: R3 + gold on 200 random log rows (seed 0).
- **Metrics:** all-fields-correct (7 fields) on `test_id` (600) and `test_drift` (200).
  - Versus A, per item: fix / negative flip / both right / both wrong same / both wrong different.
  - Negative-flip rate = flips / n, with a Wilson 95% CI.
  - Accuracy difference: paired bootstrap, 1,000 resamples.
  - Exact McNemar test on flips vs fixes; Holm correction across the rebuild models.
  - Inherited errors = share of A's wrong items where the model gives the same wrong answer, compared against R1-log's overlap.
  - Slices: new_template, heldout_wording, ambiguous_purpose, policy_affected, hard.
  - Off-task: ARC-Easy, first 500 items, paired against untuned B; descriptive only.
- **Hypotheses (from PORT_PLAN v2 §11):**
  - H1: the best R0 is ≥ 5 points below A on `test_id`.
  - H2: R3 reaches ≥ 95% of R1-log's accuracy.
  - H3: R2 reproduces more of A's errors than R1-log's overlap; R3 and R4 reduce that excess.
  - H4: R4 beats R4-rand, overall and on `policy_affected`.
  - H5: R2 and R2-KD have lower flip rates than R1-log (two-sided).
  - H6: R2-KD has a lower flip rate than R2.
  - H7: the best log-based rebuild's flip rate lies inside the A→A′ floor CI.
  - H8: R2, R2-KD and R3 stay wrong on `policy_affected`; R4 fixes part of it.
- **SHIP / REVIEW / BLOCK:**
  - SHIP: accuracy not below A (paired CI lower bound ≥ 0), flip-rate CI overlaps the floor's CI, and no slice with n ≥ 50 has a flip-rate CI above 5%.
  - REVIEW: any of these is borderline.
  - BLOCK: otherwise.
- **Reporting:** everything above is reported, including failures. Anything added later is labelled exploratory.

## FINDINGS

**F1 (log generation, 2026-10-19 02:36; `data/recipes/recipes_stats.json`):**
- **The old specialist is wrong on 595 of the 1,500 production-log inputs (39.7%),** against 20.5% wrong on its own validation split. The logs include new templates, held-out wordings and the post-launch policy, so they carry many stale or wrong answers.
- **The validators rejected 277 log rows.** Precision at finding A's real errors is 98.9% (274/277), recall 46.1% (274/595). The low recall is by design: wrong-but-valid purpose codes and missed new-beneficiary flags are invisible to them.
- **The R4 selection rule (A's lowest confidence on `purpose_code` and `risk_flag`) picked 141/200 wrong rows (70.5%).** Random selection picked 69/200 (34.5%). Confidence-targeted labelling finds errors at about twice the random rate.

These are descriptive pipeline facts, not pre-registered hypotheses.

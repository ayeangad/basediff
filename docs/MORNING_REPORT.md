# MORNING REPORT: portcheck (PORT_PLAN v2, LEAN)

Date: 2026-10-19. Everything below comes from files in this repo (`results/summary.json`, `results/table.md`, `data/recipes/recipes_stats.json`). All comparisons are paired on the same items.

## 1. What ran

| Item | Status |
|---|---|
| Old specialist A (Qwen2.5-0.5B + LoRA, seed 0) and A′ (seed 1) | done |
| A's production logs on 1,500 inputs, with top-20 log-probs | done |
| 6 rebuilds on Qwen3-0.6B: R1-log, R2, R2-KD, R3, R4, R4-rand | done (25–35 min each) |
| R0a (zero-shot prompt with the written rules) | done |
| R0b (3-shot, 300-row subset) | done (23 min at batch 1) |
| Scoring on test_id (600) and test_drift (200) | done |
| Off-task ARC-Easy (500 items, paired) | PENDING_OFFTASK |

**Problems hit and fixed overnight** (details in `docs/log.md` D1–D14):
- fp16 is 4× slower than fp32 on this GPU, so everything runs in fp32.
- A never learned `risk_flag` at first. Fixed by adding a `risk_reasons` answer field.
- Out-of-memory errors in Qwen3 training and generation. Fixed with smaller batches; the effective training batch is unchanged.
- R0 answers were cut off at 96 tokens. Fixed by raising the limit to 256 for R0.
- 3-shot R0b fits only at batch 1, so it runs on a fixed 300-row subset.

## 2. Gate decisions

- **G0 passed.** Qwen2.5 at ~500 tok/s, Qwen3 at ~300–430 tok/s, peak ≤ 3.1 GB.
- **G1 MISSED.**
  - A scores 79.5% on val, but **62.3% on test_id** (target 80–90%).
  - On test rows without drift factors: 70.3% [65.1, 75.0].
  - Policy-affected rows: 2%, which is the designed stale-policy effect.
  - The task was frozen once test results had been seen. Every comparison is paired against the same A, so the rankings stand. Absolute accuracies are low; say so.
- **G2: results reported as they came out.** See below.
- **G3:** `python -m portcheck.report` rebuilds every table from saved predictions in seconds, and 8 tests pass. The fresh-clone rebuild test is still TODO.

## 3. Results

### test_id (600 tickets; old formats, half with held-out wordings, new policy)

| Model | Human labels | Accuracy | Δ vs A [95% CI] | Negative flips [95% CI] | Copies A's errors [95% CI] | Policy rows (n=56) |
|---|---|---|---|---|---|---|
| A (old specialist) | 1,500 (orig.) | 62.3% | — | — | — | 2% |
| A′ (seed noise floor) | — | 64.7% | +2.3 [+0.3, +4.3] | **2.0% [1.1, 3.5]** | 77% [71, 82] | 7% |
| R0a: prompt only | 0 | **0.0%** | −62.3 | — | — | 0% |
| **R1-log**: gold relabel of log inputs | 1,500 | **73.2%** | **+10.8 [+6.2, +15.5]** | **13.2% [10.7, 16.1]** | 22% [17, 28] | **82%** |
| R2: raw logs | 0 | 56.5% | −5.8 [−8.7, −3.2] | 9.0% [7.0, 11.6] | **80% [74, 85]** | 0% |
| R2-KD: logs + soft labels | 0 | 57.0% | −5.3 [−8.2, −3.0] | 8.7% [6.7, 11.2] | 79% [73, 84] | 0% |
| R3: validator-filtered logs | 0 | 55.2% | −7.2 [−10.2, −4.3] | 11.2% [8.9, 13.9] | 66% [60, 72] | 0% |
| R4: R3 + 200 low-confidence labels | 200 | 59.2% | −3.2 [−6.8, +0.2] | 11.0% [8.7, 13.8] | 61% [55, 67] | 0% |
| R4-rand: R3 + 200 random labels | 200 | 59.8% | −2.5 [−5.5, +0.3] | 8.8% [6.8, 11.4] | 64% [57, 70] | 5% |

### test_drift (200 tickets in NEW formats the old model never saw)

| Model | Labels | Accuracy | Δ vs A [95% CI] | Negative flips |
|---|---|---|---|---|
| A | — | 33.5% | — | — |
| A′ | — | 31.0% | −2.5 [−6.0, +0.5] | 4.0% (floor) |
| R1-log | 1,500 | **69.0%** | **+35.5 [+28.0, +43.5]** | 6.0% |
| **R4** | **200** | **59.5%** | **+26.0 [+18.5, +34.0]** | 6.0% |
| R4-rand | 200 | 56.5% | +23.0 [+16.0, +30.5] | 5.0% |
| R2 / R2-KD / R3 | 0 | 28.5–30.0% | −3.5 to −5.0 | 6.5–8.0% |

### Pipeline facts (`data/recipes/recipes_stats.json`)
- A is wrong on 39.7% of production-log inputs.
- Validators: precision 98.9%, recall 46.1% at finding A's errors.
- Confidence-based selection picked 70.5% wrong rows, against 34.5% for random selection. It also over-sampled new-format tickets (63/200 vs 44/200).

PENDING_EXTRA

## 4. What it means (plain words)

1. **Fine-tuning still matters at this size.** The new base with written rules scores 0%; with 3 worked examples it scores 3.3% against A's 63.0% on the same 300 rows. Examples help with formats (last-4 digits 1% → 25%, dates 97%), but not with the rules (risk_flag 45%, purpose_code 50%). Caveat: one of the 3 examples had `changed_bank_details`, and the model copied that reason onto 176/300 tickets (few-shot copying), which inflates false risk flags. Other examples might score higher, but last-4 accuracy (25%) alone caps all-fields-correct at 25%, far below A's 63%. The conclusion does not depend on the choice of examples.
2. **Accuracy hides regressions.** The best rebuild (R1-log, +10.8 points) still breaks 79 tickets the old model got right (13.2%), about 6× the seed-noise floor (2.0%). Most of these (57/79) are `risk_flag`. Under the pre-registered gate, **no rebuild would SHIP.** Every one has a flip rate far above the floor, which is exactly what the diff report is for.
3. **Rebuilding from the old model's logs does not work here.**
   - R2, R2-KD and R3 all score *below* the model they replace.
   - They copy ~80% of its mistakes, as much as a retrained copy of A does (A′: 77%).
   - They keep its stale policy (0% on policy rows).
   - Soft labels (KD) changed nothing. Filtering cut copied errors (80% → 66%) but did not raise accuracy.
4. **A small label budget pays off where the business changed.** On new ticket formats, 200 labels recover 73% of the full relabel's gain (+26.0 vs +35.5 points) with 13% of the labels. On old formats it barely helps (−3.2 points vs A), and it never fixes the stale policy there: about 12 corrected policy rows cannot outvote ~100 stale ones in the logs.
5. **Confidence-targeted labels find twice as many real errors, but did not beat random labels on accuracy.** test_id: 59.2% vs 59.8%. test_drift: 59.5% vs 56.5%, McNemar p = 0.41.

## 5. Pre-registered hypotheses

| | Hypothesis | Result |
|---|---|---|
| H1 | best R0 ≥ 5 pts below A | **Supported** (R0a 0% vs 62.3%; R0b 3-shot 3.3% vs A 63.0% on the same 300 rows) |
| H2 | R3 ≥ 95% of R1-log accuracy | **Not supported** (75%: 55.2 vs 73.2) |
| H3 | R2 inherits more of A's errors than R1-log; R3/R4 reduce it | **Supported** (80% vs 22%; R3 66%, R4 61%) |
| H4 | R4 beats R4-rand (overall and policy slice) | **Not supported** (59.2 vs 59.8; policy 0% vs 5%) |
| H5 | R2 and R2-KD flip less than R1-log | **Direction as predicted** (9.0%, 8.7% vs 13.2%); the Wilson CIs only just touch, and no paired test between rebuilds was pre-registered |
| H6 | R2-KD flips less than R2 | **Not supported** (8.7% vs 9.0%) |
| H7 | best log-based rebuild's flips inside the seed floor | **Not supported** (best 8.7% [6.7, 11.2] vs floor 2.0% [1.1, 3.5]) |
| H8 | log-only rebuilds stay wrong on policy; R4 fixes part | First part **supported** (0%); second part **not supported** on test_id (0%). On test_drift R4 is 79% vs A 14%, but n = 14 (exploratory) |

## 6. The pitch, filled in (send only after your checks in §7)

> Subject: Re: [original subject] — upgrading a specialist's base model, measured
>
> Hi Hyde team,
>
> Following up on my 5 Oct note with something closer to your core loop.
>
> Every specialist built on an open base has to be rebuilt when a better base ships. I measured what happens when you do that, rebuilding a payments-ops specialist from Qwen2.5-0.5B onto Qwen3-0.6B, on a laptop GPU for $0, with a per-example audit of every rebuild:
>
> - Prompting the new base scored 0%, and 3% even with worked examples, against 63% for the old specialist, so fine-tuning still mattered.
> - Rebuilding from the old model's logs scored *below* the old model: it copied ~80% of its mistakes and its outdated risk rule. Soft labels and rule filters didn't fix that.
> - The best rebuild (full relabel) gained 11 points but broke 13% of the tickets the old model got right, 6× the noise from simply retraining with a new seed. Net accuracy hid it; the per-example diff didn't.
> - On ticket formats the old model never saw, 200 labels recovered 73% of the full relabel's gain.
>
> This builds on Trans-LoRA and Apple's MUSCLE; what's new is the per-example audit of practical rebuild recipes. Code, results and a 1-page memo: [link]
>
> I've applied for AI Builder – Research (Mumbai). Happy to run the audit on one of your recipes.
>
> Angad

## 7. What you must check by hand before sending

1. Re-read §3 against `results/table.md` after the final `report` step.
2. Verify the citations (Trans-LoRA NeurIPS 2024, MUSCLE arXiv 2407.09435, Yan et al. CVPR 2021) and the §1 Hyde facts on the live pages.
3. Read the limits: synthetic task, 0.5–0.6B models, one seed per rebuild, simulated labels, and an old specialist that missed its accuracy target (G1).
4. Push the repo, add the link and apply on Ashby before replying on the careers@ thread. I did not push anything.

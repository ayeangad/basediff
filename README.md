# basediff

**What breaks when you move a fine-tuned specialist model onto a new base model, and how to catch it before you ship.**

A specialist is a small model fine-tuned for one job. When a better base model ships (Qwen2.5 → Qwen3 → …), the specialist must be rebuilt on it. That rebuild can quietly break cases the old model got right, while the overall score goes up.

basediff:
1. rebuilds a specialist on a new base six different ways;
2. compares old and new **ticket by ticket**: fixes, negative flips (old right → new wrong), copied errors and per-slice changes, all against a seed-noise floor;
3. gives a SHIP / REVIEW / BLOCK verdict.

Everything ran on a laptop (GTX 1650 Ti, 4 GB) for $0.

## Results

Task: extract 7 fields from messy synthetic payments-ops tickets. The old specialist runs on Qwen2.5-0.5B; every rebuild runs on Qwen3-0.6B. After launch, one risk rule changed (threshold 50k → 25k) and new ticket formats appeared. Comparisons are paired, with 95% CIs. Full tables: [`results/table.md`](results/table.md). Write-up: [`docs/MORNING_REPORT.md`](docs/MORNING_REPORT.md).

**test_id (600 tickets).** The old specialist scores 62.3%. Retraining it with another seed flips 2.0% of tickets, which is the noise floor.

| Rebuild on Qwen3 | Human labels | Accuracy | Negative flips | Copies old errors | Stale-policy tickets fixed |
|---|---|---|---|---|---|
| Prompt only (zero-shot / 3-shot) | 0 | 0% / 3.3% | — | — | 0% |
| Train on old model's logs (R2) | 0 | 56.5% | 9.0% | 80% | 0% |
| … with soft labels (R2-KD) | 0 | 57.0% | 8.7% | 79% | 0% |
| … rule-filtered (R3) | 0 | 55.2% | 11.2% | 66% | 0% |
| … + 200 low-confidence labels (R4) | 200 | 59.2% | 11.0% | 61% | 0% |
| … + 200 random labels (R4-rand) | 200 | 59.8% | 8.8% | 64% | 5% |
| Full relabel of log inputs (R1-log) | 1,500 | **73.2%** | **13.2%** | 22% | 82% |

**test_drift (200 tickets in new formats).** The old specialist scores 33.5%. R1-log reaches 69.0% and R4 reaches 59.5%: 200 labels recover 73% of the full relabel's gain.

**Takeaways**
1. Fine-tuning still matters at this size: prompting the new base alone scores 0–3%.
2. Rebuilding from the old model's logs scores *below* the old model. It copies its mistakes and its outdated rule.
3. The best rebuild gains 11 points but breaks 13% of tickets that used to work, 6× the seed noise. Net accuracy hides this; the per-example diff does not.
4. A small label budget goes a long way where the business changed (new formats). It does not fix a stale rule that the logs repeat many times.
5. No rebuild loses general ability (ARC-Easy, all changes within CI of 0).

**Limits:**
- one synthetic task;
- 0.5–0.6B models;
- one seed per rebuild;
- human labels simulated with generator gold;
- the old specialist sits below its planned accuracy (62% vs a target of 80–90%; see `docs/log.md` D14).

The hypotheses were pre-registered (git tag `prereg`): 2 supported, 4 not supported, 2 partly supported. All are reported in the write-up.

## How it works

```
gold_train ──► A (Qwen2.5 + LoRA) ──► A's logs on 1,500 production inputs (+ top-20 log-probs)
                                            │
                     validators.py ─────────┤  rule checks (R3)
                     recipes.py ────────────┤  confidence-based label selection (R4)
                                            ▼
              R1-log · R2 · R2-KD · R3 · R4 · R4-rand   (Qwen3 + LoRA)
                                            │
          eval/diff.py: fixes, negative flips, copied errors, per-slice, vs A→A' seed floor
          eval/stats.py: Wilson CIs, exact McNemar, paired bootstrap, Holm
                                            ▼
                              results/table.md · results/summary.json
```

| Path | What it does |
|---|---|
| `basediff/data_gen.py` | Synthetic tickets: ambiguous purpose codes, held-out wordings, interacting risk rules, policy drift |
| `basediff/train.py` | LoRA SFT, loss on answer tokens only; optional KL to the old model's top-20 (R2-KD) |
| `basediff/generate.py` | Batched greedy generation; can save top-k log-probs |
| `basediff/validators.py` | Cheap rule checks on logged answers (no model) |
| `basediff/recipes.py` | Builds each rebuild's training file from the old model's logs |
| `basediff/eval/` | Scoring, migration diff, paired statistics, off-task (ARC-Easy) |
| `basediff/report.py` | Rebuilds every table from saved predictions (CPU, ~15 s) |
| `scripts/overnight.sh` | The whole GPU pipeline as one resumable queue |
| `docs/log.md` | Every decision, finding and the pre-registration |

## Run it

```bash
uv venv --python 3.12 && uv pip install -r requirements.lock && uv pip install -e . --no-deps
python -m basediff.data_gen --out data      # synthetic data (seeded; hashes in data/MANIFEST.json)
pytest                                       # fast tests; `pytest -m slow` checks no row is truncated
bash scripts/overnight.sh                    # train, generate logs, rebuild, evaluate (~8 h on a 4 GB GPU)
python -m basediff.report                   # tables -> results/
```

Or with make: `make data`, `make test`, `make queue`, `make report`.

Compare any two models on any labelled test set:

```bash
basediff diff --old runs/A/test_id.preds.jsonl --new runs/R1-log/test_id.preds.jsonl --gold data/test_id.jsonl
```

```
tickets        600
accuracy       old 62.3%  new 73.2%  (change +10.8%, 95% CI +6.2% to +15.5%)
fixed          144
broken         79  (13.2%, 95% CI 10.7%-16.1%)
same mistake   49  (22% of the old model's mistakes)
McNemar p      1.6e-05
  slice ambiguous_purpose  n=142  broken 26  (18.3%)
  ...
```

Add `--out diff.json` for the full result, including per-slice intervals and the broken ticket ids.

**Hardware notes (GTX 1650 Ti):**
- Train and evaluate in **fp32**: fp16 autocast measured about 4× slower on this card.
- Qwen3 training uses micro-batch 2; generation uses batch 2–6 depending on prompt length, because of the fp32 KV cache.
- Details in `docs/log.md` D1, D11 and D12.

## Prior work

This builds on, and does not claim novelty over:
- adapter transfer across base versions: Trans-LoRA (NeurIPS 2024), PortLLM, Lin et al. 2025;
- negative flips in model updates: Yan et al. (CVPR 2021), Xie et al. (ACL 2021), Apple's MUSCLE (arXiv 2407.09435).

What's new here: a per-example audit that compares practical rebuild recipes on production logs, with copied-error measurement and a seed-noise floor.

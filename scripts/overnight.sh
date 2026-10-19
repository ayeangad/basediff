#!/usr/bin/env bash
# One queued run for all GPU work after G0 (PORT_PLAN v2 §9 Day 1). Resumable: every step
# is skipped if its output already exists, so re-running after a crash continues where
# it stopped. A failed step is logged and the queue moves on (it never retries in a loop).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 4 GB GPU: avoid fragmentation OOMs
OLD=Qwen/Qwen2.5-0.5B-Instruct
NEW=Qwen/Qwen3-0.6B
LOG=runs/queue.log
mkdir -p runs
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

step() {  # step NAME DONE_FILE CMD...
  local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then say "skip $name (exists: $done)"; return 0; fi
  say "start $name"
  local t0=$SECONDS
  if "$@" >> "runs/queue.$name.log" 2>&1; then
    say "done  $name ($(( (SECONDS - t0) / 60 )) min)"
  else
    say "FAIL  $name (see runs/queue.$name.log)"
  fi
}

gen() {  # gen BASE ADAPTER(or -) DATA OUT [extra args]
  local base=$1 adapter=$2 data=$3 out=$4; shift 4
  local a=(); [ "$adapter" != "-" ] && a=(--adapter "$adapter")
  $PY -m basediff.generate --base "$base" "${a[@]}" --data "$data" --out "$out" "$@"
}

# 1. old specialist A (seed 0) and A' (seed 1)
step train_A        runs/A/adapter        $PY -m basediff.train configs/A.yaml
step train_A_prime  runs/A_prime/adapter  $PY -m basediff.train configs/A_prime.yaml

# 2. A's production logs with top-20 log-probs, then the recipe training files
step logs_A         runs/A/logs.preds.topk.pt  gen $OLD runs/A/adapter data/log_inputs.jsonl runs/A/logs.preds.jsonl --save-topk 20 --batch-size 8
step recipes        data/recipes/recipes_stats.json  $PY -m basediff.recipes

# 3. rebuild recipes on the new base
for R in R1-log R2 R2-KD R3 R4 R4-rand; do
  step "train_$R" "runs/$R/adapter" $PY -m basediff.train "configs/$R.yaml"
done

# 4. R0: choose the prompt variant on val (excluding policy-affected rows), then test
for V in bare rules rules_strict; do
  BS=2; [ "$V" = bare ] && BS=4   # fp32 KV cache: long rules prompts need small batches (D12)
  step "r0_val_$V" "runs/R0/val_$V.preds.jsonl" gen $NEW - data/val.jsonl "runs/R0/val_$V.preds.jsonl" --r0 "$V" --batch-size $BS --max-new 256
done
step r0_choose runs/R0/choice.json $PY -m basediff.eval.choose_r0
R0V=$(cat runs/R0/choice.json 2>/dev/null | $PY -c "import json,sys; print(json.load(sys.stdin)['variant'])" 2>/dev/null || echo rules)
R0BS=2; [ "$R0V" = bare ] && R0BS=4
step shots data/shots.jsonl $PY -c "
import json
rows=[json.loads(l) for l in open('data/gold_train.jsonl')]
pick=[r for r in rows if not r['slices']['policy_affected']][:3]
open('data/shots.jsonl','w').write(''.join(json.dumps(r)+'\n' for r in pick))"

# 5. evaluate every model on test_id and test_drift
for SPLIT in test_id test_drift; do
  step "eval_A_$SPLIT"       "runs/A/$SPLIT.preds.jsonl"       gen $OLD runs/A/adapter       "data/$SPLIT.jsonl" "runs/A/$SPLIT.preds.jsonl"
  step "eval_A_prime_$SPLIT" "runs/A_prime/$SPLIT.preds.jsonl" gen $OLD runs/A_prime/adapter "data/$SPLIT.jsonl" "runs/A_prime/$SPLIT.preds.jsonl"
  step "eval_R0a_$SPLIT"     "runs/R0a/$SPLIT.preds.jsonl"     gen $NEW - "data/$SPLIT.jsonl" "runs/R0a/$SPLIT.preds.jsonl" --r0 "$R0V" --batch-size $R0BS --max-new 256
  for R in R1-log R2 R2-KD R3 R4 R4-rand; do
    step "eval_${R}_$SPLIT" "runs/$R/$SPLIT.preds.jsonl" gen $NEW "runs/$R/adapter" "data/$SPLIT.jsonl" "runs/$R/$SPLIT.preds.jsonl" --batch-size 4
  done
done
# R0b (3-shot) only fits at batch 2 (~5.7 s/row), so it runs on a fixed 300-row subset of
# test_id and is scored for accuracy (H1), not diffed (docs/log.md D11)
step eval_R0b_test_id_300 runs/R0b/test_id_300.preds.jsonl gen $NEW - data/test_id_300.jsonl runs/R0b/test_id_300.preds.jsonl --r0 "$R0V" --shots-from data/shots.jsonl --k-shots 3 --batch-size 1 --max-new 256

# 6. off-task retention (ARC-Easy, 500 items, paired)
step offtask_B runs/offtask/B.json $PY -m basediff.eval.offtask --base $NEW --out runs/offtask/B.json --batch-size 4
for R in R1-log R2 R2-KD R3 R4; do
  step "offtask_$R" "runs/offtask/$R.json" $PY -m basediff.eval.offtask --base $NEW --adapter "runs/$R/adapter" --out "runs/offtask/$R.json" --batch-size 4
done

# 7. tables (CPU only)
step report results/summary.json $PY -m basediff.report
say "queue finished"

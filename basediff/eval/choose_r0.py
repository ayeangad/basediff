"""Pick the R0 system-prompt variant on val, excluding policy-affected rows (val uses the old
policy while the R0 prompt states the new one). Never looks at the test splits."""
import json
from pathlib import Path

from portcheck.eval.task import graded_rows
from portcheck.prompts import R0_VARIANTS

scores = {}
for v in R0_VARIANTS:
    rows = [r for r in graded_rows(f"runs/R0/val_{v}.preds.jsonl", "data/val.jsonl")
            if not r["slices"]["policy_affected"]]
    scores[v] = sum(r["all_correct"] for r in rows) / len(rows)
best = max(scores, key=lambda v: (scores[v], v == "rules"))
Path("runs/R0/choice.json").write_text(json.dumps({"variant": best, "val_all_correct": scores}, indent=2))
print(best, scores)

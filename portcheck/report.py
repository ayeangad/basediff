"""Build every results table from saved predictions (CPU only, no model).

Writes results/summary.json and results/table.md: accuracy, negative flips against the
seed-noise floor (A vs A'), inherited errors, the policy slice, off-task change and cost
for each model, with paired statistics and Holm-adjusted McNemar p-values.
"""
import json
from pathlib import Path

from portcheck.eval.diff import diff_files
from portcheck.eval.stats import holm, paired_bootstrap
from portcheck.eval.task import graded_rows, summarize

MODELS = ["A_prime", "R0a", "R0b", "R1-log", "R2", "R2-KD", "R3", "R4", "R4-rand"]
LABELS = {"A_prime": 0, "R0a": 0, "R0b": 0, "R1-log": 1500, "R2": 0, "R2-KD": 0, "R3": 0, "R4": 200, "R4-rand": 200}
SPLITS = ["test_id", "test_drift"]


def _cost(name):
    p = Path(f"runs/{name}/cost.json")
    return json.loads(p.read_text())["minutes"] if p.exists() else None


def _offtask(name, base):
    p, b = Path(f"runs/offtask/{name}.json"), Path(f"runs/offtask/{base}.json")
    if not (p.exists() and b.exists()):
        return None
    x, y = json.loads(b.read_text())["items"], json.loads(p.read_text())["items"]
    keys = sorted(set(x) & set(y))
    return {"acc": sum(y[k] for k in keys) / len(keys),
            "vs_untuned_B": paired_bootstrap([x[k] for k in keys], [y[k] for k in keys])}


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)
    summary = {}
    for split in SPLITS:
        gold = f"data/{split}.jsonl"
        a_preds = f"runs/A/{split}.preds.jsonl"
        if not Path(a_preds).exists():
            continue
        s = {"A": summarize(graded_rows(a_preds, gold)), "models": {}}
        for m in MODELS:
            p = f"runs/{m}/{split}.preds.jsonl"
            if Path(p).exists():
                d = diff_files(a_preds, p, gold)
                d["score"] = summarize(graded_rows(p, gold))
                d["labels"] = LABELS[m]
                d["train_min"] = _cost(m)
                d["offtask"] = _offtask(m, "B")
                s["models"][m] = d
        ps = {m: d["mcnemar_p"] for m, d in s["models"].items() if m != "A_prime"}
        for m, p in holm(ps).items():
            s["models"][m]["mcnemar_p_holm"] = p
        summary[split] = s
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=list))

    lines = []
    for split, s in summary.items():
        floor = s["models"].get("A_prime", {})
        lines += [f"## {split}", "",
                  f"Old specialist A: all-fields-correct {s['A']['all_correct']:.1%} (n={s['A']['n']}). "
                  f"Seed-noise floor (A vs A'): flips {floor.get('flip_rate', float('nan')):.1%} "
                  f"[{floor.get('flip_ci', (float('nan'),) * 2)[0]:.1%}, {floor.get('flip_ci', (float('nan'),) * 2)[1]:.1%}].", "",
                  "| model | labels | acc | Δacc vs A [95% CI] | flips [95% CI] | fixes | McNemar p (Holm) | inherited errors | policy-slice acc | train min |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for m, d in s["models"].items():
            pol = d["slices"].get("policy_affected", {})
            lines.append(
                f"| {m} | {d['labels']} | {d['new_acc']:.1%} | {d['acc_diff']['diff']:+.1%} "
                f"[{d['acc_diff']['lo']:+.1%}, {d['acc_diff']['hi']:+.1%}] | {d['flip_rate']:.1%} "
                f"[{d['flip_ci'][0]:.1%}, {d['flip_ci'][1]:.1%}] | {d['fix_rate']:.1%} | "
                f"{d.get('mcnemar_p_holm', d['mcnemar_p']):.3g} | {d['inherited']:.0%} | "
                f"{pol.get('new_acc', float('nan')):.0%} (A {pol.get('old_acc', float('nan')):.0%}, n={pol.get('n', 0)}) | "
                f"{d['train_min'] if d['train_min'] is not None else '-'} |")
        lines.append("")
    off = {m: _offtask(m, "B") for m in MODELS}
    if any(off.values()):
        lines += ["## Off-task (ARC-Easy, 500 items, paired vs untuned Qwen3-0.6B)", "",
                  "| model | acc | Δ vs untuned B [95% CI] |", "|---|---|---|"]
        for m, o in off.items():
            if o:
                v = o["vs_untuned_B"]
                lines.append(f"| {m} | {o['acc']:.1%} | {v['diff']:+.1%} [{v['lo']:+.1%}, {v['hi']:+.1%}] |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

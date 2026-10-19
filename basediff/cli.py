"""Command line: `basediff diff | report | data`.

    basediff diff --old OLD.preds.jsonl --new NEW.preds.jsonl --gold GOLD.jsonl [--out diff.json]
    basediff report                      # rebuild results/ from saved predictions
    basediff data --out data --seed 0    # regenerate the synthetic task
"""
import argparse
import json
import sys
from pathlib import Path


def _diff(a) -> int:
    from basediff.eval.diff import diff_files
    d = diff_files(a.old, a.new, a.gold)
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=2))
    c = d["counts"]
    print(f"tickets        {d['n']}")
    print(f"accuracy       old {d['old_acc']:.1%}  new {d['new_acc']:.1%}  "
          f"(change {d['acc_diff']['diff']:+.1%}, 95% CI {d['acc_diff']['lo']:+.1%} to {d['acc_diff']['hi']:+.1%})")
    print(f"fixed          {c.get('fix', 0)}")
    print(f"broken         {c.get('flip', 0)}  ({d['flip_rate']:.1%}, 95% CI {d['flip_ci'][0]:.1%}-{d['flip_ci'][1]:.1%})")
    print(f"same mistake   {c.get('both_wrong_same', 0)}  ({d['inherited']:.0%} of the old model's mistakes)")
    print(f"McNemar p      {d['mcnemar_p']:.3g}")
    for name, s in d["slices"].items():
        print(f"  slice {name:<18} n={s['n']:<4} broken {s['flips']:<3} ({s['flip_rate']:.1%})")
    return 0


def _report(a) -> int:
    from basediff.report import main
    main()
    return 0


def _data(a) -> int:
    from basediff.data_gen import generate
    print(json.dumps(generate(Path(a.out), a.seed, a.scale), indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="basediff", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="per-ticket comparison of two prediction files")
    d.add_argument("--old", required=True)
    d.add_argument("--new", required=True)
    d.add_argument("--gold", required=True)
    d.add_argument("--out")
    d.set_defaults(fn=_diff)
    r = sub.add_parser("report", help="rebuild results/ from saved predictions")
    r.set_defaults(fn=_report)
    g = sub.add_parser("data", help="regenerate the synthetic dataset")
    g.add_argument("--out", default="data")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scale", type=float, default=1.0)
    g.set_defaults(fn=_data)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

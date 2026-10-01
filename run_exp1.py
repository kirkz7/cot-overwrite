"""Exp 1: does line-shuffling hurt only when the queried state was overwritten?

usage: python run_exp1.py --model Qwen/Qwen3-4B --n 200
"""
import argparse
import json
import os
import time

import pandas as pd
from tqdm import tqdm

from probe import Scorer, load
from tasks import build_prompt, make_example, presented_lines

KS = [0, 1, 2, 4, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fmts", default="bare,chained")
    ap.add_argument("--show", action="store_true", help="print one prompt per condition and exit")
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--plain", action="store_true", help="no chat template (base models)")
    ap.add_argument("--style", default="sym", choices=["sym", "nl"])
    args = ap.parse_args()

    tok, model = load(args.model, args.four_bit)
    scorer = Scorer(tok, model)
    tag = args.model.split("/")[-1] + ("_nl" if args.style == "nl" else "")
    os.makedirs("results", exist_ok=True)
    out_path = f"results/exp1_{tag}.jsonl"

    jobs = []
    for k in KS:
        for i in range(args.n):
            ex = make_example(k, seed=k * 100_000 + i)
            queries = [("target", ex.target)] + ([("control", ex.control)] if k > 0 else [])
            for qtype, qv in queries:
                jobs.append((ex, qtype, qv, "io", "none"))
                for fmt in args.fmts.split(","):
                    for cond in ("full", "shuf"):
                        jobs.append((ex, qtype, qv, cond, fmt))

    if args.show:
        ex = make_example(4, seed=1)
        for cond, fmt in [("io", "none"), ("full", "bare"), ("shuf", "chained")]:
            print("=" * 30, cond, fmt, "| target", ex.target, ex.history(ex.target))
            print(build_prompt(ex, ex.target, cond, fmt if fmt != "none" else "bare", tok, plain=args.plain, style=args.style))
        return

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, qtype, qv, cond, fmt in tqdm(jobs):
            prefix = build_prompt(ex, qv, cond, fmt if fmt != "none" else "bare", tok, plain=args.plain, style=args.style)
            s = scorer.score(prefix)
            pred = max(s, key=s.get)
            hist = ex.history(qv)
            trace_vals = {l.value for l in ex.lines}
            # values of the queried var in the order the probe saw them
            seen = [l.value for l in presented_lines(ex, cond) if l.var == qv]
            rec = dict(k=ex.k, seed=ex.seed, qtype=qtype, cond=cond, fmt=fmt,
                       gold=hist[-1], pred=pred, correct=pred == hist[-1],
                       stale=pred in hist[:-1],
                       other_trace=(pred in trace_vals) and pred not in hist,
                       # recency heuristic: pred == the value presented last / first
                       pick_last_seen=bool(seen) and pred == seen[-1],
                       pick_first_seen=bool(seen) and pred == seen[0],
                       gold_is_last_seen=bool(seen) and hist[-1] == seen[-1],
                       gold_lp=s[hist[-1]])
            f.write(json.dumps(rec) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")

    df = pd.read_json(out_path, lines=True)
    summ = (df.groupby(["qtype", "fmt", "cond", "k"])[["correct", "stale", "other_trace"]]
              .mean().mul(100).round(1).unstack("k"))
    print(summ.to_string())
    summ.to_csv(f"results/exp1_{tag}_summary.csv")


if __name__ == "__main__":
    main()


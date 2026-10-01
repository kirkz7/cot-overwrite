"""Exp 8: recency vs primacy as the number of overwrites grows (k up to 64, 72 lines).

For each wrong answer we record where the picked value sits in the queried variable's
history, chronologically (0 = initial value) and in presented order.
usage: python run_exp8.py --model Qwen/Qwen3-4B --n 150
"""
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from probe import Scorer, load
from tasks import build_prompt, presented_lines
from tasks_long import make_long_example

KS = [0, 1, 2, 4, 8, 16, 32, 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--plain", action="store_true")
    args = ap.parse_args()
    tok, model = load(args.model, args.four_bit)
    scorer = Scorer(tok, model)
    tag = args.model.split("/")[-1]
    out_path = f"results/exp8_{tag}.jsonl"
    os.makedirs("results", exist_ok=True)
    jobs = []
    for k in KS:
        for i in range(args.n):
            ex = make_long_example(k, seed=900_000 + k * 10_000 + i)
            for qtype, qv in [("target", ex.target)] + ([("control", ex.control)] if k else []):
                for cond in ("io", "full", "shuf"):
                    if qtype == "control" and cond == "io":
                        continue
                    jobs.append((ex, qtype, qv, cond))
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, qtype, qv, cond in tqdm(jobs):
            s = scorer.score(build_prompt(ex, qv, cond, "bare", tok, plain=args.plain))
            pred = max(s, key=s.get)
            hist = ex.history(qv)
            seen = [l.value for l in presented_lines(ex, cond) if l.var == qv]
            f.write(json.dumps(dict(
                k=ex.k, seed=ex.seed, qtype=qtype, cond=cond, gold=hist[-1], pred=pred,
                correct=pred == hist[-1], stale=pred in hist[:-1],
                hist_idx=hist.index(pred) if pred in hist else -1,         # chronological position
                seen_idx=seen.index(pred) if pred in seen else -1,         # presented position
                n_hist=len(hist), pick_last_seen=bool(seen) and pred == seen[-1],
                pick_first_seen=bool(seen) and pred == seen[0])) + "\n")
    df = pd.read_json(out_path, lines=True)
    t = df[df.qtype == "target"]
    print((t.groupby(["cond", "k"])[["correct", "stale", "pick_last_seen", "pick_first_seen"]].mean().unstack("cond") * 100).round(1).to_string())
    c = df[(df.qtype == "control")]
    print("control acc by cond/k:\n", (c.groupby(["cond", "k"]).correct.mean().unstack("cond") * 100).round(1).to_string())


if __name__ == "__main__":
    main()

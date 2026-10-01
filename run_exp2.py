"""Exp 2: position vs correction markers. {ordered, shuffled} x {none, back, fwd, both}.

usage: python run_exp2.py --model Qwen/Qwen3-4B --n 200
"""
import argparse
import json
import os
import time

import pandas as pd
from tqdm import tqdm

from probe import Scorer, load
from tasks_rev import MARKERS, build_rev_prompt, make_rev_example, presented

KS = [1, 2, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    tok, model = load(args.model, args.four_bit)
    if args.show:
        ex = make_rev_example(2, seed=3)
        for cond, marker in [("io", "none"), ("full", "both"), ("shuf", "back"), ("shuf", "fwd")]:
            print("=" * 30, cond, marker, "| gold", ex.gold, "wrong", ex.wrong_vals)
            print(build_rev_prompt(ex, cond, marker, tok, args.plain))
        return

    scorer = Scorer(tok, model)
    tag = args.model.split("/")[-1]
    os.makedirs("results", exist_ok=True)
    out_path = f"results/exp2_{tag}.jsonl"
    jobs = []
    for k in KS:
        for i in range(args.n):
            ex = make_rev_example(k, seed=500_000 + k * 100_000 + i)
            jobs.append((ex, "io", "none"))
            for marker in MARKERS:
                for cond in ("full", "shuf"):
                    jobs.append((ex, cond, marker))

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for ex, cond, marker in tqdm(jobs):
            s = scorer.score(build_rev_prompt(ex, cond, marker, tok, args.plain))
            pred = max(s, key=s.get)
            seen = [l.value for l in presented(ex, cond) if l.var == ex.target]
            f.write(json.dumps(dict(
                k=ex.k, seed=ex.seed, cond=cond, marker=marker, gold=ex.gold, pred=pred,
                correct=pred == ex.gold, stale=pred in ex.wrong_vals,
                pick_last_seen=bool(seen) and pred == seen[-1],
                gold_is_last_seen=bool(seen) and ex.gold == seen[-1],
                gold_lp=s[ex.gold])) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")

    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["cond", "marker", "k"]).correct.mean().unstack("k") * 100).round(1).to_string())


if __name__ == "__main__":
    main()

"""Exp 18: explicit order cues vs presentation position (see tasks_cue.py).

usage: python run_exp18.py --model Qwen/Qwen3-4B --style sym --n 200
"""
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from probe import Scorer, load
from tasks import make_example
from tasks_cue import CONDS, build_cue_prompt

KS = [2, 4, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--style", default="sym", choices=["sym", "nl"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    tok, model = load(args.model, args.four_bit)
    if args.show:
        ex = make_example(4, seed=1)
        for c in CONDS:
            p, seen = build_cue_prompt(ex, ex.target, c, tok, args.style, args.plain)
            print("=" * 20, c, "| history", ex.history(ex.target), "| presented", seen)
            print(p[p.index("Let me"):])
        return
    scorer = Scorer(tok, model)
    tag = args.model.split("/")[-1] + ("_nl" if args.style == "nl" else "")
    out_path = f"results/exp18_{tag}.jsonl"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for k in KS:
            for i in tqdm(range(args.n), desc=f"k={k}"):
                ex = make_example(k, seed=k * 100_000 + i)     # same items as Exp 1
                hist = ex.history(ex.target)
                for c in CONDS:
                    p, seen = build_cue_prompt(ex, ex.target, c, tok, args.style, args.plain)
                    s = scorer.score(p)
                    pred = max(s, key=s.get)
                    f.write(json.dumps(dict(k=k, seed=ex.seed, cond=c, gold=hist[-1], pred=pred,
                                            correct=pred == hist[-1], stale=pred in hist[:-1],
                                            pick_last_presented=pred == seen[-1],
                                            pick_first_presented=pred == seen[0],
                                            gold_is_last_presented=hist[-1] == seen[-1])) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["cond", "k"])[["correct", "pick_last_presented", "pick_first_presented"]].mean()
           .unstack("k") * 100).round(1).to_string())
    sh = df[df.cond.isin(["shuf", "shuf_step", "shuf_neutral"])]
    print("\nshuffled, accuracy when the latest value is NOT presented last (cue vs position conflict):")
    print((sh[~sh.gold_is_last_presented].groupby(["cond", "k"]).correct.mean().unstack("k") * 100).round(1).to_string())


if __name__ == "__main__":
    main()

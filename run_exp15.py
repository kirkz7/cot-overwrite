"""Exp 15: support vs position. How many restatements does an earlier conclusion need to beat a
single later one?

Context: 30% of the derivation + competing conclusions (Exp 13 setup). The conclusion stated
once always comes last; the other one is restated r extra times at random points inside the
derivation (r = 0, 1, 2, 4), using short restatement templates.
  gold_repeated : P_g earlier (+ r gold restatements), P_x last  -> does repetition protect gold?
  x_repeated    : P_x earlier (+ r X restatements),   P_g last  -> does repetition hijack the reader?

usage: python run_exp15.py --n 300
"""
import argparse
import json
import random

import pandas as pd
import torch
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from probe import greedy, load
from run_exp13 import build_items, keep_deriv

TEMPLATES = ["Let me double-check: the result should be {v}.", "So far this points to {v}.",
             "That gives {v} again.", "Checking once more, I get {v}."]
REPEATS = [0, 1, 2, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--four_bit", action="store_true")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    items = build_items(args.n)
    tok, model = load(args.model, args.four_bit)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = f"results/exp15_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j, (r, x, pg, px, deriv) in enumerate(tqdm(items)):
            g = r["gold"]
            base = keep_deriv(deriv, j, args.frac)
            for which in ("gold_repeated", "x_repeated"):
                rep_val, first, last = (g, pg, px) if which == "gold_repeated" else (x, px, pg)
                for rpt in REPEATS:
                    rng = random.Random(j * 100 + rpt)
                    ctx = base[:]
                    for t in range(rpt):
                        ctx.insert(rng.randint(0, len(ctx)), TEMPLATES[t % len(TEMPLATES)].format(v=rep_val))
                    segs = segments(tok, r["problem"], ctx + [first, last])
                    ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                    pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                    f.write(json.dumps(dict(uuid=r["uuid"], which=which, repeats=rpt, gold=g, x=x, pred=pred,
                                            correct=pred == g, picked_x=pred == x)) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["which", "repeats"])[["correct", "picked_x"]].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()

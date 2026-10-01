"""Exp 10: read/write asymmetry. Same (possibly shuffled) CoT prefix as Exp 1, but instead of
forcing "Therefore, the final value of x is", the model continues generating freely and we
parse its final answer. Compared with the forced readout on the same prompts.

usage: python run_exp10.py --model Qwen/Qwen3-4B --n 100
"""
import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm

from probe import Scorer, greedy, load
from tasks import build_prompt, make_example, presented_lines

KS = [1, 2, 4, 8]
ANS = re.compile(r"final value of \w+ is\s*\**\s*(-?\d+)", re.I)
INT = re.compile(r"-?\d+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--four_bit", action="store_true")
    args = ap.parse_args()
    tok, model = load(args.model, args.four_bit)
    scorer = Scorer(tok, model)
    tag = args.model.split("/")[-1]
    out_path = f"results/exp10_{tag}.jsonl"
    os.makedirs("results", exist_ok=True)
    tail = "Therefore, the final value of "
    with open(out_path, "w", encoding="utf-8") as f:
        for k in tqdm(KS):
            for i in range(args.n):
                ex = make_example(k, seed=k * 100_000 + i)   # same seeds as Exp 1
                hist = ex.history(ex.target)
                for cond in ("full", "shuf"):
                    forced = build_prompt(ex, ex.target, cond, "bare", tok)
                    s = scorer.score(forced)
                    pred_read = max(s, key=s.get)
                    # write: drop the forced answer stem and let the model continue after the trace
                    prefix = forced[: forced.rindex(tail)]
                    ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                    txt = tok.decode(greedy(model, ids, args.max_new, {tok.eos_token_id}), skip_special_tokens=True)
                    m = ANS.findall(txt)
                    ints = INT.findall(txt)
                    pred_write = int(m[-1]) if m else (int(ints[-1]) if ints else None)
                    seen = [l.value for l in presented_lines(ex, cond) if l.var == ex.target]
                    f.write(json.dumps(dict(
                        k=k, seed=ex.seed, cond=cond, gold=hist[-1],
                        read=pred_read, read_ok=pred_read == hist[-1], read_last=pred_read == seen[-1],
                        write=pred_write, write_ok=pred_write == hist[-1],
                        write_stale=pred_write in hist[:-1], write_last=pred_write == seen[-1],
                        write_len=len(tok(txt).input_ids), write_text=txt)) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["cond", "k"])[["read_ok", "write_ok", "read_last", "write_last", "write_stale"]].mean() * 100).round(1).to_string())
    print(df.groupby(["cond", "k"]).write_len.median().to_string())


if __name__ == "__main__":
    main()

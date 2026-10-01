"""Exp 10b: read/write asymmetry, redone. Exp 10's greedy continuations fell into repetition loops
and hit the 320-token cap in 41-56% of shuffled cases, so its 'writing does not help' conclusion is
void. Here: Qwen3's recommended non-thinking sampling (T=0.7, top_p=0.8, top_k=20), 1024 new tokens,
batched. Same items as Exp 1/10; the forced readout ('read') is taken from Exp 10.

usage: python run_exp10b.py --model Qwen/Qwen3-4B --n 100
"""
import argparse
import json
import re

import pandas as pd
import torch
from tqdm import tqdm

from probe import load
from tasks import build_prompt, make_example, presented_lines

KS = [1, 2, 4, 8]
ANS = re.compile(r"final value of \w+ is\s*\**\s*(-?\d+)", re.I)
INT = re.compile(r"-?\d+")
TAIL = "Therefore, the final value of "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)  # padded batches fall back to math SDPA; 16 x 1.4k tokens risks VRAM spill
    args = ap.parse_args()
    tok, model = load(args.model)
    tok.padding_side = "left"
    torch.manual_seed(0)
    jobs = []
    for k in KS:
        for i in range(args.n):
            ex = make_example(k, seed=k * 100_000 + i)
            for cond in ("full", "shuf"):
                forced = build_prompt(ex, ex.target, cond, "bare", tok)
                jobs.append((ex, cond, forced[: forced.rindex(TAIL)]))
    out_path = f"results/exp10b_{args.model.split('/')[-1]}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for b in tqdm(range(0, len(jobs), args.bs)):
            batch = jobs[b:b + args.bs]
            enc = tok([p for _, _, p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.8,
                                     top_k=20, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for (ex, cond, _), txt in zip(batch, texts):
                hist = ex.history(ex.target)
                seen = [l.value for l in presented_lines(ex, cond) if l.var == ex.target]
                m, ints = ANS.findall(txt), INT.findall(txt)
                pred = int(m[-1]) if m else (int(ints[-1]) if ints else None)
                n_tok = len(tok(txt, add_special_tokens=False).input_ids)
                f.write(json.dumps(dict(k=ex.k, seed=ex.seed, cond=cond, gold=hist[-1], write=pred,
                                        write_ok=pred == hist[-1], write_stale=pred in hist[:-1],
                                        write_last=pred == seen[-1], explicit=bool(m), capped=n_tok >= args.max_new,
                                        n_tok=n_tok, text=txt)) + "\n")
    df = pd.read_json(out_path, lines=True)
    read = pd.read_json("results/exp10_Qwen3-4B.jsonl", lines=True).groupby(["cond", "k"]).read_ok.mean()
    t = df.groupby(["cond", "k"])[["write_ok", "write_last", "write_stale", "capped", "explicit"]].mean()
    t["read_ok(exp10)"] = read
    print((t * 100).round(1).to_string())


if __name__ == "__main__":
    main()

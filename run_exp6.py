"""Exp 6 step B: paragraph shuffle on real traces whose superseded answer candidates were
identified by early answering (early_answer.py), not by regex.

rev6    : full-trace answer == gold, and at some truncation point the probe answered X != gold,
          X is written in the trace before that point, and a later point answers gold.
stable6 : full-trace answer == gold, and every written early answer equals gold.

For each shuffled presentation we record whether a superseded X is mentioned after the last
mention of gold ("conflict": position favours a stale value), the real-trace analogue of Exp 2.
usage: python run_exp6.py --model Qwen/Qwen3-4B --seeds 3
"""
import argparse
import json
import random
import re
import zlib

import pandas as pd
import torch
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from probe import greedy, load


def last_para(paras, v):
    pat = re.compile(rf"(?<![\d.]){v}(?!\d|\.\d)")
    idx = [i for i, p in enumerate(paras) if pat.search(p)]
    return idx[-1] if idx else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_stable", type=int, default=300)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    data = {(r["uuid"], r["group"]): r for r in map(json.loads, open("data_exp4.jsonl", encoding="utf-8"))}
    early = [json.loads(l) for l in open(f"results/early_{tag}.jsonl", encoding="utf-8")]
    rev, stable = [], []
    for e in early:
        a, t, g = e["answers"], e["in_text"], e["gold"]
        if a[-1] != g:
            continue
        stale = sorted({a[i] for i in range(len(a) - 1)
                        if t[i] and a[i] != g and any(x == g for x in a[i + 1:])})
        written = [a[i] for i in range(len(a)) if t[i]]
        if stale:
            rev.append((e, stale))
        elif written and all(x == g for x in written):
            stable.append((e, []))
    random.Random(0).shuffle(stable)
    items = [("rev6", e, s) for e, s in rev] + [("stable6", e, s) for e, s in stable[:args.n_stable]]
    print(f"rev6={len(rev)} stable6={len(stable)} (using {min(len(stable), args.n_stable)})")

    tok, model = load(args.model)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = f"results/exp6_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for grp, e, stale in tqdm(items):
            r = data[(e["uuid"], e["group"])]
            conds = [("full", 0)] + [("shuf", s) for s in range(1, args.seeds + 1)]
            for cond, seed in conds:
                paras = r["paras"][:]
                if cond == "shuf":
                    random.Random(seed * 1_000_003 + zlib.crc32(r["uuid"].encode())).shuffle(paras)
                segs = segments(tok, r["problem"], paras)
                ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                lg = last_para(paras, r["gold"])
                ls = max([last_para(paras, x) for x in stale], default=-1)
                f.write(json.dumps(dict(uuid=r["uuid"], grp=grp, cond=cond, seed=seed, gold=r["gold"], stale=stale,
                                        pred=pred, correct=pred == r["gold"], picked_stale=pred in stale,
                                        gold_last_para=lg, stale_last_para=ls, n_paras=len(paras),
                                        conflict=ls > lg)) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["grp", "cond"])[["correct", "picked_stale"]].mean() * 100).round(1).to_string())
    rv = df[df.grp == "rev6"]
    print((rv.groupby(["cond", "conflict"])[["correct", "picked_stale"]].mean() * 100).round(1)
          .assign(n=rv.groupby(["cond", "conflict"]).size()).to_string())


if __name__ == "__main__":
    main()

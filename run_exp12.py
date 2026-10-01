"""Exp 12: competing conclusions in real R1 traces.

Exp 11 showed that a bare late mention of X does not move the reader. Here X gets the same
derivational support as the gold answer: we copy the last paragraph that states gold, replace
gold by X inside it (a counterfeit conclusion), and place it
  end        : appended after the trace (latest conclusion is X)
  start      : prepended (earliest conclusion is X)
  replace    : the original last gold paragraph itself is rewritten (gold still appears earlier)
  end_retract: appended, followed by "Wait, that's wrong." (late but explicitly retracted)
X = gold +/- d, d in 1..9, not otherwise present in the trace.

usage: python run_exp12.py --model Qwen/Qwen3-4B --n 400
"""
import argparse
import json
import random
import re

import pandas as pd
import torch
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from numutil import INT, contains, num_re, swap
from probe import greedy, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--early_from", default="Qwen3-4B", help="early-answer file used to pick correctly-read traces")
    ap.add_argument("--four_bit", action="store_true")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    data = {(r["uuid"], r["group"]): r for r in map(json.loads, open("data_exp4.jsonl", encoding="utf-8"))}
    early = [json.loads(l) for l in open(f"results/early_{args.early_from}.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    items = []
    for e in early:
        if e["answers"][-1] != e["gold"]:
            continue
        r = data[(e["uuid"], e["group"])]
        g = r["gold"]
        present = {int(x) for x in INT.findall("\n\n".join(r["paras"]))}
        gold_idx = [i for i, p in enumerate(r["paras"]) if num_re(g).search(p)]
        cands = [g + s * d for d in range(1, 10) for s in (1, -1) if (g + s * d) not in present and abs(g + s * d) >= 10]
        if not gold_idx or not cands:
            continue
        items.append((r, rng.choice(cands), gold_idx[-1], len(gold_idx)))
    rng.shuffle(items)
    items = items[:args.n]
    print("items", len(items))

    tok, model = load(args.model, args.four_bit)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = f"results/exp12_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r, x, gi, n_gold_paras in tqdm(items):
            g, paras0 = r["gold"], r["paras"]
            fake = swap(paras0[gi], g, x)
            assert not contains(fake, g), "counterfeit still states gold"
            variants = {
                "none": paras0,
                "end": paras0 + [fake],
                "start": [fake] + paras0,
                "replace": paras0[:gi] + [fake] + paras0[gi + 1:],
                "end_retract": paras0 + [fake + " Wait, that's wrong."],
            }
            for cond, paras in variants.items():
                segs = segments(tok, r["problem"], paras)
                ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                f.write(json.dumps(dict(uuid=r["uuid"], cond=cond, gold=g, x=x, pred=pred, correct=pred == g,
                                        picked_x=pred == x, n_gold_paras=n_gold_paras,
                                        gold_in_context=contains("\n\n".join(paras), g) or contains(r["problem"], g),
                                        gold_para_pos=gi / len(paras0))) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby("cond")[["correct", "picked_x"]].mean() * 100).round(1).to_string())
    rp = df[df.cond == "replace"]
    print("replace, by whether gold is still anywhere in problem+trace:")
    print((rp.groupby("gold_in_context")[["correct", "picked_x"]].mean() * 100).round(1)
          .assign(n=rp.groupby("gold_in_context").size()).to_string())
    print("replace, by #paragraphs that state gold:")
    print((rp.groupby(pd.cut(rp.n_gold_paras, [0, 1, 2, 4, 100]), observed=True)[["correct", "picked_x"]].mean() * 100).round(1)
          .assign(n=rp.groupby(pd.cut(rp.n_gold_paras, [0, 1, 2, 4, 100]), observed=True).size()).to_string())


if __name__ == "__main__":
    main()

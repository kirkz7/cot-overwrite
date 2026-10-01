"""Exp 11: does a late mention of a retracted value mislead a reader of a real R1 trace?

Traces: ones the probe reads correctly in full (from early_answer.py output). X is a superseded
answer candidate found by early answering ("stale"), or else a random other integer from the
trace ("other"). One paragraph mentioning X is inserted at the start or the end, either with an
explicit retraction or neutrally. We measure how often the reader switches to X.

usage: python run_exp11.py --model Qwen/Qwen3-4B --n 400
"""
import argparse
import json
import random
import re

import pandas as pd
import torch
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from probe import greedy, load

INT = re.compile(r"(?<![\d.])-?\d+(?!\d|\.\d)")
INSERTS = {
    "retract": "Wait, earlier I got {x}. That was wrong.",
    "neutral": "Let me double-check the key value: {x}.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    data = {(r["uuid"], r["group"]): r for r in map(json.loads, open("data_exp4.jsonl", encoding="utf-8"))}
    early = [json.loads(l) for l in open(f"results/early_{tag}.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    items = []
    for e in early:
        a, t, g = e["answers"], e["in_text"], e["gold"]
        if a[-1] != g:
            continue
        r = data[(e["uuid"], e["group"])]
        stale = sorted({a[i] for i in range(len(a) - 1) if t[i] and a[i] != g})
        if stale:
            items.append((r, rng.choice(stale), "stale"))
        else:
            others = sorted({int(x) for x in INT.findall("\n\n".join(r["paras"]))} - {g})
            others = [o for o in others if abs(o) >= 10]
            if others:
                items.append((r, rng.choice(others), "other"))
    rng.shuffle(items)
    items = [it for it in items if it[2] == "stale"] + [it for it in items if it[2] == "other"][:args.n]
    print({k: sum(1 for it in items if it[2] == k) for k in ("stale", "other")})

    tok, model = load(args.model)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = f"results/exp11_{tag}.jsonl"
    conds = [("none", None, None)] + [(f"{pos}_{kind}", pos, kind) for pos in ("end", "start") for kind in INSERTS]
    with open(out_path, "w", encoding="utf-8") as f:
        for r, x, xtype in tqdm(items):
            for name, pos, kind in conds:
                paras = r["paras"][:]
                if pos == "end":
                    paras = paras + [INSERTS[kind].format(x=x)]
                elif pos == "start":
                    paras = [INSERTS[kind].format(x=x)] + paras
                segs = segments(tok, r["problem"], paras)
                ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                f.write(json.dumps(dict(uuid=r["uuid"], xtype=xtype, cond=name, gold=r["gold"], x=x, pred=pred,
                                        correct=pred == r["gold"], picked_x=pred == x)) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["xtype", "cond"])[["correct", "picked_x"]].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()

"""Exp 14: do the recency heads found on synthetic traces (Exp 7) drive the position effect on
real traces (Exp 13)? Zero the top-N recency heads (by Exp 7 score) or N random heads, then
re-run the competing-conclusion test.

usage: python run_exp14.py --n 300
"""
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from probe import greedy, load
from run_exp13 import build_items, keep_deriv

FRACS = [0.1, 0.3, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--topn", type=int, default=32)
    args = ap.parse_args()
    items = build_items(args.n)
    tok, model = load(args.model)
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // H)
    score = np.load("results/exp7_Qwen3-4B_attn.npz")["score"]
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    top = [tuple(map(int, x)) for x in order[:args.topn]]
    rand = random.Random(0).sample([(l, h) for l in range(L) for h in range(H)], args.topn)
    ablate = {}

    def make_hook(li):
        def hook(mod, inp):
            hs = ablate.get(li)
            if not hs:
                return None
            x = inp[0].clone()
            for h in hs:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        return hook
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))

    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = "results/exp14_Qwen3-4B.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for abl_name, heads in (("none", []), (f"top{args.topn}", top), (f"random{args.topn}", rand)):
            ablate.clear()
            for l, h in heads:
                ablate.setdefault(l, []).append(h)
            for j, (r, x, pg, px, deriv) in enumerate(tqdm(items, desc=abl_name)):
                g = r["gold"]
                for fr in FRACS:
                    base = keep_deriv(deriv, j, fr)
                    for order_, tail in (("gold_last", [px, pg]), ("x_last", [pg, px])):
                        segs = segments(tok, r["problem"], base + tail)
                        ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                        pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                        f.write(json.dumps(dict(uuid=r["uuid"], ablation=abl_name, frac=fr, order=order_, gold=g, x=x,
                                                pred=pred, correct=pred == g, picked_x=pred == x)) + "\n")
    df = pd.read_json(out_path, lines=True)
    t = df.groupby(["ablation", "frac", "order"]).correct.mean().unstack("order") * 100
    t["position_effect"] = t.gold_last - t.x_last
    t["picked_x_when_x_last"] = df[df.order == "x_last"].groupby(["ablation", "frac"]).picked_x.mean() * 100
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()

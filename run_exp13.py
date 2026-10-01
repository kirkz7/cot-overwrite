"""Exp 13: copy vs compute. Dose-response of position sensitivity on derivational support.

From a real R1 trace we take the last paragraph that states gold (P_g) and a counterfeit copy
with gold -> X (P_x). The context is: a fraction f of the remaining derivation paragraphs
(order kept; paragraphs that mention gold are dropped so gold is only stated in P_g), followed
by the two conclusions in either order:
    gold_last : ... P_x, P_g
    x_last    : ... P_g, P_x
Position sensitivity = P(pred = value of the last conclusion) averaged over both orders.
f = 0 is synthetic-like (bare competing conclusions); f = 1 keeps the whole derivation.

usage: python run_exp13.py --model Qwen/Qwen3-4B --n 400
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

FRACS = [0.0, 0.1, 0.3, 0.6, 1.0]


def build_items(n, bare=False):
    """(trace, X, P_g, P_x, derivation paragraphs) for traces Qwen3-4B reads correctly in full.
    Gold is stated only in P_g; P_x is P_g with gold -> X. Deterministic for a given n."""
    data = {(r["uuid"], r["group"]): r for r in map(json.loads, open("data_exp4.jsonl", encoding="utf-8"))}
    early = [json.loads(l) for l in open("results/early_Qwen3-4B.jsonl", encoding="utf-8")]
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
        x = rng.choice(cands)
        pg = f"So the answer is {g}." if bare else r["paras"][gold_idx[-1]]
        deriv = [p for i, p in enumerate(r["paras"]) if i not in gold_idx]
        px = swap(pg, g, x)
        assert not contains(px, g) and contains(pg, g) and not any(contains(p, g) for p in deriv)
        items.append((r, x, pg, px, deriv))
    rng.shuffle(items)
    return items[:n]


def keep_deriv(deriv, j, fr):
    """The derivation subset used for item j at fraction fr (same across scripts)."""
    keep = sorted(random.Random(j * 1000 + int(fr * 100)).sample(range(len(deriv)), round(fr * len(deriv))))
    return [deriv[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--bare", action="store_true",
                    help="conclusions are bare statements 'So the answer is N.' (nothing locally checkable)")
    args = ap.parse_args()
    tag = args.model.split("/")[-1] + ("_bare" if args.bare else "")
    items = build_items(args.n, args.bare)
    print("items", len(items))

    tok, model = load(args.model, args.four_bit)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = f"results/exp13_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j, (r, x, pg, px, deriv) in enumerate(tqdm(items)):
            g = r["gold"]
            for fr in FRACS:
                base = keep_deriv(deriv, j, fr)
                for order, tail in (("gold_last", [px, pg]), ("x_last", [pg, px])):
                    segs = segments(tok, r["problem"], base + tail)
                    ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                    pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                    last_val = g if order == "gold_last" else x
                    f.write(json.dumps(dict(uuid=r["uuid"], frac=fr, order=order, gold=g, x=x, pred=pred,
                                            correct=pred == g, picked_x=pred == x, picked_last=pred == last_val,
                                            n_deriv=len(base), gold_in_problem=contains(r["problem"], g))) + "\n")
    df = pd.read_json(out_path, lines=True)
    t = df.groupby(["frac", "order"])[["correct", "picked_x"]].mean().unstack("order") * 100
    t["position_effect"] = t[("correct", "gold_last")] - t[("correct", "x_last")]
    print(t.round(1).to_string())
    print((df.groupby("frac").picked_last.mean() * 100).round(1).rename("picked last conclusion %").to_string())


if __name__ == "__main__":
    main()

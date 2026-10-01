"""Exp 19: retraction markers vs position on real traces, with a symmetric control.

Design G (gold vs counterfeit X, as Exp 13):
  gold_last            ... P_x, P_g
  x_last               ... P_g, P_x
  x_last_retracted     ... P_g, P_x + " Wait, that's wrong."     (position says X, marker says not X)
Design S (symmetric: two counterfeits A, B; gold removed from the context entirely, so prior
knowledge of the answer favours neither; A/B values alternate across items):
  sym_none             ... P_A, P_B                               (latest standing: B)
  sym_later_retracted  ... P_A, P_B + " Wait, that's wrong."      (latest standing: A)
  sym_earlier_retracted ... P_A + " Wait, that's wrong.", P_B     (latest standing: B)
A reader that follows retractions picks A only in sym_later_retracted; a pure position reader
picks B everywhere.

usage: python run_exp19.py --model Qwen/Qwen3-4B --n 300
"""
import argparse
import json
import random

import pandas as pd
import torch
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from numutil import INT, contains, swap
from probe import greedy, load
from run_exp13 import build_items, keep_deriv

FRACS = [0.1, 0.3, 1.0]
RETRACT = " Wait, that's wrong."


def second_counterfeit(r, x1):
    """Another value near gold that does not occur anywhere in the trace (same matcher as the asserts;
    INT would miss e.g. '801' inside '-801')."""
    g = r["gold"]
    text = "\n\n".join(r["paras"])
    cands = [g + s * d for d in range(1, 10) for s in (1, -1)
             if abs(g + s * d) >= 10 and (g + s * d) != x1 and not contains(text, g + s * d)]
    return random.Random(r["uuid"]).choice(cands) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--four_bit", action="store_true")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    items = build_items(args.n)
    tok, model = load(args.model, args.four_bit)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids

    def ask(problem, paras):
        segs = segments(tok, problem, paras)
        ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
        return parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))

    out_path = f"results/exp19_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j, (r, x, pg, px, deriv) in enumerate(tqdm(items)):
            g = r["gold"]
            x2 = second_counterfeit(r, x)
            if x2 is None:
                continue
            # A (earlier) / B (later) alternate between the two counterfeit values
            a, b = (x, x2) if j % 2 == 0 else (x2, x)
            pa, pb = swap(pg, g, a), swap(pg, g, b)
            assert not contains(pa, g) and not contains(pb, g) and contains(pa, a) and contains(pb, b)
            for fr in FRACS:
                base = keep_deriv(deriv, j, fr)
                conds = {
                    "gold_last": base + [px, pg],
                    "x_last": base + [pg, px],
                    "x_last_retracted": base + [pg, px + RETRACT],
                    "sym_none": base + [pa, pb],
                    "sym_later_retracted": base + [pa, pb + RETRACT],
                    "sym_earlier_retracted": base + [pa + RETRACT, pb],
                }
                for cond, paras in conds.items():
                    pred = ask(r["problem"], paras)
                    rec = dict(uuid=r["uuid"], frac=fr, cond=cond, gold=g, pred=pred, picked_gold=pred == g)
                    if cond.startswith("sym"):
                        rec.update(a=a, b=b, picked_earlier=pred == a, picked_later=pred == b)
                    else:
                        rec.update(x=x, picked_x=pred == x)
                    f.write(json.dumps(rec) + "\n")
    df = pd.read_json(out_path, lines=True)
    g_ = df[~df.cond.str.startswith("sym")]
    print("== gold vs X ==")
    print((g_.groupby(["frac", "cond"])[["picked_gold", "picked_x"]].mean().unstack("cond") * 100).round(1).to_string())
    s_ = df[df.cond.str.startswith("sym")]
    print("\n== symmetric (two counterfeits, gold absent) ==")
    print((s_.groupby(["frac", "cond"])[["picked_earlier", "picked_later", "picked_gold"]].mean().unstack("cond") * 100).round(1).to_string())


if __name__ == "__main__":
    main()

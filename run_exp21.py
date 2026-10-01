"""Exp 21: NATURAL competing conclusions in R1 traces that end wrong.

43% of wrong OpenR1 traces stated the correct answer as a candidate earlier and later concluded
something else. Holding content fixed, we only reorder the two natural conclusion paragraphs:
  orig     : ..., P_gold-candidate, ..., P_final-wrong, ...   (as written)
  swap     : the two paragraphs exchange places
  g_end    : P_gold-candidate moved to the very end
If the reader's answer follows the paragraph order, position (not content or belief) decides
between the trace's own competing conclusions.
Symmetric control (no answer-recognition advantage): traces with two distinct WRONG candidates
v (earlier) and w (final), gold never stated; swap P_v and P_w.

usage: python run_exp21.py --model Qwen/Qwen3-4B --n 400
"""
import argparse
import json
import os
import random
import re

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from early_answer import SUFFIX, parse, segments
from numutil import contains
from probe import greedy, load

# candidate answers; a number that is part of a formula ("the answer is 1991 - T") is not a candidate
CAND = re.compile(r"(?i)(?:\banswer\b[^.\n]{0,30}?(?:\bis\b|=|\bbe\b)\s*\$?(?:\\boxed\{)?|\\boxed\{)\s*(-?\d+)"
                  r"(?!\d|\.\d|\s*[-+*/×^=<>])")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
MAX_CHARS = 32_000  # wrong traces run long
DATA = "data_exp21.jsonl"


def cand_paras(paras):
    """paragraph index -> set of candidate answer values stated in it ('287,091' read as 287091)"""
    return {i: {int(v) for v in CAND.findall(THOUSANDS.sub("", p))} for i, p in enumerate(paras)}


SHARDS = [f"data/train-{i:05d}-of-00010.parquet" for i in range(10)] + \
         [f"extended/train-{i:05d}-of-00010.parquet" for i in range(10)]


def prep():
    out, n_wrong = [], 0
    for shard in SHARDS:   # one shard at a time: all 20 at once would need >10 GB of RAM
        df = pd.read_parquet(hf_hub_download("open-r1/OpenR1-Math-220k", shard, repo_type="dataset"),
                             columns=["problem", "answer", "generations"])
        df = df[df.answer.str.fullmatch(r"-?\d+") & (df.answer.str.len() <= 7)]
        before = len(out)
        n_wrong += _scan(df, out)
        print(shard, "items +", len(out) - before, flush=True)
        del df
    with open(DATA, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    kinds = pd.Series([o["kind"] for o in out]).value_counts().to_dict()
    print(kinds, "| eligible wrong traces:", n_wrong,
          "| rate gold_vs_wrong: %.1f%%" % (100 * kinds.get("gold_vs_wrong", 0) / max(n_wrong, 1)))


def _scan(df, out):
    """append items from one shard; return the number of eligible wrong traces (the denominator)"""
    n_wrong = 0
    for _, r in df.iterrows():
        for g in r.generations:
            if "</think>" not in g:
                continue
            think, after = g.split("<think>", 1)[-1].split("</think>", 1)
            fb = BOXED.findall(after)
            if not fb or len(think) > MAX_CHARS:
                continue
            # multi-part problems ("a) ... b) ...") give several boxed values: not revisions
            if len(set(fb)) > 1 or re.search(r"(?m)^\W*\(?[a-d]\)\s", r.problem) or re.search(r"(?m)^\W*\(?b\)\s*\\boxed", think):
                continue
            final, gold = int(fb[-1]), int(r.answer)
            if final == gold or abs(final) < 10 or abs(gold) < 10:
                continue
            n_wrong += 1
            # keep in-think \boxed conclusions: here the conclusions themselves are what we reorder
            paras = [p.strip() for p in think.split("\n\n") if p.strip()]
            cp = cand_paras(paras)
            w_idx = [i for i, s in cp.items() if final in s]
            if not w_idx:
                continue
            wi = w_idx[-1]
            if contains(paras[wi], gold):   # multi-part answers ("150 g and 450 g") are not revisions
                continue
            g_idx = [i for i, s in cp.items() if gold in s and i < wi and final not in s]
            if g_idx:
                out.append(dict(kind="gold_vs_wrong", problem=r.problem, gold=gold, final=final, other=gold,
                                paras=paras, i_other=g_idx[-1], i_final=wi))
                continue
            if contains("\n\n".join(paras), gold):
                continue
            v_idx = [(i, v) for i, s in cp.items() if i < wi for v in s if v != final and abs(v) >= 10]
            if v_idx:
                i, v = v_idx[-1]
                if final not in cp[i]:
                    out.append(dict(kind="wrong_vs_wrong", problem=r.problem, gold=gold, final=final, other=v,
                                    paras=paras, i_other=i, i_final=wi))
    return n_wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--four_bit", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(DATA):
        prep()
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    rng = random.Random(0)
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    items = []
    for k, rs in by.items():
        rng.shuffle(rs)
        items += rs[:args.n]
    tok, model = load(args.model, args.four_bit)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    out_path = f"results/exp21_{args.model.split('/')[-1]}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in tqdm(items):
            p, io, iw = r["paras"], r["i_other"], r["i_final"]
            swapped = p[:]
            swapped[io], swapped[iw] = p[iw], p[io]
            o_end = [x for i, x in enumerate(p) if i != io] + [p[io]]
            for cond, paras in (("orig", p), ("swap", swapped), ("other_end", o_end)):
                segs = segments(tok, r["problem"], paras)
                ids = torch.tensor([[t for s in segs for t in s] + suffix_ids], device="cuda")
                pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                text = "\n\n".join(paras)
                f.write(json.dumps(dict(kind=r["kind"], cond=cond, gold=r["gold"], final=r["final"], other=r["other"],
                                        pred=pred, picked_final=pred == r["final"], picked_other=pred == r["other"],
                                        picked_gold=pred == r["gold"],
                                        n_final=len(re.findall(rf"(?<![\d.]){r['final']}(?!\d|\.\d)", text)),
                                        n_other=len(re.findall(rf"(?<![\d.]){r['other']}(?!\d|\.\d)", text)))) + "\n")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["kind", "cond"])[["picked_final", "picked_other", "picked_gold"]].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()

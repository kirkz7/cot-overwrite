"""Exp 4: real R1 traces. {ordered, paragraph-shuffled} x {markers kept, removed}, plus IO.
The probe reads the (conclusion-stripped) trace inside <think> and greedily decodes \\boxed{...}.

usage: python run_exp4.py --model Qwen/Qwen3-4B --n_per_group 300
"""
import argparse
import json
import os
import random
import re
import time
import zlib

import pandas as pd
import torch
from tqdm import tqdm

from probe import greedy, load

MARK = re.compile(r"(?:(?<=^)|(?<=[.!?]\s)|(?<=\n))(?:But wait|Wait|Hmm+|Actually|Oh|Oops|No)[,.!]?\s+", re.M)
INT = re.compile(r"(?<![\d.])-?\d+(?!\d|\.\d)")
CONDS = [("io", False), ("full", False), ("shuf", False), ("full", True), ("shuf", True)]


def strip_markers(p):
    p = MARK.sub("", p)
    return p[:1].upper() + p[1:] if p else p


def presented(row, cond, nomark, seed=0):
    if cond == "io":
        return ""
    paras = row["paras"][:]
    if cond == "shuf":
        rng = random.Random(30_000 + seed + zlib.crc32(row["uuid"].encode()))
        rng.shuffle(paras)
    if nomark:
        paras = [strip_markers(p) for p in paras]
    return "\n\n".join(paras)


def last_candidate(text, cands):
    """which of cands has its last occurrence latest in text"""
    best, pos = None, -1
    for c in cands:
        ms = [m.start() for m in re.finditer(rf"(?<![\d.]){c}(?!\d|\.\d)", text)]
        if ms and ms[-1] > pos:
            best, pos = c, ms[-1]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_per_group", type=int, default=300)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("data_exp4.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    data = []
    for g, rs in by.items():
        rng.shuffle(rs)
        data += rs[:args.n_per_group]
    print({g: min(len(rs), args.n_per_group) for g, rs in by.items()})

    tok, model = load(args.model, args.four_bit)
    tag = args.model.split("/")[-1]
    # stop as soon as a token containing "}" is produced
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])}
    stop_ids.add(tok.eos_token_id)

    def prompt(row, cond, nomark):
        head = tok.apply_chat_template([{"role": "user", "content": row["problem"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        head = head.split("<think>")[0]  # some templates pre-insert <think>; we write our own
        return head + "<think>\n" + presented(row, cond, nomark) + "\n</think>\n\nThe final answer is $\\boxed{"

    if args.show:
        r = next(r for r in data if r["group"] == "rev")
        print(prompt(r, "shuf", True)[:3000])
        return

    os.makedirs("results", exist_ok=True)
    out_path = f"results/exp4_{tag}.jsonl"
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(data):
            for cond, nomark in CONDS:
                p = prompt(row, cond, nomark)
                ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                gen = greedy(model, ids, 14, stop_ids)
                txt = tok.decode(gen, skip_special_tokens=True)
                m = INT.search(txt.split("}")[0].replace(",", "").replace(" ", ""))
                pred = int(m.group()) if m else None
                shown = presented(row, cond, nomark)
                last = last_candidate(shown, [row["gold"]] + row["stale"]) if cond != "io" else None
                f.write(json.dumps(dict(
                    uuid=row["uuid"], group=row["group"], cond=cond, nomark=nomark, gold=row["gold"],
                    pred=pred, raw=txt, correct=pred == row["gold"], stale=pred in row["stale"],
                    in_trace=pred is not None and pred in {int(x) for x in INT.findall(shown)},
                    pick_last_cand=last is not None and pred == last,
                    gold_is_last_cand=last == row["gold"], n_tok=ids.shape[1])) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")
    df = pd.read_json(out_path, lines=True)
    print((df.groupby(["group", "nomark", "cond"])[["correct", "stale"]].mean().unstack("cond") * 100).round(1).to_string())


if __name__ == "__main__":
    main()

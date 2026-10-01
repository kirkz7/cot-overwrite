"""Natural position-vs-answer conflicts in the model's OWN chain of thought (no editing).

In Qwen3-4B's own BBH multiple-choice CoTs (Exp 22 v2 generations, answer sentence stripped),
is the correct option the LAST option mentioned? When it is not (the model discussed other options
after the right one), does re-reading the unshuffled CoT get less accurate?
Uses only the 'full' (original order) re-read; CPU only."""
import json
import re

import pandas as pd

from run_exp22 import ANS_LINE, kind

gens = [json.loads(l) for l in open("results/bbh27_gen_v2_Qwen3-4B.jsonl", encoding="utf-8")]
reads = pd.read_json("results/exp22_v2_Qwen3-4B.jsonl", lines=True)
full = reads[reads.cond == "full"].set_index("idx")
OPT = re.compile(r"\(([A-R])\)")
rows = []
for i, g in enumerate(gens):
    if i not in full.index or kind(g["task"]) != "mc":
        continue
    lines = [l.strip() for l in g["cot"].split("\n") if l.strip() and not ANS_LINE.search(l)]
    mentions = [m for l in lines for m in OPT.findall(l)]
    if len(set(mentions)) < 2:
        continue                       # needs at least two distinct options discussed
    gold = g["target"].strip()[1]
    rows.append(dict(task=g["task"], gold_last=mentions[-1] == gold, gold_mentioned=gold in mentions,
                     correct=bool(full.loc[i, "correct"])))
d = pd.DataFrame(rows)
print(f"MC examples whose own CoT discusses >=2 options (answer line stripped): {len(d)}")
print(f"  correct option is the LAST option mentioned: {d.gold_last.mean() * 100:.1f}%")
print("  re-read accuracy by whether the correct option is mentioned last:")
print((d.groupby("gold_last").correct.agg(["mean", "size"]).assign(mean=lambda x: (x["mean"] * 100).round(1))).to_string())
by = d.groupby("task").agg(n=("correct", "size"), gold_last=("gold_last", "mean"))
by = by[by.n >= 20]
by["acc_if_last"] = d[d.gold_last].groupby("task").correct.mean()
by["acc_if_not_last"] = d[~d.gold_last].groupby("task").correct.mean()
print("\nper task (n>=20):")
print((by.assign(gold_last=by.gold_last * 100, acc_if_last=by.acc_if_last * 100, acc_if_not_last=by.acc_if_not_last * 100)).round(1).to_string())

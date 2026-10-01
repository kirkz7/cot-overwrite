"""Exp 22 analysis + sanity checks: per-task generation accuracy, extraction failures, shuffle drop."""
import json
import sys

import pandas as pd

from run_exp22 import TASKS, extract, is_correct

gens = pd.DataFrame([json.loads(l) for l in open("results/bbh27_gen_v2_Qwen3-4B.jsonl", encoding="utf-8")])
gens["extracted"] = gens.gen_answer.map(lambda a: isinstance(a, str))
gens["ok"] = [is_correct(a, t) for a, t in zip(gens.gen_answer, gens.target)]
gens["n_tok_approx"] = gens.cot.str.len() / 4
s = gens.groupby("task").agg(gen_acc=("ok", "mean"), extracted=("extracted", "mean"), n=("ok", "size"),
                             cot_chars=("cot", lambda c: c.str.len().median()))
print((s.assign(gen_acc=s.gen_acc * 100, extracted=s.extracted * 100)).round(1).sort_values("gen_acc").to_string())

if "--examples" in sys.argv:
    for t in ["object_counting", "boolean_expressions", "navigate", "word_sorting"]:
        bad = gens[(gens.task == t) & (~gens.ok)].head(2)
        for _, r in bad.iterrows():
            print("=" * 80, "\n", t, "| target:", r.target, "| extracted:", repr(r.gen_answer), "\n...", r.cot[-300:])

df = pd.read_json("results/exp22_v2_Qwen3-4B.jsonl", lines=True)
t = df.groupby(["task", "cond"]).correct.mean().unstack("cond") * 100
t["drop"] = t["full"] - t["shuf"]
t["label"] = [TASKS[x] for x in t.index]
t["overwrite"] = df[df.cond == "full"].groupby("task").overwrite.mean()
t["n"] = df[df.cond == "full"].groupby("task").size()
print("\n", t.sort_values("drop", ascending=False).round(2).to_string())
print("\nmean drop by a-priori label (task-weighted):", t.groupby("label")["drop"].mean().round(1).to_dict())
print("spearman(drop, overwrite proxy) over tasks:", round(t[["drop", "overwrite"]].corr(method="spearman").iloc[0, 1], 3))

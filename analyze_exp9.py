"""Exp 9: post-training lineage. Exp1 (bare) recency and Exp2 marker sensitivity per model."""
import os

import pandas as pd

MODELS = ["OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
          "OLMo-2-1124-13B", "OLMo-2-1124-13B-Instruct",
          "Qwen2.5-Math-7B", "DeepSeek-R1-Distill-Qwen-7B", "Qwen3-4B-Base", "Qwen3-4B", "Qwen3-14B"]
rows1, rows2 = [], []
for m in MODELS:
    f1, f2 = f"results/exp1_{m}.jsonl", f"results/exp2_{m}.jsonl"
    if os.path.exists(f1):
        d = pd.read_json(f1, lines=True)
        d = d[d.fmt.isin(["bare", "none"])]
        t = d[d.qtype == "target"]
        r = dict(model=m)
        for k in (2, 4, 8):
            sh = t[(t.k == k) & (t.cond == "shuf")]
            r[f"full k{k}"] = t[(t.k == k) & (t.cond == "full")].correct.mean() * 100
            r[f"shuf k{k}"] = sh.correct.mean() * 100
            r[f"last k{k}"] = sh.pick_last_seen.mean() * 100
        r["ctrl shuf"] = d[(d.qtype == "control") & (d.cond == "shuf")].correct.mean() * 100
        rows1.append(r)
    if os.path.exists(f2):
        d = pd.read_json(f2, lines=True)
        sh = d[d.cond == "shuf"]
        r = dict(model=m)
        for mk in ("none", "fwd", "back", "both"):
            x = sh[sh.marker == mk]
            r[f"{mk} conflict"] = x[~x.gold_is_last_seen].correct.mean() * 100
            r[f"{mk} full"] = d[(d.cond == "full") & (d.marker == mk)].correct.mean() * 100
        rows2.append(r)
pd.set_option("display.width", 250)
print("== Exp1 (bare): ordered acc / shuffled acc / pick-last % ==")
print(pd.DataFrame(rows1).round(1).to_string(index=False))
print("\n== Exp2: accuracy when a wrong value is presented last (conflict), and ordered acc ==")
print(pd.DataFrame(rows2).round(1).to_string(index=False))

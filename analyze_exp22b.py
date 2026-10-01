"""Exp 22 v2, conditional metric: among examples whose ORDERED re-read is correct, how often does
the shuffled re-read stay correct? (Removes ceiling effects from tasks whose answer-stripping also
removed the last computation, e.g. 'So the total is 13'.) Writes results/exp22_v2_summary.json."""
import json

import pandas as pd

from run_exp22 import TASKS

df = pd.read_json("results/exp22_v2_Qwen3-4B.jsonl", lines=True)
full = df[df.cond == "full"].set_index("idx").correct
sh = df[df.cond == "shuf"].groupby("idx").correct.mean()
d = pd.DataFrame({"task": df[df.cond == "full"].set_index("idx").task, "full": full, "shuf": sh})
ok = d[d.full]
t = ok.groupby("task").agg(retained=("shuf", "mean"), n=("shuf", "size"))
t["retained"] *= 100
t["drop"] = 100 - t["retained"]
t["full_acc"] = d.groupby("task").full.mean() * 100
t["label"] = [TASKS[x] for x in t.index]
t = t.sort_values("drop", ascending=False)
print(t.round(1).to_string())
print("\nmean conditional drop by label (tasks with n>=20):",
      t[t.n >= 20].groupby("label")["drop"].mean().round(1).to_dict())
print("tasks with n<20 (unreliable):", t[t.n < 20].index.tolist())
json.dump([dict(task=k, label=r["label"], drop=round(r["drop"], 1), retained=round(r["retained"], 1), n=int(r["n"]),
                full_acc=round(r["full_acc"], 1)) for k, r in t.iterrows()],   # r.drop would be DataFrame.drop
          open("results/exp22_v2_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

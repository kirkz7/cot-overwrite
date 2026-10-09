"""Preliminary MemConflict read (NOT the pre-registered metric): string heuristic on the response.
new-like: the later value is mentioned and, if the earlier one is too, the earlier one comes first ("from X to Y").
old-like: the earlier value is mentioned and, if the later one is too, the later one comes first. Aggregates only."""
import sys
import pandas as pd
sys.path.insert(0, "/root/cot-overwrite")
from app_common import load_jsonl
from explore_extmem import key


def lab(r):
    t = str(r["response"]).lower()
    o, n = t.find(key(r["old"])), t.find(key(r["new"]))
    if n >= 0 and (o < 0 or o < n):
        return "new"
    if o >= 0 and (n < 0 or n < o):
        return "old"
    return "neither"


for name in sys.argv[1:]:
    df = pd.DataFrame(load_jsonl(f"/root/cot-overwrite/results/extmem_{name}.jsonl"))
    df = df[df.task == "memconf"].copy()
    df["lab"] = df.apply(lab, axis=1)
    full = df.groupby("id").cond.nunique()
    df = df[df.id.isin(full[full == 3].index)]
    tab = pd.crosstab(df.cond, df.lab, normalize="index").mul(100).round(1)
    print(f"== {name}: {df.id.nunique()} questions with all three orders")
    print(tab.reindex(["chrono", "rev", "retr"]).to_string())
    piv = df.assign(ok=df.lab == "new").pivot_table(index="id", columns="cond", values="ok", aggfunc="first").astype(float)
    print("   new-like rev - chrono %.1f, retr - chrono %.1f" % ((piv.rev - piv.chrono).mean() * 100, (piv.retr - piv.chrono).mean() * 100))

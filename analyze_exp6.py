"""Exp 6 diagnostics: what do the superseded candidates look like, and does the conflict
effect appear anywhere (e.g. large-|X| candidates, conflicts where X is far after gold)?"""
import json
import re

import pandas as pd

df = pd.read_json("results/exp6_Qwen3-4B.jsonl", lines=True)
data = {(r["uuid"]): r for r in map(json.loads, open("data_exp4.jsonl", encoding="utf-8"))}
rv = df[df.grp == "rev6"].copy()
rv["max_abs_stale"] = rv.stale.map(lambda s: max(abs(x) for x in s))
rv["gap"] = (rv.stale_last_para - rv.gold_last_para) / rv.n_paras
print("rev6 traces:", rv.uuid.nunique(), "| stale values per trace (median):", rv.stale.map(len).median())
print("share of traces whose stale candidates are all |X|<10:", round((rv.drop_duplicates("uuid").max_abs_stale < 10).mean() * 100, 1), "%")
sh = rv[rv.cond == "shuf"]
print("\nshuffled, by |X| >= 10 and conflict:")
print((sh.groupby([sh.max_abs_stale >= 10, "conflict"])[["correct", "picked_stale"]].mean() * 100).round(1)
      .assign(n=sh.groupby([sh.max_abs_stale >= 10, "conflict"]).size()).to_string())
print("\nshuffled conflicts, by how far after gold the stale value comes (fraction of trace):")
c = sh[sh.conflict]
print((c.groupby(pd.cut(c.gap, [0, .1, .3, .6, 1.0]), observed=True)[["correct", "picked_stale"]].mean() * 100).round(1)
      .assign(n=c.groupby(pd.cut(c.gap, [0, .1, .3, .6, 1.0]), observed=True).size()).to_string())
# a few examples of stale contexts
ex = rv[(rv.cond == "full") & (rv.max_abs_stale >= 10)].head(5)
for _, x in ex.iterrows():
    text = "\n\n".join(data[x.uuid]["paras"])
    s = max(x.stale, key=abs)
    m = re.search(rf"(?<![\d.]){s}(?!\d|\.\d)", text)
    print("\n--- gold", x.gold, "stale", s, "::", text[max(0, m.start() - 160): m.end() + 120].replace("\n", " ") if m else "?")

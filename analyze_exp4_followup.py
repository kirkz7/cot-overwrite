"""Exp4 follow-up: (1) frequency of gold vs stale mentions; (2) do stale and gold share a paragraph?"""
import json, re
import pandas as pd

rows = {(r["uuid"], r["group"]): r for r in map(json.loads, open("data_exp4.jsonl", encoding="utf-8"))}
df = pd.read_json("results/exp4_Qwen3-4B.jsonl", lines=True)
has = lambda text, c: re.search(rf"(?<![\d.]){c}(?!\d|\.\d)", text) is not None
cnt = lambda text, c: len(re.findall(rf"(?<![\d.]){c}(?!\d|\.\d)", text))
rev = df[(df.group == "rev") & (df.cond == "full") & (~df.nomark)].copy()
feat = []
for _, x in rev.iterrows():
    r = rows[(x.uuid, x.group)]; text = "\n\n".join(r["paras"])
    same_para = any(has(p, r["gold"]) and any(has(p, s) for s in r["stale"]) for p in r["paras"])
    gold_paras = sum(has(p, r["gold"]) for p in r["paras"])
    feat.append(dict(gold_n=cnt(text, r["gold"]), stale_n=max(cnt(text, c) for c in r["stale"]),
                     same_para=same_para, gold_paras=gold_paras, n_paras=len(r["paras"])))
rev = pd.concat([rev.reset_index(drop=True), pd.DataFrame(feat)], axis=1)
print("median mentions: gold", rev.gold_n.median(), "| most-mentioned stale", rev.stale_n.median())
print("gold mentioned more than every stale value:", round((rev.gold_n > rev.stale_n).mean() * 100, 1), "%")
print("gold and a stale value co-occur in one paragraph:", round(rev.same_para.mean() * 100, 1), "%")
print("median #paragraphs mentioning gold:", rev.gold_paras.median(), "of", rev.n_paras.median())
print((rev.groupby("same_para")[["correct", "stale"]].mean() * 100).round(1).assign(n=rev.groupby("same_para").size()))

"""usage: python analyze_exp4.py results/exp4_Qwen3-4B.jsonl"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = sys.argv[1]
tag = path.split("exp4_")[-1].replace(".jsonl", "")
df = pd.read_json(path, lines=True)
df["cell"] = df.cond + np.where(df.nomark, "-nomark", "")
cells = ["io", "full", "shuf", "full-nomark", "shuf-nomark"]

pct = lambda s: (s * 100).round(1)
print("n per group:", df[df.cell == "io"].group.value_counts().to_dict())
print("\n== accuracy % ==")
print(pct(df.groupby(["group", "cell"]).correct.mean().unstack()[cells]).to_string())
print("\n== picks a retracted answer candidate (stale) % ==")
print(pct(df.groupby(["group", "cell"]).stale.mean().unstack()[cells]).to_string())
print("\n== pred appears somewhere in the shown trace % ==")
print(pct(df.groupby(["group", "cell"]).in_trace.mean().unstack()[cells]).to_string())

rev = df[(df.group == "rev") & (df.cond != "io")]
print("\n== rev group: accuracy by whether gold is the LAST-mentioned answer candidate ==")
t = pct(rev.groupby(["cell", "gold_is_last_cand"]).correct.mean().unstack())
t.columns = ["a stale value is mentioned last", "gold mentioned last"]
print(t.assign(n_stale_last=rev.groupby("cell").gold_is_last_cand.apply(lambda s: int((~s).sum()))).to_string())
print("\n== rev group: pick the last-mentioned candidate % ==")
print(pct(rev.groupby("cell").pick_last_cand.mean()).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
w = 0.25
for i, g in enumerate(["clean", "rev", "unk"]):
    m = df[df.group == g].groupby("cell").correct.mean().reindex(cells)
    n = df[df.group == g].groupby("cell").size().reindex(cells)
    ax.bar(np.arange(len(cells)) + (i - 1) * w, m * 100, w, yerr=1.96 * np.sqrt(m * (1 - m) / n) * 100,
           capsize=2, label=f"{g} (n={int(n.iloc[0])})")
ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells); ax.set_ylabel("accuracy (%)")
ax.legend(); ax.set_title(f"Exp 4 · {tag} · real R1 traces, paragraph shuffle")
fig.tight_layout(); fig.savefig(f"results/exp4_{tag}.png", dpi=130)
print(f"\nsaved results/exp4_{tag}.png")

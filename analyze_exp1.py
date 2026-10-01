"""usage: python analyze_exp1.py results/exp1_Qwen3-4B.jsonl"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.read_json(path, lines=True)
tag = path.split("exp1_")[-1].replace(".jsonl", "")


def ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, fmt in zip(axes, ["bare", "chained"]):
    for qtype, cond, style, label in [("target", "full", "-o", "overwritten var, ordered CoT"),
                                      ("target", "shuf", "-s", "overwritten var, shuffled CoT"),
                                      ("control", "shuf", "--s", "once-assigned var, shuffled CoT"),
                                      ("target", "io", ":x", "overwritten var, no CoT (IO)")]:
        f = "none" if cond == "io" else fmt
        g = df[(df.qtype == qtype) & (df.cond == cond) & (df.fmt == f)].groupby("k").correct
        m, n = g.mean(), g.size()
        ax.errorbar(m.index, m * 100, yerr=ci(m, n), fmt=style, label=label, capsize=3)
    ks = sorted(df.k.unique())
    ax.plot(ks, [100 / (k + 1) for k in ks], color="gray", lw=0.8, label="chance among k+1 values")
    ax.set_title(f"CoT format: {fmt}")
    ax.set_xlabel("k = overwrites of the queried variable")
    ax.set_xticks(ks)
axes[0].set_ylabel("accuracy (%)")
axes[1].legend(fontsize=8, loc="center right")
fig.suptitle(f"Exp 1 · {tag} · line shuffle vs state overwrite")
fig.tight_layout()
fig.savefig(f"results/exp1_{tag}.png", dpi=130)

# what does the probe pick when the CoT is shuffled?
sh = df[(df.qtype == "target") & (df.cond == "shuf") & (df.k > 0)]
tab = sh.groupby(["fmt", "k"]).agg(acc=("correct", "mean"), stale=("stale", "mean"),
                                   pick_last_seen=("pick_last_seen", "mean"),
                                   pick_first_seen=("pick_first_seen", "mean"),
                                   gold_is_last_seen=("gold_is_last_seen", "mean"), n=("correct", "size"))
print("\n== shuffled, overwritten var: which value is picked ==")
print((tab.drop(columns="n") * 100).round(1).assign(n=tab.n).to_string())

# conditional accuracy: when gold happens to be presented last vs not
print("\n== shuffled accuracy split by whether gold is presented last ==")
print((sh.groupby(["fmt", "k", "gold_is_last_seen"]).correct.mean().unstack() * 100).round(1).to_string())

print("\n== main table (accuracy %) ==")
print((df.groupby(["qtype", "fmt", "cond", "k"]).correct.mean().unstack("k") * 100).round(1).to_string())
print(f"\nsaved results/exp1_{tag}.png")

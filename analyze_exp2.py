"""usage: python analyze_exp2.py results/exp2_Qwen3-4B.jsonl"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = sys.argv[1]
tag = path.split("exp2_")[-1].replace(".jsonl", "")
df = pd.read_json(path, lines=True)
sh = df[df.cond == "shuf"]

print("== accuracy % ==")
print((df.groupby(["cond", "marker", "k"]).correct.mean().unstack("k") * 100).round(1).to_string())
print("\n== shuffled: pick-last-seen %, stale % ==")
print((sh.groupby(["marker", "k"])[["pick_last_seen", "stale"]].mean().unstack("k") * 100).round(1).to_string())
print("\n== shuffled accuracy by whether gold is presented last (position/marker agree vs conflict) ==")
t = (sh.groupby(["marker", "k", "gold_is_last_seen"]).correct.mean().unstack() * 100).round(1)
t.columns = ["gold NOT last (conflict)", "gold last (agree)"]
print(t.to_string())

# figure: shuffled accuracy in the conflict case (position says wrong value), per marker
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
order = ["none", "fwd", "back", "both"]
for ax, agree in zip(axes, [False, True]):
    sub = sh[sh.gold_is_last_seen == agree]
    for m in order:
        g = sub[sub.marker == m].groupby("k").correct
        mu, n = g.mean(), g.size()
        ax.errorbar(mu.index, mu * 100, yerr=1.96 * np.sqrt(mu * (1 - mu) / n) * 100, fmt="-o", capsize=3, label=m)
    ax.set_xticks([1, 2, 4]); ax.set_xlabel("k = wrong attempts before the correct value")
    ax.set_title("gold presented last (position agrees)" if agree else "gold NOT presented last (position misleads)")
axes[0].set_ylabel("shuffled accuracy (%)"); axes[1].legend(title="marker")
fig.suptitle(f"Exp 2 · {tag} · self-correction markers under line shuffle")
fig.tight_layout(); fig.savefig(f"results/exp2_{tag}.png", dpi=130)
print(f"\nsaved results/exp2_{tag}.png")

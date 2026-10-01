"""Exp 13 figure: position sensitivity vs amount of derivation kept."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

runs = [("Qwen3-4B", "Qwen3-4B, counterfeit conclusion paragraph", "-o"),
        ("Qwen3-14B", "Qwen3-14B (4-bit), counterfeit paragraph", "-s"),
        ("Qwen3-4B_bare", "Qwen3-4B, bare 'So the answer is N.'", "--^")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for tag, label, style in runs:
    p = f"results/exp13_{tag}.jsonl"
    if not os.path.exists(p):
        continue
    df = pd.read_json(p, lines=True)
    g = df.groupby(["frac", "order"]).correct.agg(["mean", "size"]).unstack("order")
    eff = (g[("mean", "gold_last")] - g[("mean", "x_last")]) * 100
    se = np.sqrt(sum(g[("mean", o)] * (1 - g[("mean", o)]) / g[("size", o)] for o in ("gold_last", "x_last"))) * 100
    axes[0].errorbar(eff.index * 100, eff, yerr=1.96 * se, fmt=style, capsize=3, label=label)
    px = df[df.order == "x_last"].groupby("frac").picked_x.mean() * 100
    axes[1].plot(px.index * 100, px, style, label=label)
    print(tag, "\n", pd.DataFrame({"position_effect_pp": eff.round(1), "picked_x_when_x_last_%": px.round(1)}).to_string())
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_xlabel("% of derivation paragraphs kept"); axes[0].set_ylabel("acc(gold last) - acc(X last), pp")
axes[0].set_title("Position effect vs derivational support")
axes[1].set_xlabel("% of derivation paragraphs kept"); axes[1].set_ylabel("picked counterfeit X when X is last (%)")
axes[1].set_title("Copying the last-stated conclusion"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/exp13_dose_response.png", dpi=130)
print("saved results/exp13_dose_response.png")

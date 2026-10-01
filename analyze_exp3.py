"""Cross-model summary of Exp 1 and Exp 2. usage: python analyze_exp3.py"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["Qwen3-1.7B", "Qwen3-4B-Base", "Qwen3-4B", "Qwen3-8B", "Phi-4-mini-instruct"]
LABEL = {"Qwen3-8B": "Qwen3-8B (4-bit)"}

e1 = {f.split("exp1_")[1][:-6]: pd.read_json(f, lines=True) for f in glob.glob("results/exp1_*.jsonl")}
e2 = {f.split("exp2_")[1][:-6]: pd.read_json(f, lines=True) for f in glob.glob("results/exp2_*.jsonl")}
models = [m for m in ORDER if m in e1]

rows = []
for m in models:
    d = e1[m]
    for k in [0, 1, 2, 4, 8]:
        t = d[(d.qtype == "target") & (d.k == k)]
        c = d[(d.qtype == "control") & (d.k == k) & (d.cond == "shuf")]
        sh = t[(t.cond == "shuf") & (t.fmt == "bare")]
        rows.append(dict(model=m, k=k,
                         full=t[(t.cond == "full") & (t.fmt == "bare")].correct.mean() * 100,
                         shuf_bare=sh.correct.mean() * 100,
                         shuf_chained=t[(t.cond == "shuf") & (t.fmt == "chained")].correct.mean() * 100,
                         control_shuf=c.correct.mean() * 100 if len(c) else float("nan"),
                         io=t[t.cond == "io"].correct.mean() * 100,
                         pick_last=sh.pick_last_seen.mean() * 100))
s1 = pd.DataFrame(rows).round(1)
print("== Exp1 across models ==")
print(s1.to_string(index=False))

rows = []
for m in [m for m in ORDER if m in e2]:
    d = e2[m]
    sh = d[d.cond == "shuf"]
    for mk in ["none", "fwd", "back", "both"]:
        x = sh[sh.marker == mk]
        rows.append(dict(model=m, marker=mk,
                         full=d[(d.cond == "full") & (d.marker == mk)].correct.mean() * 100,
                         shuf=x.correct.mean() * 100,
                         conflict=x[~x.gold_is_last_seen].correct.mean() * 100,
                         agree=x[x.gold_is_last_seen].correct.mean() * 100))
s2 = pd.DataFrame(rows).round(1)
print("\n== Exp2 across models (pooled over k) ==")
print(s2.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for m in models:
    x = s1[s1.model == m]
    axes[0].plot(x.k, x.shuf_bare, "-o", label=LABEL.get(m, m))
axes[0].plot([0, 1, 2, 4, 8], [100, 50, 33.3, 20, 11.1], color="gray", lw=0.8, label="chance among k+1")
axes[0].set_xticks([0, 1, 2, 4, 8]); axes[0].set_xlabel("k overwrites"); axes[0].set_ylabel("shuffled accuracy (%)")
axes[0].set_title("Exp1: overwritten var, shuffled bare CoT"); axes[0].legend(fontsize=8)
mk = ["none", "fwd", "back", "both"]
w = 0.8 / max(1, len(s2.model.unique()))
for i, m in enumerate(s2.model.unique()):
    x = s2[s2.model == m].set_index("marker").reindex(mk)
    axes[1].bar([j + i * w for j in range(4)], x.conflict, w, label=LABEL.get(m, m))
axes[1].set_xticks([j + 0.4 - w / 2 for j in range(4)]); axes[1].set_xticklabels(mk)
axes[1].set_ylabel("accuracy when a wrong value is presented last (%)")
axes[1].set_title("Exp2: markers vs position (conflict cases)"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/exp3_cross_model.png", dpi=130)
print("\nsaved results/exp3_cross_model.png")

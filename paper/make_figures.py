"""Figures for the paper draft (paper/figures/*.pdf + *.png).

Figure 3 is computed from the desktop's results/*.jsonl (4B / Phi external sets and ConvoMem-long). The other numbers are
copied from RESULTS.md / STAGE_REPORT_EN.md / CLOUD_NOTEBOOK.md (the section is named next to each constant), so the
scale figure and the fix figure can be rebuilt on any machine. No conversation text is read or printed except our own
synthetic data.
usage: .venv\\Scripts\\python.exe paper/make_figures.py            (all figures; fig3 needs the desktop results)
       python paper/make_figures.py fig_scale fig_fix              (only the named figures)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# reference palette (dataviz skill, references/palette.md), fixed slot order; slots 1-3 validate all-pairs,
# slots 1-6 validate as adjacent pairs (lines); aqua / yellow / magenta are below 3:1 on white, so lines are direct-labelled
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"
GRAY = "#9a9994"          # reference series (base model)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 220, "font.family": "DejaVu Sans",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def jl(path):
    path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


# ------------------------------------------------------------------ loaders for 10-04 results
def extmem(tag):
    """{(task, id, cond): correct} merging raw (mabcr) and judged (memconf, locomo) rows"""
    raw = {(r["task"], r["id"], r["cond"]): r for r in jl(f"results/extmem_{tag}.jsonl")}
    jud = {(r["task"], r["id"], r["cond"]): r for r in jl(f"results/extmem_{tag}_judged.jsonl")}
    out = {}
    for k, r in raw.items():
        if k in jud:
            out[k] = (jud[k]["correct"], jud[k].get("stale", False))
        elif "correct" in r:
            out[k] = (r["correct"], r.get("stale", False))
    return out


def longconv(tag):
    """{(task, id, cond): correct}; ConvoMem-long from the judged file, PersonaMem re-parsed with pm_pred"""
    out = {(r["task"], r["id"], r["cond"]): (r["correct"], r.get("stale", False)) for r in jl(f"results/longconv_{tag}_judged.jsonl")}
    pm = [r for r in jl(f"results/longconv_{tag}.jsonl") if r["task"] == "personamem"]
    if pm:
        from explore_longconv import personamem, pm_pred
        gold = {(x["id"], x["cond"]): x["gold"] for x in personamem()}
        for r in pm:
            text = r.get("full") if isinstance(r.get("full"), str) else r["response"]
            out[("personamem", r["id"], r["cond"])] = (pm_pred(text) == gold[(r["id"], r["cond"])], False)
    return out


def acc(d, task, cond, ids=None):
    v = [c for (t, i, k), (c, s) in d.items() if t == task and k == cond and (ids is None or i in ids)]
    return 100 * np.mean(v) if v else np.nan


def paired(d, task, c1, c0):
    ids = sorted({i for (t, i, k) in d if t == task and k == c1} & {i for (t, i, k) in d if t == task and k == c0})
    if not ids:
        return np.nan, np.nan, np.nan, 0
    a = [d[(task, i, c1)][0] for i in ids]
    b = [d[(task, i, c0)][0] for i in ids]
    return (*boot(a, b), len(ids))


# ------------------------------------------------------------------ Figure 1: concept
def fig_concept():
    # schematic only (our own toy example, not a model output)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.25))
    panels = [
        ("Chronological", "oldest first", [("2024-03-02", "I live in Darwin."), ("2024-09-14", "I just moved to Melbourne.")], "Melbourne", True),
        ("Newest first", "inbox, git log, feeds", [("2024-09-14", "I just moved to Melbourne."), ("2024-03-02", "I live in Darwin.")], "Darwin", False),
        ("Retrieval order", "most relevant first (BM25)", [("2024-09-14", "I just moved to Melbourne."), ("2024-03-02", "I live in Darwin.")], "Darwin", False),
    ]
    for ax, (title, sub, rows, ans, ok) in zip(axes, panels):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.0, 1.0, title, fontsize=8, color=INK, va="top", weight="bold")
        ax.text(0.0, 0.905, sub, fontsize=6.8, color=INK2, va="top")
        for j, (date, text) in enumerate(rows):
            y = 0.67 - j * 0.23
            ax.add_patch(FancyBboxPatch((0.0, y - 0.09), 0.995, 0.18, boxstyle="round,pad=0.004,rounding_size=0.03",
                                        fc="#f4f3f0", ec="none"))
            ax.text(0.03, y + 0.04, f"Session ({date})", fontsize=6.3, color=INK2, va="center")
            ax.text(0.03, y - 0.035, f"User: {text}", fontsize=6.5, color=INK, va="center")
        ax.text(0.0, 0.17, "Q: Where does the user live now?", fontsize=6.6, color=INK, va="center")
        col = BLUE if ok else ORANGE
        ax.text(0.0, 0.05, f"Last-written value: {ans}", fontsize=6.8, color=col, va="center", weight="bold")
        ax.text(0.0, -0.06, "correct" if ok else "stale (dates ignored)", fontsize=6.6, color=col, va="center")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig1_concept")


# ------------------------------------------------------------------ Figure 2: CoT reader
def fig_cot():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.3))
    # (a) Exp 1, Qwen3-4B, synthetic 16-line program, bare format, n=200/cell (RESULTS.md "Exp 1"; picks-last from STAGE_REPORT_EN §1)
    k = np.array([0, 1, 2, 4, 8])
    ordered = [100, 100, 100, 99.5, 100]
    shuffled = [100, 79.5, 39.5, 19.0, 14.0]
    picks_last = [100, 72.0, 93.5, 97.0, 98.5]
    chance = 100 / (k + 1)
    x = np.arange(len(k))
    a.plot(x, ordered, color=BLUE, lw=1.6, marker="o", ms=4, label="original order")
    a.plot(x, shuffled, color=ORANGE, lw=1.6, marker="o", ms=4, label="lines shuffled")
    a.plot(x[1:], picks_last[1:], color=AQUA, lw=1.6, marker="s", ms=4, label="shuffled: picks last-presented")
    a.plot(x, chance, color=GRAY, lw=1.0, label="chance")
    a.set_xticks(x, [str(v) for v in k]); a.set_xlabel("times the queried variable is overwritten (k)")
    a.set_ylabel("% of items"); a.set_ylim(0, 105)
    a.set_title("(a) Shuffled CoT: the last-presented value wins", loc="left")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.6, handlelength=1.6)
    # (b) Exp 18, newest-first + step number on every line; accuracy in the conflict case (STAGE_REPORT_EN §2)
    kk = ["2", "4", "8"]
    series = [("Qwen3-14B", [90.5, 87.5, 88.5], BLUE), ("OLMo-2-13B-Instruct", [26.0, 11.5, 11.5], ORANGE),
              ("Qwen3-4B", [14.0, 3.0, 5.0], AQUA), ("OLMo-2-7B-Instruct", [1.0, 0.0, 0.0], GRAY)]
    label_y = {"Qwen3-14B": 88.5, "OLMo-2-13B-Instruct": 17.0, "Qwen3-4B": 6.0, "OLMo-2-7B-Instruct": -5.0}
    for name, ys, col in series:
        b.plot(range(3), ys, color=col, lw=1.6, marker="o", ms=4)
        b.text(2.1, label_y[name], name, color=INK, fontsize=6.8, va="center")
    b.set_xticks(range(3), kk); b.set_xlim(-0.15, 3.35); b.set_ylim(-9, 100)
    b.set_xlabel("overwrites (k)"); b.set_ylabel("% correct")
    b.set_title("(b) Newest-first CoT with a step number on every line", loc="left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_cot")


# ------------------------------------------------------------------ Figure 3: applications and external sets
def bars(ax, groups, conds, colors, labels, ylim=(0, 105), delta_from=0):
    w = 0.8 / len(conds)
    for gi, (gname, vals) in enumerate(groups):
        for ci, v in enumerate(vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.bar(gi + (ci - (len(conds) - 1) / 2) * w, v, width=w * 0.9, color=colors[ci], lw=0,
                   label=labels[ci] if gi == 0 else None)
        if delta_from is not None and len(vals) > 1 and vals[1] is not None and not np.isnan(vals[1]):
            d = vals[1] - vals[delta_from]
            ax.text(gi, max(v for v in vals if v is not None and not np.isnan(v)) + 3, f"{d:+.0f}",
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylim(*ylim)


def fmt_delta(d):
    return "±0" if abs(d) < 0.5 else f"{d:+.0f}".replace("-", "−")


def grouped(ax, rows, colors, labels, ylim=(0, 108)):
    """rows: [(dataset, model, [v_chrono, v_rev, v_retr or None])]; two-level x axis, centred bars,
    each non-chronological bar labelled with its difference from the chronological bar"""
    w, seen, x, centers = 0.27, set(), 0.0, {}
    xs = []
    prev = None
    for ds, model, vals in rows:
        if prev is not None and ds != prev:
            x += 0.45                                   # gap between datasets
        present = [i for i, v in enumerate(vals) if v is not None and not np.isnan(v)]
        for j, ci in enumerate(present):
            xb = x + (j - (len(present) - 1) / 2) * w
            ax.bar(xb, vals[ci], w * 0.92, color=colors[ci], lw=0, label=None if ci in seen else labels[ci])
            seen.add(ci)
            if ci > 0:
                ax.text(xb, vals[ci] + 2, fmt_delta(vals[ci] - vals[0]), ha="center", fontsize=6.2, color=INK)
        xs.append((x, model))
        centers.setdefault(ds, []).append(x)
        prev = ds
        x += 1.0
    ax.set_xticks([p for p, _ in xs], [m for _, m in xs], fontsize=6.4)
    for ds, cs in centers.items():
        ax.text(np.mean(cs), -0.20, ds, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=INK)
    ax.set_ylim(*ylim); ax.set_xlim(-0.6, x - 0.4)


def fig_apps():
    q4 = extmem("Qwen3-4B"); phi_x = extmem("Phi-4-mini")
    lq4, lphi = longconv("Qwen3-4B"), longconv("Phi-4-mini")
    mc_phi = [acc(phi_x, "memconf", c) for c in ("chrono", "rev", "retr")]
    rows = [
        # LongMemEval knowledge-update, full sessions, dated (RESULTS.md "应用测试" table 1)
        ("LongMemEval", "Qwen-4B", [76.9, 51.3, None]),
        ("LongMemEval", "Qwen-14B", [78.2, 53.8, None]),
        ("LongMemEval", "Phi-4", [80.8, 28.2, None]),
        # synthetic e-mail threads, every message dated (RESULTS.md "应用测试" table 2)
        ("E-mail threads", "Qwen-4B", [88.3, 58.0, None]),
        ("E-mail threads", "Phi-4", [86.3, 15.0, None]),
        ("ConvoMem-long", "Qwen-4B", [acc(lq4, "convo_long", "chrono"), acc(lq4, "convo_long", "rev"), None]),
        ("ConvoMem-long", "Phi-4", [acc(lphi, "convo_long", "chrono"), acc(lphi, "convo_long", "rev"), None]),
        ("MemConflict", "Qwen-4B", [acc(q4, "memconf", c) for c in ("chrono", "rev", "retr")]),
    ]
    if not np.isnan(mc_phi[0]):
        rows.append(("MemConflict", "Phi-4", mc_phi))
    null = [
        ("PersonaMem", "Qwen-4B", [acc(lq4, "personamem", "chrono"), acc(lq4, "personamem", "rev"), None]),
        ("PersonaMem", "Phi-4", [acc(lphi, "personamem", "chrono"), acc(lphi, "personamem", "rev"), None]),
        ("LoCoMo", "Qwen-4B", [acc(q4, "locomo", c) for c in ("chrono", "rev", "retr")]),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [len(rows) + 1.6, len(null) + 0.9]})
    labels = ["chronological", "newest first", "retrieval order (BM25)"]
    grouped(a, rows, [BLUE, ORANGE, AQUA], labels)
    a.set_ylabel("% correct"); a.set_title("(a) A conflicting older state is present", loc="left")
    grouped(b, null, [BLUE, ORANGE, AQUA], labels)
    b.set_title("(b) No conflicting state", loc="left"); b.set_yticklabels([])
    h, l = [], []
    for ax in (a, b):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_apps")


# ------------------------------------------------------------------ Figure: scale (Qwen3 4B-32B, cloud, vLLM bf16)
SIZES = ["4B", "8B", "14B", "32B"]
# newest-first (or retrieval) minus chronological, paired bootstrap 95% CI; RESULTS.md "10-07 复盘 · 规模曲线"
SCALE = {
    "MemConflict, newest first": ([-40.4, -37.1, -35.8, -21.2], [(-47.5, -33.3), (-43.8, -30.4), (-42.5, -29.2), (-27.9, -14.6)]),
    "MemConflict, retrieval order": ([-19.2, -17.9, -15.4, -10.0], [(-25.4, -13.3), (-23.8, -12.5), (-21.2, -9.6), (-15.4, -4.6)]),
    "LongMemEval, newest first (seen)": ([-24.4, -24.4, -23.1, -24.4], [(-35.9, -12.8), (-34.6, -14.1), (-33.3, -12.8), (-35.9, -14.1)]),
    "ConvoMem-long, newest first": ([-17.7, -12.1, -2.4, -6.5], [(-25.0, -10.5), (-17.7, -6.5), (-6.5, 1.6), (-11.3, -2.4)]),
    "Dated logs, newest first": ([-11.3, -13.6, -3.6, -2.9], [(-14.9, -7.9), (-17.1, -10.1), (-5.3, -1.8), (-4.7, -1.1)]),
}
# pre-registered CoT test, HF scoring, k=8, n=100 per point (RESULTS.md "10-07 复盘 · 规模曲线", CoT rows)
COT_PICK_LAST_K8 = [98.0, 99.0, 97.0, 97.0]     # shuffled, no time cue: % answering the last-presented value
COT_REV_STEP = [4.0, 62.0, 63.0, 80.0]          # newest first with a step number on every line: % correct


def fig_scale():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.55), gridspec_kw={"width_ratios": [1.55, 1]})
    style = [(BLUE, "o", "-", True), (BLUE, "o", "--", False), (ORANGE, "s", "-", True), (AQUA, "^", "-", True),
             (YELLOW, "D", "-", True)]
    x = np.arange(len(SIZES))
    a.axhline(0, color=INK2, lw=0.8)
    a.axhline(-10, color=GRAY, lw=0.8, ls=":")
    a.text(3.42, -10, "threshold", fontsize=6.2, color=INK2, va="center")
    for j, ((name, (ys, cis)), (col, mk, ls, filled)) in enumerate(zip(SCALE.items(), style)):
        xo = x + (j - 2) * 0.07
        lo = [y - c[0] for y, c in zip(ys, cis)]
        hi = [c[1] - y for y, c in zip(ys, cis)]
        a.errorbar(xo, ys, yerr=[lo, hi], color=col, lw=1.6, ls=ls, marker=mk, ms=4.2, capsize=0, elinewidth=0.8,
                   mfc=col if filled else "white", mec=col, label=name)
    a.set_xticks(x, ["Qwen3-" + s for s in SIZES]); a.set_xlim(-0.3, 3.95)
    a.set_ylim(-50, 5); a.set_ylabel("accuracy change vs. chronological (pp)")
    a.set_title("(a) Order effect on memory questions, by model size", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.4, handlelength=2.4, columnspacing=1.2)
    b.plot(x, COT_PICK_LAST_K8, color=MAGENTA, lw=1.6, marker="o", ms=4.2)
    b.plot(x, COT_REV_STEP, color=GREEN, lw=1.6, marker="s", ms=4.2)
    b.text(1.5, COT_PICK_LAST_K8[1] - 6, "shuffled, no cue:\npicks last-presented", fontsize=6.2, color=INK, va="top", ha="center")
    b.text(1.5, COT_REV_STEP[1] - 7, "newest first + step\nnumbers: correct", fontsize=6.2, color=INK, va="top", ha="center")
    b.set_xticks(x, SIZES); b.set_xlim(-0.3, 3.3); b.set_ylim(0, 112); b.set_yticks(range(0, 101, 20)); b.set_ylabel("% of items")
    b.set_title("(b) Chains of thought, k = 8 overwrites", loc="left")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_scale")


# ------------------------------------------------------------------ Figure 4: mechanism
def fig_mech():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.25), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) swap experiment, Qwen3-4B thinking mode, LongMemEval (n=78) (RESULTS.md "E1c 交换实验 + E11")
    groups = [("thinking from the\nchronological run", [80.8, 80.8, 75.6]),
              ("thinking from the\nnewest-first run", [57.7, 55.1, 52.6])]
    bars(a, groups, None or ["c", "r", "n"], [BLUE, ORANGE, GRAY],
         ["input shown chronologically", "input shown newest first", "input removed"], ylim=(0, 128), delta_from=None)
    a.set_yticks(range(0, 101, 20))
    a.set_ylabel("% correct"); a.set_title("(a) The answer follows the thinking, not the input", loc="left")
    a.legend(loc="upper right", fontsize=6.6, ncol=1, borderaxespad=0.2)
    # (b) probe vs behaviour, shuffled CoT + step numbers (RESULTS.md "E7 补充")
    names = ["base", "decoupled\nLoRA", "chronological-\nonly LoRA"]
    probe = [98.8, 97.5, 98.1]
    stale = [12.2, 4.5, 31.8]
    x = np.arange(3); w = 0.38
    b.bar(x - w / 2, probe, w * 0.9, color=BLUE, label="linear probe: which value is newer")
    b.bar(x + w / 2, stale, w * 0.9, color=ORANGE, label="behaviour: answers the stale value")
    for xi, v in zip(x, stale):
        b.text(xi + w / 2, v + 2, f"{v:.0f}", ha="center", fontsize=7, color=INK)
    b.set_xticks(x, names); b.set_ylim(0, 140); b.set_yticks(range(0, 101, 20)); b.set_ylabel("%")
    b.set_title("(b) Known but not used (Qwen3-4B, shuffled CoT)", loc="left")
    b.legend(loc="upper center", fontsize=6.6, ncol=1, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_mech")


# ------------------------------------------------------------------ Figure 5: fixes
def fig_fix():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [1.42, 1.08]})
    # (a) selection stage: answer-only LoRA, pre-registered, Qwen3-4B seed 0 (RESULTS.md "LoRA 修复"; E2 row from E2 table)
    groups = [("logs\nnewest-first", [70.3, 89.3, 100.0]),
              ("CoT rev.\n+ steps", [6.0, 2.0, 74.7]),
              ("CoT shuf.\n+ steps", [34.7, 27.3, 45.7]),
              ("list\nnew→old", [57.1, 51.4, 77.1]),
              ("LongMem-\nEval rev.", [51.3, 44.9, 48.7]),
              ("Temp-\nReason rev.", [70.3, 78.3, 71.3])]
    w = 0.27
    for gi, (g, v) in enumerate(groups):
        for ci, (val, col, lab) in enumerate(zip(v, [GRAY, ORANGE, BLUE], ["base", "chronological-only LoRA", "decoupled LoRA"])):
            a.bar(gi + (ci - 1) * w, val, w * 0.9, color=col, label=lab if gi == 0 else None)
        a.text(gi, max(v) + 3, fmt_delta(v[2] - v[1]), ha="center", fontsize=6.8, color=INK)
    a.set_xticks(range(len(groups)), [g for g, _ in groups]); a.tick_params(axis="x", labelsize=5.8)
    a.set_ylim(0, 140); a.set_yticks(range(0, 101, 20)); a.set_ylabel("% correct")
    a.set_title("(a) Selection: decoupled training (answer only)", loc="left")
    a.legend(loc="upper center", fontsize=6.6, ncol=3, borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    # (b) dated list inside thinking (E18, E18.1), thinking on; base without thinking (desktop HF). E18: RESULTS.md
    # "E18 · 清单放进思考模式：全部测试汇总"; E18.1 (cloud HF, seed 0): CLOUD_NOTEBOOK.md "P7 · E18.1 测试结果"
    groups = [("newest first", [77.4, 96.8, 96.8]),
              ("newest first", [36.2, 71.7, 77.5]),
              ("retrieval", [56.2, 70.0, 77.9]),
              ("newest first", [73.9, 56.7, 71.0])]
    for xc, ds in ((0, "ConvoMem-long"), (1.5, "MemConflict"), (3, "PersonaMem")):
        b.text(xc, -0.13, ds, transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=6.4, color=INK)
    for gi, (g, v) in enumerate(groups):
        for ci, (val, col, lab) in enumerate(zip(v, [GRAY, AQUA, BLUE], ["base", "E18 (first version)", "E18.1 (final)"])):
            b.bar(gi + (ci - 1) * w, val, w * 0.9, color=col, label=lab if gi == 0 else None)
        b.text(gi, max(v) + 3, fmt_delta(v[2] - v[0]), ha="center", fontsize=6.8, color=INK)
    b.legend(loc="upper center", fontsize=6.6, ncol=3, borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    b.set_xticks(range(len(groups)), [g for g, _ in groups]); b.tick_params(axis="x", labelsize=5.8)
    b.set_ylim(0, 140); b.set_yticks(range(0, 101, 20))
    b.set_title("(b) Dated list in thinking (held-out sets)", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_fix")


def reason_acc(tag):
    rows = [r for r in jl(f"results/explore_reason_{tag}_judged.jsonl") if r["task"] == "lme" and r["cond"] == "S_rev_dated"]
    return 100 * np.mean([r["correct"] for r in rows]) if rows else np.nan


def numbers():
    """print the computed numbers (and paired CIs) used in the text"""
    q4, lq4, lphi = extmem("Qwen3-4B"), longconv("Qwen3-4B"), longconv("Phi-4-mini")
    for name, d, task in (("MemConflict 4B", q4, "memconf"), ("LoCoMo 4B", q4, "locomo"), ("MAB-CR 4B", q4, "mabcr"),
                          ("ConvoMem-long 4B", lq4, "convo_long"), ("ConvoMem-long Phi", lphi, "convo_long"),
                          ("PersonaMem 4B", lq4, "personamem"), ("PersonaMem Phi", lphi, "personamem"),
                          ("MemConflict Phi", extmem("Phi-4-mini"), "memconf")):
        line = f"{name:20s} " + " ".join(f"{c} {acc(d, task, c):5.1f}" for c in ("chrono", "rev", "retr"))
        for c in ("rev", "retr"):
            m, lo, hi, n = paired(d, task, c, "chrono")
            if n:
                line += f" | {c}-chrono {m:+.1f} [{lo:+.1f}, {hi:+.1f}] n={n}"
        print(line)


if __name__ == "__main__":
    if sys.argv[1:]:
        for name in sys.argv[1:]:
            globals()[name]()
    else:
        fig_concept(); fig_cot(); fig_apps(); fig_scale(); fig_mech(); fig_fix(); numbers()

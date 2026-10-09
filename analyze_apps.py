"""Aggregate the four application tests into tables (markdown to stdout) and results/apps_summary.json.
Paired contrasts use item-level bootstrap (2000 resamples) 95% CIs. Prints no dataset conversation text."""
import json
import os
import re

import numpy as np
import pandas as pd

from paths import LONGMEMEVAL

MODELS = ["Qwen3-4B", "Qwen3-14B", "OLMo-2-13B-Instruct", "Phi-4-mini"]
RNG = np.random.default_rng(0)


def boot_diff(a, b, n=2000):
    """mean(a - b) over paired items with a bootstrap 95% CI (percentage points)."""
    d = (np.asarray(a, float) - np.asarray(b, float)) * 100
    if len(d) == 0:
        return None
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return dict(diff=round(float(d.mean()), 1), lo=round(float(np.percentile(m, 2.5)), 1), hi=round(float(np.percentile(m, 97.5)), 1), n=len(d))


def fmt_ci(r):
    return "—" if r is None else f"{r['diff']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]"


def memory():
    out = {}
    for m in MODELS:
        p = f"results/app_memory_{m}_judged.jsonl"
        if not os.path.exists(p):
            continue
        df = pd.read_json(p, lines=True)
        df = df[~df.skipped]
        df["correct"], df["stale"] = df.label.eq("A"), df.label.eq("B")
        tab = df.groupby("cond").agg(correct=("correct", "mean"), stale=("stale", "mean"), n=("qid", "size"))
        piv_c = df.pivot_table(index="qid", columns="cond", values="correct")
        piv_s = df.pivot_table(index="qid", columns="cond", values="stale")
        contrasts = {}
        for a, b, name in [("S_rev_dated", "S_chrono_dated", "session: newest-first vs chronological (dated)"),
                           ("S_rev_nodate", "S_chrono_nodate", "session: newest-first vs chronological (no dates)"),
                           ("S_rev_dated_header", "S_rev_dated", "session: + 'most recent first' header"),
                           ("T_oldlast_dated", "T_newlast_dated", "turn: old evidence last vs new last (dated)"),
                           ("T_oldlast_nodate", "T_newlast_nodate", "turn: old evidence last vs new last (no dates)"),
                           ("T_oldlast_dated_header", "T_oldlast_dated", "turn: + 'sorted by relevance' header")]:
            if a in piv_c and b in piv_c:
                both = piv_c[[a, b]].dropna().index
                contrasts[name] = dict(correct=boot_diff(piv_c.loc[both, a], piv_c.loc[both, b]),
                                       stale=boot_diff(piv_s.loc[both, a], piv_s.loc[both, b]))
        out[m] = dict(table={c: dict(correct=round(r.correct * 100, 1), stale=round(r.stale * 100, 1), n=int(r.n))
                             for c, r in tab.iterrows()}, contrasts=contrasts)
        print(f"\n### Memory (LongMemEval KU) — {m}\n")
        print("| condition | correct % | stale % | n |\n|---|---|---|---|")
        for c, r in tab.iterrows():
            print(f"| {c} | {r.correct * 100:.1f} | {r.stale * 100:.1f} | {r.n} |")
        print("\n| contrast | Δ correct (pp) | Δ stale (pp) |\n|---|---|---|")
        for k, v in contrasts.items():
            print(f"| {k} | {fmt_ci(v['correct'])} | {fmt_ci(v['stale'])} |")
    # judge sanity check: short reference answers literally contained in the response vs judge label A
    rows = []
    for m in MODELS:
        p = f"results/app_memory_{m}_judged.jsonl"
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    if rows:
        items = {d["question_id"]: str(d["answer"]) for d in json.load(open(
            LONGMEMEVAL, encoding="utf-8")) if d["question_type"] == "knowledge-update"}
        norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
        agree, n = 0, 0
        for r in rows:
            if r.get("skipped"):
                continue
            ans = norm(items[r["qid"]])
            if 0 < len(ans) <= 3:
                hit = " ".join(ans) in " ".join(norm(r["response"]))
                agree += hit == (r["label"] == "A")
                n += 1
        out["judge_check"] = dict(n=n, agreement=round(agree / max(n, 1) * 100, 1))
        print(f"\njudge vs string match (reference answers of <= 3 words): agreement {agree / max(n, 1):.1%} on n={n}")
    return out


def logs():
    out = {}
    for m in MODELS:
        p = f"results/app_logs_{m}.jsonl"
        if not os.path.exists(p):
            continue
        df = pd.read_json(p, lines=True)
        g = df.groupby(["fmt", "order"]).agg(correct=("correct", "mean"), stale=("stale", "mean"),
                                             last_shown=("pick_last_shown", "mean"), first_shown=("pick_first_shown", "mean"))
        byk = df.groupby(["order", "k"]).correct.mean().unstack("k")
        piv = df.pivot_table(index=["seed", "fmt"], columns="order", values="correct")
        pivs = df.pivot_table(index=["seed", "fmt"], columns="order", values="stale")
        con = dict(correct=boot_diff(piv.newest_first, piv.oldest_first), stale=boot_diff(pivs.newest_first, pivs.oldest_first))
        out[m] = dict(table={f"{f}|{o}": {k: round(v * 100, 1) for k, v in r.items()} for (f, o), r in g.iterrows()},
                      by_k={o: {int(k): round(v * 100, 1) for k, v in r.items()} for o, r in byk.iterrows()}, contrast=con)
        print(f"\n### Logs — {m}\n")
        print("| format | order | correct % | stale % | = last shown % | = first shown % |\n|---|---|---|---|---|---|")
        for (f, o), r in g.iterrows():
            print(f"| {f} | {o} | {r.correct * 100:.1f} | {r.stale * 100:.1f} | {r.last_shown * 100:.1f} | {r.first_shown * 100:.1f} |")
        print("\nby k (correct %):", {o: {int(k): round(v * 100, 1) for k, v in r.items()} for o, r in byk.iterrows()})
        print("newest-first minus oldest-first:", "correct", fmt_ci(con["correct"]), "| stale", fmt_ci(con["stale"]))
    return out


def agent():
    out = {}
    for m in MODELS:
        p = f"results/app_agent_{m}.jsonl"
        if not os.path.exists(p):
            continue
        df = pd.read_json(p, lines=True)
        g = df.groupby(["cond", "k"]).agg(correct=("correct", "mean"), initial=("picked_initial", "mean"), stale=("stale", "mean"))
        piv = df.pivot_table(index=["seed", "k"], columns="cond", values="correct")
        pivs = df.pivot_table(index=["seed", "k"], columns="cond", values="stale")
        con = {c: dict(correct=boot_diff(piv[c], piv["base"]), stale=boot_diff(pivs[c], pivs["base"]))
               for c in ["stale_mem_end", "stale_mem_start", "stale_mem_end_flag", "fresh_mem_end"] if c in piv}
        out[m] = dict(table={f"{c}|k={k}": {kk: round(v * 100, 1) for kk, v in r.items()} for (c, k), r in g.iterrows()}, contrasts=con)
        print(f"\n### Agent trajectory — {m}\n")
        print("| condition | k | correct % | = initial value % | any stale value % |\n|---|---|---|---|---|")
        for (c, k), r in g.iterrows():
            print(f"| {c} | {k} | {r.correct * 100:.1f} | {r.initial * 100:.1f} | {r.stale * 100:.1f} |")
        print("\nvs base:", {c: fmt_ci(v["correct"]) for c, v in con.items()})
    return out


def main():
    res = dict(memory=memory(), logs=logs(), agent=agent())
    json.dump(res, open("results/apps_summary.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

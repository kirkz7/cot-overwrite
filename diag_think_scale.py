"""CLOUD_PLAN P8 (10-08): does thinking alone remove the MemConflict order effect, and how does that change with size?
Criteria = EXPLORE_PLAN "10-06 复盘新增" C (written 10-06, before any MemConflict output of a base model above 4B with
thinking on), applied to every size; P8 adds only the deadline rule (fewer than 120 paired items: descriptive only).
For each pair <thinking-on tag>:<thinking-off tag> (same model, same engine), on the items with chrono and rev in both runs:
  E_think = acc(rev) - acc(chrono) with thinking on (judge parse v2), paired bootstrap 95% CI (items; users clustered too)
  E_off   = the same with thinking off, on the same items;  D = E_think - E_off
  unified threshold (thinking on): REPLICATES if E_think <= -10 and its CI excludes 0, else not replicated
  thinking: REMOVES if E_think > -5 and its CI includes 0; else WEAKENS if D >= +10 and its CI excludes 0; else NO CLEAR EFFECT
  annotations: fewer than 120 paired items -> descriptive only; more than 5% unfinished thoughts -> flagged (C planned a
  2048-token supplement for that case; there is no time for it on this lease)
Also reports retr when present (same rules), stale (judge B) rates and the share of unfinished thoughts. Aggregates only.
usage: python diag_think_scale.py [--results results] [--models Qwen3-8B-bf16~vllm,Qwen3-14B-bf16~vllm,Qwen3-32B~vllm]
       python diag_think_scale.py --pairs "Qwen3-4B+think:Qwen3-4B"   (explicit pairs, e.g. the desktop HF 4B)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

B = 10000


def load(path):
    if not os.path.exists(path) or not os.path.getsize(path):
        return pd.DataFrame()
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return pd.DataFrame(rows)


def boot(d, groups=None, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    else:
        g = pd.Series(list(groups)).astype("category").cat.codes.values
        k = g.max() + 1
        s, c = np.bincount(g, weights=d, minlength=k), np.bincount(g, minlength=k)
        idx = rng.integers(0, k, (B, k))
        m = s[idx].sum(1) / c[idx].sum(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def fmt(t):
    return f"{t[0]:+6.1f} [{t[1]:+6.1f}, {t[2]:+6.1f}]"


def labels(res, tag, judged):
    raw = load(os.path.join(res, f"extmem_{tag}.jsonl"))
    if raw.empty:
        return None, None
    raw = raw[raw.task == "memconf"]
    j = pd.DataFrame()
    for suffix in judged:
        j = load(os.path.join(res, f"extmem_{tag}{suffix}"))
        if not j.empty:
            break
    if j.empty:
        return raw, None
    j = j[j.task == "memconf"][["id", "cond", "label"]]
    df = raw.merge(j, on=["id", "cond"])
    return raw, df.pivot_table(index="id", columns="cond", values="label", aggfunc="first")


def verdict(e, d, n, unfinished):
    unified = "REPLICATES" if e[0] <= -10 and e[2] < 0 else "not replicated"
    if e[0] > -5 and e[1] <= 0 <= e[2]:
        think = "REMOVES"
    elif d[0] >= 10 and d[1] > 0:
        think = "WEAKENS"
    else:
        think = "NO CLEAR EFFECT"
    notes = (["descriptive only: n < 120"] if n < 120 else []) + (["flag: unfinished > 5%"] if unfinished > 5 else [])
    return f"unified {unified} | thinking {think}" + (" | " + "; ".join(notes) if notes else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="Qwen3-8B-bf16~vllm,Qwen3-14B-bf16~vllm,Qwen3-32B~vllm")
    ap.add_argument("--pairs", default="", help="comma list of <thinking-on tag>:<thinking-off tag>")
    args = ap.parse_args()
    pairs = [p.split(":") for p in args.pairs.split(",")] if args.pairs else [(f"{m}+think", m) for m in args.models.split(",")]
    for on_tag, off_tag in pairs:
        raw_on, on = labels(args.results, on_tag, ["_judged2.jsonl"])
        _, off = labels(args.results, off_tag, ["_judged2.jsonl", "_judged.jsonl"])
        print(f"== {on_tag}  vs  {off_tag}")
        if on is None or off is None:
            print("   missing:", "thinking-on judged (parse v2)" if on is None else "thinking-off judged")
            continue
        conds = [c for c in ("chrono", "rev", "retr") if c in on.columns and c in off.columns]
        ids = on.dropna(subset=["chrono", "rev"]).index.intersection(off.dropna(subset=["chrono", "rev"]).index)
        full = raw_on[raw_on.id.isin(ids)]["full"].astype(str)
        unfinished = (full.str.contains("<think>") & ~full.str.contains("</think>")).mean() * 100
        lab = lambda t, x: (t.loc[ids] == x).astype(float).where(t.loc[ids].notna())   # NaN = order not run (deadline)
        a_on, a_off, s_on, s_off = lab(on, "A"), lab(off, "A"), lab(on, "B"), lab(off, "B")
        cl = ids.str[:8]
        print(f"   paired items n={len(ids)} (users {cl.nunique()}); unfinished thoughts {unfinished:.1f}%")
        print("   thinking on : acc " + " / ".join(f"{c} {a_on[c].mean() * 100:5.1f}" for c in conds) +
              " | stale " + " / ".join(f"{s_on[c].mean() * 100:4.1f}" for c in conds))
        print("   thinking off: acc " + " / ".join(f"{c} {a_off[c].mean() * 100:5.1f}" for c in conds) +
              " | stale " + " / ".join(f"{s_off[c].mean() * 100:4.1f}" for c in conds))
        for c in [c for c in conds if c != "chrono"]:
            sub = a_on[[c, "chrono"]].dropna().index.intersection(a_off[[c, "chrono"]].dropna().index)
            e_on = boot(a_on.loc[sub, c] - a_on.loc[sub, "chrono"])
            e_on_cl = boot(a_on.loc[sub, c] - a_on.loc[sub, "chrono"], sub.str[:8])
            e_off = boot(a_off.loc[sub, c] - a_off.loc[sub, "chrono"])
            d = boot((a_on.loc[sub, c] - a_on.loc[sub, "chrono"]) - (a_off.loc[sub, c] - a_off.loc[sub, "chrono"]))
            print(f"   {c}-chrono (n={len(sub)}): E_think {fmt(e_on)} (users {fmt(e_on_cl)}) | E_off {fmt(e_off)} | D {fmt(d)}")
            print(f"      verdict ({c}): {verdict(e_on, d, len(sub), unfinished)}")


if __name__ == "__main__":
    main()

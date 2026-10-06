"""Review 10-06 (diagnosis, no scoring change): is the MemConflict order effect smaller at 32B than at 4B, and is the
judge equally valid on both? Same items, same engine (cloud vLLM), same Qwen3-14B 4-bit judge (parse v1; v2 identical).
Aggregates only, no conversation or response text is printed (MemConflict and LoCoMo are held out).
Runs without torch (reads jsonl only), so it also works on a CPU-only checkout of the cloud-l20 branch.
usage: python diag_scale.py [--results results] [--models Qwen3-32B~vllm,Qwen3-4B~vllm]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

B = 10000


def load(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def key(v):   # explore_extmem.key
    return str(v).split(",")[0].strip().lower()


def boot(d, groups=None, seed=0):
    """mean of d with a 95% bootstrap interval, over items or (groups given) over clusters"""
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
    return f"{d.mean() * 100:+6.1f} [{np.percentile(m, 2.5) * 100:+6.1f}, {np.percentile(m, 97.5) * 100:+6.1f}]"


def frame(res, model, task="memconf"):
    raw = load(f"{res}/extmem_{model}.jsonl")
    j = load(f"{res}/extmem_{model}_judged.jsonl")
    raw, j = raw[raw.task == task], j[j.task == task][["id", "cond", "label"]]
    df = raw.merge(j, on=["id", "cond"])
    df["ok"], df["stale"] = (df.label == "A").astype(float), (df.label == "B").astype(float)
    df["cluster"] = df.id.str[:8] if task == "memconf" else df.id.str.rsplit("-", n=1).str[0]
    return df


def piv(df, col):
    return df.pivot_table(index="id", columns="cond", values=col, aggfunc="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="Qwen3-32B~vllm,Qwen3-4B~vllm", help="the first two are compared")
    args = ap.parse_args()
    models = args.models.split(",")
    fr = {m: frame(args.results, m) for m in models}
    for m, df in fr.items():
        a, s = piv(df, "ok"), piv(df, "stale")
        cl = a.index.str[:8]
        print(f"== MemConflict {m}: n={len(a)}, users={cl.nunique()}")
        print("   acc   " + "  ".join(f"{c} {a[c].mean() * 100:5.1f}" for c in ("chrono", "rev", "retr")) +
              " | stale " + "  ".join(f"{c} {s[c].mean() * 100:4.1f}" for c in ("chrono", "rev", "retr")))
        for c in ("rev", "retr"):
            print(f"   {c}-chrono acc {boot(a[c] - a.chrono)} cluster {boot(a[c] - a.chrono, cl)} | "
                  f"stale {boot(s[c] - s.chrono)} cluster {boot(s[c] - s.chrono, cl)}")
        w = piv(df, "label")
        ok = w[w.chrono == "A"]
        print(f"   of the {len(ok)} items right in chrono, stale in rev {(ok.rev == 'B').mean() * 100:.1f}%, "
              f"in retr {(ok.retr == 'B').mean() * 100:.1f}%")
        r = df[df.cond == "retr"]
        first = r.pos_new == 0
        print(f"   retr, update session shown first (n={int(first.sum())}): acc {r[first].ok.mean() * 100:.1f}; "
              f"otherwise (n={int((~first).sum())}): {r[~first].ok.mean() * 100:.1f}")
        nt = df[df.cond == "chrono"].set_index("id").n_tok
        q = pd.qcut(nt, 3, labels=["short", "mid", "long"])
        print("   rev-chrono by length tercile: " + ", ".join(
            f"{lv} (median {int(nt[q == lv].median())} tok) {(a.loc[q[q == lv].index, 'rev'] - a.loc[q[q == lv].index, 'chrono']).mean() * 100:+.1f}"
            for lv in ("short", "mid", "long")))
        # judge validity, same subset as the paper (responses naming both values; string rule = which value comes first)
        t = df.response.astype(str).str.lower()
        o = np.array([x.find(key(v)) for x, v in zip(t, df.old)])
        n = np.array([x.find(key(v)) for x, v in zip(t, df.new)])
        both = (o >= 0) & (n >= 0)
        agree = (np.where(o < n, "A", "B")[both] == df.label.values[both]).mean() * 100
        print(f"   response words median {df.response.astype(str).str.split().str.len().median():.0f}, "
              f"max full words {df['full'].astype(str).str.split().str.len().max()}; "
              f"judge = string (which value first) on the {int(both.sum())} responses naming both: {agree:.1f}%")
    if len(models) >= 2:
        a, b = (piv(fr[m], "ok") for m in models[:2])
        sa, sb = (piv(fr[m], "stale") for m in models[:2])
        ids = a.index.intersection(b.index)
        print(f"== {models[0]} effect minus {models[1]} effect, same {len(ids)} items (+ = smaller drop / smaller stale rise)")
        for c in ("rev", "retr"):
            d = (a.loc[ids, c] - a.loc[ids, "chrono"]) - (b.loc[ids, c] - b.loc[ids, "chrono"])
            e = (sa.loc[ids, c] - sa.loc[ids, "chrono"]) - (sb.loc[ids, c] - sb.loc[ids, "chrono"])
            print(f"   {c}: acc {boot(d)} cluster {boot(d, ids.str[:8])} | stale {boot(-e)} cluster {boot(-e, ids.str[:8])}")
    for m in models:
        lc = frame(args.results, m, "locomo")
        a = piv(lc, "ok").dropna()
        if len(a) < 50:
            continue
        cl = a.index.str.rsplit("-", n=1).str[0]
        print(f"== LoCoMo {m}: n={len(a)}, conversations={cl.nunique()} | " +
              "  ".join(f"{c} {a[c].mean() * 100:5.1f}" for c in ("chrono", "rev", "retr")))
        for c in ("rev", "retr"):
            print(f"   {c}-chrono {boot(a[c] - a.chrono)} cluster {boot(a[c] - a.chrono, cl)}")


if __name__ == "__main__":
    main()

"""PREREG_TRAINING.md analysis: decoupled LoRA vs chrono-only control vs base, paired bootstrap per condition.

usage: python analyze_lora.py --base Qwen3-4B --runs q4-dec-s0 q4-chr-s0 [--runs ...]
With several seeds (--dec q4-dec-s0,q4-dec-s1 ...) the per-item accuracy is averaged over seeds first.
Writes results/lora_analysis_{base}.csv and prints a table. Holm correction over the primary family:
every decoupled-minus-control difference on a non-chronological, non-trained-cue condition, plus the
LongMemEval S_rev_dated decoupled-minus-base difference.
"""
import argparse
import os

import numpy as np
import pandas as pd

from app_common import load_jsonl

R = "results"
TRAINED_CUE = {("memory", "S_rev_dated_header"), ("cot", "rev_header")}
# chronological (side-effect) conditions: never part of the primary family
CHRONO = {"S_chrono_dated", "S_chrono_nodate", "T_newlast_dated", "T_newlast_nodate", "oldest_first", "base",
          "fresh_mem_end", "original", "sorted", "full", "qa1", "qa2", "qa3", "all"}


def load(test, tag):
    """-> DataFrame with columns key, cond, correct"""
    def f(name):
        p = os.path.join(R, name)
        return pd.DataFrame(load_jsonl(p)) if os.path.exists(p) else None
    if test == "memory":
        df = f(f"app_memory_{tag}_judged.jsonl")
        if df is None or "label" not in df:
            return None
        df = df[~df.skipped]
        return pd.DataFrame(dict(key=df.qid, cond=df.cond, correct=df.label.eq("A")))
    if test == "logs":
        df = f(f"app_logs_{tag}.jsonl")
        return None if df is None else pd.DataFrame(dict(key=df.seed.astype(str) + df.fmt, cond=df.order, correct=df.correct))
    if test == "agent":
        df = f(f"app_agent_{tag}.jsonl")
        return None if df is None else pd.DataFrame(dict(key=df.seed.astype(str), cond=df.cond, correct=df.correct))
    df = f(f"ext_{test}_{tag}.jsonl")
    if df is None:
        return None
    if test == "mab":
        src = df.src.str.extract(r"_(sh|mh)_")[0]
        return pd.DataFrame(dict(key=src + df.qi.astype(str), cond=src + ":" + df.order, correct=df.correct))
    if test == "tot":
        return pd.DataFrame(dict(key=df.qtype + df.question, cond=df.order, correct=df.correct))
    if test == "tempreason":
        return pd.DataFrame(dict(key=df["id"], cond=df.order, correct=df.correct))
    if test == "babilong":
        return pd.DataFrame(dict(key=df.i.astype(str), cond=df.task, correct=df.correct))
    if test == "cot":
        return pd.DataFrame(dict(key=df.k.astype(str) + "_" + df.i.astype(str), cond=df.cond, correct=df.correct))
    if test == "gsm8k":
        return pd.DataFrame(dict(key=df.i.astype(str), cond="all", correct=df.correct))


def load_avg(test, tags):
    """per-item accuracy averaged over seeds (only items present for every seed)"""
    dfs = [load(test, t) for t in tags]
    if any(d is None for d in dfs):
        return None
    df = pd.concat([d.assign(run=i) for i, d in enumerate(dfs)])
    g = df.groupby(["cond", "key"]).agg(correct=("correct", "mean"), n=("run", "nunique")).reset_index()
    return g[g.n == len(tags)].drop(columns="n")


def boot(a, b, n=10000, seed=0):
    """paired bootstrap of mean(a - b): (diff, lo, hi, two-sided p)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100, max(p, 1 / n)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen3-4B")
    ap.add_argument("--dec", default="q4-dec-s0", help="comma-separated run names (seeds)")
    ap.add_argument("--chr", default="q4-chr-s0")
    args = ap.parse_args()
    dec = [f"{args.base}+{r}" for r in args.dec.split(",")]
    chr_ = [f"{args.base}+{r}" for r in args.chr.split(",")]
    rows = []
    for test in ["memory", "logs", "agent", "mab", "tot", "tempreason", "babilong", "cot", "gsm8k"]:
        B, D, C = load_avg(test, [args.base]), load_avg(test, dec), load_avg(test, chr_)
        if B is None or D is None or C is None:
            print("missing", test)
            continue
        for cond in sorted(D.cond.unique()):
            m = (B[B.cond == cond].merge(D[D.cond == cond], on="key", suffixes=("_b", ""))
                 .merge(C[C.cond == cond], on="key", suffixes=("_d", "_c")))
            if not len(m):
                continue
            dc = boot(m.correct_d, m.correct_c)
            db = boot(m.correct_d, m.correct_b)
            cb = boot(m.correct_c, m.correct_b)
            kind = ("trained_cue" if (test, cond) in TRAINED_CUE else
                    "chrono" if cond in CHRONO or cond.endswith(":original") else "reordered")
            rows.append(dict(test=test, cond=cond, kind=kind, n=len(m), base=m.correct_b.mean() * 100,
                             dec=m.correct_d.mean() * 100, chr=m.correct_c.mean() * 100,
                             d_c=dc[0], d_c_lo=dc[1], d_c_hi=dc[2], d_c_p=dc[3],
                             d_b=db[0], d_b_lo=db[1], d_b_hi=db[2], d_b_p=db[3],
                             c_b=cb[0], c_b_lo=cb[1], c_b_hi=cb[2]))
    df = pd.DataFrame(rows)
    # primary family: dec-chr on every reordered condition + the LongMemEval dec-base endpoint
    fam = [(i, "d_c_p") for i in df.index[df.kind == "reordered"]]
    fam += [(i, "d_b_p") for i in df.index[(df.test == "memory") & (df.cond == "S_rev_dated")]]
    adj = holm(np.array([df.at[i, c] for i, c in fam]))
    df["d_c_holm"], df["d_b_holm"] = np.nan, np.nan
    for (i, c), a in zip(fam, adj):
        df.at[i, c.replace("_p", "_holm")] = a
    suffix = "" if args.dec.startswith("q4-dec") else "_" + args.dec.replace(",", "+")   # keep the pre-registered file name
    out = os.path.join(R, f"lora_analysis_{args.base}_{len(dec)}seed{suffix}.csv")
    df.round(4).to_csv(out, index=False)
    pd.set_option("display.width", 250)
    show = df[["test", "cond", "kind", "n", "base", "dec", "chr", "d_c", "d_c_lo", "d_c_hi", "d_c_holm",
               "d_b", "d_b_lo", "d_b_hi", "d_b_holm"]]
    print(show.round(1).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()

"""Exploration E4 (EXPLORE_PLAN.md): recency or primacy as the trace gets long (link to Guo et al. 2026, 2609.38866).

Same task as Exp 1 / 8 (a program of assignment lines plus the CoT trace of its values), but the trace is padded
with distractor assignments up to ~1k / 4k / 8k / 16k tokens. The target is assigned k+1 times (k = 2, 8).
In the chronological trace ("full") the last value is correct, so an error that picks the first value is primacy;
in the shuffled trace ("shuf") we record whether the picked value is the last or the first one presented.
Distractor values may repeat among distractors but never equal a target value, and every name is distinct.
Exploratory; the threshold is written in EXPLORE_PLAN.md before running.
usage: python explore_primacy.py run [--n 50]
       python explore_primacy.py stats
"""
import argparse
import random

import pandas as pd
from tqdm import tqdm

from app_common import JsonlAppender, load_jsonl
from probe import Scorer, load
from tasks import HI, LO, VAR_POOL, Example, Line, build_prompt, presented_lines

OUT = "results/explore_primacy_Qwen3-4B.jsonl"
LENGTHS = [72, 250, 500, 1000]           # lines; about 1k / 4k / 8k / 16k tokens with the program and the trace
KS = [2, 8]
NAMES = VAR_POOL + [a + b for a in VAR_POOL for b in "123456789"] + \
    [a + b + c for a in VAR_POOL for b in "123456789" for c in "123456789"]


def make_padded_example(k, seed, n_lines, max_step=30):
    rng = random.Random(seed)
    names = rng.sample(NAMES, n_lines - k)
    target, distractors = names[0], names[1:]
    while True:
        v0 = rng.randint(LO + 10, HI - 10)
        used, cur, t_lines = {v0}, v0, [Line(target, f"{target} = {v0}", f"{target} = {v0}", f"{target} = {v0}", v0)]
        for _ in range(k):
            opts = [v for v in range(max(LO, cur - max_step), min(HI, cur + max_step) + 1) if v not in used]
            if not opts:
                break
            new = rng.choice(opts)
            op, a = ("+", new - cur) if new > cur else ("-", cur - new)
            used.add(new)
            t_lines.append(Line(target, f"{target} = {target} {op} {a}", f"{target} = {new}",
                                f"{target} = {cur} {op} {a} = {new}", new))
            cur = new
        if len(t_lines) == k + 1:
            break
    free = [v for v in range(LO, HI + 1) if v not in used]
    d_lines = []
    for d in distractors:
        val, a, op = rng.choice(free), rng.randint(2, 9), rng.choice("+-")
        p = val - a if op == "+" else val + a
        d_lines.append(Line(d, f"{d} = {p} {op} {a}", f"{d} = {val}", f"{d} = {p} {op} {a} = {val}", val))
    slots = set(rng.sample(range(n_lines), k + 1))
    lines, ti, di = [], 0, 0
    for i in range(n_lines):
        if i in slots:
            lines.append(t_lines[ti]); ti += 1
        else:
            lines.append(d_lines[di]); di += 1
    return Example(k=k, target=target, control=rng.choice(distractors), lines=lines, seed=seed)


def run(args):
    tok, model = load("Qwen/Qwen3-4B")
    scorer = Scorer(tok, model)
    w = JsonlAppender(OUT, key=lambda r: (r["n_lines"], r["k"], r["seed"], r["cond"]))
    jobs = [(n, k, 1_300_000 + n * 1000 + k * 100 + i, cond)
            for n in LENGTHS for k in KS for i in range(args.n) for cond in ("full", "shuf")]
    todo = [j for j in jobs if j not in w.done]
    print("todo", len(todo), "of", len(jobs), flush=True)
    for n, k, seed, cond in tqdm(todo):
        ex = make_padded_example(k, seed, n)
        prompt = build_prompt(ex, ex.target, cond, "bare", tok)
        s = scorer.score(prompt)
        pred = max(s, key=s.get)
        hist = [l.value for l in ex.lines if l.var == ex.target]
        seen = [l.value for l in presented_lines(ex, cond) if l.var == ex.target]
        w.write(dict(n_lines=n, k=k, seed=seed, cond=cond, n_tok=len(tok(prompt).input_ids), pred=pred,
                     correct=pred == hist[-1], first=pred == hist[0], stale=pred in hist[:-1],
                     other=pred not in hist, pick_last_seen=pred == seen[-1], pick_first_seen=pred == seen[0]))
    w.close()


def stats(args):
    df = pd.DataFrame(load_jsonl(OUT))
    df["tokens"] = df.groupby("n_lines").n_tok.transform("median").astype(int)
    cols = ["correct", "first", "stale", "other", "pick_last_seen", "pick_first_seen"]
    t = (df.groupby(["cond", "k", "n_lines", "tokens"])[cols].mean() * 100).round(1)
    print(t.assign(n=df.groupby(["cond", "k", "n_lines", "tokens"]).size()).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats", "check"])
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    if args.stage == "check":                 # structure and lengths only
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        for n in LENGTHS:
            ex = make_padded_example(8, 1, n)
            hist = [l.value for l in ex.lines if l.var == ex.target]
            others = [l.value for l in ex.lines if l.var != ex.target]
            print(n, "lines | tokens", len(tok(build_prompt(ex, ex.target, "shuf", "bare", tok)).input_ids),
                  "| target values distinct", len(set(hist)) == len(hist),
                  "| no distractor shares a target value", not set(hist) & set(others),
                  "| names distinct", len({l.var for l in ex.lines}) == n - 8)
    else:
        {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()

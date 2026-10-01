"""Application 3: agent trajectories with context management.

An agent reads and edits a config file over several tool calls (each step numbered and timed).
Memory / context-management layers often insert a "relevant memory" block (a retrieved earlier
observation) right before the latest user turn. We test whether a STALE retrieved observation,
explicitly labeled with its (earlier) step, overrides the newer state in the transcript.

conditions:
  base               transcript only
  stale_mem_end      transcript + retrieved stale observation right before the question
  stale_mem_start    stale observation before the transcript
  stale_mem_end_flag stale observation at the end, labeled "may be outdated"
  fresh_mem_end      transcript + retrieved CURRENT observation right before the question (control)
k = number of edits to the queried key (1 or 3).
usage: python run_app_agent.py --models Qwen3-4B,Qwen3-14B --n 150
"""
import argparse
import random

import pandas as pd
from tqdm import tqdm

from app_common import tag, JsonlAppender, answer_fast, chat_prompt, first_number, free_gpu, load_reader

KEYS = ["max_connections", "pool_timeout", "statement_timeout", "idle_in_transaction_timeout", "max_overflow",
        "lock_timeout", "work_mem_mb", "checkpoint_interval"]
CONDS = ["base", "stale_mem_end", "stale_mem_start", "stale_mem_end_flag", "fresh_mem_end"]


def make_item(k, seed):
    rng = random.Random(seed)
    keys = rng.sample(KEYS, 5)
    target = keys[0]
    used = set()

    def val():
        while True:
            v = rng.randint(10, 999)
            if v not in used:
                used.add(v)
                return v
    cfg = {key: val() for key in keys}
    steps, snapshots = [], []          # snapshots: (step, dict) observed via read_file
    s = 1
    h, m = rng.randint(9, 16), rng.randint(0, 50)

    def stamp():
        nonlocal h, m
        m += rng.randint(1, 6)
        h, m = h + m // 60, m % 60
        return f"{h:02d}:{m:02d}"

    def read():
        nonlocal s
        body = "\n".join(f"{kk}: {vv}" for kk, vv in cfg.items())
        steps.append(f"[step {s} · {stamp()}] assistant → read_file(path=\"config/db.yaml\")\n[step {s}] tool ←\n```yaml\n{body}\n```")
        snapshots.append((s, dict(cfg))); s += 1

    history = [cfg[target]]

    def edit(key):
        nonlocal s
        old, new = cfg[key], val()
        cfg[key] = new
        if key == target:
            history.append(new)
        steps.append(f"[step {s} · {stamp()}] assistant → edit_file(path=\"config/db.yaml\", find=\"{key}: {old}\", replace=\"{key}: {new}\")\n[step {s}] tool ← OK (1 replacement)")
        s += 1

    def noise():
        nonlocal s
        steps.append(rng.choice([
            f"[step {s} · {stamp()}] assistant → run_tests(path=\"tests/db\")\n[step {s}] tool ← {rng.randint(10, 60)} passed, 0 failed",
            f"[step {s} · {stamp()}] assistant → read_file(path=\"README.md\")\n[step {s}] tool ← (412 lines)",
            f"[step {s} · {stamp()}] assistant → grep(pattern=\"TODO\", path=\"src/\")\n[step {s}] tool ← 3 matches"]))
        s += 1

    read()
    for i in range(k):
        noise()
        if rng.random() < 0.5:
            edit(rng.choice(keys[1:]))
        edit(target)
        if i < k - 1 or rng.random() < 0.5:
            read()                      # sometimes the newest value is only visible in the edit call
    noise()
    return dict(k=k, seed=seed, target=target, steps=steps, first_snap=snapshots[0], current=cfg[target],
                initial=snapshots[0][1][target], history=history, last_step=s - 1)


def build(item, cond):
    tr = "\n\n".join(item["steps"])
    snap_step, snap = item["first_snap"]
    stale = "\n".join(f"{k}: {v}" for k, v in snap.items())
    cur_lines = "\n".join(f"{k}: {v}" for k, v in snap.items() if k != item["target"]) + f"\n{item['target']}: {item['current']}"

    def mem(step, body, flag=False):
        note = " Note: this memory was saved at an earlier step and may be outdated." if flag else ""
        return f"Relevant memory (retrieved; saved at step {step}):{note}\n```yaml\n{body}\n```"
    head = "Below is the agent's work log for this task."
    q = f"What is the current value of {item['target']} in config/db.yaml? Answer with the number only."
    if cond == "base":
        ctx = f"{head}\n\n{tr}"
    elif cond == "stale_mem_end":
        ctx = f"{head}\n\n{tr}\n\n{mem(snap_step, stale)}"
    elif cond == "stale_mem_start":
        ctx = f"{mem(snap_step, stale)}\n\n{head}\n\n{tr}"
    elif cond == "stale_mem_end_flag":
        ctx = f"{head}\n\n{tr}\n\n{mem(snap_step, stale, flag=True)}"
    else:
        ctx = f"{head}\n\n{tr}\n\n{mem(item['last_step'], cur_lines)}"
    return ctx + "\n\n" + q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-14B,OLMo-2-13B-Instruct,Phi-4-mini")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    items = [make_item(k, 8_000_000 + k * 10_000 + i) for k in (1, 3) for i in range(args.n)]
    if args.show:
        it = items[args.n]
        print("current", it["current"], "initial", it["initial"])
        for c in ("stale_mem_end", "fresh_mem_end"):
            print("=" * 30, c, "\n", build(it, c))
        return
    jobs = [(it, cond) for it in items for cond in CONDS]
    for name in args.models.split(","):
        out = f"results/app_agent_{tag(name)}.jsonl"
        w = JsonlAppender(out, key=lambda r: (r["seed"], r["cond"]))
        todo = [j for j in jobs if (j[0]["seed"], j[1]) not in w.done]
        print(name, "done", len(jobs) - len(todo), "todo", len(todo), flush=True)
        if todo:
            tok, model = load_reader(name)
            for it, cond in tqdm(todo, desc=name):
                raw = answer_fast(tok, model, chat_prompt(tok, build(it, cond)))
                pred = first_number(raw)
                w.write(dict(model=name, k=it["k"], seed=it["seed"], cond=cond, raw=raw, pred=pred,
                             correct=pred == it["current"], picked_initial=pred == it["initial"],
                             stale=pred in it["history"][:-1]))
            del tok, model
            free_gpu()
        w.close()
        df = pd.DataFrame(w.rows)
        print(name, "\n", (df.groupby(["cond", "k"])[["correct", "picked_initial", "stale"]].mean().unstack("k") * 100).round(1).to_string(), flush=True)


if __name__ == "__main__":
    main()

"""Exploration E3 (EXPLORE_PLAN.md): do the recency heads found on CoT traces also drive the application failures?

Qwen3-4B. The 32 heads with the highest Exp 7 recency score (results/exp7_Qwen3-4B_attn.npz) are zeroed at the
input of o_proj, exactly as in Exp 14; the control zeroes 32 random heads (same seed as Exp 14).
Tasks (each in a chronological and a misleading order):
  logs    email threads, oldest_first vs newest_first          (run_app_logs.py items, email format)
  agent   stale memory at the start vs at the end of the log    (run_app_agent.py items)
  memory  LongMemEval sessions, chrono vs newest-first, dated   (run_app_memory.py items; judged by Qwen3-14B)
Exploratory; the threshold is written in EXPLORE_PLAN.md before running. No conversation text is printed.
usage: python explore_heads.py run [--n_logs 50 --n_agent 75]
       python explore_heads.py judge
       python explore_heads.py stats
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from app_common import JsonlAppender, answer_fast, chat_prompt, first_number, free_gpu, load_jsonl, load_reader
import run_app_agent as agent
import run_app_logs as logs
import run_app_memory as mem

OUT = "results/explore_heads.jsonl"
JUDGED = "results/explore_heads_judged.jsonl"
TOPN = 32


def head_sets(model):
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    score = np.load("results/exp7_Qwen3-4B_attn.npz")["score"]
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    top = [tuple(map(int, x)) for x in order[:TOPN]]
    rand = random.Random(0).sample([(l, h) for l in range(L) for h in range(H)], TOPN)
    return {"none": [], f"top{TOPN}": top, f"random{TOPN}": rand}


def install_hooks(model, ablate):
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    def make_hook(li):
        def hook(mod, inp):
            hs = ablate.get(li)
            if not hs:
                return None
            x = inp[0].clone()
            for h in hs:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        return hook
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))


def jobs(args):
    out = []
    for k in logs.KS:
        for i in range(args.n_logs):
            it = logs.make_item(k, 7_000_000 + k * 10_000 + i)
            for order in ("oldest_first", "newest_first"):
                out.append(("logs", f"{it['seed']}", order, it))
    for k in (1, 3):
        for i in range(args.n_agent):
            it = agent.make_item(k, 8_000_000 + k * 10_000 + i)
            for cond in ("stale_mem_start", "stale_mem_end"):
                out.append(("agent", f"{it['seed']}", cond, it))
    for it in mem.load_items():
        for cond in ("S_chrono_dated", "S_rev_dated"):
            out.append(("memory", it["qid"], cond, it))
    return out


@torch.no_grad()
def run(args):
    w = JsonlAppender(OUT, key=lambda r: (r["task"], r["id"], r["cond"], r["ablation"]))
    tok, model = load_reader("Qwen3-4B")
    ablate = {}
    install_hooks(model, ablate)
    all_jobs = jobs(args)
    for abl, heads in head_sets(model).items():
        ablate.clear()
        for l, h in heads:
            ablate.setdefault(l, []).append(h)
        todo = [j for j in all_jobs if (j[0], j[1], j[2], abl) not in w.done]
        print(abl, "todo", len(todo), flush=True)
        for task, iid, cond, it in tqdm(todo, desc=abl):
            rec = dict(task=task, id=iid, cond=cond, ablation=abl)
            if task == "logs":
                user, shown = logs.render(it, "email", cond)
                pred = first_number(answer_fast(tok, model, chat_prompt(tok, user)))
                h = it["history"]
                rec.update(pred=pred, correct=pred == h[-1], stale=pred in h[:-1])
            elif task == "agent":
                pred = first_number(answer_fast(tok, model, chat_prompt(tok, agent.build(it, cond))))
                rec.update(pred=pred, correct=pred == it["current"], stale=pred in it["history"][:-1])
            else:
                rec.update(response=answer_fast(tok, model, chat_prompt(tok, mem.build(it, cond)), max_new=64))
                torch.cuda.empty_cache()
            w.write(rec)
    w.close()
    del tok, model
    free_gpu()


@torch.no_grad()
def judge(args):
    items = {it["qid"]: it for it in mem.load_items()}
    rows = [r for r in load_jsonl(OUT) if r["task"] == "memory"]
    w = JsonlAppender(JUDGED, key=lambda r: (r["id"], r["cond"], r["ablation"]))
    todo = [r for r in rows if (r["id"], r["cond"], r["ablation"]) not in w.done]
    print("judge todo", len(todo), flush=True)
    if todo:
        tok, model = load_reader("Qwen3-14B")
        for r in tqdm(todo):
            it = items[r["id"]]
            text = mem.JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"] or "(not available)",
                                    new=it["new_ev"] or "(not available)", q=it["question"], a=it["answer"],
                                    r=r["response"])
            lp = mem.letter_logprobs(tok, model, text)
            label = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(dict(r, label=label, correct=label == "A", stale=label == "B"))
        del tok, model
        free_gpu()
    w.close()


def stats(args):
    df = pd.DataFrame(load_jsonl(OUT))
    df = df[df.task != "memory"]
    j = load_jsonl(JUDGED)
    if j:
        df = pd.concat([df, pd.DataFrame(j)])
    good = {"logs": "oldest_first", "agent": "stale_mem_start", "memory": "S_chrono_dated"}
    rows = []
    for (task, abl), g in df.groupby(["task", "ablation"]):
        a = g[g.cond == good[task]].correct.mean() * 100
        b = g[g.cond != good[task]].correct.mean() * 100
        s = g[g.cond != good[task]].stale.mean() * 100
        rows.append(dict(task=task, ablation=abl, n=len(g) // 2, acc_good_order=a, acc_misleading=b,
                         position_effect=a - b, stale_misleading=s))
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats"])
    ap.add_argument("--n_logs", type=int, default=50, help="items per k (k = 1, 2, 4)")
    ap.add_argument("--n_agent", type=int, default=75, help="items per k (k = 1, 3)")
    args = ap.parse_args()
    {"run": run, "judge": judge, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()

"""New clean held-out tests from the unused parts of ConvoMem (CLOUD_NOTEBOOK.md pre-registration, 10-08).
  c2x    changing_evidence/2_evidence, held-out personas 50-99, items 5+ of each persona (ConvoMem-long and the E12
         reason eval used only the first 4); 300 sampled
  c3-c6  changing_evidence/3_evidence .. 6_evidence (one value updated 3-6 times), only the personIds of held-out
         personas 50-99; 150 sampled per k
Built like explore_longconv.convo_long: k dated conversations (increasing dates in 2024, "Current date: 2024-12-01"),
shown chronological or newest first; judged by Qwen3-14B 4-bit with run_app_memory.JUDGE (old = second-to-last
evidence, new = last). Frozen held-out: print aggregates only.
usage: python explore_convo_heldout.py check | run [--think --budget 1024] --models ... | judge | stats
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from app_common import JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader, out_tag, strip_think
import run_app_memory as mem
from explore_probe import CONVO_GLOB, LONG_SDPA, sdpa_kernel
from run_app_fix import generate, generate_think

OUT = "results/convoh_{}.jsonl"
ROOT = CONVO_GLOB.split(os.sep + "core_benchmark")[0]
TASKS = ("c2x", "c3", "c4", "c5", "c6")
N = {"c2x": 300, "c3": 150, "c4": 150, "c5": 150, "c6": 150}


def heldout_pids():
    files = sorted(glob.glob(CONVO_GLOB))
    return [json.load(open(f, encoding="utf-8"))["evidence_items"][0]["personId"] for f in files], files


def eligible(ev):
    texts = [m["text"] for m in ev["message_evidences"]]
    return all(any(m["text"] == t for m in c["messages"]) for c, t in zip(ev["conversations"], texts))


def pool(task):
    pids, files2 = heldout_pids()
    held = set(pids[50:100])
    out = []
    if task == "c2x":
        for pi in range(50, 100):
            for k, ev in enumerate(json.load(open(files2[pi], encoding="utf-8"))["evidence_items"]):
                if k >= 4 and eligible(ev):
                    out.append((f"c2x-{pi}-{k}", ev))
    else:
        k_ev = int(task[1])
        for f in sorted(glob.glob(os.path.join(ROOT, "core_benchmark", "evidence_questions", "changing_evidence",
                                               f"{k_ev}_evidence", "*.json"))):
            for j, ev in enumerate(json.load(open(f, encoding="utf-8"))["evidence_items"]):
                if ev["personId"] in held and eligible(ev):
                    out.append((f"{task}-{os.path.basename(f)[:-5]}-{j}", ev))
    random.Random(f"convoh-{task}").shuffle(out)
    return out[:N[task]]


def items(tasks=TASKS):
    out = []
    for task in tasks:
        for iid, ev in pool(task):
            texts = [m["text"] for m in ev["message_evidences"]]
            convs = ev["conversations"]
            r = random.Random(f"dates-{iid}")
            days = sorted(r.sample(range(0, 330), len(convs)))                      # Jan 1 .. late Nov 2024
            dates = [(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in days]
            for cond, order in (("chrono", list(range(len(convs)))), ("rev", list(range(len(convs)))[::-1])):
                blocks = [f"### Conversation (date: {dates[o]})\n" + "\n".join(f"{m['speaker']}: {m['text']}" for m in convs[o]["messages"])
                          for o in order]
                user = ("Here are records of your past conversations with the user.\n\n" + "\n\n".join(blocks) +
                        f"\n\nCurrent date: 2024-12-01\nBased on the information above, answer the user's question in one short "
                        f"sentence.\nQuestion: {ev['question']}")
                out.append(dict(task=task, id=iid, person=ev["personId"], k=len(convs), cond=cond, user=user,
                                q=ev["question"], ref=str(ev["answer"]), ev=texts, dates=dates))
    return out


def run(args):
    its = [x for x in items() if not args.tasks or x["task"] in args.tasks.split(",")]
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(out_tag(name, args.think)), key=lambda r: (r["id"], r["cond"]))
        todo = [x for x in its if (x["id"], x["cond"]) not in w.done]
        print(name, "think" if args.think else "", "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["user"], think=args.think)
            n = len(tok(p, add_special_tokens=False).input_ids)
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                if args.think:
                    text = generate_think(tok, model, p, args.budget or 1024, (x["id"], x["cond"]))
                else:
                    text = generate(tok, model, p, args.budget or 64)
            vis = strip_think(text)
            resp = vis.rsplit("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in vis else vis.strip().split("\n")[0]
            w.write({k: v for k, v in x.items() if k not in ("user", "ev")} | dict(n_tok=n, response=resp, full=text,
                    old=x["ev"][-2], new=x["ev"][-1]))
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


@torch.no_grad()
def judge(args):
    tok = model = None
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["id"], r["cond"]))
        todo = [r for r in load_jsonl(path) if (r["id"], r["cond"]) not in w.done]
        print(os.path.basename(path), "todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader("Qwen3-14B")
        for r in tqdm(todo):
            text = mem.JUDGE.format(d0=r["dates"][-2], d1=r["dates"][-1], old=r["old"], new=r["new"], q=r["q"], a=r["ref"],
                                    r=r["response"])
            lp = mem.letter_logprobs(tok, model, text)
            lab = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(r | dict(label=lab, lp=lp, correct=lab == "A", stale=lab == "B"))
        w.close()


def boot(d, n=10000, seed=0):
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1) * 100
    return d.mean() * 100, np.percentile(m, 2.5), np.percentile(m, 97.5)


def stats(args):
    for path in sorted(glob.glob(OUT.format("*").replace(".jsonl", "_judged.jsonl"))):
        df = pd.DataFrame(load_jsonl(path))
        print("==========", os.path.basename(path)[len("convoh_"):-len("_judged.jsonl")])
        for name, g in [(t, df[df.task == t]) for t in TASKS] + [("c3-c6", df[df.task.isin(["c3", "c4", "c5", "c6"])])]:
            piv = g.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna().astype(float)
            if not len(piv):
                continue
            st = g.pivot_table(index="id", columns="cond", values="stale", aggfunc="first").astype(float).mean() * 100
            d, lo, hi = boot((piv.rev - piv.chrono).values)
            print(f"  {name:6s} n={len(piv):4d} | chrono {piv.chrono.mean() * 100:5.1f} rev {piv.rev.mean() * 100:5.1f} | "
                  f"rev - chrono {d:+6.1f} [{lo:+.1f}, {hi:+.1f}] | stale rev {st.get('rev', float('nan')):4.1f}")


def check(args):
    its = items()
    df = pd.DataFrame([{k: v for k, v in x.items() if k not in ("user", "ev", "q", "ref", "dates")} | {"chars": len(x["user"])} for x in its])
    c = df[df.cond == "chrono"]
    print("questions per task", c.task.value_counts().to_dict(), "| persons per task", c.groupby("task").person.nunique().to_dict())
    print("prompt chars median per task", c.groupby("task").chars.median().astype(int).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()
    {"run": run, "judge": judge, "stats": stats, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()

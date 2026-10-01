"""Application 1: chat-assistant memory (LongMemEval, knowledge-update questions).

Each item has two dated sessions; the later one updates a fact stated in the earlier one
(e.g. a count or a preference changed). Memory systems put retrieved sessions/snippets into the
prompt ordered by relevance or newest-first, not necessarily oldest-first. We vary only the
presentation order of the two sessions/snippets and whether dates are shown.

  session level (two full sessions, 4.7k-9.1k tokens; models with >= 32k context)
    S_chrono_dated, S_rev_dated, S_rev_dated_header, S_chrono_nodate, S_rev_nodate
  turn level (retrieved user turns: the two evidence turns + 4 distractor turns from the same sessions)
    T_newlast_dated, T_oldlast_dated, T_oldlast_dated_header, T_newlast_nodate, T_oldlast_nodate

The reader answers in one sentence; a local judge (Qwen3-14B, 4-bit) labels each answer as
A = up-to-date answer, B = the outdated earlier information, C = other.
Only aggregate numbers are printed; no conversation text is printed.
usage: python run_app_memory.py read --models Qwen3-4B,Phi-4-mini
       python run_app_memory.py judge
"""
import argparse
import glob
import json
import os
import random

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, answer_fast, chat_prompt, free_gpu, load_jsonl, load_reader

DATA = r"D:\hf_cache\hub\datasets--xiaowu0162--longmemeval-cleaned\snapshots\98d7416c24c778c2fee6e6f3006e7a073259d48f\longmemeval_oracle.json"
S_CONDS = ["S_chrono_dated", "S_rev_dated", "S_rev_dated_header", "S_chrono_nodate", "S_rev_nodate"]
T_CONDS = ["T_newlast_dated", "T_oldlast_dated", "T_oldlast_dated_header", "T_newlast_nodate", "T_oldlast_nodate"]
N_DISTRACT = 4


def load_items():
    data = json.load(open(DATA, encoding="utf-8"))
    items = []
    for d in data:
        if d["question_type"] != "knowledge-update":
            continue
        ss, ds = d["haystack_sessions"], d["haystack_dates"]
        assert len(ss) == 2 and ds == sorted(ds)
        ev = [[t["content"] for t in s if t.get("has_answer")] for s in ss]
        rng = random.Random(d["question_id"])
        pool = [(i, t["content"]) for i, s in enumerate(ss) for t in s if t["role"] == "user" and not t.get("has_answer")]
        distract = rng.sample(pool, min(N_DISTRACT, len(pool)))
        items.append(dict(qid=d["question_id"], question=d["question"], answer=str(d["answer"]), qdate=d["question_date"],
                          sessions=ss, dates=ds, old_ev=" ".join(ev[0]), new_ev=" ".join(ev[1]),
                          has_ev=bool(ev[0]) and bool(ev[1]), distract=distract, slots=sorted(rng.sample(range(N_DISTRACT + 2), 2))))
    return items


def clip(text, words=60):
    w = text.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def build(it, cond):
    dated = "_dated" in cond
    tail = (f"Current date: {it['qdate']}\n" if dated else "") + \
        f"Based on the information above, answer the user's question in one short sentence.\nQuestion: {it['question']}"
    if cond.startswith("S_"):
        order = [0, 1] if "chrono" in cond else [1, 0]
        note = " (listed from the most recent session to the oldest)" if cond.endswith("header") else ""
        blocks = []
        for i in order:
            body = "\n".join(f"{t['role']}: {t['content']}" for t in it["sessions"][i])
            blocks.append((f"### Session (date: {it['dates'][i]})" if dated else "### Session") + "\n" + body)
        return f"Here are records of your past chat sessions with the user{note}.\n\n" + "\n\n".join(blocks) + "\n\n" + tail
    # turn level: the two evidence turns sit at fixed slots among the distractors; only their order changes
    old = (0, it["old_ev"]); new = (1, it["new_ev"])
    first, second = (old, new) if "newlast" in cond else (new, old)
    mems, di = [], iter(it["distract"])
    for pos in range(N_DISTRACT + 2):
        if pos == it["slots"][0]:
            mems.append(first)
        elif pos == it["slots"][1]:
            mems.append(second)
        else:
            s, txt = next(di, (None, None))
            if s is not None:
                mems.append((s, clip(txt)))
    note = " (sorted by relevance, not by date)" if cond.endswith("header") else ""
    lines = [(f"- ({it['dates'][s]}) The user said: " if dated else "- The user said: ") + f"\"{txt}\"" for s, txt in mems]
    return f"Relevant memories retrieved from past conversations with the user{note}:\n" + "\n".join(lines) + "\n\n" + tail


def read(args):
    items = load_items()[:args.limit]
    jobs = [(it, c) for it in items for c in S_CONDS + T_CONDS if c.startswith("S_") or it["has_ev"]]
    for name in args.models.split(","):
        w = JsonlAppender(f"results/app_memory_{tag(name)}.jsonl", key=lambda r: (r["qid"], r["cond"]))
        todo = [(it, c) for it, c in jobs if (it["qid"], c) not in w.done]
        print(name, "done", len(jobs) - len(todo), "todo", len(todo), flush=True)
        if todo:
            tok, model = load_reader(name)
            ctx = getattr(model.config, "max_position_embeddings", 32768)
            for it, cond in tqdm(todo, desc=name):
                p = chat_prompt(tok, build(it, cond))
                n = len(tok(p, add_special_tokens=False).input_ids)
                if n + 64 > ctx:
                    w.write(dict(model=name, qid=it["qid"], cond=cond, n_tok=n, skipped=True))
                    continue
                raw = answer_fast(tok, model, p, max_new=64)
                w.write(dict(model=name, qid=it["qid"], cond=cond, n_tok=n, skipped=False, response=raw))
                if n > 6000:
                    torch.cuda.empty_cache()
            del tok, model
            free_gpu()
        w.close()


JUDGE = """You are grading an answer to a question about a user's personal information. The information was updated over time.
Earlier message from the user ({d0}): "{old}"
Later message from the user ({d1}): "{new}"

Question: {q}
Up-to-date correct answer: {a}
Response to grade: {r}

Which option best describes the response?
A. It gives the up-to-date correct answer (mentioning older information as well is fine if the final answer is the up-to-date one).
B. It gives the outdated information from the earlier message as the answer.
C. Neither (wrong, unrelated, or says it does not know).
Reply with a single letter: A, B, or C."""


@torch.no_grad()
def letter_logprobs(tok, model, text):
    """Log-probabilities of answering A / B / C (renormalized over the three letters)."""
    letters = [tok(x, add_special_tokens=False).input_ids[0] for x in ("A", "B", "C")]
    ids = tok(chat_prompt(tok, text), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    return model(ids, logits_to_keep=1).logits[0, -1, letters].float().log_softmax(-1).tolist()


@torch.no_grad()
def judge(args):
    items = {it["qid"]: it for it in load_items()}
    plan = []
    for path in sorted(glob.glob("results/app_memory_*.jsonl")):
        if path.endswith("_judged.jsonl"):
            continue
        rows = load_jsonl(path)
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["qid"], r["cond"]))
        todo = [r for r in rows if (r["qid"], r["cond"]) not in w.done]
        print(os.path.basename(path), "judged", len(rows) - len(todo), "todo", len(todo), flush=True)
        plan.append((path, w, todo))
    model = None
    if any(todo for _, _, todo in plan):
        tok, model = load_reader("Qwen3-14B")
    for path, w, todo in plan:
        for r in tqdm(todo, desc=os.path.basename(path)):
            if not r["skipped"]:
                it = items[r["qid"]]
                text = JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"] or "(not available)",
                                    new=it["new_ev"] or "(not available)", q=it["question"], a=it["answer"], r=r["response"])
                lp = letter_logprobs(tok, model, text)
                r["label"] = "ABC"[max(range(3), key=lambda i: lp[i])]
                r["lp"] = lp
                r["has_ev"] = it["has_ev"]
            w.write(r)
        w.close()
        df = pd.DataFrame(w.rows)
        df = df[~df.skipped]
        if df.empty:
            continue
        df["correct"], df["stale"] = df.label.eq("A"), df.label.eq("B")
        print(os.path.basename(path), "\n", (df.groupby("cond")[["correct", "stale"]].mean() * 100).round(1).assign(
            n=df.groupby("cond").size()).to_string(), flush=True)
    if model is not None:
        del model
        free_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "judge", "check"])
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-14B,OLMo-2-13B-Instruct,Phi-4-mini")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "check":     # structural checks only; prints no conversation text
        items = load_items()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        print("items", len(items), "with evidence turns", sum(it["has_ev"] for it in items))
        for cond in S_CONDS + T_CONDS:
            ps = [build(it, cond) for it in items if cond.startswith("S_") or it["has_ev"]]
            lens = sorted(len(tok(p).input_ids) for p in ps)
            print(f"{cond:24s} n={len(ps):3d} tokens min/med/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
        it = next(i for i in items if i["has_ev"])
        a, b = build(it, "T_newlast_dated"), build(it, "T_oldlast_dated")
        print("old/new evidence swap only:", sorted(a.splitlines()) == sorted(b.splitlines()), a != b,
              "| old evidence index in newlast/oldlast:",
              [x.splitlines().index(next(l for l in x.splitlines() if it["old_ev"][:30] in l)) for x in (a, b)])
        s1, s2 = build(it, "S_chrono_dated"), build(it, "S_rev_dated")
        print("session swap keeps content:", sorted(s1.splitlines()) == sorted(s2.splitlines()), s1 != s2)
    elif args.stage == "read":
        read(args)
    else:
        judge(args)


if __name__ == "__main__":
    main()

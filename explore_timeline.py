"""Exploration E1b / E2 (EXPLORE_PLAN.md): does a correct timeline in the model's own turn protect the answer?

LongMemEval knowledge-update items, session level (two dated sessions), with the timeline prompt of run_app_fix.py.
  gen_{chrono,rev}            the model writes its own timeline, then the answer (same as run_app_fix S_*_timeline)
  pre_{chrono,rev}_{fwd,bwd}  a correct timeline (the two evidence statements with their dates) is prefilled in the
                              assistant turn, listed old->new (fwd) or new->old (bwd); the model writes only the answer
Input order (chrono / rev) is crossed with the order of the prefilled list (fwd / bwd). This separates a pull
towards the input's last-presented state from a pull towards the last entry of the model's own list.
Exploratory (LongMemEval was already used); the thresholds are written in EXPLORE_PLAN.md before running.
Only aggregate numbers are printed; no conversation text is printed.
usage: python explore_timeline.py read --models Qwen3-4B,Qwen3-14B --conds pre
       python explore_timeline.py read --models Qwen3-4B@runs/q4-dec-s0/final --conds pre,gen
       python explore_timeline.py judge
       python explore_timeline.py check
"""
import argparse
import glob
import os

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_reader
from run_app_fix import build as fix_build, final_answer, generate
from run_app_memory import JUDGE, letter_logprobs, load_items

CONDS = {"pre": [f"pre_{o}_{t}" for o in ("chrono", "rev") for t in ("fwd", "bwd")],
         "gen": ["gen_chrono", "gen_rev"]}


def user_text(it, cond):
    return fix_build(it, "S_chrono_timeline" if "_chrono" in cond else "S_rev_timeline")


def prefill(it, cond):
    old = f"- ({it['dates'][0]}) The user said: \"{it['old_ev']}\""
    new = f"- ({it['dates'][1]}) The user said: \"{it['new_ev']}\""
    lines = [old, new] if cond.endswith("fwd") else [new, old]
    return "Relevant statements:\n" + "\n".join(lines) + "\n\nAnswer:"


def first_line(text):
    return next((l.strip() for l in text.split("\n") if l.strip()), "")


def jobs_for(items, groups):
    out = []
    for it in items:
        for g in groups:
            for c in CONDS[g]:
                if g == "pre" and not it["has_ev"]:
                    continue
                out.append((it, c))
    return out


def read(args):
    items = load_items()
    for name in args.models.split(","):
        w = JsonlAppender(f"results/explore_timeline_{tag(name)}.jsonl", key=lambda r: (r["qid"], r["cond"]))
        jobs = jobs_for(items, args.conds.split(","))
        todo = [(it, c) for it, c in jobs if (it["qid"], c) not in w.done]
        print(name, "done", len(jobs) - len(todo), "todo", len(todo), flush=True)
        if todo:
            tok, model = load_reader(name)
            for it, cond in tqdm(todo, desc=name):
                if cond.startswith("pre_"):
                    p = chat_prompt(tok, user_text(it, cond), prefix=prefill(it, cond))
                    text = generate(tok, model, p, 48)
                    rec = dict(response=first_line(text))
                else:
                    p = chat_prompt(tok, user_text(it, cond))
                    text = generate(tok, model, p, 320)
                    rec = dict(response=final_answer("S_timeline", text), full=text)
                n = len(tok(p, add_special_tokens=False).input_ids)
                w.write(dict(model=name, qid=it["qid"], cond=cond, n_tok=n, skipped=False, **rec))
                if n > 6000:
                    torch.cuda.empty_cache()
            del tok, model
            free_gpu()
        w.close()


@torch.no_grad()
def judge(args):
    items = {it["qid"]: it for it in load_items()}
    plan = []
    for path in sorted(glob.glob("results/explore_timeline_*.jsonl")):
        if path.endswith("_judged.jsonl"):
            continue
        from app_common import load_jsonl
        rows = load_jsonl(path)
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["qid"], r["cond"]))
        plan.append((path, w, [r for r in rows if (r["qid"], r["cond"]) not in w.done]))
    tok = model = None
    if any(todo for _, _, todo in plan):
        tok, model = load_reader("Qwen3-14B")
    for path, w, todo in plan:
        for r in tqdm(todo, desc=os.path.basename(path)):
            it = items[r["qid"]]
            text = JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"] or "(not available)",
                                new=it["new_ev"] or "(not available)", q=it["question"], a=it["answer"], r=r["response"])
            lp = letter_logprobs(tok, model, text)
            r["label"], r["lp"] = "ABC"[max(range(3), key=lambda i: lp[i])], lp
            w.write(r)
        w.close()
        df = pd.DataFrame(w.rows)
        if len(df):
            print(os.path.basename(path), "\n", (df.groupby("cond").label.value_counts(normalize=True).unstack().fillna(0)
                                                 * 100).round(1).assign(n=df.groupby("cond").size()).to_string(), flush=True)
    if model is not None:
        del tok, model
        free_gpu()


def stats(args):
    """Per model: label shares per condition, then the input-order and list-order contrasts (paired bootstrap)."""
    from analyze_lora import boot
    from app_common import load_jsonl
    frames = {}
    for path in sorted(glob.glob("results/explore_timeline_*_judged.jsonl")):
        name = os.path.basename(path)[len("explore_timeline_"):-len("_judged.jsonl")]
        df = pd.DataFrame(load_jsonl(path))
        df["correct"], df["stale"] = df.label.eq("A"), df.label.eq("B")
        frames[name] = df
        print(name, "\n", (df.groupby("cond").label.value_counts(normalize=True).unstack().fillna(0) * 100).round(1)
              .assign(n=df.groupby("cond").size()).to_string())
        piv = df.pivot_table(index="qid", columns="cond", values="stale", aggfunc="first")
        rows = []
        for label, a, b in [("input order, list old->new", "pre_rev_fwd", "pre_chrono_fwd"),
                            ("input order, list new->old", "pre_rev_bwd", "pre_chrono_bwd"),
                            ("list order, chrono input", "pre_chrono_bwd", "pre_chrono_fwd"),
                            ("list order, rev input", "pre_rev_bwd", "pre_rev_fwd"),
                            ("input order, own timeline", "gen_rev", "gen_chrono")]:
            if a in piv and b in piv:
                m = piv[[a, b]].dropna()
                d, lo, hi, p = boot(m[a].astype(float), m[b].astype(float))
                rows.append(dict(contrast=f"stale: {a} - {b}", meaning=label, n=len(m), diff=d, lo=lo, hi=hi))
        print(pd.DataFrame(rows).round(1).to_string(index=False))
    dec = next((k for k in frames if "+q4-dec" in k), None)
    chr_ = next((k for k in frames if "+q4-chr" in k), None)
    if dec and chr_:
        rows = []
        for cond in sorted(set(frames[dec].cond) & set(frames[chr_].cond)):
            m = frames[dec][frames[dec].cond == cond].merge(frames[chr_][frames[chr_].cond == cond], on="qid")
            d, lo, hi, p = boot(m.correct_x.astype(float), m.correct_y.astype(float))
            rows.append(dict(cond=cond, n=len(m), dec=m.correct_x.mean() * 100, chr=m.correct_y.mean() * 100,
                             diff=d, lo=lo, hi=hi))
        print("decoupled - control, correct %\n", pd.DataFrame(rows).round(1).to_string(index=False))


def check():
    """Structural checks only; prints no conversation text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    items = load_items()
    ev = [it for it in items if it["has_ev"]]
    print("items", len(items), "with both evidence turns", len(ev))
    it = ev[0]
    a, b = user_text(it, "pre_chrono_fwd"), user_text(it, "pre_rev_fwd")
    print("chrono/rev inputs hold the same lines:", sorted(a.splitlines()) == sorted(b.splitlines()), a != b)
    f, g = prefill(it, "pre_rev_fwd"), prefill(it, "pre_rev_bwd")
    print("fwd/bwd prefills hold the same lines:", sorted(f.splitlines()) == sorted(g.splitlines()), f != g)
    print("old date listed first in fwd:", f.index(it["dates"][0]) < f.index(it["dates"][1]),
          "| new date listed first in bwd:", g.index(it["dates"][1]) < g.index(it["dates"][0]))
    lens = sorted(len(tok(chat_prompt(tok, user_text(x, "pre_rev_fwd"), prefix=prefill(x, "pre_rev_fwd"))).input_ids)
                  for x in ev)
    print("prefilled prompt tokens min/med/max:", lens[0], lens[len(lens) // 2], lens[-1])
    for g in CONDS:
        print(g, "jobs per model:", len(jobs_for(items, [g])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "judge", "check", "stats"])
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-14B")
    ap.add_argument("--conds", default="pre", help="comma-separated groups: pre, gen")
    args = ap.parse_args()
    if args.stage == "check":
        check()
    elif args.stage == "read":
        read(args)
    elif args.stage == "stats":
        stats(args)
    else:
        judge(args)


if __name__ == "__main__":
    main()

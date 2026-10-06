"""Format check (EXPLORE_PLAN.md E17 / E18): does the reader follow the answer-format instruction on held-out genres
(letters, ship's log) and held-out instruction phrasings? Synthetic dev data only (data_train/<data>_decoupled_dev.jsonl).
Answer = app_common.answer_text (parse v2; strips a thinking block). Pre-registered checks, per style:
  value    : the answer contains the needed value and has at most (value words + 3) words
  sentence : at least 5 words, one sentence, contains every needed string (both values for a change question)
  dated    : as sentence, and also contains the record's date
  letter   : (E18) the answer is the option letter only, e.g. "(c)" or "c"
  correct  : contains every needed string, whatever the format (accuracy, separate from compliance)
E18 adds modes: every item is asked with thinking on ("think") and off ("direct"); also recorded per row: whether the
VISIBLE reply contains a dated list ("from oldest to newest"), and whether the thought was left unfinished.
usage: python explore_format_eval.py run --models Qwen3-4B@runs/e17-dec/final           (E17: bind2, thinking off)
       python explore_format_eval.py run --data bind3 --modes think,direct --models Qwen3-4B@runs/e18-dec/final | stats
"""
import argparse
import json
import re

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader, answer_text, strip_think
from explore_probe import LONG_SDPA, sdpa_kernel
from run_app_fix import generate

OUT = {"bind2": "results/e17_format_{}.jsonl", "bind3": "results/fmt3_{}.jsonl"}
N = 150
BUDGET = {"direct": 320, "think": 1024}


def items(data):
    return [json.loads(l) for l in open(f"data_train/{data}_decoupled_dev.jsonl", encoding="utf-8")][:N]


def n_sentences(s):
    s = re.sub(r"\d\.\d", "0", s)                                  # decimals ($12.50) are not sentence ends
    return len([p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", s.strip()) if p])


def score(ans, x):
    w = len(ans.split())
    has = all(n.lower() in ans.lower() for n in x["need"])
    if x["style"] == "value":
        ok = has and w <= len(str(x["value"]).split()) + 3
    elif x["style"] == "letter":
        ok = bool(re.fullmatch(r"\(?[a-d]\)?\.?", ans.strip().lower())) and ans.strip("(). ").lower() == x["final"].strip("()")
        has = ans.strip("(). ").lower()[:1] == x["final"].strip("()")
    else:
        ok = has and w >= 5 and n_sentences(ans) == 1
    return dict(comply=ok, correct=has, words=w)


@torch.no_grad()
def run(args):
    its = items(args.data)
    modes = args.modes.split(",")
    for name in args.models.split(","):
        w = JsonlAppender(OUT[args.data].format(tag(name)), key=lambda r: (r["id"], r.get("mode", "direct")))
        todo = [(x, m) for x in its for m in modes if (x["id"], m) not in w.done]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x, m in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["prompt"], think=m == "think")
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                text = generate(tok, model, p, BUDGET[m])
            ans = answer_text(text)
            w.write(dict(id=x["id"], mode=m, qtype=x["qtype"], kind=x.get("kind", "temporal"), style=x["style"],
                         order=x["order"], full=text, response=ans,
                         visible_list="from oldest to newest" in strip_think(text),
                         unfinished=text.lstrip().startswith("<think>") and "</think>" not in text) | score(ans, x))
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


def stats(args):
    for name in args.models.split(","):
        rows = load_jsonl(OUT[args.data].format(tag(name)))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "mode" not in df:
            df["mode"] = "direct"
        print("=" * 10, name, "n =", len(df))
        for m, g in df.groupby("mode"):
            print(f"--- mode {m}: comply {g.comply.mean() * 100:.1f} | correct {g.correct.mean() * 100:.1f}" +
                  (f" | visible list {g.visible_list.mean() * 100:.1f}% | unfinished thought {g.unfinished.mean() * 100:.1f}%"
                   if "visible_list" in g else ""))
            print((g.groupby("style")[["comply", "correct"]].mean() * 100).round(1).assign(n=g.groupby("style").size()).to_string())
            key = "kind" if "kind" in g else "qtype"
            print((g.groupby(key)[["comply", "correct"]].mean() * 100).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats"])
    ap.add_argument("--models", default="Qwen3-4B@runs/e17-dec/final,Qwen3-4B@runs/e13-dec/final")
    ap.add_argument("--data", default="bind2", choices=["bind2", "bind3"])
    ap.add_argument("--modes", default="direct", help="comma list of direct (thinking off) / think")
    args = ap.parse_args()
    {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()

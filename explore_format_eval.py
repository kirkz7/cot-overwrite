"""E17 format check (EXPLORE_PLAN.md): does the reader follow the answer-format instruction on held-out genres
(letters, ship's log) and held-out instruction phrasings? Synthetic dev data only (data_train/bind2_decoupled_dev.jsonl).
Answer = app_common.answer_text (parse v2). Pre-registered checks, per style:
  value    : the answer contains the needed value and has at most (value words + 3) words
  sentence : at least 5 words, one sentence, contains every needed string (both values for a change question)
  dated    : as sentence, and also contains the record's date
  correct  : contains every needed string, whatever the format (accuracy, separate from compliance)
usage: python explore_format_eval.py run --models Qwen3-4B@runs/e17-dec/final,Qwen3-4B@runs/e13-dec/final | stats
"""
import argparse
import json
import re

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader, answer_text
from explore_probe import LONG_SDPA, sdpa_kernel
from run_app_fix import generate

OUT = "results/e17_format_{}.jsonl"
N, BUDGET = 150, 320


def items():
    rows = [json.loads(l) for l in open("data_train/bind2_decoupled_dev.jsonl", encoding="utf-8")][:N]
    return rows


def n_sentences(s):
    s = re.sub(r"\d\.\d", "0", s)                                  # decimals ($12.50) are not sentence ends
    return len([p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", s.strip()) if p])


def score(ans, x):
    w = len(ans.split())
    has = all(n.lower() in ans.lower() for n in x["need"])
    if x["style"] == "value":
        ok = has and w <= len(str(x["value"]).split()) + 3
    else:
        ok = has and w >= 5 and n_sentences(ans) == 1
    return dict(comply=ok, correct=has, words=w)


@torch.no_grad()
def run(args):
    its = items()
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(tag(name)), key=lambda r: r["id"])
        todo = [x for x in its if x["id"] not in w.done]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["prompt"])
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                text = generate(tok, model, p, BUDGET)
            ans = answer_text(text)
            w.write(dict(id=x["id"], qtype=x["qtype"], style=x["style"], order=x["order"], full=text, response=ans) | score(ans, x))
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


def stats(args):
    for name in args.models.split(","):
        rows = load_jsonl(OUT.format(tag(name)))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        print("=" * 10, name, "n =", len(df))
        print((df.groupby("style")[["comply", "correct"]].mean() * 100).round(1).assign(n=df.groupby("style").size()).to_string())
        print((df.groupby("qtype")[["comply", "correct"]].mean() * 100).round(1).to_string())
        print(f"  overall comply {df.comply.mean() * 100:.1f} | correct {df.correct.mean() * 100:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats"])
    ap.add_argument("--models", default="Qwen3-4B@runs/e17-dec/final,Qwen3-4B@runs/e13-dec/final")
    args = ap.parse_args()
    {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()

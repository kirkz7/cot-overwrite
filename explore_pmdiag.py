"""PM-diag (10-04): why does PersonaMem 32k show no order effect? BASE models only, no training (EXPLORE_PLAN.md).
  D1  no-history controls on all 7 question types: "nohist" (persona + question + options) and "bare" (question +
      options only). If a type is answered as well without the sessions, it cannot show an order effect.
  D2  order effect (chrono vs newest-first sessions) on the 5 types not run in step 1, same prompt as explore_longconv.
The with-history numbers for the 2 step-1 types are read from results/longconv_*.jsonl (not re-run).
No conversation text is printed.
usage: python explore_pmdiag.py run --models Qwen3-4B,Phi-4-mini | stats
"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader
from explore_longconv import LONG_SDPA, QTYPES, OUT as LC_OUT, personamem, pm_pred, sdpa_kernel
from run_app_fix import generate

OUT = "results/pmdiag_{}.jsonl"
ALL = ("track_full_preference_evolution", "recalling_the_reasons_behind_previous_updates", "recall_user_shared_facts",
       "recalling_facts_mentioned_by_the_user", "suggest_new_ideas", "generalizing_to_new_scenarios",
       "provide_preference_aligned_recommendations")


def items():
    short = personamem(None, conds=("nohist", "bare"))
    long_ = personamem(tuple(t for t in ALL if t not in QTYPES), conds=("chrono", "rev"))
    return short + long_


@torch.no_grad()
def run(args):
    its = items()
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(tag(name)), key=lambda r: (r["id"], r["cond"]))
        todo = [x for x in its if (x["id"], x["cond"]) not in w.done]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["user"])
            n = len(tok(p, add_special_tokens=False).input_ids)
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                text = generate(tok, model, p, 16)
            pred = pm_pred(text)
            w.write(dict(id=x["id"], cond=x["cond"], qtype=x["qtype"], n_tok=n, full=text, pred=pred,
                         correct=pred == x["gold"]))
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


def stats(args):
    from analyze_lora import boot
    for name in args.models.split(","):
        path = OUT.format(tag(name))
        if not os.path.exists(path):
            continue
        df = pd.DataFrame(load_jsonl(path))
        lc = LC_OUT.format(tag(name))
        if os.path.exists(lc):
            gold = {(x["id"], x["cond"]): x["gold"] for x in personamem()}
            a = pd.DataFrame([r for r in load_jsonl(lc) if r["task"] == "personamem"])
            a["pred"] = [pm_pred(f if isinstance(f, str) else r) for f, r in zip(a.get("full", a.response), a.response)]
            a["correct"] = [p == gold[(i, c)] for p, i, c in zip(a.pred, a.id, a.cond)]
            df = pd.concat([df, a[["id", "cond", "qtype", "n_tok", "pred", "correct"]]], ignore_index=True)
        print("=" * 10, name, "| unparsed", int(df.pred.isna().sum()), "/", len(df))
        tab = (df.pivot_table(index="qtype", columns="cond", values="correct", aggfunc="mean") * 100).round(1)
        tab["n"] = df[df.cond == "bare"].groupby("qtype").size()
        print(tab[[c for c in ("bare", "nohist", "chrono", "rev", "n") if c in tab]].to_string())
        for qt, g in df.groupby("qtype"):
            piv = g.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna()
            if {"chrono", "rev", "nohist"} <= set(piv.columns):
                d1, lo1, hi1, _ = boot(piv.rev.astype(float), piv.chrono.astype(float))
                d2, lo2, hi2, _ = boot(piv.chrono.astype(float), piv.nohist.astype(float))
                print(f"  {qt:46s} rev-chrono {d1:5.1f} [{lo1:5.1f}, {hi1:5.1f}]   chrono-nohist {d2:5.1f} [{lo2:5.1f}, {hi2:5.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B,Phi-4-mini")
    args = ap.parse_args()
    if args.stage == "check":
        its = items()
        print(len(its), pd.Series([x["cond"] for x in its]).value_counts().to_dict())
        print(pd.Series([x["qtype"] for x in its if x["cond"] == "bare"]).value_counts().to_dict())
    else:
        {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()

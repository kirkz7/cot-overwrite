"""E15 (EXPLORE_PLAN.md, 10-04 review): is "later in the text = later in time" a semantic rule or mechanical recency?

Every earlier test asked for the CURRENT value, where a reader that maps text order to time and a reader that favours
whatever sits closest to the answer make the same prediction. Asking for the EARLIEST value separates them:
  semantic rule     : earliest -> the FIRST-presented record   (newest-first input: wrong; chronological: right)
  mechanical recency: earliest -> the LAST-presented record    (newest-first input: right by accident; chronological: wrong)
Design: 150 base samples from the pre-registered dev generator (formats calendar / audit, never in any test; dated;
2-4 records of the queried attribute), each rendered chronologically AND newest first, each asked three questions
(current, earliest with symmetric wording, earliest with the dev set's "first recorded" wording as a secondary check).
Answers are classified by the presentation position of the value they give. Base models only, greedy, answer only.
usage: python explore_order_rule.py run --models Qwen3-4B,Phi-4-mini | stats | check
"""
import argparse
import datetime as dt
import math
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import gen_train_data as g
from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader
from run_app_fix import generate

OUT = "results/e15_rule_{}.jsonl"
N_BASE, SEED, BUDGET = 150, 15015, 24
QUESTIONS = {
    "current": "According to these records, what is {a} at the end of the period they cover?",
    "earliest": "According to these records, what was {a} at the beginning of the period they cover?",
    "earliest_first": "According to these records, what was {a} when it was first recorded?",   # secondary
}


def base_records(rng, wiki):
    """gen_train_data.make_sample's construction with: dev formats, always dated, k = 1-3 updates (2-4 records)."""
    fmt = rng.choice(g.DEV_FORMATS)
    attrs = rng.sample(sorted(g.ATTRS), rng.randint(2, 5))
    target, others = attrs[0], attrs[1:]
    k = rng.randint(1, 3)
    vals = []
    while len(vals) < k + 1:
        v = g.ATTRS[target][1](rng)
        if v not in vals:
            vals.append(v)
    recs = [dict(attr=target, value=v, target=True) for v in vals]
    for o in others:
        recs += [dict(attr=o, value=g.ATTRS[o][1](rng)) for _ in range(rng.randint(1, 3))]
    for r in recs:
        r["tpl"] = rng.choice(g.TEMPLATES)
        r["pre"] = g.wiki_sentences(rng, wiki, rng.randint(8, 30)) if (fmt not in g.TABULAR and rng.random() < 0.3) else ""
    target_tok = rng.uniform(3000, 6000) if rng.random() < 0.25 else math.exp(rng.uniform(math.log(150), math.log(2500)))
    while g.est_tokens(fmt, recs) < target_tok:
        recs.append(g.filler_record(rng, fmt, wiki))
    t_idx = set(rng.sample(range(len(recs)), k + 1))
    rest = [r for r in recs if not r.get("target")]
    rng.shuffle(rest)
    chrono, ti, ri = [], 0, 0
    for i in range(len(recs)):
        if i in t_idx:
            chrono.append(recs[ti]); ti += 1
        else:
            chrono.append(rest[ri]); ri += 1
    t0 = dt.datetime(rng.randint(2015, 2025), rng.randint(1, 12), rng.randint(1, 28), rng.randint(7, 19), rng.randint(0, 59))
    step = rng.choice([dt.timedelta(hours=6), dt.timedelta(days=2), dt.timedelta(days=20), dt.timedelta(days=90)])
    step = step * (12 / max(12, len(chrono)))
    for r in chrono:
        r["date"] = t0
        r["tid"] = rng.randint(1000, 9999)
        t0 = t0 + step * rng.uniform(0.3, 1.7) + dt.timedelta(minutes=rng.randint(1, 59))
    datefmt = rng.choice(g.DATE_FMTS)
    shown_t = [r["date"].strftime(datefmt) for r in chrono if r.get("target")]
    if (step < dt.timedelta(days=1) or len(set(shown_t)) < len(shown_t)) and "%H" not in datefmt:
        datefmt = g.TIME_FMT                                    # target records never share a displayed date
    return fmt, target, k, chrono, datefmt


def items():
    wiki = g.load_wiki()
    rng = random.Random(SEED)
    out = []
    for idx in range(N_BASE):
        fmt, target, k, chrono, datefmt = base_records(rng, wiki)
        tvals = [r["value"] for r in chrono if r.get("target")]          # chronological
        a = g.ATTRS[target][0]
        for order in ("chrono", "reverse"):
            shown = chrono if order == "chrono" else chrono[::-1]
            text = g.render(fmt, shown, True, datefmt, False)
            shown_vals = [r["value"] for r in shown if r.get("target")]   # presentation order
            for qtype, q in QUESTIONS.items():
                gold = tvals[-1] if qtype == "current" else tvals[0]
                prompt = f"{g.INTRO[fmt]}\n\n{text}\n\n{q.format(a=a)} Answer with the value only."
                out.append(dict(id=str(idx), order=order, qtype=qtype, fmt=fmt, k=k, prompt=prompt, gold=gold,
                                shown_vals=shown_vals))
    return out


def norm(s):
    s = re.sub(r"(?<=\d),(?=\d)", "", s.lower())
    return " ".join(re.findall(r"[a-z0-9$.\-]+", s)).strip(" .")


def classify(resp, shown_vals):
    """(presentation position of the value the answer gives: first / last / middle / other, index in shown_vals or None)"""
    r = norm(resp.strip().split("\n")[0])
    vals = [norm(v) for v in shown_vals]
    hits = [i for i, v in enumerate(vals) if r == v]
    if not hits:
        hits = [i for i, v in enumerate(vals) if re.search(r"(?<![\w$])" + re.escape(v) + r"(?![\w])", r)]
    if len(hits) != 1:
        return "other", None
    i = hits[0]
    return ("first" if i == 0 else "last" if i == len(vals) - 1 else "middle"), i


@torch.no_grad()
def run(args):
    its = items()
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(tag(name)), key=lambda r: (r["id"], r["order"], r["qtype"]))
        todo = [x for x in its if (x["id"], x["order"], x["qtype"]) not in w.done]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["prompt"])
            n = len(tok(p, add_special_tokens=False).input_ids)
            text = generate(tok, model, p, BUDGET)
            pos, i = classify(text, x["shown_vals"])
            w.write(dict(id=x["id"], order=x["order"], qtype=x["qtype"], fmt=x["fmt"], k=x["k"], n_tok=n,
                         response=text, pos=pos, correct=i is not None and x["shown_vals"][i] == x["gold"]))
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


def boot_diff(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def stats(args):
    for name in args.models.split(","):
        rows = load_jsonl(OUT.format(tag(name)))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        print("=" * 12, name, "n =", len(df))
        tab = (pd.crosstab([df.qtype, df.order], df.pos, normalize="index") * 100).round(1)
        tab["acc"] = (df.groupby(["qtype", "order"]).correct.mean() * 100).round(1)
        tab["n"] = df.groupby(["qtype", "order"]).size()
        print(tab.to_string())
        d = {}
        for (qt, od), h in df.groupby(["qtype", "order"]):
            d[(qt, od)] = boot_diff(h.pos == "last", h.pos == "first")
            print(f"  {qt:15s} {od:8s} last - first = {d[(qt, od)][0]:+6.1f} [{d[(qt, od)][1]:+6.1f}, {d[(qt, od)][2]:+6.1f}]")
        cur, ear = d.get(("current", "reverse")), d.get(("earliest", "reverse"))
        if cur and ear:
            if cur[0] >= 20 and ear[0] <= -20:
                verdict = "SEMANTIC RULE (later text read as later time)"
            elif cur[0] >= 20 and ear[0] >= 20:
                verdict = "MECHANICAL RECENCY (last-presented wins for both questions)"
            else:
                verdict = "MIXED / UNDETERMINED"
            chk = d.get(("earliest", "chrono"))
            consistent = chk is None or (verdict.startswith("SEMANTIC") and chk[0] < 0) or \
                (verdict.startswith("MECHANICAL") and chk[0] > 0) or verdict.startswith("MIXED")
            print("  pre-registered verdict (newest-first input):", verdict,
                  "| consistency check (chronological, earliest):", "ok" if consistent else "CONTRADICTS -> undetermined")


def check(args):
    from transformers import AutoTokenizer
    from app_common import MODELS
    tok = AutoTokenizer.from_pretrained(MODELS["Qwen3-4B"][0])
    its = items()
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "prompt"} | {"n_tok": len(tok(x["prompt"]).input_ids)} for x in its])
    print(len(df), "prompts;", df.groupby(["qtype", "order"]).size().to_dict())
    print("tokens min/med/max", df.n_tok.min(), int(df.n_tok.median()), df.n_tok.max(), "| fmt", df.fmt.value_counts().to_dict(),
          "| records per item", df.shown_vals.map(len).value_counts().to_dict())
    # self-test of the classifier on the gold answers
    ok = all(classify(x["gold"], x["shown_vals"])[0] == ({"current": "last", "earliest": "first", "earliest_first": "first"}[x["qtype"]]
             if x["order"] == "chrono" else {"current": "first", "earliest": "last", "earliest_first": "last"}[x["qtype"]]) for x in its)
    print("classifier maps every gold answer to the expected position:", ok)
    print("example (our synthetic data):\n" + its[0]["prompt"][:700])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B,Phi-4-mini")
    args = ap.parse_args()
    {"run": run, "stats": stats, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()
